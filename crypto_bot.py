"""
AI Crypto Trading Assistant - Telegram Bot
==========================================
Dependencies: see requirements.txt
"""

import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
import ccxt
from groq import Groq
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    format="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
EXCHANGE_ID    = os.getenv("EXCHANGE", "binance")   # or "coinbasepro"

# ── Exchange & AI clients ──────────────────────────────────────────────────────
exchange: ccxt.Exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
groq_client = Groq(api_key=GROQ_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_symbol(raw: str) -> str:
    """Normalise 'btc', 'BTC', 'BTCUSDT' → 'BTC/USDT'."""
    raw = raw.upper().strip()
    if "/" not in raw:
        if raw.endswith("USDT"):
            raw = raw[:-4] + "/USDT"
        elif raw.endswith("USD"):
            raw = raw[:-3] + "/USD"
        else:
            raw = raw + "/USDT"
    return raw


def _fetch_ticker(symbol: str) -> dict:
    return exchange.fetch_ticker(symbol)


def _fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 24) -> list:
    """Returns last `limit` candles as list of [ts, O, H, L, C, V]."""
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def _ohlcv_to_text(ohlcv: list) -> str:
    """Convert candles to a compact text table for the AI prompt."""
    lines = ["Timestamp            | Open      | High      | Low       | Close     | Volume"]
    lines.append("-" * 85)
    for candle in ohlcv:
        ts = datetime.utcfromtimestamp(candle[0] / 1000).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{ts} | {candle[1]:>9.4f} | {candle[2]:>9.4f} | {candle[3]:>9.4f} | {candle[4]:>9.4f} | {candle[5]:>12.2f}")
    return "\n".join(lines)


def _ai_analyze(symbol: str, ohlcv_text: str) -> str:
    """Send OHLCV data to Groq / Llama-3 and return a structured analysis."""
    system_prompt = (
        "You are a Senior Crypto Quantitative Analyst. "
        "You receive raw OHLCV candle data and produce a precise, structured trading analysis. "
        "Always respond in the exact JSON format requested — no extra text outside the JSON block."
    )
    user_prompt = f"""Analyze the following 24-hour hourly OHLCV data for {symbol} and return ONLY a JSON object with these keys:

{{
  "sentiment": "Bullish | Bearish | Neutral",
  "sentiment_score": <integer 0-100, 50=neutral>,
  "trend": "Uptrend | Downtrend | Sideways",
  "market_structure": "<HH/HL or LH/LL observation>",
  "key_levels": {{"support": <price>, "resistance": <price>}},
  "fibonacci": {{"swing_high": <price>, "swing_low": <price>, "golden_pocket_low": <0.618 level>, "golden_pocket_high": <0.65 level>, "current_in_golden_pocket": true | false}},
  "trade_setup": {{
    "signal": "Long | Short | No Trade",
    "entry_zone": "<low>-<high>",
    "take_profit": "<price>",
    "stop_loss": "<price>",
    "risk_reward": "<ratio e.g. 1:3.2>"
  }},
  "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"]
}}

OHLCV Data:
{ohlcv_text}
"""
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _parse_analysis(raw: str) -> dict | None:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _build_trade_alert(symbol: str, data: dict) -> str:
    """Format parsed AI JSON into a Telegram-ready Trade Alert message."""
    ts   = data.get("trade_setup", {})
    fib  = data.get("fibonacci", {})
    kl   = data.get("key_levels", {})
    sent = data.get("sentiment", "N/A")
    score= data.get("sentiment_score", "N/A")
    trend= data.get("trend", "N/A")
    ms   = data.get("market_structure", "N/A")
    reasons = data.get("reasons", [])

    signal = ts.get("signal", "N/A")
    emoji_signal = "🟢 LONG" if signal == "Long" else ("🔴 SHORT" if signal == "Short" else "⚪ NO TRADE")
    emoji_sent   = "📈" if sent == "Bullish" else ("📉" if sent == "Bearish" else "➡️")

    gp_tag = "✅ YES — Price in Golden Pocket!" if fib.get("current_in_golden_pocket") else "❌ No"

    reasons_text = "\n".join(f"  • {r}" for r in reasons) if reasons else "  N/A"

    rr = ts.get("risk_reward", "N/A")

    alert = f"""
╔══════════════════════════════╗
      🤖 *AI TRADE ALERT*
╚══════════════════════════════╝

*Symbol:* `{symbol}`
*Generated:* {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

━━━━━ 📊 MARKET STRUCTURE ━━━━━
*Trend:* {trend}
*Structure:* {ms}
*Sentiment:* {emoji_sent} {sent} ({score}/100)

━━━━━ 🌀 FIBONACCI ANALYSIS ━━━━━
*Swing High:* `{fib.get("swing_high", "N/A")}`
*Swing Low:*  `{fib.get("swing_low", "N/A")}`
*Golden Pocket (0.618–0.65):*
  ↳ {fib.get("golden_pocket_low", "N/A")} – {fib.get("golden_pocket_high", "N/A")}
*Price in Golden Pocket?* {gp_tag}

━━━━━ 🎯 TRADE SETUP ━━━━━━━━━━
*Signal:*      {emoji_signal}
*Entry Zone:* `{ts.get("entry_zone", "N/A")}`
*Take Profit:* `{ts.get("take_profit", "N/A")}` 🎯
*Stop Loss:*   `{ts.get("stop_loss", "N/A")}` 🛑
*R:R Ratio:*   `{rr}` {"✅" if rr != "N/A" else ""}

━━━━━ 🔑 KEY LEVELS ━━━━━━━━━━━
*Support:*    `{kl.get("support", "N/A")}`
*Resistance:* `{kl.get("resistance", "N/A")}`

━━━━━ 💡 REASONS ━━━━━━━━━━━━━
{reasons_text}

⚠️ _This is AI-generated analysis, not financial advice. Always DYOR._
"""
    return alert.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Command Handlers
# ══════════════════════════════════════════════════════════════════════════════

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "👋 *Welcome to AI Crypto Trading Assistant!*\n\n"
        "Available commands:\n"
        "  `/price BTC` — Live price & 24h stats\n"
        "  `/analyze ETH` — Full AI technical analysis\n"
        "  `/trade BTC` — Generate trade setup (requires confirmation)\n"
        "  `/help` — Show this message\n\n"
        "_Powered by Groq · Llama-3 · ccxt_"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def price_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Usage: /price BTC"""
    if not context.args:
        await update.message.reply_text("Usage: `/price BTC`", parse_mode="Markdown")
        return

    symbol = _fmt_symbol(context.args[0])
    await update.message.reply_text(f"⏳ Fetching price for `{symbol}`...", parse_mode="Markdown")

    try:
        ticker = _fetch_ticker(symbol)
        change = ticker.get("percentage", 0) or 0
        change_emoji = "🟢" if change >= 0 else "🔴"

        msg = (
            f"💰 *{symbol} — Live Price*\n\n"
            f"  *Price:*       `{ticker['last']:,.4f} USDT`\n"
            f"  *24h Change:*  {change_emoji} `{change:+.2f}%`\n"
            f"  *24h High:*    `{ticker['high']:,.4f}`\n"
            f"  *24h Low:*     `{ticker['low']:,.4f}`\n"
            f"  *24h Volume:*  `{ticker['quoteVolume']:,.2f} USDT`\n"
            f"  *Bid / Ask:*   `{ticker['bid']:,.4f}` / `{ticker['ask']:,.4f}`\n\n"
            f"_Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    except ccxt.BadSymbol:
        await update.message.reply_text(f"❌ Symbol `{symbol}` not found on {EXCHANGE_ID}.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"price_cmd error: {e}")
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode="Markdown")


async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Usage: /analyze ETH"""
    if not context.args:
        await update.message.reply_text("Usage: `/analyze ETH`", parse_mode="Markdown")
        return

    symbol = _fmt_symbol(context.args[0])
    msg = await update.message.reply_text(
        f"🔍 Fetching 24h OHLCV for `{symbol}` and running AI analysis...\n_This may take 10–20 seconds._",
        parse_mode="Markdown",
    )

    try:
        ohlcv      = _fetch_ohlcv(symbol, timeframe="1h", limit=24)
        ohlcv_text = _ohlcv_to_text(ohlcv)
        raw_result = _ai_analyze(symbol, ohlcv_text)
        parsed     = _parse_analysis(raw_result)

        if parsed:
            alert = _build_trade_alert(symbol, parsed)
        else:
            alert = f"⚠️ AI returned unstructured output:\n\n{raw_result}"

        await msg.delete()
        await update.message.reply_text(alert, parse_mode="Markdown")

    except ccxt.BadSymbol:
        await msg.delete()
        await update.message.reply_text(f"❌ Symbol `{symbol}` not found on {EXCHANGE_ID}.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"analyze_cmd error: {e}")
        await msg.delete()
        await update.message.reply_text(f"❌ Error during analysis: `{e}`", parse_mode="Markdown")


async def trade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Usage: /trade BTC  — Shows a confirmation button before proceeding."""
    if not context.args:
        await update.message.reply_text("Usage: `/trade BTC`", parse_mode="Markdown")
        return

    symbol = _fmt_symbol(context.args[0])

    warning = (
        f"⚠️ *Trade Confirmation Required*\n\n"
        f"You are about to execute a trade for `{symbol}`.\n\n"
        f"*This bot does NOT auto-execute trades.* Pressing *Confirm* will "
        f"run a full AI analysis and show you the suggested setup. "
        f"*You must place the order manually on your exchange.*\n\n"
        f"Do you want to proceed?"
    )
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Confirm — Show Setup", callback_data=f"trade_confirm:{symbol}"),
            InlineKeyboardButton("❌ Cancel",              callback_data="trade_cancel"),
        ]
    ])
    await update.message.reply_text(warning, parse_mode="Markdown", reply_markup=keyboard)


async def trade_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == "trade_cancel":
        await query.edit_message_text("❌ Trade cancelled.")
        return

    if query.data.startswith("trade_confirm:"):
        symbol = query.data.split(":", 1)[1]
        await query.edit_message_text(
            f"⏳ Running full AI analysis for `{symbol}`...\n_Please wait 10–20 seconds._",
            parse_mode="Markdown",
        )
        try:
            ohlcv      = _fetch_ohlcv(symbol, timeframe="1h", limit=24)
            ohlcv_text = _ohlcv_to_text(ohlcv)
            raw_result = _ai_analyze(symbol, ohlcv_text)
            parsed     = _parse_analysis(raw_result)

            if parsed:
                alert = _build_trade_alert(symbol, parsed)
                alert += "\n\n🔒 *Remember: Place this order MANUALLY on your exchange.*"
            else:
                alert = f"⚠️ AI returned unstructured output:\n\n{raw_result}"

            await query.edit_message_text(alert, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"trade_callback error: {e}")
            await query.edit_message_text(f"❌ Error: `{e}`", parse_mode="Markdown")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("help",    help_cmd))
    app.add_handler(CommandHandler("price",   price_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("trade",   trade_cmd))
    app.add_handler(CallbackQueryHandler(trade_callback, pattern=r"^trade_"))

    logger.info("🤖 Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
