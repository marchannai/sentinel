import os, logging, json, requests
from datetime import datetime
from dotenv import load_dotenv
import ccxt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

load_dotenv()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXCHANGE_ID = os.getenv("EXCHANGE", "binance")
exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})

def _groq_chat(system, user):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0.2, "max_tokens": 1024},
        timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def _fmt_symbol(raw):
    raw = raw.upper().strip()
    if "/" not in raw:
        raw = (raw[:-4]+"/USDT") if raw.endswith("USDT") else (raw[:-3]+"/USD") if raw.endswith("USD") else (raw+"/USDT")
    return raw

def _ohlcv_to_text(ohlcv):
    lines = ["Timestamp            | Open      | High      | Low       | Close     | Volume", "-"*85]
    for c in ohlcv:
        ts = datetime.utcfromtimestamp(c[0]/1000).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{ts} | {c[1]:>9.4f} | {c[2]:>9.4f} | {c[3]:>9.4f} | {c[4]:>9.4f} | {c[5]:>12.2f}")
    return "\n".join(lines)

def _ai_analyze(symbol, ohlcv_text):
    system = "You are a Senior Crypto Quantitative Analyst. Return ONLY valid JSON, no extra text."
    user = f"""Analyze this 24h OHLCV data for {symbol}. Return ONLY this JSON:
{{"sentiment":"Bullish|Bearish|Neutral","sentiment_score":50,"trend":"Uptrend|Downtrend|Sideways","market_structure":"observation","key_levels":{{"support":0,"resistance":0}},"fibonacci":{{"swing_high":0,"swing_low":0,"golden_pocket_low":0,"golden_pocket_high":0,"current_in_golden_pocket":false}},"trade_setup":{{"signal":"Long|Short|No Trade","entry_zone":"0-0","take_profit":"0","stop_loss":"0","risk_reward":"1:2"}},"reasons":["r1","r2","r3"]}}
Data:\n{ohlcv_text}"""
    return _groq_chat(system, user)

def _parse(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try: return json.loads(raw)
    except: return None

def _alert(symbol, d):
    ts = d.get("trade_setup",{}); fib = d.get("fibonacci",{}); kl = d.get("key_levels",{})
    sig = ts.get("signal","N/A")
    esig = "🟢 LONG" if sig=="Long" else "🔴 SHORT" if sig=="Short" else "⚪ NO TRADE"
    esent = "📈" if d.get("sentiment")=="Bullish" else "📉" if d.get("sentiment")=="Bearish" else "➡️"
    gp = "✅ YES" if fib.get("current_in_golden_pocket") else "❌ No"
    reasons = "\n".join(f"  • {r}" for r in d.get("reasons",[]))
    return f"""╔══════════════════════════════╗
      🤖 *AI TRADE ALERT*
╚══════════════════════════════╝
*Symbol:* `{symbol}`
*Generated:* {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
━━━━━ 📊 MARKET STRUCTURE ━━━━━
*Trend:* {d.get("trend","N/A")}
*Structure:* {d.get("market_structure","N/A")}
*Sentiment:* {esent} {d.get("sentiment","N/A")} ({d.get("sentiment_score","N/A")}/100)
━━━━━ 🌀 FIBONACCI ━━━━━━━━━━━
*Swing High:* `{fib.get("swing_high","N/A")}`
*Swing Low:*  `{fib.get("swing_low","N/A")}`
*Golden Pocket:* {fib.get("golden_pocket_low","N/A")} – {fib.get("golden_pocket_high","N/A")}
*In Golden Pocket?* {gp}
━━━━━ 🎯 TRADE SETUP ━━━━━━━━━
*Signal:*      {esig}
*Entry Zone:* `{ts.get("entry_zone","N/A")}`
*Take Profit:* `{ts.get("take_profit","N/A")}` 🎯
*Stop Loss:*   `{ts.get("stop_loss","N/A")}` 🛑
*R:R Ratio:*   `{ts.get("risk_reward","N/A")}`
━━━━━ 🔑 KEY LEVELS ━━━━━━━━━━
*Support:*    `{kl.get("support","N/A")}`
*Resistance:* `{kl.get("resistance","N/A")}`
━━━━━ 💡 REASONS ━━━━━━━━━━━━
{reasons}
⚠️ _Not financial advice. Always DYOR._""".strip()

async def start(update, context):
    await update.message.reply_text("👋 *AI Crypto Trading Assistant*\n\n`/price BTC` — Live price\n`/analyze ETH` — AI analysis\n`/trade BTC` — Trade setup\n`/help` — Help", parse_mode="Markdown")

async def help_cmd(update, context): await start(update, context)

async def price_cmd(update, context):
    if not context.args:
        await update.message.reply_text("Usage: `/price BTC`", parse_mode="Markdown"); return
    symbol = _fmt_symbol(context.args[0])
    await update.message.reply_text(f"⏳ Fetching `{symbol}`...", parse_mode="Markdown")
    try:
        t = exchange.fetch_ticker(symbol)
        c = t.get("percentage",0) or 0
        await update.message.reply_text(f"💰 *{symbol}*\n\n*Price:* `{t['last']:,.4f}`\n*24h:* {'🟢' if c>=0 else '🔴'} `{c:+.2f}%`\n*High:* `{t['high']:,.4f}`\n*Low:* `{t['low']:,.4f}`\n*Volume:* `{t['quoteVolume']:,.2f}`", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode="Markdown")

async def analyze_cmd(update, context):
    if not context.args:
        await update.message.reply_text("Usage: `/analyze ETH`", parse_mode="Markdown"); return
    symbol = _fmt_symbol(context.args[0])
    msg = await update.message.reply_text(f"🔍 Analyzing `{symbol}`...", parse_mode="Markdown")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=24)
        raw = _ai_analyze(symbol, _ohlcv_to_text(ohlcv))
        parsed = _parse(raw)
        await msg.delete()
        await update.message.reply_text(_alert(symbol, parsed) if parsed else f"⚠️ Raw:\n{raw}", parse_mode="Markdown")
    except Exception as e:
        await msg.delete(); await update.message.reply_text(f"❌ Error: `{e}`", parse_mode="Markdown")

async def trade_cmd(update, context):
    if not context.args:
        await update.message.reply_text("Usage: `/trade BTC`", parse_mode="Markdown"); return
    symbol = _fmt_symbol(context.args[0])
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Confirm", callback_data=f"trade_confirm:{symbol}"), InlineKeyboardButton("❌ Cancel", callback_data="trade_cancel")]])
    await update.message.reply_text(f"⚠️ *Confirm trade setup for `{symbol}`?*\n_Bot does NOT auto-execute._", parse_mode="Markdown", reply_markup=kb)

async def trade_callback(update, context):
    q = update.callback_query; await q.answer()
    if q.data == "trade_cancel":
        await q.edit_message_text("❌ Cancelled."); return
    if q.data.startswith("trade_confirm:"):
        symbol = q.data.split(":",1)[1]
        await q.edit_message_text(f"⏳ Analyzing `{symbol}`...", parse_mode="Markdown")
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=24)
            raw = _ai_analyze(symbol, _ohlcv_to_text(ohlcv))
            parsed = _parse(raw)
            result = (_alert(symbol, parsed)+"\n\n🔒 *Place order MANUALLY on your exchange.*") if parsed else f"⚠️ {raw}"
            await q.edit_message_text(result, parse_mode="Markdown")
        except Exception as e:
            await q.edit_message_text(f"❌ Error: `{e}`", parse_mode="Markdown")

def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    if not GROQ_API_KEY: raise RuntimeError("GROQ_API_KEY not set")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("price", price_cmd))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("trade", trade_cmd))
    app.add_handler(CallbackQueryHandler(trade_callback, pattern=r"^trade_"))
    logger.info("🤖 Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
