import os, logging, json, requests, asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import ccxt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import TelegramError

load_dotenv()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
EXCHANGE_ID    = os.getenv("EXCHANGE", "binance")
exchange       = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})

# ══════════════════════════════════════════════════════════════════════════════
# Safe send — splits long messages automatically
# ══════════════════════════════════════════════════════════════════════════════
async def safe_send(update, text, parse_mode="Markdown"):
    """Send message, splitting if over Telegram's 4096 char limit."""
    try:
        if len(text) <= 4096:
            await update.message.reply_text(text, parse_mode=parse_mode)
        else:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for chunk in chunks:
                await update.message.reply_text(chunk, parse_mode=parse_mode)
    except TelegramError as e:
        logger.error(f"safe_send TelegramError: {e}")
        try:
            await update.message.reply_text(text, parse_mode=None)
        except Exception as e2:
            logger.error(f"safe_send fallback failed: {e2}")

# ══════════════════════════════════════════════════════════════════════════════
# Persistent Storage
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _path(name): return DATA_DIR / f"{name}.json"
def load_db(name):
    try:
        p = _path(name)
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception as e:
        logger.error(f"load_db {name}: {e}"); return {}
def save_db(name, data):
    try: _path(name).write_text(json.dumps(data, default=str, indent=2))
    except Exception as e: logger.error(f"save_db {name}: {e}")

def uid(u): return str(u)
def get_watchlist(u):    return load_db("watchlists").get(uid(u), [])
def save_watchlist(u,v): d=load_db("watchlists"); d[uid(u)]=v; save_db("watchlists",d)
def get_alerts(u):       return load_db("alert_levels").get(uid(u), {})
def save_alerts(u,v):    d=load_db("alert_levels"); d[uid(u)]=v; save_db("alert_levels",d)
def get_trades(u):       return load_db("paper_trades").get(uid(u), [])
def save_trades(u,v):    d=load_db("paper_trades"); d[uid(u)]=v; save_db("paper_trades",d)
def get_portfolio(u):    return load_db("portfolios").get(uid(u), [])
def save_portfolio(u,v): d=load_db("portfolios"); d[uid(u)]=v; save_db("portfolios",d)
def get_settings(u):     return load_db("settings").get(uid(u), {"risk":"moderate","notifications":True})
def save_settings(u,v):  d=load_db("settings"); d[uid(u)]=v; save_db("settings",d)

# ══════════════════════════════════════════════════════════════════════════════
# Indicators
# ══════════════════════════════════════════════════════════════════════════════
def calc_rsi(closes, period=14):
    c=np.array(closes,dtype=float); d=np.diff(c)
    g=np.where(d>0,d,0.0); l=np.where(d<0,-d,0.0)
    ag=np.mean(g[:period]); al=np.mean(l[:period])
    for i in range(period,len(g)):
        ag=(ag*(period-1)+g[i])/period; al=(al*(period-1)+l[i])/period
    return 100.0 if al==0 else round(100-(100/(1+ag/al)),2)

def calc_macd(closes):
    c=np.array(closes,dtype=float)
    def ema(d,s):
        k=2/(s+1); r=[d[0]]
        for p in d[1:]: r.append(p*k+r[-1]*(1-k))
        return np.array(r)
    ml=ema(c,12)-ema(c,26); sl=ema(ml,9)
    return round(float(ml[-1]),4), round(float(sl[-1]),4), round(float((ml-sl)[-1]),4)

def calc_bb(closes):
    a=np.array(closes[-20:],dtype=float); m=np.mean(a); s=np.std(a)
    return round(float(m+2*s),4), round(float(m),4), round(float(m-2*s),4)

def calc_ema(closes, period):
    c=np.array(closes,dtype=float); k=2/(period+1); e=c[0]
    for p in c[1:]: e=p*k+e*(1-k)
    return round(float(e),4)

def calc_stoch(highs, lows, closes):
    h=np.array(highs[-14:],dtype=float); l=np.array(lows[-14:],dtype=float)
    mn=min(l); mx=max(h)
    if mx==mn: return 50.0
    return round(100*(closes[-1]-mn)/(mx-mn),2)

def calc_atr(highs, lows, closes):
    h=np.array(highs,dtype=float); l=np.array(lows,dtype=float); c=np.array(closes,dtype=float)
    tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
    return round(float(np.mean(tr[-14:])),4)

def calc_vwap(ohlcv):
    tp=np.array([(x[2]+x[3]+x[4])/3 for x in ohlcv],dtype=float)
    v=np.array([x[5] for x in ohlcv],dtype=float)
    return round(float(np.sum(tp*v)/np.sum(v)),4)

def detect_patterns(ohlcv):
    if len(ohlcv)<3: return ["None"]
    o=[x[1] for x in ohlcv]; h=[x[2] for x in ohlcv]
    l=[x[3] for x in ohlcv]; c=[x[4] for x in ohlcv]
    pts=[]; body=abs(c[-1]-o[-1]); rng=h[-1]-l[-1]
    if rng>0:
        if body/rng<0.1: pts.append("Doji")
        lw=(o[-1]-l[-1]) if c[-1]>o[-1] else (c[-1]-l[-1])
        uw=(h[-1]-c[-1]) if c[-1]>o[-1] else (h[-1]-o[-1])
        if lw/rng>0.6 and body/rng<0.3: pts.append("Hammer")
        if uw/rng>0.6 and body/rng<0.3: pts.append("Shooting Star")
    if len(c)>=2:
        if c[-2]<o[-2] and c[-1]>o[-1] and c[-1]>o[-2] and o[-1]<c[-2]: pts.append("Bullish Engulfing")
        if c[-2]>o[-2] and c[-1]<o[-1] and c[-1]<o[-2] and o[-1]>c[-2]: pts.append("Bearish Engulfing")
    if len(c)>=3:
        if c[-3]<o[-3] and abs(c[-2]-o[-2])<(h[-2]-l[-2])*0.3 and c[-1]>o[-1]: pts.append("Morning Star")
        if c[-3]>o[-3] and abs(c[-2]-o[-2])<(h[-2]-l[-2])*0.3 and c[-1]<o[-1]: pts.append("Evening Star")
    return pts if pts else ["None"]

def build_ind(ohlcv):
    closes=[c[4] for c in ohlcv]; highs=[c[2] for c in ohlcv]
    lows=[c[3] for c in ohlcv]; volumes=[c[5] for c in ohlcv]
    price=closes[-1]
    rsi=calc_rsi(closes)
    ml,sl_v,hist=calc_macd(closes)
    bbu,bbm,bbl=calc_bb(closes)
    e9=calc_ema(closes,9); e21=calc_ema(closes,21)
    e50=calc_ema(closes,min(50,len(closes)))
    stoch=calc_stoch(highs,lows,closes)
    atr=calc_atr(highs,lows,closes)
    vwap=calc_vwap(ohlcv)
    patterns=detect_patterns(ohlcv)
    sh=max(highs[-50:]) if len(highs)>=50 else max(highs)
    sl2=min(lows[-50:]) if len(lows)>=50 else min(lows)
    fr=sh-sl2
    sup=round(float(np.mean(sorted(lows[-20:])[:3])),4)
    res=round(float(np.mean(sorted(highs[-20:],reverse=True)[:3])),4)
    vols=np.array(volumes,dtype=float); hf=len(vols)//2
    vr=np.mean(vols[hf:])/np.mean(vols[:hf]) if np.mean(vols[:hf])>0 else 1
    return {
        "price":price,"atr":atr,"vwap":vwap,
        "rsi":rsi,"rsi_s":"Overbought" if rsi>=70 else "Oversold" if rsi<=30 else "Neutral",
        "ml":ml,"sl_v":sl_v,"hist":hist,"macd_s":"Bullish" if hist>0 else "Bearish",
        "bbu":bbu,"bbm":bbm,"bbl":bbl,"bb_s":"Overbought" if price>bbu else "Oversold" if price<bbl else "Inside",
        "e9":e9,"e21":e21,"e50":e50,
        "ema_s":"Bullish" if e9>e21 else "Bearish",
        "etrend":"Bullish" if price>e50 else "Bearish",
        "stoch":stoch,"stoch_s":"Overbought" if stoch>=80 else "Oversold" if stoch<=20 else "Neutral",
        "vol":"Rising" if vr>1.1 else "Falling" if vr<0.9 else "Stable",
        "sup":sup,"res":res,"sh":sh,"sl2":sl2,
        "f382":round(sh-fr*0.382,4),"f500":round(sh-fr*0.500,4),
        "f618":round(sh-fr*0.618,4),"f650":round(sh-fr*0.650,4),
        "golden":round(sh-fr*0.650,4)<=price<=round(sh-fr*0.618,4),
        "vwap_s":"Bullish" if price>vwap else "Bearish",
        "patterns":patterns,
    }

# ══════════════════════════════════════════════════════════════════════════════
# External
# ══════════════════════════════════════════════════════════════════════════════
def get_fg():
    try:
        r=requests.get("https://api.alternative.me/fng/?limit=1",timeout=10)
        d=r.json()["data"][0]; v=int(d["value"]); lbl=d["value_classification"]
        e="😱" if v<=25 else "😨" if v<=45 else "😐" if v<=55 else "😄" if v<=75 else "🤑"
        return v,lbl,e
    except Exception as ex:
        logger.error(f"get_fg: {ex}"); return None,"Unknown","❓"

# ══════════════════════════════════════════════════════════════════════════════
# Groq
# ══════════════════════════════════════════════════════════════════════════════
def groq(system, user, max_tokens=1500):
    r=requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"},
        json={"model":"llama-3.3-70b-versatile",
              "messages":[{"role":"system","content":system},{"role":"user","content":user}],
              "temperature":0.2,"max_tokens":max_tokens},
        timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def parse_json(raw):
    raw=raw.strip()
    if raw.startswith("```"):
        raw=raw.split("```")[1]
        if raw.startswith("json"): raw=raw[4:]
    try: return json.loads(raw)
    except: return None

def ai_analyze(symbol, ind, fg=None, risk="moderate"):
    system="You are a Senior Crypto Analyst. Return ONLY valid JSON, no extra text."
    user=f"""Analyze {symbol} (risk:{risk}) return ONLY this JSON:
{{"sentiment":"Bullish|Bearish|Neutral","sentiment_score":50,"trend":"Uptrend|Downtrend|Sideways","trend_strength":"Strong|Moderate|Weak","market_structure":"observation","trade_setup":{{"signal":"Long|Short|No Trade","entry_zone":"low-high","take_profit_1":"p","take_profit_2":"p","take_profit_3":"p","stop_loss":"p","risk_reward":"1:3","timeframe":"Short-term|Medium-term","confidence":"Low|Medium|High"}},"key_confluences":["c1","c2","c3"],"reasons":["RSI reason","MACD reason","BB reason","EMA reason","Volume reason","Fib reason"],"warnings":["w1","w2"]}}
DATA: Price={ind['price']} RSI={ind['rsi']}({ind['rsi_s']}) MACD_hist={ind['hist']}({ind['macd_s']}) BB={ind['bb_s']} EMA={ind['ema_s']} Stoch={ind['stoch']}({ind['stoch_s']}) Volume={ind['vol']} Support={ind['sup']} Resistance={ind['res']} Fib618={ind['f618']} GoldenPocket={ind['golden']} Patterns={ind['patterns']} FearGreed={fg}"""
    return groq(system, user)

def ai_compare(data):
    system="You are a Crypto Analyst. Return ONLY valid JSON."
    entries="\n".join([f"{s}:RSI={d['rsi']},MACD={d['hist']},Trend={d['etrend']},BB={d['bb_s']}" for s,d in data.items()])
    user=f"""Rank coins by trade opportunity. Return ONLY:
{{"ranking":[{{"symbol":"s","score":80,"signal":"Long|Short|No Trade","reason":"brief"}}],"best_opportunity":"symbol","summary":"2 sentences"}}
DATA:\n{entries}"""
    return groq(system, user, max_tokens=600)

def ai_performance(trades):
    wins=len([t for t in trades if t.get("result")=="win"]); total=len(trades)
    system="You are a quant analyst. Return ONLY valid JSON."
    user=f"""Paper trading stats: Total={total} Wins={wins} Losses={total-wins}
Return ONLY: {{"win_rate":"{wins}/{total}","assessment":"2 sentences","improvements":["tip1","tip2"]}}"""
    return groq(system, user, max_tokens=400)

# ══════════════════════════════════════════════════════════════════════════════
# Format helpers
# ══════════════════════════════════════════════════════════════════════════════
def fmt(raw):
    raw=raw.upper().strip()
    if "/" not in raw:
        raw=(raw[:-4]+"/USDT") if raw.endswith("USDT") else (raw[:-3]+"/USD") if raw.endswith("USD") else (raw+"/USDT")
    return raw

def build_alert(symbol, d, ind, fg_val=None, fg_lbl=None, fg_e=""):
    ts=d.get("trade_setup",{}); sig=ts.get("signal","N/A"); conf=ts.get("confidence","N/A")
    esig="🟢 LONG" if sig=="Long" else "🔴 SHORT" if sig=="Short" else "⚪ NO TRADE"
    esent="📈" if d.get("sentiment")=="Bullish" else "📉" if d.get("sentiment")=="Bearish" else "➡️"
    gp="✅ YES" if ind['golden'] else "❌ No"
    ce="🟢" if conf=="High" else "🟡" if conf=="Medium" else "🔴"
    re="🔴" if ind['rsi']>=70 else "🟢" if ind['rsi']<=30 else "⚪"
    reasons="\n".join(f"  - {r}" for r in d.get("reasons",[]))
    confluences="\n".join(f"  + {c}" for c in d.get("key_confluences",[]))
    warnings="\n".join(f"  ! {w}" for w in d.get("warnings",[]))
    fg_line=f"\nFear+Greed: {fg_e} {fg_val}/100 - {fg_lbl}" if fg_val else ""
    return (
        f"AI TRADE ALERT\n"
        f"Symbol: {symbol} | Price: {ind['price']:,.4f}\n"
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}{fg_line}\n\n"
        f"MARKET\nTrend: {d.get('trend','N/A')} ({d.get('trend_strength','N/A')})\n"
        f"Structure: {d.get('market_structure','N/A')}\n"
        f"Sentiment: {esent} {d.get('sentiment','N/A')} ({d.get('sentiment_score','N/A')}/100)\n"
        f"Patterns: {', '.join(ind.get('patterns',['None']))}\n\n"
        f"INDICATORS\n"
        f"RSI: {re} {ind['rsi']} - {ind['rsi_s']}\n"
        f"MACD: {'📈' if ind['hist']>0 else '📉'} {ind['ml']} Hist:{ind['hist']} - {ind['macd_s']}\n"
        f"Stoch: {ind['stoch']} - {ind['stoch_s']}\n"
        f"BB: {ind['bbl']} / {ind['bbm']} / {ind['bbu']} - {ind['bb_s']}\n"
        f"EMA 9/21/50: {ind['e9']} / {ind['e21']} / {ind['e50']} - {ind['ema_s']}\n"
        f"VWAP: {ind['vwap']} - {ind['vwap_s']}\n"
        f"ATR: {ind['atr']} | Volume: {ind['vol']}\n\n"
        f"FIBONACCI\n"
        f"0.382: {ind['f382']} | 0.5: {ind['f500']} | 0.618: {ind['f618']}\n"
        f"Golden Pocket: {ind['f650']}-{ind['f618']} | In Pocket: {gp}\n"
        f"Support: {ind['sup']} | Resistance: {ind['res']}\n\n"
        f"TRADE SETUP\n"
        f"Signal: {esig} | Confidence: {ce} {conf}\n"
        f"Entry: {ts.get('entry_zone','N/A')}\n"
        f"TP1: {ts.get('take_profit_1','N/A')} | TP2: {ts.get('take_profit_2','N/A')} | TP3: {ts.get('take_profit_3','N/A')}\n"
        f"SL: {ts.get('stop_loss','N/A')} | R:R: {ts.get('risk_reward','N/A')}\n\n"
        f"CONFLUENCES\n{confluences}\n\n"
        f"REASONING\n{reasons}\n\n"
        f"WARNINGS\n{warnings}\n\n"
        f"Not financial advice. Always DYOR."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Background monitor
# ══════════════════════════════════════════════════════════════════════════════
async def monitor_alerts(app):
    await asyncio.sleep(60)
    while True:
        try:
            for uid_str,symbols in load_db("watchlists").items():
                s=load_db("settings").get(uid_str,{})
                if not s.get("notifications",True): continue
                for symbol in symbols[:5]:
                    try:
                        ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=100)
                        ind=build_ind(ohlcv); alerts=[]
                        if ind['rsi']>=70: alerts.append(f"RSI Overbought ({ind['rsi']})")
                        if ind['rsi']<=30: alerts.append(f"RSI Oversold ({ind['rsi']})")
                        if ind['golden']:  alerts.append("Price in Golden Pocket!")
                        if ind['stoch']>=80: alerts.append("Stochastic Overbought")
                        if ind['stoch']<=20: alerts.append("Stochastic Oversold")
                        custom=load_db("alert_levels").get(uid_str,{}).get(symbol,{})
                        if custom.get("price_above") and ind['price']>=float(custom["price_above"]):
                            alerts.append(f"Price crossed ABOVE {custom['price_above']}")
                        if custom.get("price_below") and ind['price']<=float(custom["price_below"]):
                            alerts.append(f"Price crossed BELOW {custom['price_below']}")
                        if alerts:
                            msg="Alert: "+symbol+"\n\n"+"\n".join(alerts)+f"\nPrice: {ind['price']:,.4f}"
                            await app.bot.send_message(chat_id=int(uid_str),text=msg)
                    except Exception as e: logger.error(f"Monitor {symbol}: {e}")
        except Exception as e: logger.error(f"Monitor loop: {e}")
        await asyncio.sleep(900)

# ══════════════════════════════════════════════════════════════════════════════
# Commands
# ══════════════════════════════════════════════════════════════════════════════
async def start(update, context):
    text=(
        "AI Crypto Trading Bot v3\n\n"
        "ANALYSIS\n"
        "/price BTC - Live price\n"
        "/analyze BTC - Full AI analysis\n"
        "/indicators BTC - Raw indicator values\n"
        "/multitf BTC - Multi-timeframe analysis\n"
        "/compare BTC ETH SOL - Compare coins\n"
        "/feargreed - Fear and Greed Index\n\n"
        "WATCHLIST\n"
        "/watchlist - View watchlist\n"
        "/addwatch BTC - Add coin\n"
        "/removewatch BTC - Remove coin\n"
        "/setalert BTC 95000 90000 - Price alerts\n"
        "/alerts on or off - Toggle notifications\n"
        "/scan - Scan watchlist for best setup\n\n"
        "TRADING\n"
        "/trade BTC - Trade setup with confirmation\n\n"
        "PAPER TRADING\n"
        "/papertrades - View open trades\n"
        "/closetrade BTC win or loss - Close trade\n"
        "/performance - Performance analysis\n\n"
        "PORTFOLIO\n"
        "/portfolio - Holdings and P&L\n"
        "/addholding BTC 0.5 95000 - Add holding\n"
        "/removeholding BTC - Remove holding\n\n"
        "SETTINGS\n"
        "/settings - View settings\n"
        "/setrisk low or moderate or high\n"
        "/help - This menu"
    )
    await update.message.reply_text(text)

async def help_cmd(update, context): await start(update, context)

async def price_cmd(update, context):
    if not context.args: await update.message.reply_text("Usage: /price BTC"); return
    symbol=fmt(context.args[0])
    try:
        t=exchange.fetch_ticker(symbol); c=t.get("percentage",0) or 0
        text=(f"{symbol}\nPrice: {t['last']:,.4f}\n24h: {c:+.2f}%\n"
              f"High: {t['high']:,.4f} | Low: {t['low']:,.4f}\nVolume: {t['quoteVolume']:,.2f}")
        await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"price_cmd: {e}")
        await update.message.reply_text(f"Error: {e}")

async def feargreed_cmd(update, context):
    try:
        v,lbl,e=get_fg()
        if v:
            bar="█"*(v//10)+"░"*(10-v//10)
            await update.message.reply_text(f"Fear & Greed Index\n\n{e} {v}/100 - {lbl}\n{bar}\n\n0=Extreme Fear, 100=Extreme Greed")
        else:
            await update.message.reply_text("Could not fetch Fear & Greed index. API may be down.")
    except Exception as e:
        logger.error(f"feargreed_cmd: {e}")
        await update.message.reply_text(f"Error: {e}")

async def indicators_cmd(update, context):
    if not context.args: await update.message.reply_text("Usage: /indicators BTC"); return
    symbol=fmt(context.args[0])
    msg=await update.message.reply_text(f"Calculating indicators for {symbol}...")
    try:
        ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200)
        ind=build_ind(ohlcv)
        text=(f"{symbol} Indicators\n\n"
              f"Price: {ind['price']:,.4f} | ATR: {ind['atr']} | VWAP: {ind['vwap']}\n\n"
              f"RSI(14): {ind['rsi']} - {ind['rsi_s']}\n"
              f"MACD: {ind['ml']} | Signal: {ind['sl_v']} | Hist: {ind['hist']}\n"
              f"Stochastic: {ind['stoch']} - {ind['stoch_s']}\n"
              f"BB: {ind['bbl']} / {ind['bbm']} / {ind['bbu']} - {ind['bb_s']}\n"
              f"EMA 9/21/50: {ind['e9']} / {ind['e21']} / {ind['e50']}\n"
              f"EMA Trend: {ind['etrend']} | Cross: {ind['ema_s']}\n"
              f"Support: {ind['sup']} | Resistance: {ind['res']}\n"
              f"Volume: {ind['vol']}\n"
              f"Patterns: {', '.join(ind['patterns'])}")
        await msg.delete()
        await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"indicators_cmd: {e}")
        await msg.delete(); await update.message.reply_text(f"Error: {e}")

async def multitf_cmd(update, context):
    if not context.args: await update.message.reply_text("Usage: /multitf BTC"); return
    symbol=fmt(context.args[0])
    msg=await update.message.reply_text(f"Running multi-timeframe for {symbol}...")
    try:
        lines=[f"{symbol} Multi-Timeframe Analysis\n"]
        bulls=0; total=0
        for tf,limit in [("15m",96),("1h",200),("4h",100),("1d",60)]:
            try:
                ohlcv=exchange.fetch_ohlcv(symbol,timeframe=tf,limit=limit)
                ind=build_ind(ohlcv); total+=1
                trend="Bullish" if ind['etrend']=="Bullish" else "Bearish"
                if trend=="Bullish": bulls+=1
                lines.append(f"{tf}: {trend} | RSI:{ind['rsi']} | MACD:{'Up' if ind['hist']>0 else 'Down'} | {ind['patterns'][0]}")
            except: lines.append(f"{tf}: unavailable")
        overall="BULLISH CONFLUENCE" if bulls>=3 else "BEARISH CONFLUENCE" if (total-bulls)>=3 else "MIXED SIGNALS"
        lines.append(f"\nOverall: {overall} ({bulls}/{total} bullish)")
        await msg.delete(); await update.message.reply_text("\n".join(lines))
    except Exception as e:
        logger.error(f"multitf_cmd: {e}")
        await msg.delete(); await update.message.reply_text(f"Error: {e}")

async def compare_cmd(update, context):
    if len(context.args)<2: await update.message.reply_text("Usage: /compare BTC ETH SOL"); return
    symbols=[fmt(a) for a in context.args[:5]]
    msg=await update.message.reply_text(f"Comparing {', '.join(symbols)}...")
    try:
        data={}
        for s in symbols:
            ohlcv=exchange.fetch_ohlcv(s,timeframe="1h",limit=200)
            data[s]=build_ind(ohlcv)
        raw=ai_compare(data); parsed=parse_json(raw)
        await msg.delete()
        if parsed:
            lines=[f"Coin Comparison\n\nBest: {parsed.get('best_opportunity','N/A')}\n{parsed.get('summary','')}\n\nRankings:"]
            for i,r in enumerate(parsed.get("ranking",[]),1):
                lines.append(f"{i}. {r.get('symbol')} - Score:{r.get('score')}/100 - {r.get('signal')}\n   {r.get('reason','')}")
            await update.message.reply_text("\n".join(lines))
        else:
            await update.message.reply_text(f"AI response:\n{raw[:500]}")
    except Exception as e:
        logger.error(f"compare_cmd: {e}")
        await msg.delete(); await update.message.reply_text(f"Error: {e}")

async def analyze_cmd(update, context):
    if not context.args: await update.message.reply_text("Usage: /analyze BTC"); return
    u=update.effective_user.id; symbol=fmt(context.args[0])
    settings=get_settings(u)
    msg=await update.message.reply_text(f"Analyzing {symbol}...")
    try:
        ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200)
        ind=build_ind(ohlcv); fg_val,fg_lbl,fg_e=get_fg()
        raw=ai_analyze(symbol,ind,fg_val,settings.get("risk","moderate"))
        parsed=parse_json(raw)
        await msg.delete()
        if parsed:
            ts=parsed.get("trade_setup",{})
            if ts.get("signal") in ["Long","Short"]:
                trades=get_trades(u)
                trades.append({"symbol":symbol,"signal":ts["signal"],"entry":ts.get("entry_zone",""),
                    "tp1":ts.get("take_profit_1",""),"sl":ts.get("stop_loss",""),
                    "rr":ts.get("risk_reward",""),"time":datetime.utcnow().isoformat(),
                    "status":"open","result":"pending"})
                save_trades(u,trades)
            await safe_send(update, build_alert(symbol,parsed,ind,fg_val,fg_lbl,fg_e), parse_mode=None)
        else:
            await update.message.reply_text(f"AI returned unexpected format:\n{raw[:500]}")
    except Exception as e:
        logger.error(f"analyze_cmd: {e}")
        await msg.delete(); await update.message.reply_text(f"Error: {e}")

async def trade_cmd(update, context):
    if not context.args: await update.message.reply_text("Usage: /trade BTC"); return
    symbol=fmt(context.args[0])
    kb=InlineKeyboardMarkup([[
        InlineKeyboardButton("Confirm",callback_data=f"tc:{symbol}"),
        InlineKeyboardButton("Cancel",callback_data="tx")]])
    await update.message.reply_text(f"Generate trade setup for {symbol}?\nBot does NOT auto-execute.",reply_markup=kb)

async def trade_callback(update, context):
    q=update.callback_query; await q.answer()
    if q.data=="tx": await q.edit_message_text("Cancelled."); return
    if q.data.startswith("tc:"):
        u=q.from_user.id; symbol=q.data.split(":",1)[1]
        settings=get_settings(u)
        await q.edit_message_text(f"Analyzing {symbol}...")
        try:
            ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200)
            ind=build_ind(ohlcv); fg_val,fg_lbl,fg_e=get_fg()
            raw=ai_analyze(symbol,ind,fg_val,settings.get("risk","moderate"))
            parsed=parse_json(raw)
            if parsed:
                ts=parsed.get("trade_setup",{})
                if ts.get("signal") in ["Long","Short"]:
                    trades=get_trades(u)
                    trades.append({"symbol":symbol,"signal":ts["signal"],"entry":ts.get("entry_zone",""),
                        "tp1":ts.get("take_profit_1",""),"sl":ts.get("stop_loss",""),
                        "rr":ts.get("risk_reward",""),"time":datetime.utcnow().isoformat(),
                        "status":"open","result":"pending"})
                    save_trades(u,trades)
                result=build_alert(symbol,parsed,ind,fg_val,fg_lbl,fg_e)+"\n\nPlace order MANUALLY on your exchange."
                if len(result)>4096: result=result[:4090]+"..."
            else: result=f"AI response:\n{raw[:500]}"
            await q.edit_message_text(result)
        except Exception as e:
            logger.error(f"trade_callback: {e}")
            await q.edit_message_text(f"Error: {e}")

async def scan_cmd(update, context):
    u=update.effective_user.id; wl=get_watchlist(u)
    if not wl: await update.message.reply_text("Watchlist empty. Use /addwatch BTC first."); return
    msg=await update.message.reply_text(f"Scanning {len(wl)} coins...")
    try:
        data={}
        for s in wl[:8]:
            ohlcv=exchange.fetch_ohlcv(s,timeframe="1h",limit=200)
            data[s]=build_ind(ohlcv)
        raw=ai_compare(data); parsed=parse_json(raw)
        await msg.delete()
        if parsed:
            lines=[f"Watchlist Scan\n\nBest: {parsed.get('best_opportunity','N/A')}\n{parsed.get('summary','')}\n\nRankings:"]
            for i,r in enumerate(parsed.get("ranking",[]),1):
                lines.append(f"{i}. {r.get('symbol')} Score:{r.get('score')}/100 - {r.get('signal')}\n   {r.get('reason','')}")
            await update.message.reply_text("\n".join(lines))
        else:
            await update.message.reply_text(f"AI response:\n{raw[:500]}")
    except Exception as e:
        logger.error(f"scan_cmd: {e}")
        await msg.delete(); await update.message.reply_text(f"Error: {e}")

async def watchlist_cmd(update, context):
    u=update.effective_user.id; wl=get_watchlist(u)
    if not wl: await update.message.reply_text("Watchlist empty. Use /addwatch BTC"); return
    lines=["Your Watchlist\n"]
    for s in wl:
        try:
            t=exchange.fetch_ticker(s); c=t.get("percentage",0) or 0
            lines.append(f"{'UP' if c>=0 else 'DN'} {s}: {t['last']:,.4f} ({c:+.2f}%)")
        except: lines.append(f"- {s}: unavailable")
    await update.message.reply_text("\n".join(lines))

async def addwatch_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await update.message.reply_text("Usage: /addwatch BTC"); return
    symbol=fmt(context.args[0]); wl=get_watchlist(u)
    if symbol in wl: await update.message.reply_text(f"{symbol} already in watchlist."); return
    if len(wl)>=10: await update.message.reply_text("Watchlist limit is 10."); return
    wl.append(symbol); save_watchlist(u,wl)
    await update.message.reply_text(f"Added {symbol} ({len(wl)}/10)")

async def removewatch_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await update.message.reply_text("Usage: /removewatch BTC"); return
    symbol=fmt(context.args[0]); wl=get_watchlist(u)
    if symbol in wl:
        wl.remove(symbol); save_watchlist(u,wl)
        await update.message.reply_text(f"Removed {symbol}")
    else:
        await update.message.reply_text(f"{symbol} not in watchlist.")

async def setalert_cmd(update, context):
    u=update.effective_user.id
    if len(context.args)<3: await update.message.reply_text("Usage: /setalert BTC 95000 90000"); return
    try:
        symbol=fmt(context.args[0]); above=float(context.args[1]); below=float(context.args[2])
        al=get_alerts(u); al[symbol]={"price_above":above,"price_below":below}; save_alerts(u,al)
        await update.message.reply_text(f"Alerts set for {symbol}\nAbove: {above:,.4f}\nBelow: {below:,.4f}")
    except ValueError:
        await update.message.reply_text("Invalid prices. Use numbers.")

async def alerts_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await update.message.reply_text("Usage: /alerts on or /alerts off"); return
    s=get_settings(u); s["notifications"]=(context.args[0].lower()=="on"); save_settings(u,s)
    await update.message.reply_text(f"Alerts {'ON' if s['notifications'] else 'OFF'}")

async def papertrades_cmd(update, context):
    u=update.effective_user.id; trades=get_trades(u)
    if not trades: await update.message.reply_text("No paper trades yet. Run /analyze BTC first."); return
    open_t=[t for t in trades if t.get("status")=="open"]
    closed_t=[t for t in trades if t.get("status")=="closed"]
    lines=[f"Paper Trades: {len(open_t)} open, {len(closed_t)} closed\n"]
    for t in open_t[-5:]:
        lines.append(f"{t['signal']} {t['symbol']}\nEntry:{t['entry']} TP1:{t['tp1']} SL:{t['sl']}\nR:R:{t['rr']} | {t['time'][:16]}")
    if not open_t: lines.append("No open trades.")
    await update.message.reply_text("\n".join(lines))

async def closetrade_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await update.message.reply_text("Usage: /closetrade BTC win or loss"); return
    symbol=fmt(context.args[0]); result=context.args[1].lower() if len(context.args)>1 else "pending"
    trades=get_trades(u)
    for t in reversed(trades):
        if t["symbol"]==symbol and t["status"]=="open":
            t["status"]="closed"; t["result"]=result; t["close_time"]=datetime.utcnow().isoformat()
            save_trades(u,trades)
            await update.message.reply_text(f"Closed {symbol} as {result.upper()}"); return
    await update.message.reply_text(f"No open trade for {symbol}.")

async def performance_cmd(update, context):
    u=update.effective_user.id; trades=get_trades(u)
    closed=[t for t in trades if t.get("status")=="closed"]
    if not closed: await update.message.reply_text("No closed trades yet. Use /closetrade BTC win."); return
    wins=len([t for t in closed if t.get("result")=="win"])
    rate=round(wins/len(closed)*100,1)
    msg=await update.message.reply_text("Analyzing performance...")
    try:
        raw=ai_performance(closed); parsed=parse_json(raw)
        await msg.delete()
        lines=[f"Performance Summary\n\nTotal: {len(closed)} | Wins: {wins} | Losses: {len(closed)-wins}\nWin Rate: {rate}%"]
        if parsed:
            lines.append(f"\nAssessment: {parsed.get('assessment','')}")
            for tip in parsed.get("improvements",[]): lines.append(f"- {tip}")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        logger.error(f"performance_cmd: {e}")
        await msg.delete(); await update.message.reply_text(f"Error: {e}")

async def portfolio_cmd(update, context):
    u=update.effective_user.id; pf=get_portfolio(u)
    if not pf: await update.message.reply_text("Portfolio empty. Use /addholding BTC 0.5 95000"); return
    lines=["Your Portfolio\n"]; tv=0; tc=0
    for h in pf:
        try:
            t=exchange.fetch_ticker(h["symbol"]); p=t["last"]
            val=p*h["amount"]; cost=h["buy_price"]*h["amount"]
            pnl=val-cost; pct=(pnl/cost*100) if cost>0 else 0
            tv+=val; tc+=cost
            lines.append(f"{'UP' if pnl>=0 else 'DN'} {h['symbol']} x{h['amount']}\nPrice:{p:,.4f} Value:${val:,.2f} PnL:{pct:+.2f}%")
        except: lines.append(f"- {h['symbol']}: price unavailable")
    tpnl=tv-tc; tpct=(tpnl/tc*100) if tc>0 else 0
    lines.append(f"\nTotal Value: ${tv:,.2f}\nTotal PnL: {tpct:+.2f}%")
    await update.message.reply_text("\n".join(lines))

async def addholding_cmd(update, context):
    u=update.effective_user.id
    if len(context.args)<3: await update.message.reply_text("Usage: /addholding BTC 0.5 95000"); return
    try:
        symbol=fmt(context.args[0]); amount=float(context.args[1]); price=float(context.args[2])
        pf=get_portfolio(u); pf=[h for h in pf if h["symbol"]!=symbol]
        pf.append({"symbol":symbol,"amount":amount,"buy_price":price}); save_portfolio(u,pf)
        await update.message.reply_text(f"Added {amount} {symbol} at ${price:,.4f}\nCost: ${amount*price:,.2f}")
    except ValueError:
        await update.message.reply_text("Invalid values. Use: /addholding BTC 0.5 95000")

async def removeholding_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await update.message.reply_text("Usage: /removeholding BTC"); return
    symbol=fmt(context.args[0]); pf=get_portfolio(u)
    pf=[h for h in pf if h["symbol"]!=symbol]; save_portfolio(u,pf)
    await update.message.reply_text(f"Removed {symbol} from portfolio.")

async def settings_cmd(update, context):
    u=update.effective_user.id; s=get_settings(u)
    await update.message.reply_text(
        f"Your Settings\n\nRisk: {s.get('risk','moderate')}\nNotifications: {'ON' if s.get('notifications',True) else 'OFF'}\n\nChange with:\n/setrisk low or moderate or high\n/alerts on or off")

async def setrisk_cmd(update, context):
    u=update.effective_user.id
    if not context.args or context.args[0].lower() not in ["low","moderate","high"]:
        await update.message.reply_text("Usage: /setrisk low or moderate or high"); return
    risk=context.args[0].lower(); s=get_settings(u); s["risk"]=risk; save_settings(u,s)
    desc={"low":"Conservative - wider stops","moderate":"Balanced - standard","high":"Aggressive - tight entries"}
    await update.message.reply_text(f"Risk set to {risk.upper()}\n{desc[risk]}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    if not GROQ_API_KEY:   raise RuntimeError("GROQ_API_KEY not set")
    app=Application.builder().token(TELEGRAM_TOKEN).build()
    for cmd,fn in [
        ("start",start),("help",help_cmd),("price",price_cmd),
        ("feargreed",feargreed_cmd),("indicators",indicators_cmd),
        ("multitf",multitf_cmd),("compare",compare_cmd),("analyze",analyze_cmd),
        ("trade",trade_cmd),("scan",scan_cmd),
        ("watchlist",watchlist_cmd),("addwatch",addwatch_cmd),
        ("removewatch",removewatch_cmd),("setalert",setalert_cmd),("alerts",alerts_cmd),
        ("papertrades",papertrades_cmd),("closetrade",closetrade_cmd),("performance",performance_cmd),
        ("portfolio",portfolio_cmd),("addholding",addholding_cmd),("removeholding",removeholding_cmd),
        ("settings",settings_cmd),("setrisk",setrisk_cmd),
    ]: app.add_handler(CommandHandler(cmd,fn))
    app.add_handler(CallbackQueryHandler(trade_callback,pattern=r"^t[cx]:?"))

    async def post_init(application):
        asyncio.create_task(monitor_alerts(application))
    app.post_init=post_init

    logger.info("Bot running!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__=="__main__":
    main()