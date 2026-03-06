import os, logging, json, requests, asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import ccxt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from telegram.error import TelegramError

load_dotenv()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
EXCHANGE_ID    = os.getenv("EXCHANGE", "binance")
exchange       = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})

async def safe_send(update, text):
    try:
        if len(text) <= 4096:
            await update.message.reply_text(text)
        else:
            for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
                await update.message.reply_text(chunk)
    except Exception as e:
        logger.error("safe_send error: " + str(e))

DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _path(name): return DATA_DIR / (name + ".json")
def load_db(name):
    try:
        p = _path(name)
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception as e:
        logger.error("load_db " + name + ": " + str(e)); return {}
def save_db(name, data):
    try: _path(name).write_text(json.dumps(data, default=str, indent=2))
    except Exception as e: logger.error("save_db " + name + ": " + str(e))

def sid(u): return str(u)
def get_watchlist(u):    return load_db("watchlists").get(sid(u), [])
def save_watchlist(u,v): d=load_db("watchlists"); d[sid(u)]=v; save_db("watchlists",d)
def get_alerts(u):       return load_db("alert_levels").get(sid(u), {})
def save_alerts(u,v):    d=load_db("alert_levels"); d[sid(u)]=v; save_db("alert_levels",d)
def get_trades(u):       return load_db("paper_trades").get(sid(u), [])
def save_trades(u,v):    d=load_db("paper_trades"); d[sid(u)]=v; save_db("paper_trades",d)
def get_portfolio(u):    return load_db("portfolios").get(sid(u), [])
def save_portfolio(u,v): d=load_db("portfolios"); d[sid(u)]=v; save_db("portfolios",d)
def get_settings(u):     return load_db("settings").get(sid(u), {"risk":"moderate","notifications":True})
def save_settings(u,v):  d=load_db("settings"); d[sid(u)]=v; save_db("settings",d)
def get_account(u):      return load_db("accounts").get(sid(u), {"balance":0,"currency":"USDT","risk_pct":1.0,"max_leverage":10})
def save_account(u,v):   d=load_db("accounts"); d[sid(u)]=v; save_db("accounts",d)

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
    rsi=calc_rsi(closes); ml,sl_v,hist=calc_macd(closes)
    bbu,bbm,bbl=calc_bb(closes)
    e9=calc_ema(closes,9); e21=calc_ema(closes,21)
    e50=calc_ema(closes,min(50,len(closes)))
    stoch=calc_stoch(highs,lows,closes)
    atr=calc_atr(highs,lows,closes); vwap=calc_vwap(ohlcv)
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

def get_fg():
    try:
        r=requests.get("https://api.alternative.me/fng/?limit=1",timeout=10)
        d=r.json()["data"][0]; v=int(d["value"]); lbl=d["value_classification"]
        e="😱" if v<=25 else "😨" if v<=45 else "😐" if v<=55 else "😄" if v<=75 else "🤑"
        return v,lbl,e
    except Exception as ex:
        logger.error("get_fg: "+str(ex)); return None,"Unknown","?"

def get_news(symbol):
    coin=symbol.replace("/USDT","").replace("/USD","").upper()
    headlines=[]
    try:
        url="https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies="+coin+"&kind=news&public=true"
        r=requests.get(url,timeout=8)
        if r.status_code==200:
            for item in r.json().get("results",[])[:6]:
                title=item.get("title","")
                votes=item.get("votes",{})
                b=votes.get("positive",0); bear=votes.get("negative",0)
                e="🟢" if b>bear else "🔴" if bear>b else "⚪"
                headlines.append(e+" "+title)
    except Exception as ex:
        logger.error("get_news: "+str(ex))
    return headlines if headlines else ["No recent news found"]

def groq_call(system, user, max_tokens=1500):
    r=requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization":"Bearer "+GROQ_API_KEY,"Content-Type":"application/json"},
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
    user=(
        "Analyze "+symbol+" (risk:"+risk+") return ONLY this JSON:\n"
        '{"sentiment":"Bullish|Bearish|Neutral","sentiment_score":50,"trend":"Uptrend|Downtrend|Sideways",'
        '"trend_strength":"Strong|Moderate|Weak","market_structure":"observation",'
        '"trade_setup":{"signal":"Long|Short|No Trade","entry_zone":"low-high",'
        '"take_profit_1":"p","take_profit_2":"p","take_profit_3":"p",'
        '"stop_loss":"p","risk_reward":"1:3","timeframe":"Short-term|Medium-term","confidence":"Low|Medium|High"},'
        '"key_confluences":["c1","c2","c3"],'
        '"reasons":["RSI","MACD","BB","EMA","Volume","Fib"],'
        '"warnings":["w1","w2"]}\n'
        "DATA: Price="+str(ind["price"])+" RSI="+str(ind["rsi"])+"("+ind["rsi_s"]+") "
        "MACD_hist="+str(ind["hist"])+"("+ind["macd_s"]+") BB="+ind["bb_s"]+" "
        "EMA="+ind["ema_s"]+" Stoch="+str(ind["stoch"])+"("+ind["stoch_s"]+") "
        "Volume="+ind["vol"]+" Support="+str(ind["sup"])+" Resistance="+str(ind["res"])+" "
        "Fib618="+str(ind["f618"])+" GoldenPocket="+str(ind["golden"])+" "
        "Patterns="+str(ind["patterns"])+" FearGreed="+str(fg)
    )
    return groq_call(system, user)

def ai_news_sentiment(symbol, headlines):
    system="You are a crypto news analyst. Return ONLY valid JSON."
    user=(
        "Analyze these headlines for "+symbol+" return ONLY:\n"
        '{"news_sentiment":"Bullish|Bearish|Neutral","news_score":50,'
        '"impact":"High|Medium|Low","summary":"1 sentence","trade_bias":"Buy the dip|Sell the rally|Hold|No clear bias"}\n'
        "HEADLINES:\n"+"\n".join(headlines[:6])
    )
    return groq_call(system, user, max_tokens=300)

def ai_compare(data):
    system="You are a Crypto Analyst. Return ONLY valid JSON."
    entries="\n".join([s+":RSI="+str(d["rsi"])+",MACD="+str(d["hist"])+",Trend="+d["etrend"]+",BB="+d["bb_s"] for s,d in data.items()])
    user=(
        "Rank coins by trade opportunity. Return ONLY:\n"
        '{"ranking":[{"symbol":"s","score":80,"signal":"Long|Short|No Trade","reason":"brief"}],'
        '"best_opportunity":"symbol","summary":"2 sentences"}\n'
        "DATA:\n"+entries
    )
    return groq_call(system, user, max_tokens=600)

def ai_performance(trades):
    wins=len([t for t in trades if t.get("result")=="win"]); total=len(trades)
    system="You are a quant analyst. Return ONLY valid JSON."
    user=(
        "Paper trading: Total="+str(total)+" Wins="+str(wins)+" Losses="+str(total-wins)+"\n"
        'Return ONLY: {"win_rate":"x/y","assessment":"2 sentences","improvements":["tip1","tip2"]}'
    )
    return groq_call(system, user, max_tokens=400)

def calc_position(account, entry_str, sl_str, signal):
    try:
        balance=float(account.get("balance",0))
        risk_pct=float(account.get("risk_pct",1.0))
        max_lev=float(account.get("max_leverage",10))
        if balance<=0: return None
        risk_amount=balance*(risk_pct/100)
        entry=float(str(entry_str).split("-")[0].strip())
        sl=float(str(sl_str).strip())
        if entry<=0 or sl<=0: return None
        sl_dist_pct=abs(entry-sl)/entry*100
        if sl_dist_pct<=0: return None
        pos_size=risk_amount/(sl_dist_pct/100)
        margin_to_use=balance*0.20
        rec_lev=min(round(pos_size/margin_to_use,1),max_lev)
        rec_lev=max(1.0,rec_lev)
        margin_needed=round(pos_size/rec_lev,2)
        qty=round(pos_size/entry,4)
        liq=round(entry*(1-1/rec_lev*0.9),4) if signal=="Long" else round(entry*(1+1/rec_lev*0.9),4)
        return {
            "balance":balance,"currency":account.get("currency","USDT"),
            "risk_pct":risk_pct,"risk_amount":round(risk_amount,2),
            "entry_price":entry,"stop_loss":sl,
            "sl_distance_pct":round(sl_dist_pct,2),
            "position_size_usdt":round(pos_size,2),
            "margin_needed":margin_needed,
            "recommended_leverage":rec_lev,
            "qty":qty,"liq_price":liq,"signal":signal,
        }
    except Exception as e:
        logger.error("calc_position: "+str(e)); return None

def fmt(raw):
    raw=raw.upper().strip()
    if "/" not in raw:
        raw=(raw[:-4]+"/USDT") if raw.endswith("USDT") else (raw[:-3]+"/USD") if raw.endswith("USD") else (raw+"/USDT")
    return raw

def build_alert(symbol, d, ind, fg_val=None, fg_lbl=None, fg_e="", news=None, pos=None):
    ts=d.get("trade_setup",{}); sig=ts.get("signal","N/A"); conf=ts.get("confidence","N/A")
    sig_e="🟢 LONG" if sig=="Long" else "🔴 SHORT" if sig=="Short" else "⚪ NO TRADE"
    sent_e="📈" if d.get("sentiment")=="Bullish" else "📉" if d.get("sentiment")=="Bearish" else "➡️"
    gp="✅ YES" if ind["golden"] else "❌ No"
    conf_e="🟢" if conf=="High" else "🟡" if conf=="Medium" else "🔴"
    rsi_e="🔴" if ind["rsi"]>=70 else "🟢" if ind["rsi"]<=30 else "⚪"
    macd_e="📈" if ind["hist"]>0 else "📉"
    stoch_e="🔴" if ind["stoch"]>=80 else "🟢" if ind["stoch"]<=20 else "⚪"
    reasons="\n".join("  - "+r for r in d.get("reasons",[]))
    confluences="\n".join("  + "+c for c in d.get("key_confluences",[]))
    warnings="\n".join("  ! "+w for w in d.get("warnings",[]))
    patterns=", ".join(ind.get("patterns",["None"]))
    fg_line="\n😱 Fear & Greed: "+fg_e+" "+str(fg_val)+"/100 - "+str(fg_lbl) if fg_val else ""

    out = (
        "=== 🤖 AI TRADE ALERT ===\n"
        "📌 "+symbol+"  💰 "+f"{ind['price']:,.4f}"+"\n"
        "🕐 "+datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")+fg_line+"\n"
        "\n--- 📊 MARKET ---\n"
        "📈 Trend: "+d.get("trend","N/A")+" ("+d.get("trend_strength","N/A")+")\n"
        "🏗 Structure: "+d.get("market_structure","N/A")+"\n"
        "🧠 Sentiment: "+sent_e+" "+d.get("sentiment","N/A")+" ("+str(d.get("sentiment_score","N/A"))+"/100)\n"
        "🕯 Patterns: "+patterns+"\n"
        "\n--- 📉 INDICATORS ---\n"
        "RSI: "+rsi_e+" "+str(ind["rsi"])+" - "+ind["rsi_s"]+"\n"
        "MACD: "+macd_e+" "+str(ind["ml"])+"  Hist: "+str(ind["hist"])+" - "+ind["macd_s"]+"\n"
        "Stoch: "+stoch_e+" "+str(ind["stoch"])+" - "+ind["stoch_s"]+"\n"
        "BB: "+str(ind["bbl"])+" / "+str(ind["bbm"])+" / "+str(ind["bbu"])+" - "+ind["bb_s"]+"\n"
        "EMA 9/21/50: "+str(ind["e9"])+" / "+str(ind["e21"])+" / "+str(ind["e50"])+" - "+ind["ema_s"]+"\n"
        "VWAP: "+str(ind["vwap"])+" - "+ind["vwap_s"]+"\n"
        "ATR: "+str(ind["atr"])+"  Volume: "+ind["vol"]+"\n"
        "\n--- 🌀 FIBONACCI ---\n"
        "0.382: "+str(ind["f382"])+"  0.5: "+str(ind["f500"])+"  0.618: "+str(ind["f618"])+"\n"
        "✨ Golden Pocket: "+str(ind["f650"])+" - "+str(ind["f618"])+"\n"
        "In Golden Pocket: "+gp+"\n"
        "🔑 Support: "+str(ind["sup"])+"  Resistance: "+str(ind["res"])+"\n"
        "\n--- 🎯 TRADE SETUP ---\n"
        "Signal: "+sig_e+"  Confidence: "+conf_e+" "+conf+"\n"
        "Entry: "+ts.get("entry_zone","N/A")+"\n"
        "TP1: "+ts.get("take_profit_1","N/A")+" 🎯\n"
        "TP2: "+ts.get("take_profit_2","N/A")+" 🎯🎯\n"
        "TP3: "+ts.get("take_profit_3","N/A")+" 🎯🎯🎯\n"
        "SL: "+ts.get("stop_loss","N/A")+" 🛑  R:R: "+ts.get("risk_reward","N/A")+"\n"
        "\n--- ✅ CONFLUENCES ---\n"+confluences+"\n"
        "\n--- 💡 REASONING ---\n"+reasons+"\n"
        "\n--- ⚠️ WARNINGS ---\n"+warnings+"\n"
    )

    if news:
        ns_e="📈" if news.get("news_sentiment")=="Bullish" else "📉" if news.get("news_sentiment")=="Bearish" else "➡️"
        out += (
            "\n--- 📰 NEWS SENTIMENT ---\n"
            "Sentiment: "+ns_e+" "+news.get("news_sentiment","N/A")+" ("+str(news.get("news_score","N/A"))+"/100)\n"
            "Impact: "+news.get("impact","N/A")+"  Bias: "+news.get("trade_bias","N/A")+"\n"
            "Summary: "+news.get("summary","N/A")+"\n"
        )

    if pos:
        pos_e="🟢 LONG" if pos["signal"]=="Long" else "🔴 SHORT"
        out += (
            "\n--- 💰 POSITION SIZING ---\n"
            "Account: $"+f"{pos['balance']:,.2f}"+" "+pos["currency"]+"\n"
            "Risk: "+str(pos["risk_pct"])+"% = $"+str(pos["risk_amount"])+" max loss\n"
            "SL Distance: "+str(pos["sl_distance_pct"])+"%\n"
            "\n"
            "Position Size: $"+f"{pos['position_size_usdt']:,.2f}"+"\n"
            "Margin to Use: $"+f"{pos['margin_needed']:,.2f}"+"\n"
            "Leverage: "+str(pos["recommended_leverage"])+"x 📊\n"
            "Quantity: "+str(pos["qty"])+" coins\n"
            "Est. Liquidation: $"+str(pos["liq_price"])+" 💀\n"
            "Max Loss if SL hit: $"+str(pos["risk_amount"])+"\n"
        )

    out += "\n⚠️ Not financial advice. Always DYOR."
    return out

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
                        if ind["rsi"]>=70: alerts.append("RSI Overbought "+str(ind["rsi"]))
                        if ind["rsi"]<=30: alerts.append("RSI Oversold "+str(ind["rsi"]))
                        if ind["golden"]:  alerts.append("Price in Golden Pocket!")
                        if ind["stoch"]>=80: alerts.append("Stochastic Overbought")
                        if ind["stoch"]<=20: alerts.append("Stochastic Oversold")
                        custom=load_db("alert_levels").get(uid_str,{}).get(symbol,{})
                        if custom.get("price_above") and ind["price"]>=float(custom["price_above"]):
                            alerts.append("Price crossed ABOVE "+str(custom["price_above"]))
                        if custom.get("price_below") and ind["price"]<=float(custom["price_below"]):
                            alerts.append("Price crossed BELOW "+str(custom["price_below"]))
                        if alerts:
                            msg="🔔 Alert: "+symbol+"\n\n"+"\n".join(alerts)+"\nPrice: "+f"{ind['price']:,.4f}"
                            await app.bot.send_message(chat_id=int(uid_str),text=msg)
                    except Exception as e: logger.error("Monitor "+symbol+": "+str(e))
        except Exception as e: logger.error("Monitor loop: "+str(e))
        await asyncio.sleep(900)

async def start(update, context):
    await safe_send(update,
        "🤖 AI Crypto Trading Bot\n\n"
        "📊 ANALYSIS\n"
        "/price BTC\n"
        "/analyze BTC - Full AI analysis + news + position sizing\n"
        "/indicators BTC\n"
        "/multitf BTC - Multi-timeframe\n"
        "/compare BTC ETH SOL\n"
        "/feargreed\n"
        "/news BTC - Latest news + AI sentiment\n\n"
        "📋 WATCHLIST\n"
        "/watchlist\n"
        "/addwatch BTC\n"
        "/removewatch BTC\n"
        "/setalert BTC 95000 90000\n"
        "/alerts on or off\n"
        "/scan\n\n"
        "📈 TRADING\n"
        "/trade BTC\n\n"
        "📒 PAPER TRADING\n"
        "/papertrades\n"
        "/closetrade BTC win or loss\n"
        "/performance\n\n"
        "💼 PORTFOLIO\n"
        "/portfolio\n"
        "/addholding BTC 0.5 95000\n"
        "/removeholding BTC\n\n"
        "💰 ACCOUNT & RISK\n"
        "/setaccount 1000 1 10 - balance, risk%, max leverage\n"
        "/account - View account\n"
        "/possize BTC 95000 92000 - Calculate position\n\n"
        "⚙️ SETTINGS\n"
        "/settings\n"
        "/setrisk low or moderate or high\n"
        "/help"
    )

async def help_cmd(update, context): await start(update, context)

async def price_cmd(update, context):
    if not context.args: await safe_send(update, "Usage: /price BTC"); return
    symbol=fmt(context.args[0])
    try:
        t=exchange.fetch_ticker(symbol); c=t.get("percentage",0) or 0
        await safe_send(update,
            symbol+"\nPrice: "+f"{t['last']:,.4f}"+"\n24h: "+f"{c:+.2f}%"+
            "\nHigh: "+f"{t['high']:,.4f}"+"  Low: "+f"{t['low']:,.4f}"+
            "\nVolume: "+f"{t['quoteVolume']:,.2f}"
        )
    except Exception as e:
        logger.error("price_cmd: "+str(e)); await safe_send(update,"Error: "+str(e))

async def feargreed_cmd(update, context):
    try:
        v,lbl,e=get_fg()
        if v:
            bar="X"*(v//10)+"-"*(10-v//10)
            await safe_send(update,"😱 Fear & Greed Index\n\n"+e+" "+str(v)+"/100 - "+lbl+"\n["+bar+"]\n\n0=Extreme Fear  100=Extreme Greed")
        else:
            await safe_send(update,"Could not fetch Fear & Greed index.")
    except Exception as e:
        logger.error("feargreed_cmd: "+str(e)); await safe_send(update,"Error: "+str(e))

async def news_cmd(update, context):
    if not context.args: await safe_send(update, "Usage: /news BTC"); return
    symbol=fmt(context.args[0])
    msg=await update.message.reply_text("📰 Fetching news for "+symbol+"...")
    try:
        headlines=get_news(symbol)
        news_parsed=None
        if headlines[0]!="No recent news found":
            try:
                raw=ai_news_sentiment(symbol,headlines)
                news_parsed=parse_json(raw)
            except Exception as e: logger.error("news ai: "+str(e))
        await msg.delete()
        lines=["📰 "+symbol+" News\n"]
        if news_parsed:
            ns_e="📈" if news_parsed.get("news_sentiment")=="Bullish" else "📉" if news_parsed.get("news_sentiment")=="Bearish" else "➡️"
            lines.append("AI Sentiment: "+ns_e+" "+news_parsed.get("news_sentiment","N/A")+" ("+str(news_parsed.get("news_score","N/A"))+"/100)")
            lines.append("Impact: "+news_parsed.get("impact","N/A")+"  Bias: "+news_parsed.get("trade_bias","N/A"))
            lines.append("Summary: "+news_parsed.get("summary","N/A")+"\n")
        lines.append("Latest Headlines:")
        for h in headlines[:6]: lines.append(h)
        await safe_send(update,"\n".join(lines))
    except Exception as e:
        logger.error("news_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def indicators_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /indicators BTC"); return
    symbol=fmt(context.args[0])
    msg=await update.message.reply_text("Calculating indicators for "+symbol+"...")
    try:
        ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200)
        ind=build_ind(ohlcv)
        await msg.delete()
        await safe_send(update,
            symbol+" Indicators\n\n"
            "Price: "+f"{ind['price']:,.4f}"+"  ATR: "+str(ind["atr"])+"  VWAP: "+str(ind["vwap"])+"\n\n"
            "RSI(14): "+str(ind["rsi"])+" - "+ind["rsi_s"]+"\n"
            "MACD: "+str(ind["ml"])+"  Signal: "+str(ind["sl_v"])+"  Hist: "+str(ind["hist"])+"\n"
            "Stochastic: "+str(ind["stoch"])+" - "+ind["stoch_s"]+"\n"
            "BB: "+str(ind["bbl"])+" / "+str(ind["bbm"])+" / "+str(ind["bbu"])+"\n"
            "EMA 9/21/50: "+str(ind["e9"])+" / "+str(ind["e21"])+" / "+str(ind["e50"])+"\n"
            "EMA Trend: "+ind["etrend"]+"  Cross: "+ind["ema_s"]+"\n"
            "Support: "+str(ind["sup"])+"  Resistance: "+str(ind["res"])+"\n"
            "Volume: "+ind["vol"]+"\n"
            "Patterns: "+", ".join(ind["patterns"])
        )
    except Exception as e:
        logger.error("indicators_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def multitf_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /multitf BTC"); return
    symbol=fmt(context.args[0])
    msg=await update.message.reply_text("Running multi-timeframe for "+symbol+"...")
    try:
        lines=[symbol+" Multi-Timeframe\n"]; bulls=0; total=0
        for tf,limit in [("15m",96),("1h",200),("4h",100),("1d",60)]:
            try:
                ohlcv=exchange.fetch_ohlcv(symbol,timeframe=tf,limit=limit)
                ind=build_ind(ohlcv); total+=1
                if ind["etrend"]=="Bullish": bulls+=1
                lines.append(tf+": "+ind["etrend"]+"  RSI:"+str(ind["rsi"])+"  MACD:"+("Up" if ind["hist"]>0 else "Down")+"  "+ind["patterns"][0])
            except: lines.append(tf+": unavailable")
        overall="🟢 BULLISH" if bulls>=3 else "🔴 BEARISH" if (total-bulls)>=3 else "⚪ MIXED"
        lines.append("\nOverall: "+overall+" ("+str(bulls)+"/"+str(total)+" bullish)")
        await msg.delete(); await safe_send(update,"\n".join(lines))
    except Exception as e:
        logger.error("multitf_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def compare_cmd(update, context):
    if len(context.args)<2: await safe_send(update,"Usage: /compare BTC ETH SOL"); return
    symbols=[fmt(a) for a in context.args[:5]]
    msg=await update.message.reply_text("Comparing "+", ".join(symbols)+"...")
    try:
        data={}
        for s in symbols:
            ohlcv=exchange.fetch_ohlcv(s,timeframe="1h",limit=200); data[s]=build_ind(ohlcv)
        raw=ai_compare(data); parsed=parse_json(raw)
        await msg.delete()
        if parsed:
            lines=["⚖️ Coin Comparison\n\nBest: "+parsed.get("best_opportunity","N/A")+"\n"+parsed.get("summary","")+"\n\nRankings:"]
            for i,r in enumerate(parsed.get("ranking",[]),1):
                lines.append(str(i)+". "+r.get("symbol","")+" Score:"+str(r.get("score"))+"/100 - "+r.get("signal","")+"\n   "+r.get("reason",""))
            await safe_send(update,"\n".join(lines))
        else: await safe_send(update,"AI response:\n"+raw[:500])
    except Exception as e:
        logger.error("compare_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def analyze_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /analyze BTC"); return
    u=update.effective_user.id; symbol=fmt(context.args[0])
    settings=get_settings(u)
    msg=await update.message.reply_text("🔍 Analyzing "+symbol+"...")
    try:
        ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200)
        ind=build_ind(ohlcv); fg_val,fg_lbl,fg_e=get_fg()
        await msg.edit_text("📰 Fetching news for "+symbol+"...")
        headlines=get_news(symbol)
        news_parsed=None
        if headlines[0]!="No recent news found":
            try:
                news_raw=ai_news_sentiment(symbol,headlines)
                news_parsed=parse_json(news_raw)
            except Exception as ne: logger.error("news sentiment: "+str(ne))
        await msg.edit_text("🤖 Running AI analysis for "+symbol+"...")
        raw=ai_analyze(symbol,ind,fg_val,settings.get("risk","moderate"))
        parsed=parse_json(raw)
        await msg.delete()
        if parsed:
            ts=parsed.get("trade_setup",{})
            account=get_account(u); pos=None
            if account.get("balance",0)>0 and ts.get("signal") in ["Long","Short"]:
                pos=calc_position(account,ts.get("entry_zone","0"),ts.get("stop_loss","0"),ts.get("signal"))
            if ts.get("signal") in ["Long","Short"]:
                trades=get_trades(u)
                trades.append({"symbol":symbol,"signal":ts["signal"],"entry":ts.get("entry_zone",""),
                    "tp1":ts.get("take_profit_1",""),"sl":ts.get("stop_loss",""),
                    "rr":ts.get("risk_reward",""),"time":datetime.utcnow().isoformat(),"status":"open","result":"pending"})
                save_trades(u,trades)
            await safe_send(update,build_alert(symbol,parsed,ind,fg_val,fg_lbl,fg_e,news_parsed,pos))
            if headlines[0]!="No recent news found":
                await safe_send(update,"📰 "+symbol+" Headlines:\n\n"+"\n".join(headlines[:6]))
        else:
            await safe_send(update,"AI format error:\n"+raw[:500])
    except Exception as e:
        logger.error("analyze_cmd: "+str(e))
        try: await msg.delete()
        except: pass
        await safe_send(update,"Error: "+str(e))

async def trade_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /trade BTC"); return
    symbol=fmt(context.args[0])
    kb=InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Confirm",callback_data="tc:"+symbol),
        InlineKeyboardButton("❌ Cancel",callback_data="tx")]])
    await update.message.reply_text("Generate trade setup for "+symbol+"?\nBot does NOT auto-execute.",reply_markup=kb)

async def trade_callback(update, context):
    q=update.callback_query; await q.answer()
    if q.data=="tx": await q.edit_message_text("Cancelled."); return
    if q.data.startswith("tc:"):
        u=q.from_user.id; symbol=q.data[3:]; settings=get_settings(u)
        await q.edit_message_text("Analyzing "+symbol+"...")
        try:
            ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200)
            ind=build_ind(ohlcv); fg_val,fg_lbl,fg_e=get_fg()
            raw=ai_analyze(symbol,ind,fg_val,settings.get("risk","moderate"))
            parsed=parse_json(raw)
            if parsed:
                ts=parsed.get("trade_setup",{})
                account=get_account(u); pos=None
                if account.get("balance",0)>0 and ts.get("signal") in ["Long","Short"]:
                    pos=calc_position(account,ts.get("entry_zone","0"),ts.get("stop_loss","0"),ts.get("signal"))
                if ts.get("signal") in ["Long","Short"]:
                    trades=get_trades(u)
                    trades.append({"symbol":symbol,"signal":ts["signal"],"entry":ts.get("entry_zone",""),
                        "tp1":ts.get("take_profit_1",""),"sl":ts.get("stop_loss",""),
                        "rr":ts.get("risk_reward",""),"time":datetime.utcnow().isoformat(),"status":"open","result":"pending"})
                    save_trades(u,trades)
                result=build_alert(symbol,parsed,ind,fg_val,fg_lbl,fg_e,None,pos)+"\n\nPlace order MANUALLY."
                if len(result)>4096: result=result[:4090]+"..."
            else: result="AI format error:\n"+raw[:500]
            await q.edit_message_text(result)
        except Exception as e:
            logger.error("trade_callback: "+str(e)); await q.edit_message_text("Error: "+str(e))

async def scan_cmd(update, context):
    u=update.effective_user.id; wl=get_watchlist(u)
    if not wl: await safe_send(update,"Watchlist empty. Use /addwatch BTC first."); return
    msg=await update.message.reply_text("🔍 Scanning "+str(len(wl))+" coins...")
    try:
        data={}
        for s in wl[:8]:
            ohlcv=exchange.fetch_ohlcv(s,timeframe="1h",limit=200); data[s]=build_ind(ohlcv)
        raw=ai_compare(data); parsed=parse_json(raw)
        await msg.delete()
        if parsed:
            lines=["🔍 Watchlist Scan\n\nBest: "+parsed.get("best_opportunity","N/A")+"\n"+parsed.get("summary","")+"\n\nRankings:"]
            for i,r in enumerate(parsed.get("ranking",[]),1):
                lines.append(str(i)+". "+r.get("symbol","")+" Score:"+str(r.get("score"))+"/100 - "+r.get("signal","")+"\n   "+r.get("reason",""))
            await safe_send(update,"\n".join(lines))
        else: await safe_send(update,"AI response:\n"+raw[:500])
    except Exception as e:
        logger.error("scan_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def watchlist_cmd(update, context):
    u=update.effective_user.id; wl=get_watchlist(u)
    if not wl: await safe_send(update,"Watchlist empty. Use /addwatch BTC"); return
    lines=["📋 Your Watchlist\n"]
    for s in wl:
        try:
            t=exchange.fetch_ticker(s); c=t.get("percentage",0) or 0
            lines.append(("🟢" if c>=0 else "🔴")+" "+s+": "+f"{t['last']:,.4f}"+" ("+f"{c:+.2f}%"+")")
        except: lines.append("- "+s+": unavailable")
    await safe_send(update,"\n".join(lines))

async def addwatch_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /addwatch BTC"); return
    symbol=fmt(context.args[0]); wl=get_watchlist(u)
    if symbol in wl: await safe_send(update,symbol+" already in watchlist."); return
    if len(wl)>=10: await safe_send(update,"Watchlist limit is 10."); return
    wl.append(symbol); save_watchlist(u,wl)
    await safe_send(update,"✅ Added "+symbol+" ("+str(len(wl))+"/10)")

async def removewatch_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /removewatch BTC"); return
    symbol=fmt(context.args[0]); wl=get_watchlist(u)
    if symbol in wl:
        wl.remove(symbol); save_watchlist(u,wl)
        await safe_send(update,"✅ Removed "+symbol)
    else: await safe_send(update,symbol+" not in watchlist.")

async def setalert_cmd(update, context):
    u=update.effective_user.id
    if len(context.args)<3: await safe_send(update,"Usage: /setalert BTC 95000 90000"); return
    try:
        symbol=fmt(context.args[0]); above=float(context.args[1]); below=float(context.args[2])
        al=get_alerts(u); al[symbol]={"price_above":above,"price_below":below}; save_alerts(u,al)
        await safe_send(update,"✅ Alerts set for "+symbol+"\nAbove: "+f"{above:,.4f}"+"\nBelow: "+f"{below:,.4f}")
    except ValueError: await safe_send(update,"Invalid prices. Use numbers.")

async def alerts_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /alerts on or /alerts off"); return
    s=get_settings(u); s["notifications"]=(context.args[0].lower()=="on"); save_settings(u,s)
    await safe_send(update,"🔔 Alerts "+("ON ✅" if s["notifications"] else "OFF ❌"))

async def papertrades_cmd(update, context):
    u=update.effective_user.id; trades=get_trades(u)
    if not trades: await safe_send(update,"No paper trades yet. Run /analyze BTC first."); return
    open_t=[t for t in trades if t.get("status")=="open"]
    closed_t=[t for t in trades if t.get("status")=="closed"]
    lines=["📒 Paper Trades: "+str(len(open_t))+" open, "+str(len(closed_t))+" closed\n"]
    for t in open_t[-5:]:
        lines.append(("🟢" if t["signal"]=="Long" else "🔴")+" "+t["symbol"]+" - "+t["signal"]+"\nEntry:"+t["entry"]+" TP1:"+t["tp1"]+" SL:"+t["sl"]+"\nR:R:"+t["rr"]+" | "+t["time"][:16])
    if not open_t: lines.append("No open trades.")
    await safe_send(update,"\n".join(lines))

async def closetrade_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /closetrade BTC win or loss"); return
    symbol=fmt(context.args[0]); result=context.args[1].lower() if len(context.args)>1 else "pending"
    trades=get_trades(u)
    for t in reversed(trades):
        if t["symbol"]==symbol and t["status"]=="open":
            t["status"]="closed"; t["result"]=result; t["close_time"]=datetime.utcnow().isoformat()
            save_trades(u,trades)
            await safe_send(update,"✅ Closed "+symbol+" as "+result.upper()); return
    await safe_send(update,"No open trade for "+symbol+".")

async def performance_cmd(update, context):
    u=update.effective_user.id; trades=get_trades(u)
    closed=[t for t in trades if t.get("status")=="closed"]
    if not closed: await safe_send(update,"No closed trades yet. Use /closetrade BTC win."); return
    wins=len([t for t in closed if t.get("result")=="win"])
    rate=round(wins/len(closed)*100,1)
    msg=await update.message.reply_text("📊 Analyzing performance...")
    try:
        raw=ai_performance(closed); parsed=parse_json(raw)
        await msg.delete()
        lines=["📊 Performance\n\nTotal: "+str(len(closed))+"  Wins: "+str(wins)+"  Losses: "+str(len(closed)-wins)+"\nWin Rate: "+str(rate)+"%"]
        if parsed:
            lines.append("\nAssessment: "+parsed.get("assessment",""))
            for tip in parsed.get("improvements",[]): lines.append("- "+tip)
        await safe_send(update,"\n".join(lines))
    except Exception as e:
        logger.error("performance_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def portfolio_cmd(update, context):
    u=update.effective_user.id; pf=get_portfolio(u)
    if not pf: await safe_send(update,"Portfolio empty. Use /addholding BTC 0.5 95000"); return
    lines=["💼 Your Portfolio\n"]; tv=0; tc=0
    for h in pf:
        try:
            t=exchange.fetch_ticker(h["symbol"]); p=t["last"]
            val=p*h["amount"]; cost=h["buy_price"]*h["amount"]
            pnl=val-cost; pct=(pnl/cost*100) if cost>0 else 0
            tv+=val; tc+=cost
            lines.append(("🟢" if pnl>=0 else "🔴")+" "+h["symbol"]+" x"+str(h["amount"])+"\nPrice:"+f"{p:,.4f}"+" Value:$"+f"{val:,.2f}"+" PnL:"+f"{pct:+.2f}%")
        except: lines.append("- "+h["symbol"]+": unavailable")
    tpct=((tv-tc)/tc*100) if tc>0 else 0
    lines.append("\nTotal Value: $"+f"{tv:,.2f}"+"\nTotal PnL: "+f"{tpct:+.2f}%")
    await safe_send(update,"\n".join(lines))

async def addholding_cmd(update, context):
    u=update.effective_user.id
    if len(context.args)<3: await safe_send(update,"Usage: /addholding BTC 0.5 95000"); return
    try:
        symbol=fmt(context.args[0]); amount=float(context.args[1]); price=float(context.args[2])
        pf=get_portfolio(u); pf=[h for h in pf if h["symbol"]!=symbol]
        pf.append({"symbol":symbol,"amount":amount,"buy_price":price}); save_portfolio(u,pf)
        await safe_send(update,"✅ Added "+str(amount)+" "+symbol+" at $"+f"{price:,.4f}"+"\nCost: $"+f"{amount*price:,.2f}")
    except ValueError: await safe_send(update,"Invalid values.")

async def removeholding_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /removeholding BTC"); return
    symbol=fmt(context.args[0]); pf=get_portfolio(u)
    pf=[h for h in pf if h["symbol"]!=symbol]; save_portfolio(u,pf)
    await safe_send(update,"✅ Removed "+symbol)

async def settings_cmd(update, context):
    u=update.effective_user.id; s=get_settings(u)
    await safe_send(update,
        "⚙️ Settings\n\nRisk: "+s.get("risk","moderate")+"\nNotifications: "+("ON ✅" if s.get("notifications",True) else "OFF ❌")+
        "\n\nChange:\n/setrisk low or moderate or high\n/alerts on or off"
    )

async def setrisk_cmd(update, context):
    u=update.effective_user.id
    if not context.args or context.args[0].lower() not in ["low","moderate","high"]:
        await safe_send(update,"Usage: /setrisk low or moderate or high"); return
    risk=context.args[0].lower(); s=get_settings(u); s["risk"]=risk; save_settings(u,s)
    desc={"low":"Conservative - wider stops","moderate":"Balanced","high":"Aggressive - tight entries"}
    await safe_send(update,"✅ Risk: "+risk.upper()+"\n"+desc[risk])

async def setaccount_cmd(update, context):
    u=update.effective_user.id
    if not context.args:
        await safe_send(update,"Usage: /setaccount 1000 1 10\n(balance, risk%, max leverage)\nExample: /setaccount 500 1 5"); return
    try:
        balance=float(context.args[0])
        risk_pct=min(max(float(context.args[1]) if len(context.args)>1 else 1.0, 0.1), 5.0)
        max_lev=min(max(float(context.args[2]) if len(context.args)>2 else 10.0, 1.0), 125.0)
        account=get_account(u)
        account.update({"balance":balance,"risk_pct":risk_pct,"max_leverage":max_lev,"currency":"USDT"})
        save_account(u,account)
        risk_amount=round(balance*risk_pct/100,2)
        await safe_send(update,
            "✅ Account configured!\n\n"
            "💰 Balance: $"+f"{balance:,.2f}"+" USDT\n"
            "⚠️ Risk per trade: "+str(risk_pct)+"% = $"+str(risk_amount)+"\n"
            "📊 Max leverage: "+str(max_lev)+"x\n\n"
            "Now /analyze BTC will include position sizing!"
        )
    except ValueError: await safe_send(update,"Invalid values. Use: /setaccount 1000 1 10")

async def account_cmd(update, context):
    u=update.effective_user.id; account=get_account(u)
    if account.get("balance",0)<=0:
        await safe_send(update,"No account set. Use /setaccount 1000"); return
    balance=account["balance"]; risk_pct=account["risk_pct"]
    await safe_send(update,
        "💼 Your Account\n\n"
        "Balance: $"+f"{balance:,.2f}"+" "+account.get("currency","USDT")+"\n"
        "Risk per trade: "+str(risk_pct)+"% = $"+str(round(balance*risk_pct/100,2))+"\n"
        "Max leverage: "+str(account.get("max_leverage",10))+"x\n\n"
        "Change: /setaccount balance risk% maxleverage"
    )

async def possize_cmd(update, context):
    u=update.effective_user.id
    if len(context.args)<3:
        await safe_send(update,"Usage: /possize BTC 95000 92000\n(symbol, entry, stop loss)"); return
    account=get_account(u)
    if account.get("balance",0)<=0:
        await safe_send(update,"Set your account first: /setaccount 1000"); return
    try:
        symbol=fmt(context.args[0]); entry=float(context.args[1]); sl=float(context.args[2])
        signal="Long" if entry>sl else "Short"
        pos=calc_position(account,entry,sl,signal)
        if pos:
            sig_e="🟢 LONG" if signal=="Long" else "🔴 SHORT"
            await safe_send(update,
                "💰 Position Size Calculator\n\n"
                "Symbol: "+symbol+"\n"
                "Signal: "+sig_e+"\n"
                "Entry: $"+f"{entry:,.4f}"+"\n"
                "Stop Loss: $"+f"{sl:,.4f}"+"\n"
                "SL Distance: "+str(pos["sl_distance_pct"])+"%\n\n"
                "Account: $"+f"{pos['balance']:,.2f}"+"\n"
                "Risk Amount: $"+str(pos["risk_amount"])+" ("+str(pos["risk_pct"])+"%)\n\n"
                "Position Size: $"+f"{pos['position_size_usdt']:,.2f}"+"\n"
                "Margin to Use: $"+f"{pos['margin_needed']:,.2f}"+"\n"
                "Leverage: "+str(pos["recommended_leverage"])+"x 📊\n"
                "Quantity: "+str(pos["qty"])+" coins\n"
                "Est. Liquidation: $"+str(pos["liq_price"])+" 💀\n\n"
                "Max Loss if SL hit: $"+str(pos["risk_amount"])
            )
        else: await safe_send(update,"Could not calculate. Check inputs.")
    except ValueError: await safe_send(update,"Invalid values. Use numbers.")

def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    if not GROQ_API_KEY:   raise RuntimeError("GROQ_API_KEY not set")
    app=Application.builder().token(TELEGRAM_TOKEN).build()
    for cmd,fn in [
        ("start",start),("help",help_cmd),("price",price_cmd),
        ("feargreed",feargreed_cmd),("news",news_cmd),
        ("indicators",indicators_cmd),("multitf",multitf_cmd),
        ("compare",compare_cmd),("analyze",analyze_cmd),
        ("trade",trade_cmd),("scan",scan_cmd),
        ("watchlist",watchlist_cmd),("addwatch",addwatch_cmd),
        ("removewatch",removewatch_cmd),("setalert",setalert_cmd),("alerts",alerts_cmd),
        ("papertrades",papertrades_cmd),("closetrade",closetrade_cmd),("performance",performance_cmd),
        ("portfolio",portfolio_cmd),("addholding",addholding_cmd),("removeholding",removeholding_cmd),
        ("settings",settings_cmd),("setrisk",setrisk_cmd),
        ("setaccount",setaccount_cmd),("account",account_cmd),("possize",possize_cmd),
    ]:
        app.add_handler(CommandHandler(cmd,fn))
    app.add_handler(CallbackQueryHandler(trade_callback,pattern=r"^t[cx]:?"))
    async def post_init(application):
        asyncio.create_task(monitor_alerts(application))
    app.post_init=post_init
    logger.info("Bot running!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__=="__main__":
    main()
