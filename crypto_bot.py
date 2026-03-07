import os, logging, json, requests, asyncio
from datetime import datetime, timezone
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
        for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
            await update.message.reply_text(chunk)
    except Exception as e:
        logger.error("safe_send: " + str(e))

# ── Storage ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _path(n): return DATA_DIR / (n + ".json")
def load_db(n):
    try:
        p = _path(n); return json.loads(p.read_text()) if p.exists() else {}
    except Exception as e: logger.error("load_db "+n+": "+str(e)); return {}
def save_db(n, d):
    try: _path(n).write_text(json.dumps(d, default=str, indent=2))
    except Exception as e: logger.error("save_db "+n+": "+str(e))

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
def get_account(u):      return load_db("accounts").get(sid(u), {"balance":0,"currency":"USDT","risk_pct":1.0,"max_leverage":10,"daily_loss":0,"daily_loss_limit":3.0})
def save_account(u,v):   d=load_db("accounts"); d[sid(u)]=v; save_db("accounts",d)
def get_journal(u):      return load_db("journal").get(sid(u), [])
def save_journal(u,v):   d=load_db("journal"); d[sid(u)]=v; save_db("journal",d)

# ── Technical Indicators ──────────────────────────────────────────────────────
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

def calc_obv(closes, volumes):
    obv=0; prev=closes[0]
    for c,v in zip(closes[1:],volumes[1:]):
        obv += v if c>prev else (-v if c<prev else 0); prev=c
    return round(obv,2)

def calc_williams_r(highs, lows, closes, period=14):
    h=max(highs[-period:]); l=min(lows[-period:])
    if h==l: return -50
    return round(-100*(h-closes[-1])/(h-l),2)

def calc_cci(highs, lows, closes, period=20):
    tp=[(h+l+c)/3 for h,l,c in zip(highs[-period:],lows[-period:],closes[-period:])]
    m=np.mean(tp); md=np.mean([abs(x-m) for x in tp])
    if md==0: return 0
    return round((tp[-1]-m)/(0.015*md),2)

def calc_roc(closes, period=10):
    if len(closes)<period+1: return 0
    return round((closes[-1]-closes[-period-1])/closes[-period-1]*100,4)

def detect_rsi_divergence(closes, highs, lows, period=14):
    if len(closes)<30: return "None"
    rsi_vals=[]
    for i in range(len(closes)-20, len(closes)):
        rsi_vals.append(calc_rsi(closes[:i+1], period))
    price_highs=[max(highs[i-5:i+1]) for i in range(14,20)]
    price_lows=[min(lows[i-5:i+1]) for i in range(14,20)]
    rsi_highs=rsi_vals[:6]; rsi_lows=rsi_vals[:6]
    if price_highs[-1]>price_highs[0] and rsi_highs[-1]<rsi_highs[0]:
        return "Bearish Divergence (price up, RSI down = reversal warning)"
    if price_lows[-1]<price_lows[0] and rsi_lows[-1]>rsi_lows[0]:
        return "Bullish Divergence (price down, RSI up = reversal signal)"
    return "None"

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
        if c[-3]>o[-3] and c[-2]>o[-2] and c[-1]>o[-1]: pts.append("Three White Soldiers")
        if c[-3]<o[-3] and c[-2]<o[-2] and c[-1]<o[-1]: pts.append("Three Black Crows")
    return pts if pts else ["None"]

def build_ind(ohlcv):
    closes=[c[4] for c in ohlcv]; highs=[c[2] for c in ohlcv]
    lows=[c[3] for c in ohlcv]; volumes=[c[5] for c in ohlcv]
    price=closes[-1]
    rsi=calc_rsi(closes); ml,sl_v,hist=calc_macd(closes)
    bbu,bbm,bbl=calc_bb(closes)
    e9=calc_ema(closes,9); e21=calc_ema(closes,21)
    e50=calc_ema(closes,min(50,len(closes))); e200=calc_ema(closes,min(200,len(closes)))
    stoch=calc_stoch(highs,lows,closes)
    atr=calc_atr(highs,lows,closes); vwap=calc_vwap(ohlcv)
    obv=calc_obv(closes,volumes)
    wr=calc_williams_r(highs,lows,closes)
    cci=calc_cci(highs,lows,closes)
    roc=calc_roc(closes)
    divergence=detect_rsi_divergence(closes,highs,lows)
    patterns=detect_patterns(ohlcv)
    sh=max(highs[-50:]) if len(highs)>=50 else max(highs)
    sl2=min(lows[-50:]) if len(lows)>=50 else min(lows)
    fr=sh-sl2
    sup=round(float(np.mean(sorted(lows[-20:])[:3])),4)
    res=round(float(np.mean(sorted(highs[-20:],reverse=True)[:3])),4)
    vols=np.array(volumes,dtype=float); hf=len(vols)//2
    vr=np.mean(vols[hf:])/np.mean(vols[:hf]) if np.mean(vols[:hf])>0 else 1
    bb_width=round((bbu-bbl)/bbm*100,2) if bbm>0 else 0
    bb_squeeze=bb_width<2.0
    # ATR-based dynamic SL
    atr_sl_long=round(price-2*atr,4); atr_sl_short=round(price+2*atr,4)
    # Confluence scoring
    bull_pts=0; bear_pts=0
    if rsi<40: bull_pts+=1
    if rsi>60: bear_pts+=1
    if hist>0: bull_pts+=1
    else: bear_pts+=1
    if price<bbl: bull_pts+=1
    if price>bbu: bear_pts+=1
    if e9>e21: bull_pts+=1
    else: bear_pts+=1
    if price>e50: bull_pts+=1
    else: bear_pts+=1
    if price>e200: bull_pts+=1
    else: bear_pts+=1
    if stoch<25: bull_pts+=1
    if stoch>75: bear_pts+=1
    if wr<-80: bull_pts+=1
    if wr>-20: bear_pts+=1
    if cci<-100: bull_pts+=1
    if cci>100: bear_pts+=1
    if vr>1.2 and hist>0: bull_pts+=1
    if vr>1.2 and hist<0: bear_pts+=1
    if "Bullish" in str(divergence): bull_pts+=2
    if "Bearish" in str(divergence): bear_pts+=2
    if "Bullish Engulfing" in patterns or "Morning Star" in patterns or "Hammer" in patterns: bull_pts+=1
    if "Bearish Engulfing" in patterns or "Evening Star" in patterns or "Shooting Star" in patterns: bear_pts+=1
    return {
        "price":price,"atr":atr,"vwap":vwap,"obv":obv,"wr":wr,"cci":cci,"roc":roc,
        "rsi":rsi,"rsi_s":"Overbought" if rsi>=70 else "Oversold" if rsi<=30 else "Neutral",
        "ml":ml,"sl_v":sl_v,"hist":hist,"macd_s":"Bullish" if hist>0 else "Bearish",
        "bbu":bbu,"bbm":bbm,"bbl":bbl,"bb_s":"Overbought" if price>bbu else "Oversold" if price<bbl else "Inside",
        "bb_width":bb_width,"bb_squeeze":bb_squeeze,
        "e9":e9,"e21":e21,"e50":e50,"e200":e200,
        "ema_s":"Bullish" if e9>e21 else "Bearish","etrend":"Bullish" if price>e50 else "Bearish",
        "above200":"Yes" if price>e200 else "No",
        "stoch":stoch,"stoch_s":"Overbought" if stoch>=80 else "Oversold" if stoch<=20 else "Neutral",
        "vol":"Rising" if vr>1.1 else "Falling" if vr<0.9 else "Stable",
        "sup":sup,"res":res,"sh":sh,"sl2":sl2,
        "f236":round(sh-fr*0.236,4),"f382":round(sh-fr*0.382,4),
        "f500":round(sh-fr*0.500,4),"f618":round(sh-fr*0.618,4),"f650":round(sh-fr*0.650,4),
        "golden":round(sh-fr*0.650,4)<=price<=round(sh-fr*0.618,4),
        "vwap_s":"Bullish" if price>vwap else "Bearish",
        "patterns":patterns,"divergence":divergence,
        "atr_sl_long":atr_sl_long,"atr_sl_short":atr_sl_short,
        "bull_pts":bull_pts,"bear_pts":bear_pts,
    }

def build_multi_tf(symbol):
    results={}
    for tf,limit in [("15m",96),("1h",200),("4h",100),("1d",60)]:
        try:
            ohlcv=exchange.fetch_ohlcv(symbol,timeframe=tf,limit=limit)
            results[tf]=build_ind(ohlcv)
        except Exception as e: logger.error("multitf "+tf+": "+str(e))
    return results

def get_mtf_confluence(results):
    bull=sum(1 for r in results.values() if r["bull_pts"]>r["bear_pts"])
    bear=sum(1 for r in results.values() if r["bear_pts"]>r["bull_pts"])
    total=len(results)
    if bull>=3: return "Strong Bullish ("+str(bull)+"/"+str(total)+" TFs agree)"
    if bear>=3: return "Strong Bearish ("+str(bear)+"/"+str(total)+" TFs agree)"
    if bull>bear: return "Weak Bullish ("+str(bull)+"/"+str(total)+" TFs)"
    return "Mixed/Neutral"

# ── All External Data Sources ─────────────────────────────────────────────────
def get_fg():
    try:
        r=requests.get("https://api.alternative.me/fng/?limit=1",timeout=10)
        d=r.json()["data"][0]; v=int(d["value"]); lbl=d["value_classification"]
        e="😱" if v<=25 else "😨" if v<=45 else "😐" if v<=55 else "😄" if v<=75 else "🤑"
        return v,lbl,e
    except Exception as ex: logger.error("get_fg: "+str(ex)); return None,"Unknown","?"

def get_news(symbol):
    coin=symbol.replace("/USDT","").replace("/USD","").upper()
    headlines=[]
    try:
        url="https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies="+coin+"&kind=news&public=true"
        r=requests.get(url,timeout=8)
        if r.status_code==200:
            for item in r.json().get("results",[])[:8]:
                title=item.get("title","")
                votes=item.get("votes",{}); b=votes.get("positive",0); bear=votes.get("negative",0)
                e="🟢" if b>bear else "🔴" if bear>b else "⚪"
                headlines.append(e+" "+title)
    except Exception as ex: logger.error("get_news: "+str(ex))
    return headlines if headlines else ["No recent news found"]

def get_onchain(symbol):
    coin_map={"BTC":"bitcoin","ETH":"ethereum","SOL":"solana","BNB":"binancecoin",
              "ADA":"cardano","XRP":"ripple","DOGE":"dogecoin","AVAX":"avalanche-2",
              "DOT":"polkadot","MATIC":"matic-network","LINK":"chainlink","UNI":"uniswap",
              "ATOM":"cosmos","LTC":"litecoin","BCH":"bitcoin-cash","NEAR":"near"}
    coin=symbol.replace("/USDT","").replace("/USD","").upper()
    coin_id=coin_map.get(coin,""); data={}
    if coin_id:
        try:
            url="https://api.coingecko.com/api/v3/coins/"+coin_id+"?localization=false&tickers=false&community_data=true&developer_data=false"
            r=requests.get(url,timeout=10)
            if r.status_code==200:
                d=r.json(); md=d.get("market_data",{})
                data["market_cap_rank"]=d.get("market_cap_rank","N/A")
                data["market_cap"]=md.get("market_cap",{}).get("usd",0)
                data["volume_24h"]=md.get("total_volume",{}).get("usd",0)
                data["price_change_7d"]=round(md.get("price_change_percentage_7d",0) or 0,2)
                data["price_change_30d"]=round(md.get("price_change_percentage_30d",0) or 0,2)
                data["ath"]=md.get("ath",{}).get("usd",0)
                data["ath_change_pct"]=round(md.get("ath_change_percentage",{}).get("usd",0) or 0,2)
                data["circulating_supply"]=md.get("circulating_supply",0)
                data["max_supply"]=md.get("max_supply",0)
                comm=d.get("community_data",{})
                data["twitter_followers"]=comm.get("twitter_followers",0)
                data["reddit_subscribers"]=comm.get("reddit_subscribers",0)
                try:
                    tr=requests.get("https://api.coingecko.com/api/v3/search/trending",timeout=8)
                    if tr.status_code==200:
                        trending=[c["item"]["symbol"].upper() for c in tr.json().get("coins",[])]
                        data["trending"]=coin in trending
                except: data["trending"]=False
        except Exception as ex: logger.error("onchain coingecko: "+str(ex))
    if coin=="BTC":
        try:
            r=requests.get("https://blockchain.info/stats?format=json",timeout=8)
            if r.status_code==200:
                d=r.json()
                data["btc_hashrate"]=round(d.get("hash_rate",0)/1e9,2)
                data["btc_mempool_size"]=d.get("mempool_size",0)
                data["btc_txs_per_day"]=d.get("n_tx",0)
        except Exception as ex: logger.error("onchain btc: "+str(ex))
    return data

def get_dominance():
    try:
        r=requests.get("https://api.coingecko.com/api/v3/global",timeout=10)
        if r.status_code==200:
            d=r.json().get("data",{})
            return {
                "btc_dom":round(d.get("market_cap_percentage",{}).get("btc",0),2),
                "eth_dom":round(d.get("market_cap_percentage",{}).get("eth",0),2),
                "total_mcap":d.get("total_market_cap",{}).get("usd",0),
                "mcap_change_24h":round(d.get("market_cap_change_percentage_24h_usd",0),2),
            }
    except Exception as ex: logger.error("dominance: "+str(ex))
    return {}

def get_funding_rate(symbol):
    try:
        coin=symbol.replace("/USDT","").replace("/USD","")
        r=requests.get("https://fapi.binance.com/fapi/v1/fundingRate?symbol="+coin+"USDT&limit=8",timeout=8)
        if r.status_code==200:
            rates=r.json()
            if rates:
                latest=float(rates[-1].get("fundingRate",0))*100
                avg=round(sum(float(x.get("fundingRate",0)) for x in rates)/len(rates)*100,6)
                sent="Bullish" if latest<-0.005 else "Bearish" if latest>0.01 else "Neutral"
                return {"funding_rate":round(latest,6),"avg_funding":avg,"funding_sentiment":sent}
    except Exception as ex: logger.error("funding: "+str(ex))
    return {}

def get_open_interest(symbol):
    try:
        coin=symbol.replace("/USDT","").replace("/USD","")
        r=requests.get("https://fapi.binance.com/fapi/v1/openInterest?symbol="+coin+"USDT",timeout=8)
        if r.status_code==200:
            return {"open_interest":round(float(r.json().get("openInterest",0)),2)}
    except Exception as ex: logger.error("oi: "+str(ex))
    return {}

def get_long_short_ratio(symbol):
    try:
        coin=symbol.replace("/USDT","").replace("/USD","")
        r=requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol="+coin+"USDT&period=1h&limit=1",timeout=8)
        if r.status_code==200:
            d=r.json()
            if d:
                ls=float(d[0].get("longShortRatio",1))
                long_pct=round(float(d[0].get("longAccount",0))*100,2)
                short_pct=round(float(d[0].get("shortAccount",0))*100,2)
                sent="Crowded Long (fade warning)" if ls>2.0 else "Crowded Short (squeeze potential)" if ls<0.5 else "Balanced"
                return {"ls_ratio":round(ls,3),"long_pct":long_pct,"short_pct":short_pct,"ls_sentiment":sent}
    except Exception as ex: logger.error("ls_ratio: "+str(ex))
    return {}

def get_taker_volume(symbol):
    try:
        coin=symbol.replace("/USDT","").replace("/USD","")
        r=requests.get("https://fapi.binance.com/futures/data/takerlongshortRatio?symbol="+coin+"USDT&period=1h&limit=3",timeout=8)
        if r.status_code==200:
            d=r.json()
            if d:
                buy=round(float(d[-1].get("buyVol",0)),2)
                sell=round(float(d[-1].get("sellVol",0)),2)
                ratio=round(buy/sell,3) if sell>0 else 1
                aggressor="Buyers aggressive" if ratio>1.1 else "Sellers aggressive" if ratio<0.9 else "Balanced"
                return {"taker_buy":buy,"taker_sell":sell,"taker_ratio":ratio,"taker_sentiment":aggressor}
    except Exception as ex: logger.error("taker_vol: "+str(ex))
    return {}

def get_liquidations(symbol):
    try:
        coin=symbol.replace("/USDT","").replace("/USD","")
        r=requests.get("https://fapi.binance.com/fapi/v1/allForceOrders?symbol="+coin+"USDT&limit=20",timeout=8)
        if r.status_code==200:
            orders=r.json()
            long_liqs=sum(float(o.get("origQty",0)) for o in orders if o.get("side")=="SELL")
            short_liqs=sum(float(o.get("origQty",0)) for o in orders if o.get("side")=="BUY")
            return {"long_liquidations":round(long_liqs,2),"short_liquidations":round(short_liqs,2)}
    except Exception as ex: logger.error("liqs: "+str(ex))
    return {}

def get_orderbook_imbalance(symbol):
    try:
        ob=exchange.fetch_order_book(symbol,limit=20)
        bid_vol=sum(x[1] for x in ob["bids"][:10])
        ask_vol=sum(x[1] for x in ob["asks"][:10])
        total=bid_vol+ask_vol
        if total==0: return {}
        imbalance=round((bid_vol-ask_vol)/total*100,2)
        return {"bid_vol":round(bid_vol,4),"ask_vol":round(ask_vol,4),
                "imbalance_pct":imbalance,"orderbook_bias":"Bullish" if imbalance>10 else "Bearish" if imbalance<-10 else "Neutral"}
    except Exception as ex: logger.error("orderbook: "+str(ex)); return {}

def get_coinbase_premium(symbol):
    try:
        if "BTC" not in symbol and "ETH" not in symbol: return {}
        coin=symbol.replace("/USDT","").replace("/USD","")
        cb=ccxt.coinbase({"enableRateLimit":True})
        bn_ticker=exchange.fetch_ticker(symbol)
        cb_ticker=cb.fetch_ticker(coin+"/USD")
        premium=round((cb_ticker["last"]-bn_ticker["last"])/bn_ticker["last"]*100,4)
        return {"coinbase_price":cb_ticker["last"],"binance_price":bn_ticker["last"],
                "cb_premium_pct":premium,"cb_signal":"Institutional Buying" if premium>0.1 else "Institutional Selling" if premium<-0.1 else "Neutral"}
    except Exception as ex: logger.error("cb_premium: "+str(ex)); return {}

def get_google_trends(symbol):
    try:
        coin=symbol.replace("/USDT","").replace("/USD","").lower()
        r=requests.get("https://trends.google.com/trends/api/dailytrends?hl=en-US&tz=-480&geo=US",timeout=8)
        if r.status_code==200:
            text=r.text[5:] if r.text.startswith(")]}',") else r.text
            data=json.loads(text)
            trending=data.get("default",{}).get("trendingSearchesDays",[])
            for day in trending:
                for search in day.get("trendingSearches",[]):
                    title=search.get("title",{}).get("query","").lower()
                    if coin in title or coin.upper() in title.upper():
                        traffic=search.get("formattedTraffic","0")
                        return {"google_trending":True,"traffic":traffic,"signal":"FOMO Alert - Retail interest spiking!"}
        return {"google_trending":False,"signal":"Not trending on Google"}
    except Exception as ex: logger.error("google_trends: "+str(ex)); return {"google_trending":False,"signal":"Unavailable"}

def get_multi_exchange_price(symbol):
    prices={}
    for exch_id in ["binance","bybit","okx"]:
        try:
            ex=getattr(ccxt,exch_id)({"enableRateLimit":True})
            t=ex.fetch_ticker(symbol); prices[exch_id]=t["last"]
        except: pass
    return prices

# ── Groq AI ───────────────────────────────────────────────────────────────────
def groq_call(system, user, max_tokens=2000):
    r=requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization":"Bearer "+GROQ_API_KEY,"Content-Type":"application/json"},
        json={"model":"llama-3.3-70b-versatile",
              "messages":[{"role":"system","content":system},{"role":"user","content":user}],
              "temperature":0.1,"max_tokens":max_tokens},
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

def ai_full_analyze(symbol, ind, fg=None, risk="moderate", mtf_confluence=None, onchain=None, dom=None, funding=None, ls=None, taker=None, oi=None, liqs=None, ob=None, cb_premium=None, news_parsed=None, divergence=None):
    system="You are an elite crypto quant analyst with access to technical, on-chain, sentiment, derivatives and social data. Return ONLY valid JSON."
    extra=""
    if mtf_confluence: extra+=" MTF_Confluence="+str(mtf_confluence)
    if divergence and divergence!="None": extra+=" RSI_Divergence="+str(divergence)
    if onchain:
        extra+=" MCap_Rank="+str(onchain.get("market_cap_rank","N/A"))
        extra+=" 7d="+str(onchain.get("price_change_7d","N/A"))+"% 30d="+str(onchain.get("price_change_30d","N/A"))+"%"
        extra+=" ATH_dist="+str(onchain.get("ath_change_pct","N/A"))+"% Trending="+str(onchain.get("trending","N/A"))
        if onchain.get("btc_hashrate"): extra+=" BTC_Hash="+str(onchain.get("btc_hashrate"))+"GH/s"
    if dom: extra+=" BTC_Dom="+str(dom.get("btc_dom","N/A"))+"% MCap_Change24h="+str(dom.get("mcap_change_24h","N/A"))+"%"
    if funding: extra+=" Funding="+str(funding.get("funding_rate","N/A"))+"% FundingBias="+str(funding.get("funding_sentiment","N/A"))
    if ls: extra+=" LS_Ratio="+str(ls.get("ls_ratio","N/A"))+" Longs="+str(ls.get("long_pct","N/A"))+"% "+str(ls.get("ls_sentiment","N/A"))
    if taker: extra+=" TakerRatio="+str(taker.get("taker_ratio","N/A"))+" "+str(taker.get("taker_sentiment","N/A"))
    if oi: extra+=" OpenInterest="+str(oi.get("open_interest","N/A"))
    if liqs: extra+=" LongLiqs="+str(liqs.get("long_liquidations","N/A"))+" ShortLiqs="+str(liqs.get("short_liquidations","N/A"))
    if ob: extra+=" OB_Imbalance="+str(ob.get("imbalance_pct","N/A"))+"% OB_Bias="+str(ob.get("orderbook_bias","N/A"))
    if cb_premium and cb_premium.get("cb_premium_pct"): extra+=" CB_Premium="+str(cb_premium.get("cb_premium_pct","N/A"))+"% "+str(cb_premium.get("cb_signal","N/A"))
    if news_parsed: extra+=" News_Sentiment="+str(news_parsed.get("news_sentiment","N/A"))+" News_Score="+str(news_parsed.get("news_score","N/A"))+" News_Impact="+str(news_parsed.get("impact","N/A"))
    tech="Price="+str(ind["price"])+" RSI="+str(ind["rsi"])+"("+ind["rsi_s"]+")"+" MACD="+str(ind["hist"])+"("+ind["macd_s"]+")"+" BB="+ind["bb_s"]+" BBsqueeze="+str(ind["bb_squeeze"])+" EMA9/21/50/200="+str(ind["e9"])+"/"+str(ind["e21"])+"/"+str(ind["e50"])+"/"+str(ind["e200"])+" Above200="+ind["above200"]+" Stoch="+str(ind["stoch"])+"("+ind["stoch_s"]+")"+" WilliamsR="+str(ind["wr"])+" CCI="+str(ind["cci"])+" ROC="+str(ind["roc"])+" OBV="+str(ind["obv"])+" Volume="+ind["vol"]+" Support="+str(ind["sup"])+" Resistance="+str(ind["res"])+" Fib618="+str(ind["f618"])+" GoldenPocket="+str(ind["golden"])+" ATR_SL_Long="+str(ind["atr_sl_long"])+" ATR_SL_Short="+str(ind["atr_sl_short"])+" BullPts="+str(ind["bull_pts"])+" BearPts="+str(ind["bear_pts"])+" Patterns="+str(ind["patterns"])+" FearGreed="+str(fg)
    schema='{"sentiment":"Bullish|Bearish|Neutral","sentiment_score":50,"trend":"Uptrend|Downtrend|Sideways","trend_strength":"Strong|Moderate|Weak","market_structure":"1 sentence","confluence_rating":"Strong Buy|Buy|Neutral|Sell|Strong Sell","trade_setup":{"signal":"Long|Short|No Trade","entry_zone":"low-high","take_profit_1":"p","take_profit_2":"p","take_profit_3":"p","stop_loss":"p","risk_reward":"1:3","timeframe":"Short-term|Medium-term","confidence":"Low|Medium|High","invalidation":"condition"},"key_confluences":["c1","c2","c3","c4"],"reasons":["RSI","MACD","BB","EMA","Volume","Fib","OnChain/Derivatives","News/Sentiment"],"warnings":["w1","w2"],"market_context":"1 sentence"}'
    user="Analyze "+symbol+" risk:"+risk+" return ONLY:\n"+schema+"\nTECH: "+tech+" EXTRA: "+extra
    return groq_call(system, user, max_tokens=2000)

def ai_news_sentiment(symbol, headlines):
    system="You are a crypto news analyst. Return ONLY valid JSON."
    user="Analyze headlines for "+symbol+" return ONLY:\n"+'{"news_sentiment":"Bullish|Bearish|Neutral","news_score":50,"impact":"High|Medium|Low","summary":"1 sentence","trade_bias":"Buy the dip|Sell the rally|Hold|No clear bias"}'+"\nHEADLINES:\n"+"\n".join(headlines[:6])
    return groq_call(system, user, max_tokens=300)

def ai_compare(data):
    system="You are a Crypto Analyst. Return ONLY valid JSON."
    entries="\n".join([s+":RSI="+str(d["rsi"])+",MACD="+str(d["hist"])+",Trend="+d["etrend"]+",BB="+d["bb_s"]+",BullPts="+str(d["bull_pts"])+",BearPts="+str(d["bear_pts"]) for s,d in data.items()])
    user="Rank coins by trade opportunity. Return ONLY:\n"+'{"ranking":[{"symbol":"s","score":80,"signal":"Long|Short|No Trade","reason":"brief"}],"best_opportunity":"symbol","summary":"2 sentences"}'+"\nDATA:\n"+entries
    return groq_call(system, user, max_tokens=600)

def ai_performance(trades):
    wins=len([t for t in trades if t.get("result")=="win"]); total=len(trades)
    system="You are a quant analyst. Return ONLY valid JSON."
    user="Paper trading: Total="+str(total)+" Wins="+str(wins)+" Losses="+str(total-wins)+"\nReturn ONLY: "+'{"win_rate":"x/y","assessment":"2 sentences","improvements":["tip1","tip2"]}'
    return groq_call(system, user, max_tokens=400)

# ── Position Sizing ───────────────────────────────────────────────────────────
def calc_position(account, entry_str, sl_str, signal):
    try:
        balance=float(account.get("balance",0)); risk_pct=float(account.get("risk_pct",1.0))
        max_lev=float(account.get("max_leverage",10))
        if balance<=0: return None
        risk_amount=balance*(risk_pct/100)
        entry=float(str(entry_str).split("-")[0].strip()); sl=float(str(sl_str).strip())
        if entry<=0 or sl<=0: return None
        sl_dist=abs(entry-sl)/entry*100
        if sl_dist<=0: return None
        pos_size=risk_amount/(sl_dist/100)
        rec_lev=min(round(pos_size/(balance*0.20),1),max_lev); rec_lev=max(1.0,rec_lev)
        margin=round(pos_size/rec_lev,2); qty=round(pos_size/entry,6)
        liq=round(entry*(1-1/rec_lev*0.9),4) if signal=="Long" else round(entry*(1+1/rec_lev*0.9),4)
        return {"balance":balance,"currency":account.get("currency","USDT"),"risk_pct":risk_pct,
                "risk_amount":round(risk_amount,2),"entry_price":entry,"stop_loss":sl,
                "sl_distance_pct":round(sl_dist,2),"position_size_usdt":round(pos_size,2),
                "margin_needed":margin,"recommended_leverage":rec_lev,"qty":qty,"liq_price":liq,"signal":signal}
    except Exception as e: logger.error("calc_position: "+str(e)); return None

def check_daily_loss(account, u):
    limit=float(account.get("daily_loss_limit",3.0))
    daily=float(account.get("daily_loss",0))
    balance=float(account.get("balance",0))
    if balance<=0: return False, ""
    loss_pct=daily/balance*100
    if loss_pct>=limit:
        return True, "DAILY LOSS LIMIT HIT ("+str(round(loss_pct,2))+"% of account). STOP TRADING TODAY."
    return False, ""

def fmt(raw):
    raw=raw.upper().strip()
    if "/" not in raw:
        raw=(raw[:-4]+"/USDT") if raw.endswith("USDT") else (raw[:-3]+"/USD") if raw.endswith("USD") else (raw+"/USDT")
    return raw

# ── Alert Builder ─────────────────────────────────────────────────────────────
def build_alert(symbol, d, ind, fg_val=None, fg_lbl=None, fg_e="", news=None, pos=None, onchain=None, funding=None, ob=None, dom=None, ls=None, taker=None, cb=None, mtf=None):
    ts=d.get("trade_setup",{}); sig=ts.get("signal","N/A"); conf=ts.get("confidence","N/A")
    sig_e="🟢 LONG" if sig=="Long" else "🔴 SHORT" if sig=="Short" else "⚪ NO TRADE"
    sent_e="📈" if d.get("sentiment")=="Bullish" else "📉" if d.get("sentiment")=="Bearish" else "➡️"
    gp="✅ YES" if ind["golden"] else "❌ No"
    conf_e="🟢" if conf=="High" else "🟡" if conf=="Medium" else "🔴"
    rsi_e="🔴" if ind["rsi"]>=70 else "🟢" if ind["rsi"]<=30 else "⚪"
    macd_e="📈" if ind["hist"]>0 else "📉"
    stoch_e="🔴" if ind["stoch"]>=80 else "🟢" if ind["stoch"]<=20 else "⚪"
    cr=d.get("confluence_rating","N/A")
    cr_e="🔥" if "Strong Buy" in cr else "🟢" if cr=="Buy" else "🔴" if "Strong Sell" in cr else "🟠" if cr=="Sell" else "⚪"
    reasons="\n".join("  - "+r for r in d.get("reasons",[]))
    confluences="\n".join("  + "+c for c in d.get("key_confluences",[]))
    warnings="\n".join("  ! "+w for w in d.get("warnings",[]))
    patterns=", ".join(ind.get("patterns",["None"]))
    fg_line="\n😱 Fear & Greed: "+fg_e+" "+str(fg_val)+"/100 - "+str(fg_lbl) if fg_val else ""

    out=(
        "=== 🤖 AI TRADE ALERT ULTRA ===\n"
        "📌 "+symbol+"  💰 "+f"{ind['price']:,.4f}"+"\n"
        "🕐 "+datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")+fg_line+"\n"
        "Rating: "+cr_e+" "+cr+"\n"
        "Bull/Bear Score: "+str(ind["bull_pts"])+" / "+str(ind["bear_pts"])+"\n"
    )
    if mtf: out+="MTF Confluence: "+mtf+"\n"
    if ind.get("divergence") and ind["divergence"]!="None":
        out+="RSI Divergence: ⚡ "+ind["divergence"]+"\n"
    out+=(
        "\n--- 📊 MARKET ---\n"
        "Trend: "+d.get("trend","N/A")+" ("+d.get("trend_strength","N/A")+")\n"
        "Structure: "+d.get("market_structure","N/A")+"\n"
        "Sentiment: "+sent_e+" "+d.get("sentiment","N/A")+" ("+str(d.get("sentiment_score","N/A"))+"/100)\n"
        "Context: "+d.get("market_context","N/A")+"\n"
        "Patterns: "+patterns+"\n"
        "\n--- 📉 TECHNICALS ---\n"
        "RSI: "+rsi_e+" "+str(ind["rsi"])+" ("+ind["rsi_s"]+")\n"
        "MACD: "+macd_e+" Hist:"+str(ind["hist"])+" ("+ind["macd_s"]+")\n"
        "Stoch: "+stoch_e+" "+str(ind["stoch"])+" ("+ind["stoch_s"]+")\n"
        "Williams R: "+str(ind["wr"])+"  CCI: "+str(ind["cci"])+"  ROC: "+str(ind["roc"])+"%\n"
        "BB: "+str(ind["bbl"])+" / "+str(ind["bbm"])+" / "+str(ind["bbu"])+" ("+ind["bb_s"]+")\n"
        "BB Squeeze: "+("🔴 YES - Breakout incoming!" if ind["bb_squeeze"] else "No")+"\n"
        "EMA 9/21/50/200: "+str(ind["e9"])+" / "+str(ind["e21"])+" / "+str(ind["e50"])+" / "+str(ind["e200"])+"\n"
        "Above 200 EMA: "+ind["above200"]+"  VWAP: "+str(ind["vwap"])+" ("+ind["vwap_s"]+")\n"
        "OBV: "+str(ind["obv"])+"  ATR: "+str(ind["atr"])+"  Volume: "+ind["vol"]+"\n"
        "ATR Stop Loss: Long="+str(ind["atr_sl_long"])+" / Short="+str(ind["atr_sl_short"])+"\n"
        "\n--- 🌀 FIBONACCI ---\n"
        "0.236: "+str(ind["f236"])+"  0.382: "+str(ind["f382"])+"\n"
        "0.500: "+str(ind["f500"])+"  0.618: "+str(ind["f618"])+"\n"
        "Golden Pocket: "+str(ind["f650"])+" - "+str(ind["f618"])+" | In Pocket: "+gp+"\n"
        "Support: "+str(ind["sup"])+"  Resistance: "+str(ind["res"])+"\n"
    )
    if funding:
        fr_val=funding.get("funding_rate",0)
        fr_e="🟢" if fr_val<-0.005 else "🔴" if fr_val>0.01 else "⚪"
        out+=("\n--- 📊 DERIVATIVES ---\n"
              "Funding Rate: "+fr_e+" "+str(fr_val)+"% ("+funding.get("funding_sentiment","N/A")+")\n"
              "Avg Funding (8x): "+str(funding.get("avg_funding","N/A"))+"%\n")
    if ls:
        out+=("Long/Short Ratio: "+str(ls.get("ls_ratio","N/A"))+" | Longs: "+str(ls.get("long_pct","N/A"))+"% Shorts: "+str(ls.get("short_pct","N/A"))+"%\n"
              "LS Signal: "+ls.get("ls_sentiment","N/A")+"\n")
    if taker:
        tk_e="🟢" if taker.get("taker_ratio",1)>1.1 else "🔴" if taker.get("taker_ratio",1)<0.9 else "⚪"
        out+=("Taker Volume: "+tk_e+" "+taker.get("taker_sentiment","N/A")+" (ratio:"+str(taker.get("taker_ratio","N/A"))+")\n")
    if ob:
        ob_e="🟢" if ob.get("orderbook_bias")=="Bullish" else "🔴" if ob.get("orderbook_bias")=="Bearish" else "⚪"
        out+=("Orderbook: "+ob_e+" "+ob.get("orderbook_bias","N/A")+" (imbalance:"+str(ob.get("imbalance_pct","N/A"))+"%)\n")
    if cb and cb.get("cb_premium_pct"):
        cb_e="🟢" if cb.get("cb_premium_pct",0)>0.1 else "🔴" if cb.get("cb_premium_pct",0)<-0.1 else "⚪"
        out+=("Coinbase Premium: "+cb_e+" "+str(cb.get("cb_premium_pct","N/A"))+"% - "+cb.get("cb_signal","N/A")+"\n")
    if onchain and onchain.get("market_cap_rank"):
        out+=("\n--- 🔗 ON-CHAIN ---\n"
              "Rank: #"+str(onchain.get("market_cap_rank","N/A"))+" | 7d: "+str(onchain.get("price_change_7d","N/A"))+"% | 30d: "+str(onchain.get("price_change_30d","N/A"))+"%\n"
              "ATH Distance: "+str(onchain.get("ath_change_pct","N/A"))+"% | Trending: "+("🔥 YES" if onchain.get("trending") else "No")+"\n")
        if onchain.get("btc_hashrate"):
            out+=("BTC Hashrate: "+str(onchain.get("btc_hashrate","N/A"))+" GH/s | Mempool: "+str(onchain.get("btc_mempool_size","N/A"))+"\n")
    if dom:
        out+=("\n--- 🌍 MARKET ---\n"
              "BTC Dom: "+str(dom.get("btc_dom","N/A"))+"% | ETH Dom: "+str(dom.get("eth_dom","N/A"))+"% | MCap 24h: "+str(dom.get("mcap_change_24h","N/A"))+"%\n")
    if news:
        ns_e="📈" if news.get("news_sentiment")=="Bullish" else "📉" if news.get("news_sentiment")=="Bearish" else "➡️"
        out+=("\n--- 📰 NEWS ---\n"
              "Sentiment: "+ns_e+" "+news.get("news_sentiment","N/A")+" ("+str(news.get("news_score","N/A"))+"/100) | Impact: "+news.get("impact","N/A")+"\n"
              "Bias: "+news.get("trade_bias","N/A")+" | "+news.get("summary","N/A")+"\n")
    out+=(
        "\n--- 🎯 TRADE SETUP ---\n"
        "Signal: "+sig_e+"  Confidence: "+conf_e+" "+conf+"\n"
        "Entry Zone: "+ts.get("entry_zone","N/A")+"\n"
        "TP1: "+ts.get("take_profit_1","N/A")+" 🎯\n"
        "TP2: "+ts.get("take_profit_2","N/A")+" 🎯🎯\n"
        "TP3: "+ts.get("take_profit_3","N/A")+" 🎯🎯🎯\n"
        "SL: "+ts.get("stop_loss","N/A")+" 🛑  R:R: "+ts.get("risk_reward","N/A")+"\n"
        "Invalidation: "+ts.get("invalidation","N/A")+"\n"
        "\n--- ✅ CONFLUENCES ---\n"+confluences+"\n"
        "\n--- 💡 REASONING ---\n"+reasons+"\n"
        "\n--- ⚠️ WARNINGS ---\n"+warnings+"\n"
    )
    if pos:
        out+=(
            "\n--- 💰 POSITION SIZING ---\n"
            "Account: $"+f"{pos['balance']:,.2f}"+" "+pos["currency"]+"\n"
            "Risk: "+str(pos["risk_pct"])+"% = $"+str(pos["risk_amount"])+" max loss\n"
            "SL Distance: "+str(pos["sl_distance_pct"])+"%\n"
            "Position Size: $"+f"{pos['position_size_usdt']:,.2f}"+"\n"
            "Margin: $"+f"{pos['margin_needed']:,.2f}"+"  Leverage: "+str(pos["recommended_leverage"])+"x\n"
            "Quantity: "+str(pos["qty"])+" coins\n"
            "Est. Liquidation: $"+str(pos["liq_price"])+" 💀\n"
            "Max Loss: $"+str(pos["risk_amount"])+"\n"
        )
    out+="\n⚠️ Not financial advice. Always DYOR."
    return out

# ── Background Monitor ────────────────────────────────────────────────────────
def quality_score(ind, funding=None, ls=None, taker=None, ob=None, mtf=None, news=None):
    score=0; reasons=[]
    bull=ind["bull_pts"]; bear=ind["bear_pts"]
    # Need strong technical edge
    if bull>=7: score+=3; reasons.append("Strong bull confluence ("+str(bull)+" pts)")
    elif bull>=5: score+=2; reasons.append("Good bull confluence ("+str(bull)+" pts)")
    if bear>=7: score-=3
    elif bear>=5: score-=2
    # Divergence is powerful
    if "Bullish" in str(ind.get("divergence","")): score+=2; reasons.append("Bullish RSI divergence")
    if "Bearish" in str(ind.get("divergence","")): score-=2
    # BB squeeze = breakout setup
    if ind["bb_squeeze"]: score+=1; reasons.append("BB squeeze breakout setup")
    # Golden pocket = high probability entry
    if ind["golden"]: score+=2; reasons.append("Price in Golden Pocket")
    # Above 200 EMA = healthy trend
    if ind["above200"]=="Yes": score+=1; reasons.append("Above 200 EMA")
    # MTF confluence
    if mtf and "Strong Bullish" in str(mtf): score+=2; reasons.append("Strong MTF confluence")
    elif mtf and "Strong Bearish" in str(mtf): score-=2
    # Funding - negative = bullish (shorts paying longs)
    if funding:
        fr=float(funding.get("funding_rate",0))
        if fr<-0.005: score+=1; reasons.append("Negative funding (shorts paying)")
        if fr>0.02: score-=1
    # LS ratio - crowded one side = fade signal
    if ls:
        ratio=float(ls.get("ls_ratio",1))
        if ratio>2.5: score-=1; reasons.append("Crowded long (fade risk)")
        if ratio<0.4: score+=1; reasons.append("Crowded short (squeeze risk)")
    # Taker aggression
    if taker:
        tr=float(taker.get("taker_ratio",1))
        if tr>1.2: score+=1; reasons.append("Aggressive buyers (taker ratio "+str(tr)+")")
        if tr<0.8: score-=1
    # Orderbook
    if ob:
        imp=float(ob.get("imbalance_pct",0))
        if imp>20: score+=1; reasons.append("Strong buy wall in orderbook")
        if imp<-20: score-=1
    # News
    if news:
        if news.get("news_sentiment")=="Bullish" and news.get("impact")=="High":
            score+=1; reasons.append("Bullish high-impact news")
        if news.get("news_sentiment")=="Bearish" and news.get("impact")=="High":
            score-=1
    return score, reasons

async def auto_scanner(app):
    await asyncio.sleep(90)
    logger.info("Auto-scanner started")
    # Track last signal time per user+symbol to avoid spam
    last_signal={}
    while True:
        try:
            all_watchlists=load_db("watchlists")
            all_settings=load_db("settings")
            all_autoscan=load_db("autoscan")
            for uid_str, symbols in all_watchlists.items():
                user_settings=all_settings.get(uid_str,{})
                if not user_settings.get("notifications",True): continue
                autoscan_cfg=all_autoscan.get(uid_str,{})
                if not autoscan_cfg.get("enabled",False): continue
                min_score=int(autoscan_cfg.get("min_score",6))
                scan_interval=int(autoscan_cfg.get("interval",60))
                for symbol in symbols[:10]:
                    try:
                        sig_key=uid_str+":"+symbol
                        last=last_signal.get(sig_key,0)
                        if (datetime.utcnow().timestamp()-last) < scan_interval*60:
                            continue
                        logger.info("Auto-scanning "+symbol+" for user "+uid_str)
                        ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200)
                        ind=build_ind(ohlcv)
                        # Quick score check before expensive API calls
                        quick_score,_=quality_score(ind)
                        if quick_score<3: continue
                        # Full data fetch
                        funding=get_funding_rate(symbol); ls=get_long_short_ratio(symbol)
                        taker=get_taker_volume(symbol); ob=get_orderbook_imbalance(symbol)
                        mtf_results=build_multi_tf(symbol); mtf=get_mtf_confluence(mtf_results)
                        headlines=get_news(symbol); news_parsed=None
                        if headlines[0]!="No recent news found":
                            try: news_parsed=parse_json(ai_news_sentiment(symbol,headlines))
                            except: pass
                        score,score_reasons=quality_score(ind,funding,ls,taker,ob,mtf,news_parsed)
                        if score<min_score:
                            logger.info(symbol+" score="+str(score)+" below threshold "+str(min_score)+", skip")
                            continue
                        # Score is high enough - run full AI analysis
                        logger.info(symbol+" score="+str(score)+" ABOVE threshold, running AI...")
                        user_risk=user_settings.get("risk","moderate")
                        fg_val,fg_lbl,fg_e=get_fg(); dom=get_dominance()
                        onchain=get_onchain(symbol); oi=get_open_interest(symbol)
                        liqs=get_liquidations(symbol); cb=get_coinbase_premium(symbol)
                        raw=ai_full_analyze(symbol,ind,fg_val,user_risk,mtf,onchain,dom,funding,ls,taker,oi,liqs,ob,cb,news_parsed,ind["divergence"])
                        parsed=parse_json(raw)
                        if not parsed: continue
                        ts=parsed.get("trade_setup",{})
                        signal=ts.get("signal","No Trade")
                        if signal not in ["Long","Short"]: continue
                        conf=ts.get("confidence","Low")
                        if conf=="Low": continue
                        # Build and send the alert
                        account=load_db("accounts").get(uid_str,{"balance":0})
                        pos=None
                        if float(account.get("balance",0))>0:
                            pos=calc_position(account,ts.get("entry_zone","0"),ts.get("stop_loss","0"),signal)
                        alert_text=(
                            "🚨 AUTO-SCAN ALERT 🚨\n"
                            "Quality Score: "+str(score)+"/10\n"
                            "Reasons: "+", ".join(score_reasons[:3])+"\n\n"
                            +build_alert(symbol,parsed,ind,fg_val,fg_lbl,fg_e,news_parsed,pos,onchain,funding,ob,dom,ls,taker,cb,mtf)
                        )
                        # Send in chunks if needed
                        for chunk in [alert_text[i:i+4000] for i in range(0,len(alert_text),4000)]:
                            await app.bot.send_message(chat_id=int(uid_str),text=chunk)
                        if headlines[0]!="No recent news found":
                            await app.bot.send_message(chat_id=int(uid_str),text="📰 "+symbol+" Headlines:\n\n"+"\n".join(headlines[:4]))
                        last_signal[sig_key]=datetime.utcnow().timestamp()
                        logger.info("Auto-scan alert sent for "+symbol+" to "+uid_str)
                        # Save to journal
                        journal=load_db("journal").get(uid_str,[])
                        journal.append({"symbol":symbol,"signal":signal,"entry":ts.get("entry_zone",""),
                            "tp1":ts.get("take_profit_1",""),"sl":ts.get("stop_loss",""),
                            "rr":ts.get("risk_reward",""),"conf":conf,"auto":True,
                            "score":score,"time":datetime.utcnow().isoformat(),"status":"open","result":"pending",
                            "indicators":{"rsi":ind["rsi"],"bull_pts":ind["bull_pts"],"bear_pts":ind["bear_pts"],"divergence":ind["divergence"],"mtf":mtf}})
                        d=load_db("journal"); d[uid_str]=journal; save_db("journal",d)
                    except Exception as e: logger.error("Auto-scan "+symbol+": "+str(e))
                await asyncio.sleep(5)
        except Exception as e: logger.error("Auto-scanner loop: "+str(e))
        await asyncio.sleep(300)

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
                        if ind["golden"]: alerts.append("Price in Golden Pocket!")
                        if ind["stoch"]>=80: alerts.append("Stochastic Overbought")
                        if ind["stoch"]<=20: alerts.append("Stochastic Oversold")
                        if ind["bb_squeeze"]: alerts.append("BB Squeeze - Breakout incoming!")
                        if abs(ind["cci"])>150: alerts.append("CCI Extreme: "+str(ind["cci"]))
                        if ind["divergence"]!="None": alerts.append("RSI Divergence: "+ind["divergence"])
                        custom=load_db("alert_levels").get(uid_str,{}).get(symbol,{})
                        if custom.get("price_above") and ind["price"]>=float(custom["price_above"]):
                            alerts.append("Price crossed ABOVE "+str(custom["price_above"]))
                        if custom.get("price_below") and ind["price"]<=float(custom["price_below"]):
                            alerts.append("Price crossed BELOW "+str(custom["price_below"]))
                        try:
                            ls=get_long_short_ratio(symbol)
                            if ls.get("ls_ratio",1)>2.5: alerts.append("Crowded Long! LS Ratio: "+str(ls.get("ls_ratio")))
                            if ls.get("ls_ratio",1)<0.4: alerts.append("Short Squeeze Potential! LS: "+str(ls.get("ls_ratio")))
                        except: pass
                        if alerts:
                            msg="🔔 "+symbol+"\n\n"+"\n".join(alerts)+"\nPrice: "+f"{ind['price']:,.4f}"
                            await app.bot.send_message(chat_id=int(uid_str),text=msg)
                    except Exception as e: logger.error("Monitor "+symbol+": "+str(e))
        except Exception as e: logger.error("Monitor loop: "+str(e))
        await asyncio.sleep(900)

# ── Commands ──────────────────────────────────────────────────────────────────
async def start(update, context):
    await safe_send(update,
        "🤖 AI Crypto Bot ULTRA\n\n"
        "📊 ANALYSIS\n"
        "/analyze BTC - Full analysis (everything)\n"
        "/price BTC\n"
        "/indicators BTC\n"
        "/multitf BTC\n"
        "/compare BTC ETH SOL\n"
        "/feargreed\n"
        "/news BTC\n"
        "/market\n"
        "/derivatives BTC - Funding, LS ratio, OI\n"
        "/orderbook BTC\n\n"
        "📋 WATCHLIST\n"
        "/watchlist\n"
        "/addwatch BTC\n"
        "/removewatch BTC\n"
        "/setalert BTC 95000 90000\n"
        "/alerts on or off\n"
        "/scan\n"
        "/autoscan - Auto trade alerts (set and forget!)\n\n"
        "📈 TRADING\n"
        "/trade BTC\n\n"
        "📒 PAPER TRADING\n"
        "/papertrades\n"
        "/closetrade BTC win or loss\n"
        "/performance\n"
        "/journal - View trade journal\n\n"
        "💼 PORTFOLIO\n"
        "/portfolio\n"
        "/addholding BTC 0.5 95000\n"
        "/removeholding BTC\n\n"
        "💰 ACCOUNT\n"
        "/setaccount 1000 1 10 3\n"
        "  (balance, risk%, max lev, daily loss limit%)\n"
        "/account\n"
        "/possize BTC 95000 92000\n\n"
        "⚙️ SETTINGS\n"
        "/settings\n"
        "/setrisk low or moderate or high\n"
        "/help"
    )

async def help_cmd(update, context): await start(update, context)

async def price_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /price BTC"); return
    symbol=fmt(context.args[0])
    try:
        t=exchange.fetch_ticker(symbol); c=t.get("percentage",0) or 0
        prices=get_multi_exchange_price(symbol)
        price_line="\n\nMulti-Exchange:\n"+"  ".join([ex+": "+f"{p:,.4f}" for ex,p in prices.items() if p])
        await safe_send(update,
            symbol+"\nPrice: "+f"{t['last']:,.4f}"+"\n24h: "+f"{c:+.2f}%"+
            "\nHigh: "+f"{t['high']:,.4f}"+"  Low: "+f"{t['low']:,.4f}"+
            "\nVolume: "+f"{t['quoteVolume']:,.2f}"+price_line
        )
    except Exception as e: logger.error("price_cmd: "+str(e)); await safe_send(update,"Error: "+str(e))

async def market_cmd(update, context):
    msg=await update.message.reply_text("Fetching global market data...")
    try:
        dom=get_dominance(); fg_val,fg_lbl,fg_e=get_fg()
        await msg.delete()
        out="🌍 Global Crypto Market\n\n"
        if dom:
            mcap=dom.get("total_mcap",0)
            out+="Total Market Cap: $"+f"{mcap/1e9:.1f}"+"B\n"
            out+="24h Change: "+str(dom.get("mcap_change_24h","N/A"))+"%\n"
            out+="BTC Dominance: "+str(dom.get("btc_dom","N/A"))+"%\n"
            out+="ETH Dominance: "+str(dom.get("eth_dom","N/A"))+"%\n"
        if fg_val:
            out+="\nFear & Greed: "+fg_e+" "+str(fg_val)+"/100 - "+fg_lbl
        await safe_send(update,out)
    except Exception as e:
        logger.error("market_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def feargreed_cmd(update, context):
    try:
        v,lbl,e=get_fg()
        if v:
            bar="X"*(v//10)+"-"*(10-v//10)
            await safe_send(update,"😱 Fear & Greed\n\n"+e+" "+str(v)+"/100 - "+lbl+"\n["+bar+"]\n\n0=Extreme Fear  100=Extreme Greed\n\nHigh fear = potential buy\nHigh greed = potential sell")
        else: await safe_send(update,"Could not fetch.")
    except Exception as e: await safe_send(update,"Error: "+str(e))

async def news_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /news BTC"); return
    symbol=fmt(context.args[0])
    msg=await update.message.reply_text("Fetching news for "+symbol+"...")
    try:
        headlines=get_news(symbol); news_parsed=None
        if headlines[0]!="No recent news found":
            try: news_parsed=parse_json(ai_news_sentiment(symbol,headlines))
            except Exception as e: logger.error("news ai: "+str(e))
        await msg.delete()
        lines=["📰 "+symbol+" News\n"]
        if news_parsed:
            ns_e="📈" if news_parsed.get("news_sentiment")=="Bullish" else "📉" if news_parsed.get("news_sentiment")=="Bearish" else "➡️"
            lines.append("AI Sentiment: "+ns_e+" "+news_parsed.get("news_sentiment","N/A")+" ("+str(news_parsed.get("news_score","N/A"))+"/100)")
            lines.append("Impact: "+news_parsed.get("impact","N/A")+"  Bias: "+news_parsed.get("trade_bias","N/A"))
            lines.append("Summary: "+news_parsed.get("summary","N/A")+"\n")
        lines.append("Latest Headlines:")
        for h in headlines[:8]: lines.append(h)
        await safe_send(update,"\n".join(lines))
    except Exception as e:
        logger.error("news_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def derivatives_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /derivatives BTC"); return
    symbol=fmt(context.args[0])
    msg=await update.message.reply_text("Fetching derivatives data for "+symbol+"...")
    try:
        funding=get_funding_rate(symbol); ls=get_long_short_ratio(symbol)
        oi=get_open_interest(symbol); taker=get_taker_volume(symbol); liqs=get_liquidations(symbol)
        await msg.delete()
        out="📊 "+symbol+" Derivatives\n\n"
        if funding:
            fr=funding.get("funding_rate",0)
            fr_e="🟢" if fr<-0.005 else "🔴" if fr>0.01 else "⚪"
            out+="Funding Rate: "+fr_e+" "+str(fr)+"% ("+funding.get("funding_sentiment","N/A")+")\n"
            out+="Avg Funding (8x): "+str(funding.get("avg_funding","N/A"))+"%\n\n"
        if ls:
            out+="Long/Short Ratio: "+str(ls.get("ls_ratio","N/A"))+"\n"
            out+="Longs: "+str(ls.get("long_pct","N/A"))+"% | Shorts: "+str(ls.get("short_pct","N/A"))+"%\n"
            out+="Signal: "+ls.get("ls_sentiment","N/A")+"\n\n"
        if taker:
            tk_e="🟢" if taker.get("taker_ratio",1)>1.1 else "🔴" if taker.get("taker_ratio",1)<0.9 else "⚪"
            out+="Taker Buy/Sell: "+tk_e+" Ratio="+str(taker.get("taker_ratio","N/A"))+"\n"
            out+=taker.get("taker_sentiment","N/A")+"\n\n"
        if oi: out+="Open Interest: "+str(oi.get("open_interest","N/A"))+" contracts\n"
        if liqs:
            out+="\nRecent Liquidations:\n"
            out+="Long Liqs: "+str(liqs.get("long_liquidations","N/A"))+"\n"
            out+="Short Liqs: "+str(liqs.get("short_liquidations","N/A"))+"\n"
        await safe_send(update,out)
    except Exception as e:
        logger.error("derivatives_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def orderbook_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /orderbook BTC"); return
    symbol=fmt(context.args[0])
    try:
        ob=exchange.fetch_order_book(symbol,limit=10)
        ob_data=get_orderbook_imbalance(symbol)
        out="📖 "+symbol+" Order Book\n\n"
        out+="Top Asks (Sell walls):\n"
        for p,v in list(reversed(ob["asks"][:5])):
            out+="  $"+f"{p:,.4f}"+" | "+f"{v:.4f}"+"\n"
        out+="\nBest Bid/Ask: $"+f"{ob['bids'][0][0]:,.4f}"+" / $"+f"{ob['asks'][0][0]:,.4f}"+"\n\nTop Bids (Buy walls):\n"
        for p,v in ob["bids"][:5]:
            out+="  $"+f"{p:,.4f}"+" | "+f"{v:.4f}"+"\n"
        if ob_data:
            ob_e="🟢" if ob_data.get("orderbook_bias")=="Bullish" else "🔴" if ob_data.get("orderbook_bias")=="Bearish" else "⚪"
            out+="\nImbalance: "+ob_e+" "+ob_data.get("orderbook_bias","N/A")+" ("+str(ob_data.get("imbalance_pct","N/A"))+"%)\n"
        await safe_send(update,out)
    except Exception as e: logger.error("orderbook_cmd: "+str(e)); await safe_send(update,"Error: "+str(e))

async def indicators_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /indicators BTC"); return
    symbol=fmt(context.args[0])
    msg=await update.message.reply_text("Calculating indicators for "+symbol+"...")
    try:
        ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200); ind=build_ind(ohlcv)
        await msg.delete()
        await safe_send(update,
            symbol+" Indicators\n\n"
            "Price: "+f"{ind['price']:,.4f}"+"  ATR: "+str(ind["atr"])+"  VWAP: "+str(ind["vwap"])+"\n\n"
            "RSI(14): "+str(ind["rsi"])+" - "+ind["rsi_s"]+"\n"
            "MACD Hist: "+str(ind["hist"])+" - "+ind["macd_s"]+"\n"
            "Stochastic: "+str(ind["stoch"])+" - "+ind["stoch_s"]+"\n"
            "Williams R: "+str(ind["wr"])+" | CCI: "+str(ind["cci"])+" | ROC: "+str(ind["roc"])+"%\n"
            "OBV: "+str(ind["obv"])+"\n\n"
            "BB: "+str(ind["bbl"])+" / "+str(ind["bbm"])+" / "+str(ind["bbu"])+" ("+ind["bb_s"]+")\n"
            "BB Width: "+str(ind["bb_width"])+"% | Squeeze: "+str(ind["bb_squeeze"])+"\n\n"
            "EMA 9/21/50/200: "+str(ind["e9"])+" / "+str(ind["e21"])+" / "+str(ind["e50"])+" / "+str(ind["e200"])+"\n"
            "Above 200 EMA: "+ind["above200"]+"\n\n"
            "RSI Divergence: "+ind["divergence"]+"\n\n"
            "ATR Stop Loss: Long="+str(ind["atr_sl_long"])+" Short="+str(ind["atr_sl_short"])+"\n\n"
            "Support: "+str(ind["sup"])+"  Resistance: "+str(ind["res"])+"\n"
            "Volume: "+ind["vol"]+" | Patterns: "+", ".join(ind["patterns"])+"\n\n"
            "Bull Score: "+str(ind["bull_pts"])+" / Bear Score: "+str(ind["bear_pts"])
        )
    except Exception as e:
        logger.error("indicators_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def multitf_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /multitf BTC"); return
    symbol=fmt(context.args[0])
    msg=await update.message.reply_text("Running multi-timeframe analysis for "+symbol+"...")
    try:
        results=build_multi_tf(symbol); mtf=get_mtf_confluence(results)
        lines=[symbol+" Multi-Timeframe\n\nOverall: "+mtf+"\n"]
        for tf,ind in results.items():
            lines.append(tf+": "+ind["etrend"]+"  RSI:"+str(ind["rsi"])+"  MACD:"+("Up" if ind["hist"]>0 else "Down")+"  Bull:"+str(ind["bull_pts"])+" Bear:"+str(ind["bear_pts"])+"  Div:"+ind["divergence"][:20])
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
            lines=["Comparison\n\nBest: "+parsed.get("best_opportunity","N/A")+"\n"+parsed.get("summary","")+"\n\nRankings:"]
            for i,r in enumerate(parsed.get("ranking",[]),1):
                lines.append(str(i)+". "+r.get("symbol","")+" Score:"+str(r.get("score"))+"/100 - "+r.get("signal","")+"\n   "+r.get("reason",""))
            await safe_send(update,"\n".join(lines))
        else: await safe_send(update,"AI response:\n"+raw[:500])
    except Exception as e:
        logger.error("compare_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def analyze_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /analyze BTC"); return
    u=update.effective_user.id; symbol=fmt(context.args[0]); settings=get_settings(u)
    account=get_account(u)
    limit_hit, limit_msg=check_daily_loss(account,u)
    if limit_hit: await safe_send(update,"🛑 "+limit_msg); return
    msg=await update.message.reply_text("Step 1/8: Fetching price data...")
    try:
        ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200); ind=build_ind(ohlcv)
        await msg.edit_text("Step 2/8: Multi-timeframe analysis...")
        mtf_results=build_multi_tf(symbol); mtf_confluence=get_mtf_confluence(mtf_results)
        await msg.edit_text("Step 3/8: Fetching news...")
        headlines=get_news(symbol); news_parsed=None
        if headlines[0]!="No recent news found":
            try: news_parsed=parse_json(ai_news_sentiment(symbol,headlines))
            except Exception as ne: logger.error("news: "+str(ne))
        await msg.edit_text("Step 4/8: Fetching Fear & Greed + Market dominance...")
        fg_val,fg_lbl,fg_e=get_fg(); dom=get_dominance()
        await msg.edit_text("Step 5/8: Fetching on-chain data...")
        onchain=get_onchain(symbol)
        await msg.edit_text("Step 6/8: Fetching derivatives...")
        funding=get_funding_rate(symbol); ls=get_long_short_ratio(symbol)
        taker=get_taker_volume(symbol); oi=get_open_interest(symbol)
        liqs=get_liquidations(symbol)
        await msg.edit_text("Step 7/8: Orderbook + Coinbase premium...")
        ob=get_orderbook_imbalance(symbol); cb=get_coinbase_premium(symbol)
        await msg.edit_text("Step 8/8: Running AI analysis...")
        raw=ai_full_analyze(symbol,ind,fg_val,settings.get("risk","moderate"),mtf_confluence,onchain,dom,funding,ls,taker,oi,liqs,ob,cb,news_parsed,ind["divergence"])
        parsed=parse_json(raw)
        await msg.delete()
        if parsed:
            ts=parsed.get("trade_setup",{})
            pos=None
            if account.get("balance",0)>0 and ts.get("signal") in ["Long","Short"]:
                pos=calc_position(account,ts.get("entry_zone","0"),ts.get("stop_loss","0"),ts.get("signal"))
            if ts.get("signal") in ["Long","Short"]:
                trades=get_trades(u)
                entry={"symbol":symbol,"signal":ts["signal"],"entry":ts.get("entry_zone",""),
                    "tp1":ts.get("take_profit_1",""),"tp2":ts.get("take_profit_2",""),
                    "tp3":ts.get("take_profit_3",""),"sl":ts.get("stop_loss",""),
                    "rr":ts.get("risk_reward",""),"conf":ts.get("confidence",""),
                    "time":datetime.utcnow().isoformat(),"status":"open","result":"pending",
                    "indicators":{"rsi":ind["rsi"],"macd":ind["hist"],"bull_pts":ind["bull_pts"],"bear_pts":ind["bear_pts"],"divergence":ind["divergence"],"mtf":mtf_confluence}}
                trades.append(entry)
                save_trades(u,trades)
                journal=get_journal(u); journal.append(entry); save_journal(u,journal)
            await safe_send(update,build_alert(symbol,parsed,ind,fg_val,fg_lbl,fg_e,news_parsed,pos,onchain,funding,ob,dom,ls,taker,cb,mtf_confluence))
            if headlines[0]!="No recent news found":
                await safe_send(update,"📰 "+symbol+" Headlines:\n\n"+"\n".join(headlines[:6]))
        else: await safe_send(update,"AI format error:\n"+raw[:500])
    except Exception as e:
        logger.error("analyze_cmd: "+str(e))
        try: await msg.delete()
        except: pass
        await safe_send(update,"Error: "+str(e))

async def trade_cmd(update, context):
    if not context.args: await safe_send(update,"Usage: /trade BTC"); return
    symbol=fmt(context.args[0])
    kb=InlineKeyboardMarkup([[InlineKeyboardButton("Confirm",callback_data="tc:"+symbol),InlineKeyboardButton("Cancel",callback_data="tx")]])
    await update.message.reply_text("Generate trade setup for "+symbol+"?\nBot does NOT auto-execute.",reply_markup=kb)

async def trade_callback(update, context):
    q=update.callback_query; await q.answer()
    if q.data=="tx": await q.edit_message_text("Cancelled."); return
    if q.data.startswith("tc:"):
        u=q.from_user.id; symbol=q.data[3:]; settings=get_settings(u)
        await q.edit_message_text("Analyzing "+symbol+"...")
        try:
            ohlcv=exchange.fetch_ohlcv(symbol,timeframe="1h",limit=200); ind=build_ind(ohlcv)
            mtf_results=build_multi_tf(symbol); mtf_confluence=get_mtf_confluence(mtf_results)
            fg_val,fg_lbl,fg_e=get_fg(); funding=get_funding_rate(symbol)
            ls=get_long_short_ratio(symbol); taker=get_taker_volume(symbol)
            ob=get_orderbook_imbalance(symbol)
            raw=ai_full_analyze(symbol,ind,fg_val,settings.get("risk","moderate"),mtf_confluence,None,None,funding,ls,taker,None,None,ob,None,None,ind["divergence"])
            parsed=parse_json(raw)
            if parsed:
                ts=parsed.get("trade_setup",{}); account=get_account(u); pos=None
                if account.get("balance",0)>0 and ts.get("signal") in ["Long","Short"]:
                    pos=calc_position(account,ts.get("entry_zone","0"),ts.get("stop_loss","0"),ts.get("signal"))
                if ts.get("signal") in ["Long","Short"]:
                    trades=get_trades(u)
                    trades.append({"symbol":symbol,"signal":ts["signal"],"entry":ts.get("entry_zone",""),
                        "tp1":ts.get("take_profit_1",""),"sl":ts.get("stop_loss",""),
                        "rr":ts.get("risk_reward",""),"time":datetime.utcnow().isoformat(),"status":"open","result":"pending"})
                    save_trades(u,trades)
                result=build_alert(symbol,parsed,ind,fg_val,fg_lbl,fg_e,None,pos,None,funding,ob,None,ls,taker,None,mtf_confluence)+"\n\nPlace order MANUALLY."
                if len(result)>4096: result=result[:4090]+"..."
            else: result="AI format error:\n"+raw[:500]
            await q.edit_message_text(result)
        except Exception as e:
            logger.error("trade_callback: "+str(e)); await q.edit_message_text("Error: "+str(e))

async def scan_cmd(update, context):
    u=update.effective_user.id; wl=get_watchlist(u)
    if not wl: await safe_send(update,"Watchlist empty. Use /addwatch BTC"); return
    msg=await update.message.reply_text("Scanning "+str(len(wl))+" coins...")
    try:
        data={}
        for s in wl[:8]:
            ohlcv=exchange.fetch_ohlcv(s,timeframe="1h",limit=200); data[s]=build_ind(ohlcv)
        raw=ai_compare(data); parsed=parse_json(raw)
        await msg.delete()
        if parsed:
            lines=["Scan Results\n\nBest: "+parsed.get("best_opportunity","N/A")+"\n"+parsed.get("summary","")+"\n\nRankings:"]
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
    lines=["📋 Watchlist\n"]
    for s in wl:
        try:
            t=exchange.fetch_ticker(s); c=t.get("percentage",0) or 0
            lines.append(("🟢" if c>=0 else "🔴")+" "+s+": "+f"{t['last']:,.4f}"+" ("+f"{c:+.2f}%"+")")
        except: lines.append("- "+s)
    await safe_send(update,"\n".join(lines))

async def addwatch_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /addwatch BTC"); return
    symbol=fmt(context.args[0]); wl=get_watchlist(u)
    if symbol in wl: await safe_send(update,symbol+" already in watchlist."); return
    if len(wl)>=10: await safe_send(update,"Max 10 coins."); return
    wl.append(symbol); save_watchlist(u,wl); await safe_send(update,"Added "+symbol)

async def removewatch_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /removewatch BTC"); return
    symbol=fmt(context.args[0]); wl=get_watchlist(u)
    if symbol in wl:
        wl.remove(symbol); save_watchlist(u,wl); await safe_send(update,"Removed "+symbol)
    else: await safe_send(update,symbol+" not in watchlist.")

async def setalert_cmd(update, context):
    u=update.effective_user.id
    if len(context.args)<3: await safe_send(update,"Usage: /setalert BTC 95000 90000"); return
    try:
        symbol=fmt(context.args[0]); above=float(context.args[1]); below=float(context.args[2])
        al=get_alerts(u); al[symbol]={"price_above":above,"price_below":below}; save_alerts(u,al)
        await safe_send(update,"Alerts set for "+symbol+"\nAbove: "+f"{above:,.4f}"+"\nBelow: "+f"{below:,.4f}")
    except ValueError: await safe_send(update,"Invalid prices.")

async def alerts_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /alerts on or off"); return
    s=get_settings(u); s["notifications"]=(context.args[0].lower()=="on"); save_settings(u,s)
    await safe_send(update,"Alerts "+("ON" if s["notifications"] else "OFF"))

async def papertrades_cmd(update, context):
    u=update.effective_user.id; trades=get_trades(u)
    if not trades: await safe_send(update,"No paper trades yet."); return
    open_t=[t for t in trades if t.get("status")=="open"]
    closed_t=[t for t in trades if t.get("status")=="closed"]
    lines=["Paper Trades: "+str(len(open_t))+" open, "+str(len(closed_t))+" closed\n"]
    for t in open_t[-5:]:
        lines.append(("🟢" if t["signal"]=="Long" else "🔴")+" "+t["symbol"]+" "+t["signal"]+"\nEntry:"+t["entry"]+" SL:"+t["sl"]+" R:R:"+t["rr"]+"\n"+t["time"][:16])
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
            if result=="loss":
                account=get_account(u)
                try:
                    pos=calc_position(account,t["entry"],t["sl"],"Long")
                    if pos: account["daily_loss"]=account.get("daily_loss",0)+pos["risk_amount"]; save_account(u,account)
                except: pass
            await safe_send(update,"Closed "+symbol+" as "+result.upper()); return
    await safe_send(update,"No open trade for "+symbol+".")

async def performance_cmd(update, context):
    u=update.effective_user.id; trades=get_trades(u)
    closed=[t for t in trades if t.get("status")=="closed"]
    if not closed: await safe_send(update,"No closed trades yet."); return
    wins=len([t for t in closed if t.get("result")=="win"]); rate=round(wins/len(closed)*100,1)
    msg=await update.message.reply_text("Analyzing performance...")
    try:
        raw=ai_performance(closed); parsed=parse_json(raw)
        await msg.delete()
        lines=["Performance\n\nTotal: "+str(len(closed))+"  Wins: "+str(wins)+"  Losses: "+str(len(closed)-wins)+"\nWin Rate: "+str(rate)+"%"]
        if parsed:
            lines.append("\nAssessment: "+parsed.get("assessment",""))
            for tip in parsed.get("improvements",[]): lines.append("- "+tip)
        await safe_send(update,"\n".join(lines))
    except Exception as e:
        logger.error("performance_cmd: "+str(e))
        await msg.delete(); await safe_send(update,"Error: "+str(e))

async def journal_cmd(update, context):
    u=update.effective_user.id; journal=get_journal(u)
    if not journal: await safe_send(update,"No journal entries yet. Run /analyze to start logging."); return
    lines=["📓 Trade Journal (last 10)\n"]
    for t in journal[-10:]:
        ind_snap=t.get("indicators",{})
        lines.append(("🟢" if t["signal"]=="Long" else "🔴")+" "+t["symbol"]+" "+t["signal"]+" ("+t.get("conf","N/A")+" conf)\n"
                     "Entry:"+t["entry"]+" SL:"+t["sl"]+" R:R:"+t["rr"]+"\n"
                     "RSI:"+str(ind_snap.get("rsi","N/A"))+" Bull:"+str(ind_snap.get("bull_pts","N/A"))+" Bear:"+str(ind_snap.get("bear_pts","N/A"))+"\n"
                     "Divergence:"+str(ind_snap.get("divergence","N/A"))[:30]+"\n"
                     "MTF:"+str(ind_snap.get("mtf","N/A"))[:40]+"\n"
                     "Status:"+t.get("status","open")+" Result:"+t.get("result","pending")+"\n"
                     "Time:"+t["time"][:16]+"\n")
    await safe_send(update,"\n".join(lines))

async def portfolio_cmd(update, context):
    u=update.effective_user.id; pf=get_portfolio(u)
    if not pf: await safe_send(update,"Portfolio empty. Use /addholding BTC 0.5 95000"); return
    lines=["Portfolio\n"]; tv=0; tc=0
    for h in pf:
        try:
            t=exchange.fetch_ticker(h["symbol"]); p=t["last"]
            val=p*h["amount"]; cost=h["buy_price"]*h["amount"]
            pnl=val-cost; pct=(pnl/cost*100) if cost>0 else 0
            tv+=val; tc+=cost
            lines.append(("🟢" if pnl>=0 else "🔴")+" "+h["symbol"]+" x"+str(h["amount"])+"\n"+f"{p:,.4f}"+" | $"+f"{val:,.2f}"+" | "+f"{pct:+.2f}%")
        except: lines.append("- "+h["symbol"])
    tpct=((tv-tc)/tc*100) if tc>0 else 0
    lines.append("\nTotal: $"+f"{tv:,.2f}"+"  PnL: "+f"{tpct:+.2f}%")
    await safe_send(update,"\n".join(lines))

async def addholding_cmd(update, context):
    u=update.effective_user.id
    if len(context.args)<3: await safe_send(update,"Usage: /addholding BTC 0.5 95000"); return
    try:
        symbol=fmt(context.args[0]); amount=float(context.args[1]); price=float(context.args[2])
        pf=get_portfolio(u); pf=[h for h in pf if h["symbol"]!=symbol]
        pf.append({"symbol":symbol,"amount":amount,"buy_price":price}); save_portfolio(u,pf)
        await safe_send(update,"Added "+str(amount)+" "+symbol+" at $"+f"{price:,.4f}")
    except ValueError: await safe_send(update,"Invalid values.")

async def removeholding_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /removeholding BTC"); return
    symbol=fmt(context.args[0]); pf=get_portfolio(u)
    pf=[h for h in pf if h["symbol"]!=symbol]; save_portfolio(u,pf)
    await safe_send(update,"Removed "+symbol)

async def settings_cmd(update, context):
    u=update.effective_user.id; s=get_settings(u)
    await safe_send(update,"Settings\n\nRisk: "+s.get("risk","moderate")+"\nNotifications: "+("ON" if s.get("notifications",True) else "OFF")+"\n\n/setrisk low or moderate or high\n/alerts on or off")

async def setrisk_cmd(update, context):
    u=update.effective_user.id
    if not context.args or context.args[0].lower() not in ["low","moderate","high"]:
        await safe_send(update,"Usage: /setrisk low or moderate or high"); return
    risk=context.args[0].lower(); s=get_settings(u); s["risk"]=risk; save_settings(u,s)
    await safe_send(update,"Risk: "+risk.upper())

async def setaccount_cmd(update, context):
    u=update.effective_user.id
    if not context.args: await safe_send(update,"Usage: /setaccount 1000 1 10 3\n(balance, risk%, max leverage, daily loss limit%)"); return
    try:
        balance=float(context.args[0])
        risk_pct=min(max(float(context.args[1]) if len(context.args)>1 else 1.0,0.1),5.0)
        max_lev=min(max(float(context.args[2]) if len(context.args)>2 else 10.0,1.0),125.0)
        daily_limit=min(max(float(context.args[3]) if len(context.args)>3 else 3.0,0.5),20.0)
        account=get_account(u)
        account.update({"balance":balance,"risk_pct":risk_pct,"max_leverage":max_lev,"currency":"USDT","daily_loss_limit":daily_limit,"daily_loss":0})
        save_account(u,account)
        await safe_send(update,
            "Account set!\n\n"
            "Balance: $"+f"{balance:,.2f}"+"\n"
            "Risk per trade: "+str(risk_pct)+"% = $"+str(round(balance*risk_pct/100,2))+"\n"
            "Max Leverage: "+str(max_lev)+"x\n"
            "Daily Loss Limit: "+str(daily_limit)+"% = $"+str(round(balance*daily_limit/100,2))+"\n\n"
            "Bot will STOP alerting if you hit daily loss limit!"
        )
    except ValueError: await safe_send(update,"Invalid values.")

async def account_cmd(update, context):
    u=update.effective_user.id; account=get_account(u)
    if account.get("balance",0)<=0: await safe_send(update,"No account set. Use /setaccount 1000"); return
    b=account["balance"]; rp=account["risk_pct"]
    daily_loss=account.get("daily_loss",0); daily_limit=account.get("daily_loss_limit",3.0)
    loss_pct=round(daily_loss/b*100,2) if b>0 else 0
    await safe_send(update,
        "Account\n\n"
        "Balance: $"+f"{b:,.2f}"+"\n"
        "Risk: "+str(rp)+"% = $"+str(round(b*rp/100,2))+" per trade\n"
        "Max Leverage: "+str(account.get("max_leverage",10))+"x\n"
        "Daily Loss Limit: "+str(daily_limit)+"% = $"+str(round(b*daily_limit/100,2))+"\n"
        "Today's Loss: $"+str(round(daily_loss,2))+" ("+str(loss_pct)+"%)\n\n"
        "Use /setaccount to update."
    )

async def possize_cmd(update, context):
    u=update.effective_user.id
    if len(context.args)<3: await safe_send(update,"Usage: /possize BTC 95000 92000"); return
    account=get_account(u)
    if account.get("balance",0)<=0: await safe_send(update,"Set account first: /setaccount 1000"); return
    try:
        symbol=fmt(context.args[0]); entry=float(context.args[1]); sl=float(context.args[2])
        signal="Long" if entry>sl else "Short"
        pos=calc_position(account,entry,sl,signal)
        if pos:
            await safe_send(update,
                "Position Calculator\n\n"
                "Symbol: "+symbol+" | Signal: "+("🟢 LONG" if signal=="Long" else "🔴 SHORT")+"\n"
                "Entry: $"+f"{entry:,.4f}"+" | SL: $"+f"{sl:,.4f}"+"\n"
                "SL Distance: "+str(pos["sl_distance_pct"])+"%\n\n"
                "Position Size: $"+f"{pos['position_size_usdt']:,.2f}"+"\n"
                "Margin: $"+f"{pos['margin_needed']:,.2f}"+"  Leverage: "+str(pos["recommended_leverage"])+"x\n"
                "Qty: "+str(pos["qty"])+" coins\n"
                "Liquidation: $"+str(pos["liq_price"])+" 💀\n"
                "Max Loss: $"+str(pos["risk_amount"])
            )
        else: await safe_send(update,"Could not calculate.")
    except ValueError: await safe_send(update,"Invalid values.")

async def autoscan_cmd(update, context):
    u=update.effective_user.id
    d=load_db("autoscan"); cfg=d.get(sid(u),{"enabled":False,"min_score":6,"interval":60})
    # No args = show status
    if not context.args:
        wl=get_watchlist(u)
        status="ON ✅" if cfg.get("enabled") else "OFF ❌"
        await safe_send(update,
            "🤖 Auto-Scanner\n\n"
            "Status: "+status+"\n"
            "Min Quality Score: "+str(cfg.get("min_score",6))+"/10\n"
            "Scan Interval: every "+str(cfg.get("interval",60))+" min\n"
            "Watching: "+(", ".join(wl) if wl else "Nothing - add coins with /addwatch")+"\n\n"
            "Commands:\n"
            "/autoscan on - Start auto-scanner\n"
            "/autoscan off - Stop\n"
            "/autoscan score 7 - Only alert on score 7+/10\n"
            "/autoscan interval 30 - Scan every 30 min\n\n"
            "Score guide:\n"
            "5 = Decent setup\n"
            "6 = Good setup (default)\n"
            "7 = Strong setup\n"
            "8+ = Exceptional only"
        ); return
    arg=context.args[0].lower()
    if arg=="on":
        wl=get_watchlist(u)
        if not wl: await safe_send(update,"Add coins to watchlist first!\n/addwatch BTC\n/addwatch ETH\netc."); return
        cfg["enabled"]=True; d[sid(u)]=cfg; save_db("autoscan",d)
        await safe_send(update,
            "✅ Auto-scanner ENABLED!\n\n"
            "Watching: "+", ".join(wl)+"\n"
            "Min quality score: "+str(cfg["min_score"])+"/10\n"
            "Checking every: "+str(cfg["interval"])+" minutes\n\n"
            "I will message you automatically whenever I find\n"
            "a quality trade setup. No need to ask!\n\n"
            "To stop: /autoscan off"
        )
    elif arg=="off":
        cfg["enabled"]=False; d[sid(u)]=cfg; save_db("autoscan",d)
        await safe_send(update,"❌ Auto-scanner stopped.\n\nUse /autoscan on to restart.")
    elif arg=="score" and len(context.args)>1:
        try:
            score=int(context.args[1])
            if not 3<=score<=10: await safe_send(update,"Score must be 3-10"); return
            cfg["min_score"]=score; d[sid(u)]=cfg; save_db("autoscan",d)
            freq="very frequent" if score<=4 else "frequent" if score<=5 else "moderate" if score<=6 else "selective" if score<=7 else "very selective"
            await safe_send(update,"✅ Min score: "+str(score)+"/10 ("+freq+" signals)")
        except ValueError: await safe_send(update,"Use: /autoscan score 7")
    elif arg=="interval" and len(context.args)>1:
        try:
            mins=int(context.args[1])
            if mins<15: await safe_send(update,"Minimum 15 minutes"); return
            if mins>480: await safe_send(update,"Maximum 480 minutes"); return
            cfg["interval"]=mins; d[sid(u)]=cfg; save_db("autoscan",d)
            await safe_send(update,"✅ Scan interval: every "+str(mins)+" minutes")
        except ValueError: await safe_send(update,"Use: /autoscan interval 60")
    else:
        await safe_send(update,"Options: on, off, score <3-10>, interval <15-480>")

def main():
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    if not GROQ_API_KEY:   raise RuntimeError("GROQ_API_KEY not set")
    app=Application.builder().token(TELEGRAM_TOKEN).build()
    for cmd,fn in [
        ("start",start),("help",help_cmd),("price",price_cmd),("market",market_cmd),
        ("feargreed",feargreed_cmd),("news",news_cmd),("indicators",indicators_cmd),
        ("derivatives",derivatives_cmd),("orderbook",orderbook_cmd),
        ("multitf",multitf_cmd),("compare",compare_cmd),("analyze",analyze_cmd),
        ("trade",trade_cmd),("scan",scan_cmd),("watchlist",watchlist_cmd),
        ("addwatch",addwatch_cmd),("removewatch",removewatch_cmd),
        ("setalert",setalert_cmd),("alerts",alerts_cmd),
        ("papertrades",papertrades_cmd),("closetrade",closetrade_cmd),
        ("performance",performance_cmd),("journal",journal_cmd),
        ("portfolio",portfolio_cmd),("addholding",addholding_cmd),("removeholding",removeholding_cmd),
        ("settings",settings_cmd),("setrisk",setrisk_cmd),
        ("setaccount",setaccount_cmd),("account",account_cmd),("possize",possize_cmd),
        ("autoscan",autoscan_set_cmd),
    ]:
        app.add_handler(CommandHandler(cmd,fn))
    app.add_handler(CallbackQueryHandler(trade_callback,pattern=r"^t[cx]:?"))
    async def post_init(application):
        asyncio.create_task(monitor_alerts(application))
        asyncio.create_task(auto_scanner(application))
    app.post_init=post_init
    logger.info("Bot ULTRA running!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__=="__main__":
    main()
