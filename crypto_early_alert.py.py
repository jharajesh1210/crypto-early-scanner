import os, requests, pandas as pd, numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL="https://api.coindcx.com"; INTERVAL="15m"; KLINE_LIMIT=120; MIN_SCORE=4; MAX_WORKERS=20
TELEGRAM_ENABLED=True
TELEGRAM_BOT_TOKEN=os.getenv("BOT_TOKEN",""); TELEGRAM_CHAT_ID=os.getenv("CHAT_ID","")
OUTPUT_FILE="crypto_early_signals.csv"; ALERT_FILE="telegram_early_alerted_signals.csv"

# Minimum 24-hour volume in USDT (100 Million)
MIN_24H_VOLUME = 100_000_000 

def get_symbols():
    r=requests.get(f"{BASE_URL}/exchange/v1/markets_details",timeout=30); r.raise_for_status()
    
    # 24h ticker data fetch kar rahe hain volume filter ke liye
    ticker_res = requests.get(f"{BASE_URL}/exchange/ticker", timeout=30)
    ticker_data = {}
    if ticker_res.status_code == 200:
        ticker_data = {x.get("market"): float(x.get("volume", 0) or 0) * float(x.get("last_price", 0) or 0) for x in ticker_res.json()}

    out=[]; seen=set()
    for x in r.json():
        symbol=str(x.get("coindcx_name") or x.get("symbol") or "").upper()
        pair=str(x.get("pair") or "")
        
        # Check 24-hour USDT volume
        vol_24h = ticker_data.get(pair, 0)
        
        if str(x.get("status","")).lower()=="active" and str(x.get("base_currency_short_name","")).upper()=="USDT" and symbol and pair:
            if vol_24h >= MIN_24H_VOLUME:  # 100M Volume Condition
                if (symbol,pair) not in seen: 
                    seen.add((symbol,pair))
                    out.append((symbol,pair))
    return out

def get_klines(pair):
    r=requests.get(f"{BASE_URL}/market_data/candles",params={"pair":pair,"interval":INTERVAL,"limit":KLINE_LIMIT},timeout=20)
    if r.status_code!=200: return None
    data=r.json()
    if not isinstance(data,list) or len(data)<3: return None
    df=pd.DataFrame(data); req=["open","high","low","close","volume","time"]
    if not all(c in df.columns for c in req): return None
    for c in ["open","high","low","close","volume","time"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=req).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df["open_time"]=pd.to_datetime(df["time"],unit="ms",utc=True).dt.tz_localize(None)
    df["close_time"]=df["open_time"]+pd.Timedelta(minutes=15)
    return df

def rsi(close,p=14):
    d=close.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    rs=g.ewm(alpha=1/p,adjust=False).mean()/l.ewm(alpha=1/p,adjust=False).mean().replace(0,np.nan)
    return 100-(100/(1+rs))

def indicators(df):
    df["EMA20"]=df.close.ewm(span=20,adjust=False).mean(); df["EMA50"]=df.close.ewm(span=50,adjust=False).mean()
    df["RSI14"]=rsi(df.close); e12=df.close.ewm(span=12,adjust=False).mean(); e26=df.close.ewm(span=26,adjust=False).mean()
    df["MACD"]=e12-e26; df["MACD_SIGNAL"]=df.MACD.ewm(span=9,adjust=False).mean()
    df["BB_MIDDLE"]=df.close.rolling(20).mean(); s=df.close.rolling(20).std()
    df["BB_UPPER"]=df.BB_MIDDLE+2*s; df["BB_LOWER"]=df.BB_MIDDLE-2*s
    df["VOLUME_AVG20"]=df.volume.rolling(20).mean(); df["VOLUME_RATIO"]=df.volume/df.VOLUME_AVG20
    return df

def poc(df,bins=50):
    if len(df)<20:return None
    lo,hi=df.low.min(),df.high.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:return None
    edges=np.linspace(lo,hi,bins+1); vp=np.zeros(bins)
    for _,c in df.iterrows():
        if c.high<=c.low or not np.isfinite(c.volume):continue
        for i in range(bins):
            overlap=max(0,min(c.high,edges[i+1])-max(c.low,edges[i]))
            if overlap>0:vp[i]+=c.volume*overlap/(c.high-c.low)
    if vp.sum()<=0:return None
    i=int(np.argmax(vp)); return (edges[i]+edges[i+1])/2

def send_telegram(msg):
    if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:return False
    try:
        r=requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",data={"chat_id":TELEGRAM_CHAT_ID,"text":msg},timeout=15)
        if r.status_code==200: print("Telegram alert sent successfully."); return True
        print("Telegram send failed:",r.text[:300])
    except Exception as e: print("Telegram error:",e)
    return False

def analyze(symbol,pair):
    df=get_klines(pair)
    if df is None or len(df)<100:return None
    df=indicators(df); sig=df.iloc[-2]; prev=df.iloc[-3]; point=poc(df.iloc[-101:-1].copy())
    if point is None:return None
    price=sig.close; gain=(sig.close-sig.open)/sig.open*100
    a=52<=sig.RSI14<=68
    b=sig.MACD>sig.MACD_SIGNAL and prev.MACD<=prev.MACD_SIGNAL
    c=price>point and price<=point*1.03
    d=sig.EMA20>sig.EMA50
    e=price>=sig.BB_MIDDLE and price<=sig.BB_UPPER*1.01
    f=sig.VOLUME_RATIO>=1.10
    score=sum([a,b,c,d,e,f])
    if not (a and b and c and f and gain<=3.0) or score<MIN_SCORE:return None
    return {"SYMBOL":symbol,"PAIR":pair,"TIME":sig.close_time,"PRICE":round(float(price),8),"SCORE":int(score),
            "RSI14":round(float(sig.RSI14),2),"MACD_BULLISH":bool(b),"PRICE_ABOVE_POC":bool(c),
            "EMA20_ABOVE_EMA50":bool(d),"BB_BREAKOUT":bool(e),"VOLUME_RATIO":round(float(sig.VOLUME_RATIO),2),
            "VOLUME_CONFIRM":bool(f),"POC":round(float(point),8)}

def alert(row):
    symbol=str(row["SYMBOL"]); st=str(row["TIME"])
    try: old=pd.read_csv(ALERT_FILE) if os.path.exists(ALERT_FILE) else pd.DataFrame(columns=["SYMBOL","TIME"])
    except: old=pd.DataFrame(columns=["SYMBOL","TIME"])
    if not old.empty and ((old.SYMBOL.astype(str)==symbol)&(old.TIME.astype(str)==st)).any():return
    ist=(pd.to_datetime(row["TIME"])+pd.Timedelta(hours=5,minutes=30)).strftime("%d-%m-%Y %I:%M:%S %p")
    msg=(f"EARLY CRYPTO ALERT - CoinDCX\n\nSymbol: {symbol}\nTime (IST): {ist}\nPrice: {row['PRICE']}\n"
         f"Score: {row['SCORE']}/6\nRSI: {row['RSI14']}\nVolume Ratio: {row['VOLUME_RATIO']}x\nPOC: {row['POC']}\n\n"
         "Core confirmation:\nRSI Bullish = YES\nFresh MACD Crossover = YES\nPrice Above POC = YES\nVolume Confirm = YES")
    if send_telegram(msg):
        pd.concat([old,pd.DataFrame([{"SYMBOL":symbol,"TIME":st}])],ignore_index=True).to_csv(ALERT_FILE,index=False)

def run_scan():
    print("\n"+"="*70+"\nCOINDCX 15-MINUTE EARLY SIGNAL SCANNER\n"+"="*70)
    print("Time (UTC):",datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    print("\nGetting CoinDCX active USDT Spot markets (Min 100M 24h Vol)...")
    markets=get_symbols(); print("Total qualifying USDT markets:",len(markets))
    if not markets: raise RuntimeError("No active CoinDCX USDT markets found matching volume criteria.")
    signals=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        jobs={ex.submit(analyze,s,p):(s,p) for s,p in markets}
        for n,fut in enumerate(as_completed(jobs),1):
            s,p=jobs[fut]; print(f"\rScanning {n}/{len(markets)} {s}             ",end="",flush=True)
            try:
                x=fut.result()
                if x is not None: signals.append(x); alert(x)
            except Exception as e: print(f"\nError scanning {s} ({p}): {e}")
    print("\n")
    if not signals: print("No qualifying early setup found.\nWait for next CLOSED 15-minute candle."); return
    out=pd.DataFrame(signals).sort_values(["SCORE","VOLUME_RATIO","RSI14"],ascending=[False,False,False])
    out.to_csv(OUTPUT_FILE,index=False); out.to_csv("crypto_early_signal_history.csv",index=False)
    print("="*70); print("EARLY SIGNALS FOUND:",len(out)); print("="*70); print(out.to_string(index=False)); print("\nCSV saved as:",OUTPUT_FILE)

if __name__=="__main__":
    try: run_scan()
    except KeyboardInterrupt: print("\nScanner stopped by user."); raise
    except Exception as e: print("\nFATAL ERROR:",e); raise
