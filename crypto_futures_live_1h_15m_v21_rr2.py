"""CoinDCX Top-50 Futures: 1H trend + 15M paper signals. NO REAL ORDERS.

Dependencies: pip install pandas numpy requests
Persistent files: live_1h15m_state.json, live_1h15m_trades.csv
GitHub Actions must persist these files between runs; otherwise paper trades are lost.
All prices/fees are simulations; OHLC stops cannot establish real fill prices.
"""
import os, json, time, math, csv
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
import pandas as pd

API = 'https://public.coindcx.com/market_data/candles'
DETAILS_API = 'https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments'
COINS = ('BTC ETH BNB XRP SOL DOGE ADA TRX AVAX LINK SUI XLM HBAR TON SHIB DOT LTC BCH UNI NEAR '
         'APT ICP ETC POL ARB OP FIL ATOM AAVE ALGO VET RENDER INJ SEI TIA IMX GRT THETA MKR '
         'RUNE FET KAS STX LDO QNT JUP BONK WIF PEPE ENA').split()
COST_PCT = 0.18
MIN_SCORE = 5
VOLUME_MULT = 1.5
MAX_HOLD_MINUTES = 24 * 60
MAX_OPEN_POSITIONS = 4
RR_RATIO = 2.0
WORKERS = 4
STATE_PATH = 'live_1h15m_state.json'
TRADES_PATH = 'live_1h15m_trades.csv'
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
FIELDS = ['pair','side','signal_time','entry_time','entry_price','stop_price','target_price','risk_pct','exit_time','exit_price',
          'exit_reason','gross_pct','net_pct','score','reasons','h1_trend','profile_shape','poc','vah','val']


def api_candles(pair, resolution, limit=210):
    # CoinDCX public market-data endpoint: timestamps are in milliseconds.
    end = int(time.time())
    start = end - limit * resolution * 60 - 3600
    params = {'pair': pair, 'from': start, 'to': end, 'resolution': str(resolution), 'pcode': 'f'}
    r = requests.get(API, params=params, timeout=22)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = data.get('data', data.get('candles', []))
    if not isinstance(data, list) or not data:
        raise ValueError(f'No candle data for {pair} {resolution}m: {str(data)[:100]}')
    df = pd.DataFrame(data)
    if 'time' not in df and 'timestamp' in df:
        df = df.rename(columns={'timestamp': 'time'})
    for k in ('open','high','low','close','volume','time'):
        if k not in df:
            raise ValueError(f'{pair} {resolution}m missing column {k}; keys={list(df.columns)}')
        df[k] = pd.to_numeric(df[k], errors='coerce')
    df = df.dropna(subset=['time','open','high','low','close','volume']).copy()
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df = df.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    # Strictly closed candles; avoid using an unfinished bar.
    now = pd.Timestamp.now(tz='UTC')
    df = df[df['time'] + pd.Timedelta(minutes=resolution) <= now - pd.Timedelta(seconds=8)]
    return df.reset_index(drop=True)


def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def indicators(df):
    d = df.copy()
    close = d['close']
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d['rsi'] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    d['macd'] = ema(close, 12) - ema(close, 26)
    d['macd_hist'] = d['macd'] - ema(d['macd'], 9)
    d['ema20'] = ema(close, 20)
    d['ema50'] = ema(close, 50)
    std = close.rolling(20).std()
    d['bb_up'] = d['ema20'] + 2*std
    d['bb_low'] = d['ema20'] - 2*std
    d['bb_width'] = (d['bb_up']-d['bb_low'])/d['ema20']
    d['bb_expand'] = d['bb_width'] > d['bb_width'].shift(1)
    d['bb_squeeze'] = d['bb_width'] <= d['bb_width'].rolling(100, min_periods=50).quantile(.2)
    d['volume_ratio'] = d['volume']/d['volume'].shift(1).rolling(20).mean().replace(0,np.nan)
    d['bull_trend'] = (close > d['ema50']) & (d['ema20'] > d['ema50'])
    d['bear_trend'] = (close < d['ema50']) & (d['ema20'] < d['ema50'])
    return d


def profile(df, lookback=96, bins=24):
    w = df.tail(lookback)
    if len(w) < 35: return None
    lo, hi = float(w.low.min()), float(w.high.max())
    if not hi > lo: return None
    # Approximate profile: allocates each candle volume to its typical-price bin.
    # This is NOT tick-by-tick traded volume-at-price.
    edges = np.linspace(lo,hi,bins+1)
    typ = (w.high.to_numpy()+w.low.to_numpy()+w.close.to_numpy())/3
    weights = np.histogram(typ, bins=edges, weights=w.volume.to_numpy())[0]
    if weights.sum() <= 0: return None
    p = int(np.argmax(weights)); left=right=p; accum=weights[p]
    while accum < .70*weights.sum() and (left>0 or right<bins-1):
        lv = weights[left-1] if left>0 else -1
        rv = weights[right+1] if right<bins-1 else -1
        if rv>lv: right+=1; accum+=weights[right]
        else: left-=1; accum+=weights[left]
    center = np.average(np.linspace(0,1,bins), weights=weights)
    peaks = np.argsort(weights)[-2:]
    shape = ('B' if abs(int(peaks[-1])-int(peaks[-2]))>=7 and weights[peaks[-2]]>=.65*weights[peaks[-1]]
             else 'P' if center>=.58 else 'b' if center<=.42 else 'D')
    return {'poc':float((edges[p]+edges[p+1])/2),'vah':float(edges[right+1]),
            'val':float(edges[left]),'shape':shape}


def h1_direction(d):
    if len(d)<100: return 'NEUTRAL',None
    r=d.iloc[-1]; p=profile(d)
    if not p or not np.isfinite(r.rsi) or not np.isfinite(r.macd_hist): return 'NEUTRAL',p
    # Trend, MACD, RSI, BB and profile are all checked on the last closed 1H candle.
    bull=(bool(r.bull_trend) and r.macd_hist>0 and r.rsi>=50 and
          r.close>p['poc'] and r.close>=r.ema20 and
          (bool(r.bb_expand) or r.close>r.bb_up))
    bear=(bool(r.bear_trend) and r.macd_hist<0 and r.rsi<=50 and
          r.close<p['poc'] and r.close<=r.ema20 and
          (bool(r.bb_expand) or r.close<r.bb_low))
    return ('LONG' if bull else 'SHORT' if bear else 'NEUTRAL'),p


def entry_signal(d, side):
    if len(d)<110 or side=='NEUTRAL': return None
    r=d.iloc[-1]; prev=d.iloc[-2]; p=profile(d)
    if not p or not np.isfinite(r.rsi) or not np.isfinite(r.volume_ratio): return None
    vol=r.volume_ratio>=VOLUME_MULT
    bb=bool(r.bb_expand) or bool(r.bb_squeeze)
    rng=max(float(r.high-r.low),1e-12)
    lower=(min(r.open,r.close)-r.low)/rng
    upper=(r.high-max(r.open,r.close))/rng
    if side=='LONG':
        # 15m mandatory trend, volume and price action.
        price_action=(r.close>prev.high or (lower>=.55 and r.close>r.open))
        checks={'TREND':bool(r.bull_trend),'VOLUME_1.5X':vol,'PRICE_ACTION':price_action,
                'ABOVE_POC':r.close>p['poc'],'BB':bb,'RSI_RISING':30<=r.rsi<=70 and r.rsi>prev.rsi,
                'MACD_POSITIVE':r.macd_hist>0}
        stop=float(r.low)
    else:
        price_action=(r.close<prev.low or (upper>=.55 and r.close<r.open))
        checks={'TREND':bool(r.bear_trend),'VOLUME_1.5X':vol,'PRICE_ACTION':price_action,
                'BELOW_POC':r.close<p['poc'],'BB':bb,'RSI_FALLING':30<=r.rsi<=75 and r.rsi<prev.rsi,
                'MACD_NEGATIVE':r.macd_hist<0}
        stop=float(r.high)
    if not all(checks[k] for k in ('TREND','VOLUME_1.5X','PRICE_ACTION')): return None
    score=sum(bool(v) for v in checks.values())
    if score<MIN_SCORE: return None
    return {'side':side,'score':score,'reasons':'|'.join(k for k,v in checks.items() if v),
            'signal_time':r['time'].isoformat(),'stop_price':stop,'target_price':None,
            'profile_shape':p['shape'],'poc':p['poc'],'vah':p['vah'],'val':p['val']}


def load_state():
    if not os.path.exists(STATE_PATH): return {'positions':{},'last_signal':{}}
    with open(STATE_PATH,encoding='utf-8') as f: return json.load(f)


def save_state(s):
    temp=STATE_PATH+'.tmp'
    with open(temp,'w',encoding='utf-8') as f: json.dump(s,f,indent=2)
    os.replace(temp,STATE_PATH)


def record_trade(t):
    exists=os.path.exists(TRADES_PATH)
    with open(TRADES_PATH,'a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore')
        if not exists: w.writeheader()
        w.writerow(t)


def notify(message):
    print(message,flush=True)
    if BOT_TOKEN and CHAT_ID:
        try:
            r=requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',
                            data={'chat_id':CHAT_ID,'text':message},timeout=12)
            r.raise_for_status()
        except Exception as e: print('Telegram failed:',str(e)[:120],flush=True)


def evaluate_position(pos, bars):
    """Paper fills based on completed 15m OHLC bars; stop-first if both touched.

    Entry is modeled at the next candle open, but becomes observable only
    after that candle closes. Live execution and gap fills can differ.
    """
    signal_end = pd.Timestamp(pos['signal_time']) + pd.Timedelta(minutes=15)
    eligible = bars[bars.time >= signal_end]
    if eligible.empty:
        return pos, None
    if not pos.get('entry_time'):
        first = eligible.iloc[0]
        pos['entry_time'] = first['time'].isoformat()
        pos['entry_price'] = float(first.open)
    entry_time = pd.Timestamp(pos['entry_time'])
    since = bars[bars.time >= entry_time]
    if since.empty:
        return pos, None
    entry = float(pos['entry_price'])
    stop = float(pos['stop_price'])
    side = pos['side']
    if (side == 'LONG' and stop >= entry) or (side == 'SHORT' and stop <= entry):
        pos.update(exit_time=entry_time.isoformat(), exit_price=entry,
                   exit_reason='INVALID_STOP_AT_ENTRY', gross_pct=0.0,
                   net_pct=-COST_PCT)
        return pos, pos
    risk = abs(entry - stop)
    target = entry + RR_RATIO * risk if side == 'LONG' else entry - RR_RATIO * risk
    pos['target_price'] = round(target, 10)
    pos['risk_pct'] = round(risk / entry * 100, 5)
    for _, bar in since.iterrows():
        sl_hit = (float(bar.low) <= stop) if side == 'LONG' else (float(bar.high) >= stop)
        tp_hit = (float(bar.high) >= target) if side == 'LONG' else (float(bar.low) <= target)
        aged = (bar.time + pd.Timedelta(minutes=15) - entry_time) >= pd.Timedelta(minutes=MAX_HOLD_MINUTES)
        # Without tick data, both-hit ordering is unknown: conservative SL-first.
        if sl_hit:
            price = min(stop, float(bar.open)) if side == 'LONG' else max(stop, float(bar.open))
            reason = 'STOP_AND_TP_SAME_BAR_SL_FIRST' if tp_hit else 'STOP_LOSS_HIT'
        elif tp_hit:
            # Gap through TP: conservatively book the target, not a better price.
            price = target
            reason = 'TAKE_PROFIT_RR_2'
        elif aged:
            price = float(bar.close)
            reason = 'TIME_EXIT_24H'
        else:
            continue
        gross = (price / entry - 1) * 100 * (1 if side == 'LONG' else -1)
        pos.update(exit_time=(bar.time + pd.Timedelta(minutes=15)).isoformat(),
                   exit_price=round(price, 10), exit_reason=reason,
                   gross_pct=round(gross, 5), net_pct=round(gross - COST_PCT, 5))
        return pos, pos
    return pos, None


def active_pairs():
    try:
        r=requests.get(DETAILS_API,timeout=20);r.raise_for_status();data=r.json()
        if isinstance(data,dict):data=data.get('data',data.get('instruments',[]))
        available=set(x if isinstance(x,str) else x.get('pair','') for x in data)
        found=[f'B-{c}_USDT' for c in COINS if f'B-{c}_USDT' in available]
        if found: return found
    except Exception as e: print('Instrument discovery warning:',str(e)[:120])
    # No guessing that all pairs are available: try candidates and skip missing API markets.
    return [f'B-{c}_USDT' for c in COINS]


def scan_pair(pair):
    h1=indicators(api_candles(pair,60,210))
    m15=indicators(api_candles(pair,15,210))
    if len(h1)<110 or len(m15)<110: return pair,None,m15,'Insufficient closed candles'
    # Last completed 1H candle must be closed at or before the 15M signal close.
    signal_end=m15.iloc[-1]['time']+pd.Timedelta(minutes=15)
    h1=h1[h1.time+pd.Timedelta(hours=1)<=signal_end]
    direction,pr=h1_direction(h1)
    return pair,entry_signal(m15,direction),m15,None


def main():
    print('LIVE 1H -> 15M PAPER TEST | NO REAL ORDERS | COST 0.18% | RR 1:2 | MAX 4',flush=True)
    state=load_state(); state.setdefault('positions',{});state.setdefault('last_signal',{})
    pairs=active_pairs();print('Candidates:',len(pairs),flush=True)
    errors=0;new_signals=0;exits=0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(scan_pair,p):p for p in pairs}
        for f in as_completed(futures):
            pair=futures[f]
            try:
                pair,signal,bars,err=f.result()
                if err: print(pair,err);continue
                position=state['positions'].get(pair)
                if position:
                    position,closed=evaluate_position(position,bars)
                    if closed:
                        record_trade(closed);state['positions'].pop(pair,None);exits+=1
                        notify(f"PAPER EXIT {pair} {closed['side']} | {closed['exit_reason']} | net {closed['net_pct']:+.3f}%")
                    else: state['positions'][pair]=position
                # One signal per pair per closed candle. No duplicate/re-entry on same candle.
                if (signal and pair not in state['positions']
                        and len(state['positions']) < MAX_OPEN_POSITIONS
                        and state['last_signal'].get(pair) != signal['signal_time']):
                    signal['pair']=pair;signal['h1_trend']=signal['side']
                    signal.update(entry_time=None,entry_price=None)
                    state['positions'][pair]=signal
                    state['last_signal'][pair]=signal['signal_time']
                    new_signals+=1
                    notify(f"PAPER SIGNAL {pair} {signal['side']} | score {signal['score']} | "
                           f"signal candle {signal['signal_time']} | stop {signal['stop_price']:.8g} | "
                           f"next 15m open entry | RR 1:2 (simulation)")
                save_state(state)
            except Exception as e:
                errors+=1;print(f'ERROR {pair}: {str(e)[:250]}',flush=True)
    save_state(state)
    print(f'Finished: new_signals={new_signals}, exits={exits}, open={len(state["positions"])}, errors={errors}',flush=True)
    if errors==len(pairs): raise RuntimeError('All pairs failed: verify CoinDCX endpoint/data schema')

if __name__=='__main__': main()
