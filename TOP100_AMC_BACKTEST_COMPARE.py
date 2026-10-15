"""Independent CoinDCX futures AMC/POC historical comparison. PAPER RESEARCH ONLY.
Does not read/write live paper state. Run: python TOP100_AMC_BACKTEST_COMPARE.py
Env: BACKTEST_DAYS=30, BACKTEST_SYMBOLS=ZEC,BTC (optional), BACKTEST_MAX_COINS=60.
"""
import os, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests

DAYS = int(os.getenv('BACKTEST_DAYS','30'))
MAX_COINS = int(os.getenv('BACKTEST_MAX_COINS','60'))
SYMBOLS = {s.strip().upper() for s in os.getenv('BACKTEST_SYMBOLS','').split(',') if s.strip()}
HOURS = (1,2,4,6)
RANGE_LIMIT = 2.0
MIN_24H_TURNOVER = 500_000.0
FEE_SIDE, SLIP_SIDE = .0005, .0002
OUT = Path('TOP100_AMC_BACKTEST_RESULTS')
OUT.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
LOG = logging.getLogger('AMC_BACKTEST')
SESSION = requests.Session()
SESSION.headers.update({'User-Agent':'AMC-comparison-research/1.0'})

def get(url, params=None):
    for i in range(4):
        try:
            r=SESSION.get(url,params=params,timeout=35)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            LOG.warning('HTTP retry %d %s: %s',i+1,url,str(e)[:100]);time.sleep(2*(i+1))
    raise RuntimeError('Could not fetch '+url)

def universe():
    markets=get('https://api.coingecko.com/api/v3/coins/markets',{'vs_currency':'usd','order':'market_cap_desc','per_page':100,'page':1,'sparkline':'false'})
    pairs=get('https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments')
    if isinstance(pairs,dict):pairs=pairs.get('data',pairs.get('instruments',[]))
    mapping={}
    for item in pairs:
        p=(item if isinstance(item,str) else item.get('pair') or item.get('instrument') or item.get('symbol') or item.get('coindcx_name') or '').upper()
        if p.endswith('_USDT'):
            if not p.startswith('B-'):p='B-'+p
            mapping[p[2:-5]]=p
    excluded={'USDT','USDC','DAI','FDUSD','TUSD','USDE','USDD','PYUSD','USDS','RLUSD','EURC'}
    result=[]
    for m in markets:
        s=str(m.get('symbol','')).upper()
        if s in mapping and s not in excluded and (not SYMBOLS or s in SYMBOLS):result.append((s,mapping[s]))
    if not result:raise RuntimeError('No matching futures symbols')
    return result[:MAX_COINS]

def fetch_5m(pair):
    end=int(time.time())
    start=end-(DAYS+3)*86400  # warm-up for 24h volume and accumulation
    frames=[]
    # Request <= 2-day chunks to avoid API candle limits.
    while start<end:
        finish=min(start+2*86400,end)
        raw=get('https://public.coindcx.com/market_data/candlesticks',{'pair':pair,'pcode':'f','resolution':'5','from':start,'to':finish})
        arr=raw.get('data',[]) if isinstance(raw,dict) else []
        if arr:frames.extend(arr)
        start=finish
        time.sleep(.08)
    if not frames:return pd.DataFrame()
    df=pd.DataFrame(frames)
    required=['time','open','high','low','close','volume']
    if any(c not in df for c in required):return pd.DataFrame()
    for c in required:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=required)
    if df.empty:return df
    unit='ms' if df.time.median()>1e11 else 's'
    df['time']=pd.to_datetime(df.time,unit=unit,utc=True)
    df=df.drop_duplicates('time').set_index('time').sort_index()
    df=df[(df.high>=df.low)&(df.close>0)&(df.volume>=0)]
    now=pd.Timestamp.now(tz='UTC')
    return df[df.index+pd.Timedelta(minutes=5)<=now].copy()

def aggregate(df,minutes):
    group=df.resample(f'{minutes}min',label='left',closed='left')
    agg=group.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    return agg[group.close.count()==minutes//5].dropna()

def poc(frame):
    typical=((frame.high+frame.low+frame.close)/3).to_numpy()
    volume=frame.volume.to_numpy()
    if not len(typical):return float('nan')
    if volume.sum()<=0 or np.ptp(typical)==0:return float(typical[-1])
    hist,edges=np.histogram(typical,bins=30,weights=volume)
    j=int(np.argmax(hist));return float((edges[j]+edges[j+1])/2)

def settle(entry_idx,side,entry,stop,df5):
    """Walk future 5m bars, stop-before-target conservative, new trail effective next bar."""
    sign=1 if side=='LONG' else -1
    risk=abs(entry-stop)
    if risk<=0:return None
    target=entry+sign*4*risk
    trail=stop; lock=0
    for bar_time,bar in df5.iloc[entry_idx:].iterrows():
        sl_hit=bar.low<=trail if sign==1 else bar.high>=trail
        tp_hit=bar.high>=target if sign==1 else bar.low<=target
        if sl_hit or tp_hit:
            raw=trail if sl_hit else target
            exit_price=raw*(1-SLIP_SIDE if sign==1 else 1+SLIP_SIDE)
            net=(sign*(exit_price/entry-1)-2*FEE_SIDE)*100
            return {'exit_time':bar_time.isoformat(),'exit':exit_price,'result':('TRAIL_SL' if lock else 'SL') if sl_hit else 'TP_4R','net_return_pct':net,'locked_rr':lock}
        best=bar.high if sign==1 else bar.low
        reached=sign*(best-entry)/risk
        if reached>=3 and lock<3:lock=3;trail=entry+sign*3*risk
        elif reached>=2 and lock<2:lock=2;trail=entry+sign*2*risk
    return None

def simulate(symbol,pair,df5,hours):
    bars=aggregate(df5,15);hourly=aggregate(df5,60)
    if len(bars)<10 or len(hourly)<30:return [],{'symbol':symbol,'hours':hours,'eligible_bars':0,'accum_pass':0,'sweeps':0,'confirmations':0,'entries':0}
    hourly=hourly.copy()
    hourly['turnover']=hourly.close*hourly.volume
    # Last 24 fully closed hourly bars, ending no later than sweep start.
    hourly['turnover24']=hourly.turnover.rolling(24,min_periods=24).sum()
    htimes=hourly.index
    entries=[];pending=None
    counts={'symbol':symbol,'hours':hours,'eligible_bars':0,'accum_pass':0,'sweeps':0,'confirmations':0,'entries':0}
    cutoff=pd.Timestamp.now(tz='UTC')-pd.Timedelta(days=DAYS)
    last_exit=None
    for start,bar in bars.iterrows():
        if start<cutoff:continue
        # Use only completed hourly candles; no future information.
        completed_end=htimes.searchsorted(start-pd.Timedelta(hours=1),side='right')
        prior=hourly.iloc[:completed_end]
        if len(prior)<max(hours,24):pending=None;continue
        if not np.isfinite(prior.turnover24.iloc[-1]) or prior.turnover24.iloc[-1]<MIN_24H_TURNOVER:
            pending=None;continue
        counts['eligible_bars']+=1
        # Confirm only immediately following 15m bar.
        if pending is not None:
            if start==pending['start']+pd.Timedelta(minutes=15):
                side=pending['side']
                confirmed=(bar.close>pending['sweep_high']*1.0008) if side=='LONG' else (bar.close<pending['sweep_low']*.9992)
                if confirmed:
                    counts['confirmations']+=1
                    end=start+pd.Timedelta(minutes=15)
                    if last_exit is None or end>last_exit:
                        entry=float(bar.close)*(1+SLIP_SIDE if side=='LONG' else 1-SLIP_SIDE)
                        stop=pending['stop'];risk=abs(entry-stop)
                        if entry>0 and ((side=='LONG' and stop<entry) or (side=='SHORT' and stop>entry)) and .0005<=risk/entry<=.10:
                            idx=df5.index.searchsorted(end)
                            outcome=settle(idx,side,entry,stop,df5)
                            if outcome:
                                entries.append({'symbol':symbol,'pair':pair,'hours':hours,'direction':side,'entry_time':end.isoformat(),'entry':entry,'stop':stop,'target4':entry+(1 if side=='LONG' else -1)*4*risk,'poc':pending['poc'],**outcome})
                                counts['entries']+=1
                                last_exit=pd.Timestamp(outcome['exit_time'])
            pending=None
        region=prior.tail(hours)
        hi=float(region.high.max());lo=float(region.low.min());avg=float(region.close.mean())
        if avg<=0 or (hi-lo)/avg*100>=RANGE_LIMIT:continue
        counts['accum_pass']+=1
        side=None
        if bar.low<lo and lo<bar.close<=hi:side='LONG'
        elif bar.high>hi and lo<=bar.close<hi:side='SHORT'
        if side:
            counts['sweeps']+=1
            pending={'start':start,'side':side,'sweep_high':float(bar.high),'sweep_low':float(bar.low),'stop':float(bar.low if side=='LONG' else bar.high),'poc':poc(region)}
    return entries,counts

def summary(trades,diagnostics):
    results=[]
    for h in HOURS:
        group=trades[trades.hours==h] if not trades.empty else pd.DataFrame()
        vals=group.net_return_pct.to_numpy() if len(group) else np.array([])
        wins=int((vals>0).sum())
        # Non-compounded cumulative per-trade percent; not portfolio return.
        curve=np.r_[0,np.cumsum(vals)]
        maxdd=float(np.max(np.maximum.accumulate(curve)-curve))
        results.append({'accum_hours':h,'closed_trades':len(vals),'wins':wins,'losses':int((vals<=0).sum()),'win_rate_pct':round(100*wins/len(vals),2) if len(vals) else 0,'sum_net_trade_return_pct':round(float(vals.sum()),3),'avg_net_trade_return_pct':round(float(vals.mean()),3) if len(vals) else 0,'max_drawdown_pct_points':round(maxdd,3),'sweeps':sum(x['sweeps'] for x in diagnostics if x['hours']==h),'confirmations':sum(x['confirmations'] for x in diagnostics if x['hours']==h)})
    return pd.DataFrame(results)

def main():
    LOG.info('START historical comparison days=%d hours=%s; NO LIVE TRADING',DAYS,HOURS)
    selected=universe();LOG.info('Selected %d coins',len(selected))
    all_trades=[];diags=[];errors=[]
    def worker(item):
        symbol,pair=item
        df=fetch_5m(pair)
        if len(df)<500:raise RuntimeError(f'Only {len(df)} 5m candles')
        return symbol,pair,df
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_map={pool.submit(worker,item):item for item in selected}
        for f in as_completed(future_map):
            symbol,pair=future_map[f]
            try:
                _,_,df=f.result()
                for h in HOURS:
                    trades,diag=simulate(symbol,pair,df,h)
                    all_trades.extend(trades);diags.append(diag)
                LOG.info('DONE %s candles=%d',symbol,len(df))
            except Exception as exc:
                errors.append({'symbol':symbol,'error':str(exc)})
                LOG.error('FAILED %s %s',symbol,exc)
    if not diags:raise RuntimeError('No usable data. Backtest incomplete.')
    trades=pd.DataFrame(all_trades)
    if not trades.empty:trades=trades.sort_values(['hours','exit_time']).reset_index(drop=True)
    trades.to_csv(OUT/'trades.csv',index=False)
    pd.DataFrame(diags).to_csv(OUT/'diagnostics.csv',index=False)
    pd.DataFrame(errors).to_csv(OUT/'errors.csv',index=False)
    result=summary(trades,diags)
    result.to_csv(OUT/'comparison.csv',index=False)
    print('\n=== 1H / 2H / 4H / 6H ACCUMULATION COMPARISON ===')
    print(result.to_string(index=False))
    print(f'\nCoins with usable data: {len(set(x["symbol"] for x in diags))}/{len(selected)}. Failed: {len(errors)}')
    print('Results saved in:',OUT.resolve())
    if errors:LOG.warning('Some coins failed; inspect errors.csv. Do not generalize results to all Top100.')

if __name__=='__main__':main()
