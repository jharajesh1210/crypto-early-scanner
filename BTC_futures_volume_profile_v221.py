"""BTC Futures book-inspired volume-profile paper scanner. No real orders.
Runs periodically; collects real trades at prices. Requires accumulated trade history.
"""
import csv, json, os, time
from datetime import datetime, timezone
from pathlib import Path
import requests

PAIR='B-BTC_USDT'
TRADES_URL='https://api.coindcx.com/exchange/v1/derivatives/futures/data/trades'
CANDLES_URL='https://public.coindcx.com/market_data/candlesticks'
STATE=Path('btc_v221_state.json')
JOURNAL=Path('btc_v221_paper_trades.csv')
REPORT=Path('btc_v221_report.csv')
COST_PCT=0.18  # assumed total fees/friction, not from book
MAX_TRADE_ROWS=30000
MIN_TRADES_FOR_PROFILE=500
MIN_SPAN_HOURS=8
LEVEL_TOLERANCE_PCT=0.08  # implementation choice; book gives no numerical value
STOP_BUFFER_PCT=0.05     # implementation choice
MAX_LEVEL_AGE_HOURS=72
MAX_POSITION_HOURS=48     # safety policy, not a book rule

def now_ms(): return int(time.time()*1000)
def request_json(url, params):
    r=requests.get(url,params=params,timeout=25)
    r.raise_for_status()
    return r.json()
def fetch_trades():
    raw=request_json(TRADES_URL,{'pair':PAIR})
    if isinstance(raw,dict): raw=raw.get('data',raw.get('trades',[]))
    if not isinstance(raw,list): raise ValueError('Unexpected trades response')
    out=[]
    for x in raw:
        try:
            p=float(x.get('price')); q=float(x.get('quantity')); t=int(float(x.get('timestamp')))
            if p>0 and q>0 and t>1_000_000_000_000: out.append({'t':t,'p':p,'q':q})
        except (TypeError,ValueError): pass
    if not out: raise RuntimeError('No valid BTC futures trade data')
    return out

def fetch_candles():
    end=int(time.time()); start=end-10*86400
    payload=request_json(CANDLES_URL,{'pair':PAIR,'from':start,'to':end,'resolution':'5','pcode':'f'})
    if isinstance(payload,dict):
        if payload.get('s') not in (None,'ok'): raise RuntimeError(str(payload)[:200])
        payload=payload.get('data',[])
    rows=[]
    for x in payload:
        try:
            row={k:float(x[k]) for k in ('open','high','low','close','volume')}
            row['t']=int(x['time']); rows.append(row)
        except (KeyError,TypeError,ValueError): continue
    rows.sort(key=lambda x:x['t'])
    return [r for r in rows if r['t']+300000<=now_ms()-5000]

def load():
    if not STATE.exists(): return {'trades':[],'levels':[],'position':None,'closed_ids':[]}
    s=json.loads(STATE.read_text());
    for k,v in [('trades',[]),('levels',[]),('position',None),('closed_ids',[])]: s.setdefault(k,v)
    return s

def save(s):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(STATE)

def telegram(message):
    print(message,flush=True)
    token=os.getenv('TELEGRAM_BOT_TOKEN',''); chat=os.getenv('TELEGRAM_CHAT_ID','')
    if token and chat:
        try:
            r=requests.post(f'https://api.telegram.org/bot{token}/sendMessage',data={'chat_id':chat,'text':message},timeout=12)
            r.raise_for_status()
        except requests.RequestException as e: print('Telegram error:',str(e)[:120])

def price_profile(trades,lo,hi):
    """Trade-price volume bins; unlike candle-typical-price approximation."""
    if hi<=lo: return None
    n=40; hist=[0.0]*n
    for x in trades:
        if lo<=x['p']<=hi:
            i=min(n-1,max(0,int((x['p']-lo)/(hi-lo)*n)))
            hist[i]+=x['q']
    if sum(hist)<=0: return None
    ix=max(range(n),key=lambda i:hist[i]);poc=lo+(ix+.5)*(hi-lo)/n
    left=right=ix; accum=hist[ix]; total=sum(hist)
    while accum<.70*total and (left>0 or right<n-1):
        a=hist[left-1] if left>0 else -1
        b=hist[right+1] if right<n-1 else -1
        if b>a: right+=1;accum+=hist[right]
        else: left-=1;accum+=hist[left]
    return {'poc':poc,'vah':lo+(right+1)*(hi-lo)/n,'val':lo+left*(hi-lo)/n,'total':total}

def zone_trades(trades,start,end):return [x for x in trades if start<=x['t']<end]

def create_levels(s,bars):
    """Three book setups. Numerical pattern thresholds are explicit proxies, not book claims."""
    if len(bars)<90:return []
    trades=s['trades']; results=[]
    # Use closed candles, detect historical formation ending on most recent closed candle.
    b=bars[-1]; prior=bars[-13:-1]
    if len(prior)<12:return []
    lo=min(x['low'] for x in prior);hi=max(x['high'] for x in prior)
    span=max(hi-lo,1e-9); base=sum(x['close'] for x in prior)/len(prior)
    volavg=sum(x['volume'] for x in prior)/12
    def add(kind,side,start,end,zlo,zhi,stop):
        sample=zone_trades(trades,start,end)
        if len(sample)<25:return
        prof=price_profile(sample,zlo,zhi)
        if not prof:return
        level={'id':f'{kind}:{side}:{start}:{end}', 'kind':kind,'side':side,
               'created':b['t']+300000,'price':prof['poc'],'stop':stop,
               'poc':prof['poc'],'vah':prof['vah'],'val':prof['val'],'used':False}
        results.append(level)
    # 1 accumulation: compact rotation followed by decisive breakout
    if span/base<.018 and b['volume']>volavg*1.5:
        if b['close']>hi and b['close']>b['open']:
            add('ACCUMULATION','LONG',prior[0]['t'],prior[-1]['t']+300000,lo,hi,lo)
        elif b['close']<lo and b['close']<b['open']:
            add('ACCUMULATION','SHORT',prior[0]['t'],prior[-1]['t']+300000,lo,hi,hi)
    # 2 trend: strong directional move, pause, continuation
    old=bars[-17:-7];pause=bars[-7:-1]
    if len(old)==10 and len(pause)==6:
        p_lo=min(x['low'] for x in pause);p_hi=max(x['high'] for x in pause)
        oldmove=(old[-1]['close']/old[0]['open']-1)
        if abs(oldmove)>.012 and (p_hi-p_lo)/base<.009:
            if oldmove>0 and b['close']>p_hi:
                add('TREND','LONG',pause[0]['t'],pause[-1]['t']+300000,p_lo,p_hi,p_lo)
            if oldmove<0 and b['close']<p_lo:
                add('TREND','SHORT',pause[0]['t'],pause[-1]['t']+300000,p_lo,p_hi,p_hi)
    # 3 rejection: sharp wick and forceful close away from rejected extreme
    candle_range=max(b['high']-b['low'],1e-9)
    lower=(min(b['open'],b['close'])-b['low'])/candle_range
    upper=(b['high']-max(b['open'],b['close']))/candle_range
    if b['volume']>volavg*1.5:
        if lower>.55 and b['close']>b['open'] and b['low']<=lo:
            add('REJECTION','LONG',b['t'],b['t']+300000,b['low'],min(b['open'],b['close']),b['low'])
        if upper>.55 and b['close']<b['open'] and b['high']>=hi:
            add('REJECTION','SHORT',b['t'],b['t']+300000,max(b['open'],b['close']),b['high'],b['high'])
    return results

def journal(row):
    cols=['id','setup','side','entry_time','entry','stop','target','exit_time','exit','reason','gross_pct','net_pct']
    exists=JOURNAL.exists()
    with JOURNAL.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader() if not exists else None;w.writerow({k:row.get(k,'') for k in cols})

def manage(s,bars):
    pos=s['position']
    if not pos:return
    since=[x for x in bars if x['t']>=pos['entry_time']]
    for b in since:
        stop_hit=b['low']<=pos['stop'] if pos['side']=='LONG' else b['high']>=pos['stop']
        target_hit=b['high']>=pos['target'] if pos['side']=='LONG' else b['low']<=pos['target']
        expired=b['t']-pos['entry_time']>=MAX_POSITION_HOURS*3600000
        if not (stop_hit or target_hit or expired):continue
        if stop_hit:
            price=min(pos['stop'],b['open']) if pos['side']=='LONG' else max(pos['stop'],b['open'])
            reason='SL_FIRST_IF_BOTH' if target_hit else 'STRUCTURAL_STOP'
        elif target_hit: price=pos['target'];reason='VOLUME_TARGET'
        else: price=b['close'];reason='SAFETY_TIME_EXIT'
        gross=(price/pos['entry']-1)*100*(1 if pos['side']=='LONG' else -1)
        pos.update(exit_time=b['t']+300000,exit=price,reason=reason,gross_pct=round(gross,4),net_pct=round(gross-COST_PCT,4))
        journal(pos);s['position']=None
        telegram(f"BTC V22 PAPER EXIT {pos['side']} {reason} net {pos['net_pct']:+.3f}%")
        break

def process_levels(s,bars):
    """First-touch only. Use completed candles; entry at touched level is an idealized fill."""
    if s['position']:return
    now=now_ms()
    for level in s['levels']:
        if level['used'] or now-level['created']>MAX_LEVEL_AGE_HOURS*3600000:continue
        future=[b for b in bars if b['t']>=level['created']]
        if not future:continue
        p=level['price'];tol=p*LEVEL_TOLERANCE_PCT/100
        # reject level if market invalidates structural origin before first touch
        for b in future:
            if level['side']=='LONG' and b['low']<level['stop']:
                level['used']=True;break
            if level['side']=='SHORT' and b['high']>level['stop']:
                level['used']=True;break
            if b['low']<=p+tol and b['high']>=p-tol:
                level['used']=True
                # No historical/backdated entries: only a newly closed 5m candle may trigger.
                if b['t'] != bars[-1]['t']:
                    print('Past first touch observed; level retired',level['id']);break
                stop=level['stop']*(1-STOP_BUFFER_PCT/100 if level['side']=='LONG' else 1+STOP_BUFFER_PCT/100)
                if (level['side']=='LONG' and stop>=p) or (level['side']=='SHORT' and stop<=p):break
                # Book supports fixed or volume targets; only use an opposing heavy-volume zone.
                candidates=[z['price'] for z in s['levels'] if z['side']!=level['side'] and not z['used'] and ((z['price']>p) if level['side']=='LONG' else (z['price']<p))]
                if not candidates:
                    print('Level touched but no opposing volume target; skip',level['id']);break
                target=min(candidates) if level['side']=='LONG' else max(candidates)
                pos={'id':level['id'],'setup':level['kind'],'side':level['side'],
                     'entry_time':b['t'],'entry':p,'stop':stop,'target':target}
                s['position']=pos
                telegram(f"BTC V22 PAPER {pos['side']} {pos['setup']} entry {p:.2f} SL {stop:.2f} volume TP {target:.2f}")
                # Evaluate same bar conservatively for possible immediate stop.
                manage(s,[b]);return

def ensure_journal():
    if not JOURNAL.exists():
        with JOURNAL.open('w', newline='') as f:
            csv.writer(f).writerow(['id','setup','side','entry_time','entry','stop','target','exit_time','exit','reason','gross_pct','net_pct'])

def write_report(s, fresh_count, new_count):
    with JOURNAL.open(newline='') as f:
        closed=list(csv.DictReader(f))
    returns=[float(r['net_pct']) for r in closed if r.get('net_pct')]
    wins=sum(v>0 for v in returns)
    span=(s['trades'][-1]['t']-s['trades'][0]['t'])/3600000 if len(s['trades'])>1 else 0
    metrics={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
             'market_trades_stored':len(s['trades']),'api_trades_received':fresh_count,
             'new_unique_trades':new_count,'trade_span_hours':round(span,3),
             'warmup_ready':int(len(s['trades'])>=MIN_TRADES_FOR_PROFILE and span>=MIN_SPAN_HOURS),
             'levels':len(s['levels']),'open_position':int(bool(s['position'])),
             'closed_paper_trades':len(closed),'wins':wins,
             'win_rate_pct':round(100*wins/len(returns),2) if returns else '',
             'sum_net_return_pct':round(sum(returns),4) if returns else ''}
    with REPORT.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics));w.writeheader();w.writerow(metrics)
    print('V22.1 STATUS:',metrics,flush=True)
    if new_count==0:print('WARNING: API provided no NEW unique market trades; inspect timestamps/API history.',flush=True)

def main():
    ensure_journal()
    s=load();recent=fetch_trades()
    previous_count=len(s['trades'])
    merged={ (x['t'],x['p'],x['q']):x for x in s['trades']}
    previous_keys=set(merged)
    new_count=sum((x['t'],x['p'],x['q']) not in previous_keys for x in recent)
    for x in recent:merged[(x['t'],x['p'],x['q'])]=x
    s['trades']=sorted(merged.values(),key=lambda x:x['t'])[-MAX_TRADE_ROWS:]
    bars=fetch_candles()
    print('BTC V22.1 trades accumulated:',len(s['trades']),'previous:',previous_count,'new:',new_count,'closed 5m candles:',len(bars))
    manage(s,bars)
    if len(s['trades'])<MIN_TRADES_FOR_PROFILE or (s['trades'][-1]['t']-s['trades'][0]['t'])<MIN_SPAN_HOURS*3600000:
        print('WARMUP: collecting trade-level volume data; no signals until sufficient coverage')
        save(s);write_report(s,len(recent),new_count);return
    existing={z['id'] for z in s['levels']}
    for z in create_levels(s,bars):
        if z['id'] not in existing:s['levels'].append(z)
    s['levels']=[z for z in s['levels'] if now_ms()-z['created']<=MAX_LEVEL_AGE_HOURS*3600000]
    process_levels(s,bars)
    save(s)
    write_report(s,len(recent),new_count)
    print('BTC V22.1 completed. Levels:',len(s['levels']),'open:',bool(s['position']))
if __name__=='__main__':main()