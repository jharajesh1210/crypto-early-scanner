# MULTI-STRATEGY PAPER SCANNER - independent BTC SMC / AMC POC / ASIAN SESSION
# CoinDCX fixed user-selected 20 coins; no real orders.
# Replaces HYBRID_SMC_AMC_SESSION_PAPER.py; existing hybrid state is preserved.
import os, csv, json, time, logging
from pathlib import Path
from datetime import datetime, timezone
import requests

PREFIX = "HYBRID_SMC_AMC_SESSION_PAPER"
STATE = Path(PREFIX + "_state.json")
TRADES = Path(PREFIX + "_trades.csv")
SIGNALS = Path(PREFIX + "_signals.csv")
API = "https://public.coindcx.com/market_data/candlesticks"
ACTIVE_API = "https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments"
MARGIN = 100.0  # per independent strategy position; up to 3 per coin
LEVERAGE = 30.0
FEE_RATE = 0.0005
SLIP_SIDE = 0.0002
MAX_ASIA_RANGE_PCT = 0.07
MIN_RISK_PCT = 0.001
MAX_RISK_PCT = 0.025
LOG = logging.getLogger("hybrid_three_independent")
REQUESTED_COINS = ('BTC','ETH','USDT','BNB','XRP','USDC','SOL','TRX','HYPE','ZEC',
                   'DOGE','XMR','LINK','LEO','ADA','NEAR','XLM','BCH','USDE','LTC')
STRATEGIES = ('BTC_SMC','AMC_POC','ASIAN_LONDON_NY')

def utc(ts): return datetime.fromtimestamp(ts/1000,timezone.utc)
def iso(ts): return utc(ts).isoformat()

def append_csv(path,row):
    # Preserve older hybrid CSV rows even when this version adds strategy columns.
    if path.exists() and path.stat().st_size:
        with path.open('r',encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            old_fields=reader.fieldnames or []
            old_rows=list(reader)
        fields=list(dict.fromkeys(old_fields+list(row)))
        if fields!=old_fields:
            tmp=path.with_suffix('.csv.tmp')
            with tmp.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(old_rows)
            tmp.replace(path)
    else:
        fields=list(row)
    with path.open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        if f.tell()==0:writer.writeheader()
        writer.writerow(row)

def load_state():
    if STATE.exists():
        with STATE.open(encoding='utf-8') as f: s=json.load(f)
    else: s={}
    for key,default in [('positions',[]),('last_closed_bar',{}),('entered_days',{}),
                        ('smc_setup',{}),('amc_pending',{}),('last_15m',{})]:
        s.setdefault(key,default)
    return s

def save_state(s):
    tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(s,indent=2),encoding='utf-8')
    tmp.replace(STATE)

def notify(message):
    token,chat=os.getenv('BOT_TOKEN'),os.getenv('CHAT_ID')
    if not token or not chat:
        LOG.info('Telegram not configured: %s',message);return
    try:
        r=requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
            json={'chat_id':chat,'text':message},timeout=12);r.raise_for_status()
    except requests.RequestException as e: LOG.warning('Telegram failed: %s',e)


def top20_symbols():
    """Scan only requested coins that have an active CoinDCX USDT futures contract."""
    response = requests.get(ACTIVE_API, timeout=30)
    response.raise_for_status()
    active = response.json()
    if isinstance(active, dict):
        active = active.get('data', active.get('instruments', []))
    if not isinstance(active, list):
        raise ValueError('CoinDCX active futures list unavailable')
    pairs = set()
    for item in active:
        if isinstance(item, str):
            pair = item.upper()
        elif isinstance(item, dict):
            pair = str(item.get('pair') or item.get('instrument') or
                       item.get('symbol') or item.get('coindcx_name') or '').upper()
        else:
            continue
        if pair.endswith('_USDT'):
            pairs.add(pair if pair.startswith('B-') else 'B-' + pair)
    selected, unavailable = [], []
    for coin in REQUESTED_COINS:
        pair = f'B-{coin}_USDT'
        if pair in pairs:
            selected.append(f'{coin}USDT')
        else:
            unavailable.append(coin)
    LOG.info('USER SELECTED COINS: %s', ', '.join(REQUESTED_COINS))
    LOG.info('AVAILABLE COINDCX FUTURES (%d): %s', len(selected), ', '.join(selected) or 'NONE')
    if unavailable:
        LOG.warning('SKIPPED (no active USDT futures contract): %s', ', '.join(unavailable))
    if not selected:
        raise ValueError('None of the 20 requested coins have active CoinDCX USDT futures contracts')
    return selected

def candles(symbol, resolution='5', days=4):
    """Fetch CoinDCX futures candles, chunked for dependable coverage."""
    now = int(time.time())
    start = now - days * 86400
    step = 2 * 86400 if resolution == '5' else 10 * 86400
    rows = []
    for left in range(start, now, step):
        r = requests.get(API, params={'pair': 'B-' + symbol[:-4] + '_USDT',
                         'pcode': 'f', 'resolution': resolution,
                         'from': left, 'to': min(left + step, now)}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
            raise ValueError('Invalid CoinDCX candle response')
        rows.extend(payload['data'])
    duration = int(resolution) * 60 * 1000
    result = {}
    for row in rows:
        try:
            t = int(float(row['time']))
            if t < 100_000_000_000:
                t *= 1000
            if t + duration > int(time.time() * 1000):
                continue
            bar = {'t': t, 'o': float(row['open']), 'h': float(row['high']),
                   'l': float(row['low']), 'c': float(row['close']),
                   'v': float(row.get('volume', 0))}
            if bar['h'] >= bar['l'] and bar['c'] > 0:
                result[t] = bar
        except (ValueError, KeyError, TypeError):
            continue
    return [result[t] for t in sorted(result)]

def approximate_poc(bars, bins=24):
    low, high = min(x['l'] for x in bars), max(x['h'] for x in bars)
    if high <= low:
        return low
    volume = [0.0] * bins
    for b in bars:
        # Uniform volume allocation over candle high-low is only an approximation.
        start = max(0, min(bins - 1, int((b['l'] - low) / (high - low) * bins)))
        end = max(0, min(bins - 1, int((b['h'] - low) / (high - low) * bins)))
        per_bin = b['v'] / (end - start + 1)
        for j in range(start, end + 1):
            volume[j] += per_bin
    return low + (volume.index(max(volume)) + 0.5) * (high - low) / bins

def find_setup(bars, last_ts):
    """Find one current-day completed retest. No future bars in decisions."""
    if not bars:
        return None
    day = utc(bars[-1]['t']).date()
    today = [b for b in bars if utc(b['t']).date() == day]
    asia = [b for b in today if 0 <= utc(b['t']).hour < 8]
    # Strictly wait for all 96 Asian candles, not a partial range.
    if len(asia) != 96:
        return None
    ah, al = max(b['h'] for b in asia), min(b['l'] for b in asia)
    if al <= 0 or (ah - al) / al > MAX_ASIA_RANGE_PCT:
        return None
    poc = approximate_poc(asia)
    london = [b for b in today if 8 <= utc(b['t']).hour < 10]
    if not london:
        return None
    # Scan latest qualified sweep first. A 1-3 bar wick reversal is preferred;
    # require close back inside Asian range within 3 candles.
    for i, b in enumerate(london):
        if b['l'] < al:
            direction, extreme, boundary = 'BUY', b['l'], al
            returned = next((j for j in range(i, min(i + 3, len(london)))
                             if london[j]['c'] > boundary), None)
        elif b['h'] > ah:
            direction, extreme, boundary = 'SELL', b['h'], ah
            returned = next((j for j in range(i, min(i + 3, len(london)))
                             if london[j]['c'] < boundary), None)
        else:
            continue
        if returned is None:
            continue
        rt = london[returned]['t']
        # A CHOCH proxy: close breaks the 3 pre-sweep candle swing in reversal direction.
        prior = [x for x in today if x['t'] < b['t']][-3:]
        if len(prior) < 3:
            continue
        swing = max(x['h'] for x in prior) if direction == 'BUY' else min(x['l'] for x in prior)
        post = [x for x in today if x['t'] > rt and 8 <= utc(x['t']).hour < 18]
        choch = next((x for x in post if (x['c'] > swing if direction == 'BUY'
                                         else x['c'] < swing)), None)
        if choch is None:
            continue
        # FVG created after CHOCH: three completed candles, with displacement.
        for k in range(2, len(today)):
            c1, c2, c3 = today[k-2:k+1]
            if c3['t'] < choch['t'] or c1['t'] < rt:
                continue
            if not 8 <= utc(c3['t']).hour < 18:
                continue
            if direction == 'BUY':
                lo, hi = c1['h'], c3['l']
                displacement = c2['c'] > c2['o'] and (c2['c'] - c2['o']) > 0.35 * (c2['h'] - c2['l'])
            else:
                lo, hi = c3['h'], c1['l']
                displacement = c2['c'] < c2['o'] and (c2['o'] - c2['c']) > 0.35 * (c2['h'] - c2['l'])
            if hi <= lo or not displacement:
                continue
            # Retest must be a NEW closed candle, after FVG creation.
            for retest in today[k+1:]:
                if retest['t'] <= last_ts or not 8 <= utc(retest['t']).hour < 18:
                    continue
                if retest['l'] > hi or retest['h'] < lo:
                    continue
                entry = retest['c']
                if direction == 'BUY' and (entry <= lo or entry <= extreme):
                    continue
                if direction == 'SELL' and (entry >= hi or entry >= extreme):
                    continue
                stop = min(extreme, retest['l']) if direction == 'BUY' else max(extreme, retest['h'])
                risk = entry - stop if direction == 'BUY' else stop - entry
                if entry <= 0 or not MIN_RISK_PCT <= risk / entry <= MAX_RISK_PCT:
                    continue
                # POC boosts score; not a hard gate.
                poc_confirms = entry > poc if direction == 'BUY' else entry < poc
                return {'day': str(day), 'direction': direction, 'entry': entry,
                        'stop': stop, 'risk': risk, 'poc': poc,
                        'poc_confirms': poc_confirms, 'asian_high': ah,
                        'asian_low': al, 'bar_t': retest['t']}
    return None

def direction_sign(direction): return 1 if direction in ('LONG','BUY') else -1

def open_trade(state,symbol,strategy,direction,bar,stop,poc=None):
    if strategy == "BTC_SMC" and symbol != "BTCUSDT":
        return False
    # One open position PER STRATEGY per coin; no partial exits.
    if any(p.get('symbol')==symbol and p.get('strategy')==strategy for p in state['positions']):return False
    sign=direction_sign(direction)
    raw=bar['c']; entry=raw*(1+sign*SLIP_SIDE)
    risk=sign*(entry-stop)
    if entry<=0 or risk<=0 or not 0.0005<=risk/entry<=0.10:return False
    target=entry+sign*4*risk
    position={'symbol':symbol,'strategy':strategy,'direction':direction,'entry':entry,
        'stop':stop,'initial_stop':stop,'target':target,'risk':risk,'locked_rr':0,
        'entry_time_utc':iso(bar['t']),'entry_t':bar['t'],
        'last_checked_t':bar['t'],'margin_usdt':MARGIN,'rr':4,
        'id':f'{symbol}-{strategy}-{bar["t"]}','poc':poc}
    state['positions'].append(position)
    append_csv(SIGNALS,{'trade_id':position['id'],'symbol':symbol,'strategy':strategy,
        'direction':direction,'entry_time_utc':position['entry_time_utc'],
        'entry':entry,'stop':stop,'target':target,'rr':4,'margin_usdt':MARGIN})
    notify(f'HYBRID PAPER ENTRY | {strategy}\n{symbol} {direction}\nEntry {entry:.8g}\n'
           f'SL {stop:.8g}\n2R/3R lock full position; final 4R {target:.8g}\n'
           f'Margin ${MARGIN:.0f} leverage {LEVERAGE:.0f}x. PAPER ONLY')
    LOG.info('ENTRY %s %s %s',strategy,symbol,direction)
    return True


def settle(state,symbol,bars):
    remaining=[]
    for pos in state['positions']:
        if pos.get('symbol')!=symbol:remaining.append(pos);continue
        sign=direction_sign(pos['direction'])
        entry=float(pos['entry']); original=float(pos.get('initial_stop',pos['stop']))
        risk=float(pos.get('risk',abs(entry-original)))
        outcome=None
        for bar in bars:
            if bar['t']<=int(pos.get('last_checked_t',pos.get('entry_t',0))):continue
            stop=float(pos['stop']); target=float(pos['target'])
            hit_stop=(bar['l']<=stop if sign==1 else bar['h']>=stop)
            hit_target=(bar['h']>=target if sign==1 else bar['l']<=target)
            # Existing stop has priority when OHLC cannot reveal intrabar order.
            if hit_stop or hit_target:
                outcome=('TRAIL_SL' if hit_stop and pos.get('locked_rr',0)>0 else
                         'STOP' if hit_stop else 'TP_RR4',stop if hit_stop else target,bar['t'])
                break
            # New positions use full-position 2R/3R lock. Legacy hybrid legs
            # retain their original per-leg stop/target behavior.
            if not pos.get('strategy'):
                leg=pos.get('leg','')
                if leg in ('3R','4R'):
                    threshold=2 if leg=='3R' else 3
                    progress=sign*((bar['h'] if sign==1 else bar['l'])-entry)
                    if progress>=threshold*risk:
                        proposed=entry+sign*(threshold-1)*risk
                        pos['stop']=max(pos['stop'],proposed) if sign==1 else min(pos['stop'],proposed)
                pos['last_checked_t']=bar['t']
                continue
            best=bar['h'] if sign==1 else bar['l']
            reached=sign*(best-entry)/risk if risk>0 else 0
            locked=int(pos.get('locked_rr',0))
            new_lock=3 if reached>=3 else 2 if reached>=2 else locked
            if new_lock>locked:
                pos['locked_rr']=new_lock
                pos['stop']=entry+sign*new_lock*risk
                notify(f'HYBRID PAPER TRAIL | {pos.get("strategy","LEGACY")} {symbol}\n'
                       f'1:{new_lock} reached; full position SL => {pos["stop"]:.8g} (next candle)')
            pos['last_checked_t']=bar['t']
        if outcome is None:
            remaining.append(pos);continue
        reason,raw_exit,t=outcome
        exit_price=raw_exit*(1-sign*SLIP_SIDE)
        gross_pct=sign*(exit_price/entry-1)*100
        net_pct=gross_pct-(FEE_RATE*(1+exit_price/entry)*100)
        pnl=net_pct/100*float(pos.get('margin_usdt',MARGIN))*LEVERAGE
        append_csv(TRADES,{'trade_id':pos.get('id',''), 'symbol':symbol,
            'strategy':pos.get('strategy','LEGACY_HYBRID'), 'direction':pos['direction'],
            'entry_time_utc':pos.get('entry_time_utc',''),'exit_time_utc':iso(t),
            'entry_price':entry,'exit_price':exit_price,'stop_loss':original,
            'target':target,'exit_reason':reason,'rr':pos.get('rr',4),
            'margin_usdt':pos.get('margin_usdt',MARGIN),
            'net_return_pct_per_leg':round(net_pct,6),'paper_pnl_usdt':round(pnl,2)})
        notify(f'HYBRID PAPER EXIT | {pos.get("strategy","LEGACY")} {symbol}\n'
               f'{reason} | net {net_pct:+.3f}% | paper P/L {pnl:+.2f} USDT')
        LOG.info('EXIT %s %s %s %.2f USDT',pos.get('strategy'),symbol,reason,pnl)
    state['positions']=remaining


def swing_points(bars,n=3):
    highs=[];lows=[]
    for i in range(n,len(bars)-n):
        window=bars[i-n:i+n+1];b=bars[i]
        if b['h']==max(x['h'] for x in window) and sum(x['h']==b['h'] for x in window)==1:highs.append(b['h'])
        if b['l']==min(x['l'] for x in window) and sum(x['l']==b['l'] for x in window)==1:lows.append(b['l'])
    return highs,lows

def h1_bias(hourly,bar):
    eligible=[b for b in hourly if b['t']+3600000<=bar['t']+300000]
    hi,lo=swing_points(eligible)
    if len(hi)<2 or len(lo)<2:return 'NEUTRAL'
    if hi[-1]>hi[-2] and lo[-1]>lo[-2]:return 'LONG'
    if hi[-1]<hi[-2] and lo[-1]<lo[-2]:return 'SHORT'
    return 'NEUTRAL'

def btc_smc_step(state,symbol,history,hourly,bar):
    # Original BTC V23: 5m swing sweep -> CHOCH -> FVG -> rejection retest;
    # confirmed 1H HH/HL or LH/LL. Setup expiry 18 bars; FVG expiry 12 bars.
    bias=h1_bias(hourly,bar)
    pending=state['smc_setup'].get(symbol)
    if pending and (bar['t']-pending['sweep_t']>18*300000 or pending['side']!=bias):pending=None
    if pending:
        long=pending['side']=='LONG'
        if (bar['l']<pending['extreme'] if long else bar['h']>pending['extreme']):pending=None
        elif pending['stage']=='SWEEP':
            level=pending['break_level']
            if level is not None and (bar['c']>level if long else bar['c']<level):pending['stage']='CHOCH'
        elif pending['stage']=='CHOCH' and len(history)>=3:
            first=history[-3]
            if (bar['l']>first['h'] if long else bar['h']<first['l']):
                pending.update({'stage':'FVG','fvg_low':first['h'] if long else bar['h'],
                    'fvg_high':bar['l'] if long else first['l'],'fvg_t':bar['t']})
        elif pending['stage']=='FVG':
            if bar['t']-pending['fvg_t']>12*300000:pending=None
            else:
                touched=bar['l']<=pending['fvg_high'] and bar['h']>=pending['fvg_low']
                rejection=(bar['c']>bar['o'] and bar['c']>pending['fvg_high']) if long else (bar['c']<bar['o'] and bar['c']<pending['fvg_low'])
                if touched and rejection:
                    entry=bar['c'];stop=pending['extreme'];risk=abs(entry-stop)/entry*100
                    if 0.05<=risk<=2.50 and (stop<entry if long else stop>entry):
                        open_trade(state,symbol,'BTC_SMC',pending['side'],bar,stop)
                    pending=None
    if pending is None and bias in ('LONG','SHORT') and not any(p.get('symbol')==symbol and p.get('strategy')=='BTC_SMC' for p in state['positions']):
        previous=history[:-1];hi,lo=swing_points(previous)
        if bias=='LONG' and lo and bar['l']<lo[-1] and bar['c']>lo[-1]:
            pending={'side':'LONG','stage':'SWEEP','sweep_t':bar['t'],'extreme':bar['l'],'break_level':hi[-1] if hi else None}
        elif bias=='SHORT' and hi and bar['h']>hi[-1] and bar['c']<hi[-1]:
            pending={'side':'SHORT','stage':'SWEEP','sweep_t':bar['t'],'extreme':bar['h'],'break_level':lo[-1] if lo else None}
    if pending is None:state['smc_setup'].pop(symbol,None)
    else:state['smc_setup'][symbol]=pending

def bars_15m(bars):
    groups={}
    for b in bars:
        start=b['t']//900000*900000
        groups.setdefault(start,[]).append(b)
    result=[]
    for t,grp in sorted(groups.items()):
        if len(grp)!=3 or [b['t'] for b in grp]!=[t,t+300000,t+600000]:continue
        result.append({'t':t,'o':grp[0]['o'],'h':max(x['h'] for x in grp),
                       'l':min(x['l'] for x in grp),'c':grp[-1]['c'],
                       'v':sum(x['v'] for x in grp)})
    return result

def amc_poc_hourly(segment):
    # AMC original: volume-weighted histogram of typical (H+L+C)/3, 30 bins.
    typical=[(b['h']+b['l']+b['c'])/3 for b in segment]
    if not typical:return 0.0
    lo,hi=min(typical),max(typical)
    total=sum(b['v'] for b in segment)
    if hi<=lo or total<=0:return typical[-1]
    bins=[0.0]*30
    for price,b in zip(typical,segment):
        index=min(29,max(0,int((price-lo)/(hi-lo)*30)))
        bins[index]+=b['v']
    idx=max(range(30),key=lambda i:bins[i])
    return lo+(idx+0.5)*(hi-lo)/30

def amc_region(hourly,start):
    # Original AMC: last 4 fully completed 1H bars, range <2%, approximate volume POC.
    prior=[x for x in hourly if x['t']+3600000<=start]
    if len(prior)<4:return None
    segment=prior[-4:];hi=max(x['h'] for x in segment);lo=min(x['l'] for x in segment)
    avg=sum(x['c'] for x in segment)/4
    if avg<=0 or (hi-lo)/avg*100>=2.0:return None
    return {'high':hi,'low':lo,'poc':amc_poc_hourly(segment)}

def amc_step(state,symbol,bar,hourly):
    # Original AMC: 15m sweep of 4h accumulation range, close back inside;
    # immediately NEXT 15m candle close breaks sweep high/low by 0.08%.
    pending=state['amc_pending'].pop(symbol,None)
    if pending and pending['t']==bar['t']-900000:
        long=pending['side']=='LONG';buffer=.0008
        confirmed=bar['c']>pending['sweep_high']*(1+buffer) if long else bar['c']<pending['sweep_low']*(1-buffer)
        if confirmed:
            end=bar['t']+900000
            # Only act on recently completed candle, not old backfill.
            if 0<=(time.time()*1000-end)/60000<=5:
                entrybar={'t':end,'c':bar['c']}
                open_trade(state,symbol,'AMC_POC',pending['side'],entrybar,pending['stop'],pending['poc'])
    region=amc_region(hourly,bar['t'])
    if region:
        hi,lo=region['high'],region['low']
        if bar['l']<lo and lo<bar['c']<=hi:side='LONG'
        elif bar['h']>hi and lo<=bar['c']<hi:side='SHORT'
        else:side=None
        if side:
            state['amc_pending'][symbol]={'t':bar['t'],'side':side,
                'sweep_high':bar['h'],'sweep_low':bar['l'],
                'stop':bar['l'] if side=='LONG' else bar['h'],'poc':region['poc']}

def main():
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    state=load_state()
    selected=top20_symbols()
    LOG.info("STRATEGY UNIVERSE: BTC_SMC=BTCUSDT only; AMC_POC and ASIAN_LONDON_NY=selected CoinDCX coins")
    held={p['symbol'] for p in state['positions']}
    symbols=list(dict.fromkeys(selected+sorted(held)))
    failures=[]; scanned=0;entries_before=len(state['positions'])
    for symbol in symbols:
        try:
            bars=candles(symbol,'5',days=6)
            hourly=candles(symbol,'60',days=18)
            if len(bars)<100 or len(hourly)<30:
                LOG.warning('%s insufficient candles',symbol);continue
            settle(state,symbol,bars)
            # first run per coin initializes; no retroactive trades
            last=int(state['last_closed_bar'].get(symbol,0))
            if not last:
                state['last_closed_bar'][symbol]=bars[-1]['t']
                complete15=bars_15m(bars)
                if complete15:state['last_15m'][symbol]=complete15[-1]['t']
                LOG.info('%s initialized',symbol);save_state(state);scanned+=1;continue
            new=[i for i,b in enumerate(bars) if b['t']>last]
            if len(new)>12:
                LOG.warning('%s missed %d bars; reset pending, no backfill entries',symbol,len(new))
                state['smc_setup'].pop(symbol,None);state['amc_pending'].pop(symbol,None)
                new=[]
            if symbol == "BTCUSDT":
                for i in new:
                    btc_smc_step(state,symbol,bars[:i+1],hourly,bars[i])
            else:
                # Clear obsolete pending SMC signals, but preserve existing positions.
                state["smc_setup"].pop(symbol, None)
            complete15=bars_15m(bars)
            if symbol not in state['last_15m'] and complete15:
                state['last_15m'][symbol]=complete15[-1]['t']
            last15=int(state['last_15m'].get(symbol,0))
            for b in complete15:
                if b['t']>last15 and b['t']+900000<=int(time.time()*1000):
                    amc_step(state,symbol,b,hourly)
            if complete15:state['last_15m'][symbol]=complete15[-1]['t']
            # Session source original exact find_setup; only enter on newest completed bar.
            today=str(utc(bars[-1]['t']).date())
            if state['entered_days'].get(symbol)!=today:
                setup=find_setup(bars,last)
                if setup and setup['bar_t']==bars[-1]['t']:
                    if open_trade(state,symbol,'ASIAN_LONDON_NY',setup['direction'],
                                  {'t':setup['bar_t'],'c':setup['entry']},setup['stop'],setup['poc']):
                        state['entered_days'][symbol]=today
            state['last_closed_bar'][symbol]=bars[-1]['t']
            scanned+=1
            save_state(state)
            LOG.info('%s scan OK; open=%d',symbol,sum(p.get('symbol')==symbol for p in state['positions']))
        except (requests.RequestException,ValueError,KeyError,TypeError,ZeroDivisionError) as exc:
            failures.append(f'{symbol}: {exc}')
            LOG.exception('%s scan failed',symbol)
            save_state(state)
    LOG.info('COMPLETE selected=%d scanned=%d failures=%d open_positions=%d',len(selected),scanned,len(failures),len(state['positions']))
    if failures:LOG.error('Failed coins: %s','; '.join(failures))
    if scanned==0:raise RuntimeError('No coins successfully scanned')

if __name__=='__main__':main()
