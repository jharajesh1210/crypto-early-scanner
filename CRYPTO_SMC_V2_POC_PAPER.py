"""CoinDCX futures crypto SMC V2 research scanner. PAPER TRADING ONLY.
15m sweep -> MSS -> FVG -> retest; 1h EMA20 and swing trend; rolling 24h POC.
A single $100-margin, 30x paper position per symbol; 2R/3R stop lock, 4R full exit.
Conservative candle simulation; no live orders, no guaranteed returns.
"""
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = 'CRYPTO_SMC_V2_POC_PAPER'
STATE = Path(BASE + '_state.json')
TRADES = Path(BASE + '_trades.csv')
SIGNALS = Path(BASE + '_signals.csv')
CANDLE_URL = 'https://public.coindcx.com/market_data/candlesticks'
ACTIVE_URL = 'https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments'
COINS = ['BTC','ETH','USDT','BNB','XRP','USDC','SOL','TRX','HYPE','ZEC',
         'DOGE','XMR','LINK','LEO','ADA','NEAR','XLM','BCH','USDE','LTC']
MARGIN = 100.0
LEVERAGE = 30.0
FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002
MAX_OPEN = 5
MAX_NEW_PER_UTC_DAY = 5
MIN_RISK_PCT = 0.001
MAX_RISK_PCT = 0.025
MAX_POC_DISTANCE_PCT = 0.015  # price must be near 24h POC
MAX_15M_CANDLE_AGE_SECONDS = 1200
LOG = logging.getLogger('crypto_smc_v2')


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def send(text):
    token, chat = os.getenv('BOT_TOKEN'), os.getenv('CHAT_ID')
    if not token or not chat:
        LOG.info('Telegram not configured: %s', text.replace('\n', ' | '))
        return
    try:
        r = requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                          json={'chat_id': chat, 'text': text}, timeout=15)
        r.raise_for_status()
    except requests.RequestException as exc:
        LOG.warning('Telegram alert failed: %s', exc)


def write_csv(path, row):
    # Append-only with stable columns, keeping existing history if schema expands.
    old = []
    fields = list(row)
    if path.exists() and path.stat().st_size:
        with path.open(newline='', encoding='utf-8-sig') as f:
            rd = csv.DictReader(f)
            fields = list(dict.fromkeys((rd.fieldnames or []) + fields))
            old = list(rd)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(old)
        wr.writerow(row)
    tmp.replace(path)


def state_load():
    if not STATE.exists():
        return {'positions': {}, 'last_bar': {}, 'setups': {}, 'daily_entries': {}}
    d = json.loads(STATE.read_text(encoding='utf-8'))
    for k in ('positions','last_bar','setups','daily_entries'):
        d.setdefault(k, {})
    return d


def state_save(s):
    tmp = STATE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(s, indent=2), encoding='utf-8')
    tmp.replace(STATE)


def symbols():
    r = requests.get(ACTIVE_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = data.get('data', data.get('instruments', []))
    if not isinstance(data, list):
        raise ValueError('Unexpected active futures instruments response')
    available = set()
    for item in data:
        name = item if isinstance(item, str) else (item.get('pair') or item.get('instrument') or item.get('symbol') or item.get('coindcx_name') or '') if isinstance(item, dict) else ''
        name = str(name).upper()
        if name.endswith('_USDT'):
            available.add(name if name.startswith('B-') else 'B-' + name)
    selected = [c + 'USDT' for c in COINS if f'B-{c}_USDT' in available]
    skipped = [c for c in COINS if f'B-{c}_USDT' not in available]
    LOG.info('Selected=%d %s; skipped=%s', len(selected), ','.join(selected), ','.join(skipped))
    if not selected:
        raise RuntimeError('No selected CoinDCX futures instruments found')
    return selected


def fetch(symbol, resolution, days):
    now = int(time.time())
    start = now - days * 86400
    step = 2 * 86400 if resolution == '15' else 9 * 86400
    out = {}
    for left in range(start, now, step):
        r = requests.get(CANDLE_URL, params={
            'pair': f'B-{symbol[:-4]}_USDT', 'pcode': 'f',
            'resolution': resolution, 'from': left, 'to': min(now, left + step)
        }, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
            raise ValueError('Unexpected candle API response')
        for v in payload['data']:
            try:
                t = int(float(v['time']))
                if t < 100_000_000_000:
                    t *= 1000
                duration = int(resolution) * 60_000
                if t + duration > int(time.time() * 1000):
                    continue  # only completed candles
                b = {'t': t, 'o': float(v['open']), 'h': float(v['high']),
                     'l': float(v['low']), 'c': float(v['close']), 'v': float(v.get('volume', 0))}
                if b['l'] > 0 and b['h'] >= b['l'] and b['v'] >= 0:
                    out[t] = b
            except (KeyError, TypeError, ValueError):
                continue
    return [out[t] for t in sorted(out)]


def ema(values, n=20):
    if len(values) < n + 2:
        return None
    result = sum(values[:n]) / n
    alpha = 2 / (n + 1)
    for value in values[n:]:
        result = alpha * value + (1 - alpha) * result
    return result


def swing_levels(bars, wing=2):
    highs, lows = [], []
    for i in range(wing, len(bars) - wing):
        window = bars[i-wing:i+wing+1]
        if bars[i]['h'] > max(b['h'] for j,b in enumerate(window) if j != wing):
            highs.append(bars[i]['h'])
        if bars[i]['l'] < min(b['l'] for j,b in enumerate(window) if j != wing):
            lows.append(bars[i]['l'])
    return highs, lows


def bias(hourly, at):
    past = [b for b in hourly if b['t'] + 3_600_000 <= at]
    if len(past) < 35:
        return None
    e = ema([b['c'] for b in past])
    hs, ls = swing_levels(past[-80:])
    if len(hs) < 2 or len(ls) < 2:
        return None
    if past[-1]['c'] > e and hs[-1] > hs[-2] and ls[-1] > ls[-2]:
        return 'LONG'
    if past[-1]['c'] < e and hs[-1] < hs[-2] and ls[-1] < ls[-2]:
        return 'SHORT'
    return None


def volume_poc(bars, bins=32):
    """OHLCV-derived approximation, not exchange trade-by-price volume profile."""
    if len(bars) < 90 or sum(b['v'] for b in bars) <= 0:
        return None
    lo, hi = min(b['l'] for b in bars), max(b['h'] for b in bars)
    if hi <= lo:
        return None
    hist = [0.] * bins
    for b in bars:
        start = max(0, min(bins-1, int((b['l']-lo)/(hi-lo)*bins)))
        end = max(start, min(bins-1, int((b['h']-lo)/(hi-lo)*bins)))
        for i in range(start, end+1):
            hist[i] += b['v'] / (end-start+1)
    i = max(range(bins), key=lambda k: hist[k])
    return lo + (i + .5) * (hi-lo) / bins


def atr(bars, period=14):
    if len(bars) < period+1:
        return None
    ranges = []
    for a,b in zip(bars[-period-1:-1], bars[-period:]):
        ranges.append(max(b['h']-b['l'], abs(b['h']-a['c']), abs(b['l']-a['c'])))
    return sum(ranges)/period


def detect(s, symbol, bars, hourly, index):
    """Sequential closed 15m bars: sweep -> MSS -> FVG -> retest rejection."""
    b = bars[index]
    if index < 105:
        return None
    now = b['t']
    trend = bias(hourly, now)
    p = s['setups'].get(symbol)
    if p and (now - p['sweep_t'] > 16 * 900_000 or p['side'] != trend):
        p = None
    previous = bars[:index]
    poc = volume_poc(previous[-96:])
    a = atr(bars[:index+1])
    if poc is None or a is None or a <= 0:
        s['setups'].pop(symbol, None)
        return None
    if p:
        side = p['side']
        sign = 1 if side == 'LONG' else -1
        if (sign == 1 and b['l'] < p['sweep_extreme']) or (sign == -1 and b['h'] > p['sweep_extreme']):
            p = None
        elif p['stage'] == 'SWEEP':
            if (sign == 1 and b['c'] > p['mss_level']) or (sign == -1 and b['c'] < p['mss_level']):
                p['stage'] = 'MSS'
                p['mss_t'] = now
        elif p['stage'] == 'MSS' and index >= 2:
            first, middle = bars[index-2], bars[index-1]
            body = abs(middle['c']-middle['o'])
            displacement = body >= a and ((middle['c'] > middle['o']) if sign == 1 else (middle['c'] < middle['o']))
            gap_lo, gap_hi = (first['h'], b['l']) if sign == 1 else (b['h'], first['l'])
            if displacement and gap_hi > gap_lo and middle['t'] >= p['mss_t']:
                p.update({'stage':'FVG', 'fvg_low':gap_lo, 'fvg_high':gap_hi, 'fvg_t':now})
        elif p['stage'] == 'FVG' and now > p['fvg_t']:
            touch = b['l'] <= p['fvg_high'] and b['h'] >= p['fvg_low']
            rejection = ((b['c'] > b['o'] and b['c'] > p['fvg_high']) if sign == 1
                         else (b['c'] < b['o'] and b['c'] < p['fvg_low']))
            close = b['c']
            near_poc = abs(close - poc) / close <= MAX_POC_DISTANCE_PCT
            poc_side = close > poc if sign == 1 else close < poc
            risk = sign * (close - p['sweep_extreme'])
            risk_ok = MIN_RISK_PCT <= risk / close <= MAX_RISK_PCT
            if touch and rejection and near_poc and poc_side and risk_ok:
                s['setups'].pop(symbol, None)
                return {'side':side, 'stop':p['sweep_extreme'], 'poc':poc, 't':now, 'entry':close}
    # Do not overwrite an active setup with another sweep.
    if p is None and trend:
        # Liquidity reference: recent 12 completed 15m candles, excluding sweep candle.
        lookback = previous[-12:]
        hi, lo = max(x['h'] for x in lookback), min(x['l'] for x in lookback)
        if trend == 'LONG' and b['l'] < lo and b['c'] > lo:
            p = {'stage':'SWEEP','side':'LONG','sweep_t':now,
                 'sweep_extreme':b['l'],'mss_level':hi}
        elif trend == 'SHORT' and b['h'] > hi and b['c'] < hi:
            p = {'stage':'SWEEP','side':'SHORT','sweep_t':now,
                 'sweep_extreme':b['h'],'mss_level':lo}
    if p:
        s['setups'][symbol] = p
    else:
        s['setups'].pop(symbol, None)
    return None


def close_position(s, symbol, pos, price, bar_t, reason):
    sign = 1 if pos['side'] == 'LONG' else -1
    entry = pos['entry']
    exit_price = price * (1 - sign * SLIPPAGE_PER_SIDE)
    gross_pct = sign * (exit_price/entry - 1) * 100
    net_pct = gross_pct - 100 * FEE_PER_SIDE * (1 + exit_price/entry)
    pnl = pos['margin_usdt'] * pos['leverage'] * net_pct / 100
    write_csv(TRADES, {'trade_id':pos['id'], 'symbol':symbol, 'strategy':'SMC_V2_POC',
        'direction':pos['side'], 'entry_time_utc':iso(pos['entry_t']), 'exit_time_utc':iso(bar_t),
        'entry_price':entry, 'exit_price':exit_price, 'initial_stop':pos['initial_stop'],
        'exit_reason':reason, 'locked_rr':pos['locked_rr'], 'margin_usdt':pos['margin_usdt'],
        'leverage':pos['leverage'], 'net_return_pct_per_leg':round(net_pct,6),
        'paper_pnl_usdt':round(pnl,2)})
    del s['positions'][symbol]
    send(f'CRYPTO SMC V2 PAPER EXIT {symbol} {pos["side"]}\n{reason}\nP/L {pnl:+.2f} USDT')
    LOG.info('%s EXIT %s P/L %.2f', symbol, reason, pnl)


def manage(s, symbol, bars):
    pos = s['positions'].get(symbol)
    if not pos:
        return
    sign = 1 if pos['side'] == 'LONG' else -1
    for b in bars:
        if b['t'] <= pos['last_t']:
            continue
        # Stop before target: conservative when OHLC path is unknown.
        stop_hit = b['l'] <= pos['stop'] if sign == 1 else b['h'] >= pos['stop']
        target_hit = b['h'] >= pos['target'] if sign == 1 else b['l'] <= pos['target']
        if stop_hit:
            close_position(s, symbol, pos, pos['stop'], b['t'], 'TRAIL_SL' if pos['locked_rr'] else 'STOP')
            return
        if target_hit:
            close_position(s, symbol, pos, pos['target'], b['t'], 'TP_4R')
            return
        favorable = b['h'] if sign == 1 else b['l']
        reached = sign * (favorable - pos['entry']) / pos['risk']
        new_lock = 3 if reached >= 3 else 2 if reached >= 2 else pos['locked_rr']
        if new_lock > pos['locked_rr']:
            pos['locked_rr'] = new_lock
            pos['stop'] = pos['entry'] + sign * new_lock * pos['risk']
            send(f'CRYPTO SMC V2 PAPER TRAIL {symbol}\n{new_lock}R reached; SL at {new_lock}R: {pos["stop"]:.8g}\nEffective next 15m candle')
        pos['last_t'] = b['t']


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    s = state_load()
    selected = symbols()
    universe = list(dict.fromkeys(selected + list(s['positions'])))
    now_ms = int(time.time() * 1000)
    ok, errors = 0, []
    for symbol in universe:
        try:
            bars = fetch(symbol, '15', 6)
            hourly = fetch(symbol, '60', 16)
            if len(bars) < 110 or len(hourly) < 40:
                raise ValueError(f'Insufficient candles 15m={len(bars)} 1h={len(hourly)}')
            manage(s, symbol, bars)
            last = int(s['last_bar'].get(symbol, 0))
            if last == 0:
                s['last_bar'][symbol] = bars[-1]['t']
                LOG.info('%s initialized (no retroactive entries)', symbol)
                state_save(s)
                ok += 1
                continue
            indices = [i for i,b in enumerate(bars) if b['t'] > last]
            if len(indices) > 8:
                LOG.warning('%s missed %d bars; reset pending setup, no backfill entries', symbol, len(indices))
                s['setups'].pop(symbol, None)
                indices = []
            for i in indices:
                bar = bars[i]
                setup = detect(s, symbol, bars, hourly, i)
                if not setup or symbol in s['positions']:
                    continue
                if now_ms - (bar['t'] + 900_000) > MAX_15M_CANDLE_AGE_SECONDS * 1000:
                    LOG.info('%s stale signal ignored', symbol)
                    continue
                day = iso(bar['t'])[:10]
                if s['daily_entries'].get(day, 0) >= MAX_NEW_PER_UTC_DAY or len(s['positions']) >= MAX_OPEN:
                    LOG.info('%s risk limit prevents entry', symbol)
                    continue
                sign = 1 if setup['side'] == 'LONG' else -1
                entry = setup['entry'] * (1 + sign * SLIPPAGE_PER_SIDE)
                risk = sign * (entry - setup['stop'])
                if risk <= 0 or not MIN_RISK_PCT <= risk / entry <= MAX_RISK_PCT:
                    continue
                p = {'id':f'{symbol}-{bar["t"]}', 'side':setup['side'], 'entry':entry,
                     'entry_t':bar['t'], 'initial_stop':setup['stop'], 'stop':setup['stop'],
                     'risk':risk, 'target':entry + sign * 4 * risk, 'locked_rr':0,
                     'last_t':bar['t'], 'margin_usdt':MARGIN, 'leverage':LEVERAGE}
                s['positions'][symbol] = p
                s['daily_entries'][day] = s['daily_entries'].get(day,0) + 1
                write_csv(SIGNALS, {'trade_id':p['id'], 'symbol':symbol, 'strategy':'SMC_V2_POC',
                    'direction':p['side'], 'entry_time_utc':iso(bar['t']), 'entry':entry,
                    'stop':p['stop'], 'target_4r':p['target'], 'poc_24h':setup['poc'],
                    'margin_usdt':MARGIN, 'leverage':LEVERAGE})
                send(f'CRYPTO SMC V2 PAPER ENTRY {symbol} {p["side"]}\nEntry {entry:.8g}\nSL {p["stop"]:.8g}\nFinal 4R {p["target"]:.8g}\n$100 margin x30, no partial exit')
                LOG.info('%s ENTRY %s', symbol, p['side'])
            s['last_bar'][symbol] = bars[-1]['t']
            state_save(s)
            ok += 1
            LOG.info('%s scan OK; open=%s', symbol, int(symbol in s['positions']))
        except (requests.RequestException, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            LOG.exception('%s scan failed: %s', symbol, exc)
            errors.append(symbol)
            state_save(s)
    LOG.info('COMPLETE selected=%d scanned=%d errors=%d open_positions=%d',
             len(selected), ok, len(errors), len(s['positions']))
    if not ok or errors:
        raise RuntimeError(f'Scanner incomplete: scanned={ok} failed={errors}')


if __name__ == '__main__':
    main()
