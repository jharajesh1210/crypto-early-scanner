"""Top-100 CoinGecko market-cap / CoinDCX USDT futures SMC paper scanner.
1h HH/HL or LH/LL bias; 5m sweep -> CHOCH -> FVG -> retest/rejection.
One paper position per pair; stop locks at 2R, 3R; final take profit 4R.
No exchange orders. Run every 5 minutes via GitHub Actions.
"""
import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

NAME = '100AND_BTC_paper'
STATE = Path(NAME + '_state.json')
TRADES = Path(NAME + '_trades.csv')
MIN_RISK_PERCENT, MAX_RISK_PERCENT = 0.05, 2.50
SWING_LENGTH, SETUP_EXPIRY, FVG_EXPIRY = 3, 18, 12
FEE_ROUND_TRIP, SLIPPAGE_SIDE = 0.001, 0.0002
MAX_WORKERS = 5
MAX_NEW_BARS = 12
LOG = logging.getLogger(NAME)


def api_get(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=22,
                             headers={'User-Agent': '100AND-BTC-SMC-Paper/1.0'})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            LOG.warning('API failed (%s) attempt %d: %s', url.split('/')[2], attempt + 1, type(e).__name__)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return None


def telegram(msg):
    token, chat = os.getenv('BOT_TOKEN'), os.getenv('CHAT_ID')
    if not token or not chat:
        LOG.warning('Telegram secrets missing')
        return
    try:
        r = requests.post('https://api.telegram.org/bot' + token + '/sendMessage',
                          json={'chat_id': chat, 'text': msg[:4000]}, timeout=15)
        r.raise_for_status()
        if not r.json().get('ok'):
            LOG.warning('Telegram did not confirm success')
    except (requests.RequestException, ValueError):
        LOG.warning('Telegram delivery failed (credentials not logged)')


def top100_pairs():
    data = api_get('https://api.coingecko.com/api/v3/coins/markets',
                   {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 100,
                    'page': 1, 'sparkline': 'false'})
    if not isinstance(data, list) or len(data) < 50:
        raise RuntimeError('CoinGecko Top100 unavailable: safe stop')
    active = api_get('https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments')
    if isinstance(active, dict):
        active = active.get('data', active.get('instruments', []))
    if not isinstance(active, list) or not active:
        raise RuntimeError('CoinDCX futures list unavailable: safe stop')
    futures = {}
    for item in active:
        pair = item if isinstance(item, str) else (item.get('pair') or item.get('instrument') or item.get('symbol') or item.get('coindcx_name') or '') if isinstance(item, dict) else ''
        pair = str(pair).upper()
        if pair.endswith('_USDT'):
            pair = pair if pair.startswith('B-') else 'B-' + pair
            futures[pair[2:-5]] = pair
    stable = {'USDT','USDC','DAI','FDUSD','TUSD','USDE','USDD','PYUSD','USDS','RLUSD','EURC'}
    result = []
    seen = set()
    for coin in data:
        symbol = str(coin.get('symbol', '')).upper()
        if symbol in futures and symbol not in stable and symbol not in seen:
            result.append(futures[symbol])
            seen.add(symbol)
    if not result:
        raise RuntimeError('No matching futures: safe stop')
    return result


def candles(pair, resolution, days):
    end = int(time.time())
    raw = api_get('https://public.coindcx.com/market_data/candlesticks',
                  {'pair': pair, 'pcode': 'f', 'resolution': str(resolution),
                   'from': end - days * 86400, 'to': end})
    if not isinstance(raw, dict) or not isinstance(raw.get('data'), list):
        return pd.DataFrame()
    df = pd.DataFrame(raw['data'])
    columns = ['time','open','high','low','close','volume']
    if df.empty or any(c not in df for c in columns):
        return pd.DataFrame()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=columns)
    if df.empty:
        return df
    unit = 'ms' if df['time'].median() > 1e11 else 's'
    df['time'] = pd.to_datetime(df['time'], unit=unit, utc=True)
    df = df.drop_duplicates('time').set_index('time').sort_index()
    df = df[(df.high >= df.low) & (df.close > 0)]
    return df[df.index + pd.Timedelta(minutes=resolution) <= pd.Timestamp.now(tz='UTC')].copy()


def swings(df, length=SWING_LENGTH):
    hi, lo = [], []
    for i in range(length, len(df)-length):
        window = df.iloc[i-length:i+length+1]
        bar = df.iloc[i]
        if bar.high == window.high.max() and (window.high == bar.high).sum() == 1:
            hi.append((i, float(bar.high)))
        if bar.low == window.low.min() and (window.low == bar.low).sum() == 1:
            lo.append((i, float(bar.low)))
    return hi, lo


def bias_at(hourly, bar_end):
    # Hourly candle must have closed before the 5m bar closes.
    h = hourly[hourly.index + pd.Timedelta(hours=1) <= bar_end]
    hi, lo = swings(h)
    if len(hi) < 2 or len(lo) < 2:
        return 'NEUTRAL'
    if hi[-1][1] > hi[-2][1] and lo[-1][1] > lo[-2][1]:
        return 'LONG'
    if hi[-1][1] < hi[-2][1] and lo[-1][1] < lo[-2][1]:
        return 'SHORT'
    return 'NEUTRAL'


def initial_state():
    return {'version': 1, 'coins': {}, 'positions': [], 'closed': 0,
            'wins': 0, 'losses': 0, 'net_pct_sum': 0.0}


def load_state():
    if not STATE.exists():
        return initial_state()
    data = json.loads(STATE.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('coins'), dict) or not isinstance(data.get('positions'), list):
        raise RuntimeError('State invalid; refusing to overwrite')
    for k, v in initial_state().items():
        data.setdefault(k, v)
    return data


def save_state(state):
    tmp = STATE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(STATE)


def append_trade(row):
    fields = ['pair','side','entry_time','exit_time','entry','initial_stop','exit','reason','r_multiple','gross_pct','net_pct']
    new = not TRADES.exists() or TRADES.stat().st_size == 0
    with TRADES.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in fields})


def close_position(state, pos, exit_price, reason, when):
    direction = 1 if pos['side'] == 'LONG' else -1
    gross = (exit_price / pos['entry'] - 1) * direction * 100
    net = gross - (FEE_ROUND_TRIP + 2 * SLIPPAGE_SIDE) * 100
    r_mult = (exit_price - pos['entry']) * direction / pos['risk']
    state['closed'] += 1
    state['wins' if net > 0 else 'losses'] += 1
    state['net_pct_sum'] += net
    append_trade({'pair': pos['pair'], 'side': pos['side'], 'entry_time': pos['entry_time'],
                  'exit_time': when.isoformat(), 'entry': pos['entry'],
                  'initial_stop': pos['initial_stop'], 'exit': exit_price,
                  'reason': reason, 'r_multiple': round(r_mult, 4),
                  'gross_pct': round(gross, 4), 'net_pct': round(net, 4)})
    telegram(f'{NAME} PAPER EXIT\n{pos["pair"]} {pos["side"]}\n{reason}\nExit: {exit_price:.8g}\nNet: {net:+.3f}%\nPAPER ONLY')
    LOG.info('PAPER EXIT %s %s %s', pos['pair'], pos['side'], reason)


def settle_pair(state, pair, bars):
    positions = [p for p in state['positions'] if p['pair'] == pair]
    if not positions:
        return
    pos = positions[0]
    newer = bars[bars.index > pd.Timestamp(pos['last_checked'])]
    for t, bar in newer.iterrows():
        side = pos['side']
        stop = pos['stop']
        hit_stop = bar.low <= stop if side == 'LONG' else bar.high >= stop
        hit_4r = bar.high >= pos['target4'] if side == 'LONG' else bar.low <= pos['target4']
        # Conservative ordering: stop active at start of candle is checked first.
        if hit_stop or hit_4r:
            close_position(state, pos, stop if hit_stop else pos['target4'],
                           'TRAILING SL' if hit_stop and pos['stage'] else 'STOP LOSS' if hit_stop else 'FINAL 4R TARGET', t)
            state['positions'].remove(pos)
            return
        # Trail advances only after this bar finishes; active from the NEXT candle.
        new_stage = 0
        if (bar.high >= pos['level3'] if side == 'LONG' else bar.low <= pos['level3']):
            new_stage = 3
        elif (bar.high >= pos['level2'] if side == 'LONG' else bar.low <= pos['level2']):
            new_stage = 2
        if new_stage > pos['stage']:
            pos['stage'] = new_stage
            pos['stop'] = pos['level3'] if new_stage == 3 else pos['level2']
            telegram(f'{NAME} PAPER TRAILING SL\n{pair} {side}\nReached {new_stage}R; next-candle SL locked at {pos["stop"]:.8g}\nPAPER ONLY')
            LOG.info('TRAIL %s to %dR', pair, new_stage)
        pos['last_checked'] = t.isoformat()


def detect_sweep(bars, i, side):
    hi, lo = swings(bars.iloc[:i])
    bar = bars.iloc[i]
    if side == 'LONG' and lo and bar.low < lo[-1][1] and bar.close > lo[-1][1]:
        return {'side': side, 'stage': 'SWEEP', 'sweep_time': bars.index[i].isoformat(),
                'extreme': float(bar.low), 'break_level': hi[-1][1] if hi else None}
    if side == 'SHORT' and hi and bar.high > hi[-1][1] and bar.close < hi[-1][1]:
        return {'side': side, 'stage': 'SWEEP', 'sweep_time': bars.index[i].isoformat(),
                'extreme': float(bar.high), 'break_level': lo[-1][1] if lo else None}
    return None


def process_smc(state, pair, bars, hourly, i, stats):
    t = bars.index[i]
    bar = bars.iloc[i]
    end = t + pd.Timedelta(minutes=5)
    bias = bias_at(hourly, end)
    coin = state['coins'][pair]
    setup = coin.get('setup')
    if setup:
        age = (t - pd.Timestamp(setup['sweep_time'])).total_seconds() / 300
        if age > SETUP_EXPIRY or setup['side'] != bias:
            setup = None
        elif (bar.low < setup['extreme'] if setup['side'] == 'LONG' else bar.high > setup['extreme']):
            setup = None
    if setup:
        long = setup['side'] == 'LONG'
        if setup['stage'] == 'SWEEP':
            level = setup['break_level']
            if level is not None and (bar.close > level if long else bar.close < level):
                setup['stage'] = 'CHOCH'
                stats['choch'] += 1
        elif setup['stage'] == 'CHOCH' and i >= 2:
            first = bars.iloc[i-2]
            fvg = bar.low > first.high if long else bar.high < first.low
            if fvg:
                setup.update({'stage': 'FVG', 'fvg_low': float(first.high if long else bar.high),
                              'fvg_high': float(bar.low if long else first.low), 'fvg_time': t.isoformat()})
                stats['fvg'] += 1
        elif setup['stage'] == 'FVG':
            if (t - pd.Timestamp(setup['fvg_time'])).total_seconds() / 300 > FVG_EXPIRY:
                setup = None
            else:
                touched = bar.low <= setup['fvg_high'] and bar.high >= setup['fvg_low']
                rejected = (bar.close > bar.open and bar.close > setup['fvg_high']) if long else (bar.close < bar.open and bar.close < setup['fvg_low'])
                if touched and rejected:
                    stats['retests'] += 1
                    entry, stop = float(bar.close), float(setup['extreme'])
                    risk = abs(entry - stop)
                    valid = (stop < entry if long else stop > entry) and MIN_RISK_PERCENT <= risk / entry * 100 <= MAX_RISK_PERCENT
                    if valid and not any(p['pair'] == pair for p in state['positions']):
                        # No retroactive fills: only act on the latest fully closed candle, within 5 minutes.
                        age_minutes = (pd.Timestamp.now(tz='UTC') - end).total_seconds() / 60
                        if i == len(bars) - 1 and 0 <= age_minutes <= 5:
                            direction = 1 if long else -1
                            pos = {'pair': pair, 'side': setup['side'], 'entry_time': end.isoformat(),
                                   'entry': entry, 'initial_stop': stop, 'stop': stop, 'risk': risk,
                                   'level2': entry + direction * 2 * risk,
                                   'level3': entry + direction * 3 * risk,
                                   'target4': entry + direction * 4 * risk,
                                   'stage': 0, 'last_checked': t.isoformat()}
                            state['positions'].append(pos)
                            stats['entries'] += 1
                            telegram(f'{NAME} PAPER ENTRY\n{pair} {pos["side"]}\n1H Bias: {bias}\nEntry: {entry:.8g}\nSL: {stop:.8g}\n2R: {pos["level2"]:.8g}\n3R: {pos["level3"]:.8g}\nFINAL 4R: {pos["target4"]:.8g}\nPAPER ONLY')
                            LOG.info('PAPER ENTRY %s %s', pair, pos['side'])
                        else:
                            stats['late_signals'] += 1
                    else:
                        stats['risk_rejected'] += 1
                    setup = None
    if setup is None and bias in ('LONG', 'SHORT') and not any(p['pair'] == pair for p in state['positions']):
        setup = detect_sweep(bars, i, bias)
        if setup:
            stats['sweeps'] += 1
    coin['setup'] = setup


def scan_pair(pair):
    return pair, candles(pair, 5, 3), candles(pair, 60, 18)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    state = load_state()
    pairs = top100_pairs()
    # Include pairs with existing paper positions even if market-cap rank changes.
    pairs = list(dict.fromkeys(pairs + [p['pair'] for p in state['positions']]))
    LOG.info('START %s | TOP100 futures=%d | PAPER ONLY', NAME, len(pairs))
    stats = {k: 0 for k in ('scanned','initialized','sweeps','choch','fvg','retests','entries','late_signals','risk_rejected','gaps','data_failures')}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_pair, p): p for p in pairs}
        for future in as_completed(futures):
            pair = futures[future]
            try:
                _, bars, hourly = future.result()
                if len(bars) < 100 or len(hourly) < 30:
                    stats['data_failures'] += 1
                    continue
                stats['scanned'] += 1
                coin = state['coins'].setdefault(pair, {'last_bar': None, 'setup': None})
                # Existing positions: settle only completed 5m bars.
                settle_pair(state, pair, bars)
                last = coin.get('last_bar')
                if not last:
                    coin['last_bar'] = bars.index[-1].isoformat()
                    coin['setup'] = None
                    stats['initialized'] += 1
                    save_state(state)
                    continue
                last_time = pd.Timestamp(last)
                new_indices = [i for i, t in enumerate(bars.index) if t > last_time]
                if len(new_indices) > MAX_NEW_BARS:
                    stats['gaps'] += 1
                    LOG.warning('Too many missed 5m candles %s; reset setup without backfilling entries', pair)
                    coin['setup'] = None
                    coin['last_bar'] = bars.index[-1].isoformat()
                    save_state(state)
                    continue
                for i in new_indices:
                    t = bars.index[i]
                    if t != last_time + pd.Timedelta(minutes=5):
                        coin['setup'] = None
                        stats['gaps'] += 1
                    process_smc(state, pair, bars.iloc[:i+1], hourly, i, stats)
                    coin['last_bar'] = t.isoformat()
                    last_time = t
                    save_state(state)
            except Exception:
                stats['data_failures'] += 1
                LOG.exception('Pair scan failed %s', pair)
    LOG.info('DIAGNOSTICS %s', stats)
    LOG.info('COMPLETE open_positions=%d closed=%d wins=%d losses=%d',
             len(state['positions']), state['closed'], state['wins'], state['losses'])
    save_state(state)


if __name__ == '__main__':
    main()
