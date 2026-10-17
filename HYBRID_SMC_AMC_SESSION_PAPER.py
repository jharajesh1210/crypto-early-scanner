"""Experimental HYBRID: BTC SMC + AMC approximate POC + Asian/London session.
CoinDCX USDT futures only. New independent paper state; no live orders.
No backtest or profitability claims.
"""
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

PREFIX = 'HYBRID_SMC_AMC_SESSION_PAPER'
STATE = Path(PREFIX + '_state.json')
TRADES = Path(PREFIX + '_trades.csv')
SIGNALS = Path(PREFIX + '_signals.csv')
TOP_N = 20  # Top 20 active USDT futures by approx. 24h USDT turnover
API = 'https://public.coindcx.com/market_data/candlesticks'
ACTIVE_API = 'https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments'
PRICES_API = 'https://public.coindcx.com/market_data/v3/current_prices/futures/rt'
MARGIN = 100.0
LEVERAGE = 30.0
FEE_RATE = 0.0005  # illustrative taker fee each side, adjust to account
MIN_RISK_PCT = 0.001
MAX_RISK_PCT = 0.025
MIN_RR = 1.2
MAX_ASIA_RANGE_PCT = 0.07
LOG = logging.getLogger('amd_paper')


def utc(ts):
    return datetime.fromtimestamp(ts / 1000, timezone.utc)


def iso(ts):
    return utc(ts).isoformat(timespec='minutes')


def append_csv(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def load_state():
    if STATE.exists():
        with STATE.open(encoding='utf-8') as f:
            s = json.load(f)
        s.setdefault('positions', [])
        s.setdefault('entered_days', {})
        s.setdefault('last_closed_bar', {})
        return s
    return {'positions': [], 'entered_days': {}, 'last_closed_bar': {}}


def save_state(state):
    tmp = STATE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    tmp.replace(STATE)


def notify(message):
    token, chat = os.getenv('BOT_TOKEN'), os.getenv('CHAT_ID')
    if not token or not chat:
        LOG.info('Telegram not configured: %s', message)
        return
    try:
        r = requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                          json={'chat_id': chat, 'text': message}, timeout=12)
        r.raise_for_status()
    except requests.RequestException as exc:
        LOG.warning('Telegram failed: %s', exc)


def top20_symbols():
    """Select active CoinDCX USDT futures by approximate 24h quote turnover.

    CoinDCX's futures price snapshot reports base volume (v) and last price (ls).
    Their product is an approximate USDT turnover, not an audited volume rank.
    Fail closed if live rankings cannot be retrieved.
    """
    active_response = requests.get(
        ACTIVE_API, params={'margin_currency_short_name[]': 'USDT'}, timeout=30
    )
    active_response.raise_for_status()
    active = active_response.json()
    if isinstance(active, dict):
        active = active.get('data', active.get('instruments', []))
    if not isinstance(active, list):
        raise ValueError('CoinDCX active futures list unavailable')
    active_pairs = {
        pair for pair in active
        if isinstance(pair, str) and pair.startswith('B-') and pair.endswith('_USDT')
    }
    if len(active_pairs) < TOP_N:
        raise ValueError(f'Only {len(active_pairs)} active USDT futures returned')

    prices_response = requests.get(PRICES_API, timeout=30)
    prices_response.raise_for_status()
    prices = prices_response.json()
    # The documented API provides a dict keyed by instrument pair.
    if isinstance(prices, dict) and isinstance(prices.get('data'), dict):
        prices = prices['data']
    if not isinstance(prices, dict):
        raise ValueError('CoinDCX futures price snapshot unavailable')

    ranked = []
    for pair in active_pairs:
        tick = prices.get(pair)
        if not isinstance(tick, dict):
            continue
        try:
            volume = float(tick['v'])
            last = float(tick['ls'])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 < volume < float('inf') and 0 < last < float('inf')):
            continue
        ranked.append((pair, volume * last))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    if len(ranked) < TOP_N:
        raise ValueError(f'Only {len(ranked)} usable CoinDCX futures volume quotes')
    chosen = [pair[2:-5] + 'USDT' for pair, _ in ranked[:TOP_N]]
    LOG.info('TOP20 by approximate 24h USDT turnover: %s', ', '.join(chosen))
    return chosen


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


def hourly_bias(hourly, at_t):
    """Confirmed 1H HH/HL or LH/LL, without using future bars."""
    h = [x for x in hourly if x['t'] + 3600000 <= at_t + 300000]
    if len(h) < 20:
        return 'NEUTRAL'
    highs, lows = [], []
    for i in range(3, len(h) - 3):
        window = h[i-3:i+4]
        if h[i]['h'] == max(x['h'] for x in window) and sum(x['h'] == h[i]['h'] for x in window) == 1:
            highs.append(h[i]['h'])
        if h[i]['l'] == min(x['l'] for x in window) and sum(x['l'] == h[i]['l'] for x in window) == 1:
            lows.append(h[i]['l'])
    if len(highs) < 2 or len(lows) < 2:
        return 'NEUTRAL'
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return 'BUY'
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return 'SELL'
    return 'NEUTRAL'


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


def find_setup(bars, hourly, last_ts):
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
                if retest['t'] <= last_ts or retest['t'] != bars[-1]['t'] or not 8 <= utc(retest['t']).hour < 18:
                    continue
                if retest['l'] > hi or retest['h'] < lo:
                    continue
                entry = retest['c']
                # BTC SMC: rejection candle in intended direction.
                if direction == 'BUY' and not (entry > retest['o'] and entry > hi):
                    continue
                if direction == 'SELL' and not (entry < retest['o'] and entry < lo):
                    continue
                # BTC SMC: 1H structure must agree at entry time.
                if hourly_bias(hourly, retest['t']) != direction:
                    continue
                # AMC POC: approximate Asian volume control point as a hard filter.
                if not (entry > poc if direction == 'BUY' else entry < poc):
                    continue
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


def settle(state, symbol, bars):
    remaining = []
    for pos in state['positions']:
        if pos['symbol'] != symbol:
            remaining.append(pos)
            continue
        direction = pos['direction']
        for bar in bars:
            if bar['t'] <= pos['last_checked_t']:
                continue
            # Conservative intrabar: if stop and target both touched, stop wins.
            stop_hit = bar['l'] <= pos['stop'] if direction == 'BUY' else bar['h'] >= pos['stop']
            target_hit = bar['h'] >= pos['target'] if direction == 'BUY' else bar['l'] <= pos['target']
            if stop_hit or target_hit:
                exit_price = pos['stop'] if stop_hit else pos['target']
                gross_pct = (exit_price / pos['entry'] - 1) * 100
                if direction == 'SELL':
                    gross_pct = -gross_pct
                net_pct = gross_pct - (FEE_RATE * (1 + exit_price / pos['entry']) * 100)
                append_csv(TRADES, {
                    'symbol': symbol, 'direction': direction, 'entry_time_utc': pos['entry_time_utc'],
                    'exit_time_utc': iso(bar['t']), 'leg': pos['leg'],
                    'entry_price': pos['entry'], 'exit_price': exit_price,
                    'stop_loss': pos['initial_stop'], 'target': pos['target'],
                    'exit_reason': 'STOP' if stop_hit else 'TARGET',
                    'net_return_pct_per_leg': round(net_pct, 6),
                    'paper_pnl_usdt': round(net_pct * MARGIN * LEVERAGE / 100, 2)
                })
                notify(f'PAPER EXIT {symbol} {direction} {pos["leg"]}: '
                       f'{"STOP" if stop_hit else "TARGET"} | net {net_pct:.3f}%')
                break
            # Trailing after 2R / 3R for higher legs (only effective next candle).
            if pos['leg'] in ('3R', '4R'):
                progress = (bar['h'] - pos['entry']) if direction == 'BUY' else (pos['entry'] - bar['l'])
                threshold = 2 if pos['leg'] == '3R' else 3
                if progress >= threshold * pos['risk']:
                    proposed = pos['entry'] + (threshold - 1) * pos['risk'] * (1 if direction == 'BUY' else -1)
                    pos['stop'] = max(pos['stop'], proposed) if direction == 'BUY' else min(pos['stop'], proposed)
            pos['last_checked_t'] = bar['t']
        else:
            remaining.append(pos)
    state['positions'] = remaining


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    state = load_state()
    try:
        ranked_symbols = top20_symbols()
    except (requests.RequestException, ValueError) as exc:
        LOG.error('TOP20 selection failed; no new entries: %s', exc)
        # Still process existing positions to avoid losing exit tracking.
        ranked_symbols = []

    # Continue monitoring existing positions even if a coin drops out of Top 20.
    held_symbols = {p['symbol'] for p in state['positions']}
    symbols = list(dict.fromkeys(ranked_symbols + sorted(held_symbols)))
    if not symbols:
        raise RuntimeError('No symbols available; scanner stopped safely')
    for symbol in symbols:
        try:
            bars = candles(symbol, '5', 4)
            hourly = candles(symbol, '60', 18)
            if not bars:
                continue
            # Settle first; old entries remain safe across missed workflow runs.
            settle(state, symbol, bars)
            day = str(utc(bars[-1]['t']).date())
            if symbol not in ranked_symbols:
                state['last_closed_bar'][symbol] = bars[-1]['t']
                save_state(state)
                continue
            if state['entered_days'].get(symbol) == day:
                state['last_closed_bar'][symbol] = bars[-1]['t']
                save_state(state)
                continue
            last = int(state['last_closed_bar'].get(symbol, 0))
            if last == 0:
                # First-ever run: do not retroactively enter on old signals.
                state['last_closed_bar'][symbol] = bars[-1]['t']
                save_state(state)
                LOG.info('%s initialized at %s', symbol, iso(bars[-1]['t']))
                continue
            setup = find_setup(bars, hourly, last)
            state['last_closed_bar'][symbol] = bars[-1]['t']
            if setup and not any(p['symbol'] == symbol for p in state['positions']):
                for rr in (2, 3, 4):
                    target = setup['entry'] + (rr * setup['risk'] * (1 if setup['direction'] == 'BUY' else -1))
                    state['positions'].append({
                        'symbol': symbol, 'direction': setup['direction'], 'leg': f'{rr}R',
                        'entry': setup['entry'], 'stop': setup['stop'],
                        'initial_stop': setup['stop'], 'target': target,
                        'risk': setup['risk'], 'entry_time_utc': iso(setup['bar_t']),
                        'last_checked_t': setup['bar_t']
                    })
                state['entered_days'][symbol] = day
                append_csv(SIGNALS, {'symbol': symbol, 'time_utc': iso(setup['bar_t']),
                                     'direction': setup['direction'], 'entry': setup['entry'],
                                     'stop': setup['stop'], 'asian_high': setup['asian_high'],
                                     'asian_low': setup['asian_low'], 'approx_poc': round(setup['poc'], 8),
                                     'poc_confirms': setup['poc_confirms']})
                notify(f'HYBRID 1H SMC + ASIAN AMC POC PAPER ENTRY {symbol} {setup["direction"]} @ {setup["entry"]} '
                       f'SL {setup["stop"]} | 2R/3R/4R | POC {setup["poc_confirms"]}')
                LOG.info('%s PAPER ENTRY %s', symbol, setup['direction'])
            else:
                LOG.info('%s no fresh qualified setup; open legs=%s', symbol,
                         sum(p['symbol'] == symbol for p in state['positions']))
            save_state(state)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            LOG.error('%s: %s', symbol, exc)
    save_state(state)
    LOG.info('COMPLETE selected_top20=%s scanned=%s open_legs=%s', len(ranked_symbols), len(symbols), len(state['positions']))


if __name__ == '__main__':
    main()
