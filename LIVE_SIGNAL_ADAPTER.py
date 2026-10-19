"""Read-only bridge from three ORIGINAL paper scanners to DRY-RUN controller.

Never runs scanners, modifies their files, or sends exchange orders.
Commands:
  python LIVE_SIGNAL_ADAPTER.py signals
  python LIVE_SIGNAL_ADAPTER.py prices
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT_SIGNALS = ROOT / 'LIVE_SIGNALS.jsonl'
OUT_PRICES = ROOT / 'LIVE_PRICES.jsonl'
CONTROLLER_STATE = ROOT / 'LIVE_RISK_STATE.json'
CONFIG = ROOT / 'LIVE_RISK_CONFIG.json'
API = 'https://public.coindcx.com/market_data/candlesticks'
MAX_SIGNAL_AGE = 120


def timestamp(value):
    if isinstance(value, (float, int)):
        return datetime.fromtimestamp(float(value) / 1000, timezone.utc)
    d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Timezone required')
    return d.astimezone(timezone.utc)


def iso(d):
    return d.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def csv_rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def signal(scanner, symbol, side, entry, stop, completed, key):
    entry, stop = float(entry), float(stop)
    if not all(math.isfinite(x) and x > 0 for x in (entry, stop)):
        raise ValueError('Invalid price')
    if side not in ('BUY', 'SELL', 'LONG', 'SHORT'):
        raise ValueError('Invalid side')
    return {'signal_id': f'{scanner}:{symbol}:{key}', 'scanner': scanner,
            'symbol': symbol, 'side': side, 'entry': entry, 'stop_loss': stop,
            'timestamp_utc': iso(completed)}


def gather_signals():
    now = datetime.now(timezone.utc)
    cfg = read_json(CONFIG)
    if not cfg or cfg.get('mode') != 'DRY_RUN' or cfg.get('live_orders_enabled') is not False:
        raise RuntimeError('DRY_RUN safety config required')
    window = min(MAX_SIGNAL_AGE, int(cfg['max_signal_age_seconds']))
    candidates = []
    # BTC state position entry_time is 5m candle START, so close = start + 5m.
    btc = read_json(ROOT / 'BTC_SMC_Paper_state.json')
    if btc:
        for p in btc.get('positions', []):
            if p.get('trade_mode') != 'SINGLE_TRAIL_4R':
                continue  # legacy positions not imported
            start = timestamp(p['entry_time'])
            completed = start + timedelta(minutes=5)
            candidates.append(signal('BTC_SMC_Paper', 'BTCUSDT', p['side'],
                                     p['entry'], p['initial_stop'], completed,
                                     f'{int(p["entry_time"])}'))
    # TOP100 signal CSV entry_time is 15m candle END (confirmed time).
    for r in csv_rows(ROOT / 'TOP100_CRIPTO_AMC_POC_signals.csv'):
        pair = r['pair'].upper()
        if not (pair.startswith('B-') and pair.endswith('_USDT')):
            continue
        symbol = pair[2:-5] + 'USDT'
        completed = timestamp(r['entry_time'])
        candidates.append(signal('TOP100_CRIPTO_AMC_POC', symbol, r['direction'],
                                 r['entry'], r['stop'], completed,
                                 r.get('trade_id') or iso(completed)))
    # Asian signal CSV time_utc is 5m candle START, close = start + 5m.
    for r in csv_rows(ROOT / 'ASIAN_LONDON_NY_PAPER_signals.csv'):
        start = timestamp(r['time_utc'])
        completed = start + timedelta(minutes=5)
        candidates.append(signal('ASIAN_LONDON_NY_PAPER', r['symbol'].upper(),
                                 r['direction'], r['entry'], r['stop'], completed,
                                 f'{r["symbol"]}:{iso(start)}'))
    fresh = {}
    for s in candidates:
        age = (now - timestamp(s['timestamp_utc'])).total_seconds()
        if -30 <= age <= window:
            fresh[s['signal_id']] = s
    OUT_SIGNALS.write_text(''.join(json.dumps(v) + '\n' for v in fresh.values()), encoding='utf-8')
    print(f'Adapter: {len(fresh)} fresh signals; original scanners untouched')


def gather_prices():
    cfg = read_json(CONFIG)
    if not cfg or cfg.get('mode') != 'DRY_RUN' or cfg.get('live_orders_enabled') is not False:
        raise RuntimeError('DRY_RUN safety config required')
    state = read_json(CONTROLLER_STATE) or {}
    positions = state.get('positions', {})
    if not isinstance(positions, dict):
        raise RuntimeError('Invalid controller state')
    result = []
    now_ms = int(time.time() * 1000)
    for symbol, pos in positions.items():
        if not symbol.endswith('USDT') or not symbol[:-4].isalnum():
            raise RuntimeError('Invalid position symbol')
        pair = f'B-{symbol[:-4]}_USDT'
        opened = timestamp(pos['opened_utc'])
        last = (state.get('last_price_time') or {}).get(symbol)
        since = timestamp(last) if last else opened
        # Fail closed if downtime exceeds the available price-history window.
        if (datetime.now(timezone.utc) - since).total_seconds() > 48 * 3600:
            raise RuntimeError(f'{symbol}: >48h candle gap; manual review required')
        from_s = max(0, int(since.timestamp()) - 600)
        resp = requests.get(API, params={'pair': pair, 'resolution': '5',
                            'pcode': 'f', 'from': from_s, 'to': int(time.time())}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict) or payload.get('s') != 'ok' or not isinstance(payload.get('data'), list):
            raise RuntimeError(f'{symbol}: invalid CoinDCX candles')
        candles = {}
        for x in payload['data']:
            t = int(x['time'])
            if t < 10**11:
                t *= 1000
            # controller expects timestamp of CLOSED candle, not opening time
            close_t = t + 300000
            if close_t > now_ms - 1500 or close_t <= int(since.timestamp() * 1000):
                continue
            hi, lo, close = float(x['high']), float(x['low']), float(x['close'])
            if not all(math.isfinite(v) and v > 0 for v in (hi, lo, close)) or hi < lo or not lo <= close <= hi:
                raise RuntimeError(f'{symbol}: invalid OHLC')
            candles[close_t] = {'symbol': symbol, 'high': hi, 'low': lo,
                                'close': close, 'timestamp_utc': iso(datetime.fromtimestamp(close_t / 1000, timezone.utc))}
        result.extend(candles.values())
    result.sort(key=lambda x: x['timestamp_utc'])
    OUT_PRICES.write_text(''.join(json.dumps(x) + '\n' for x in result), encoding='utf-8')
    print(f'Adapter: {len(result)} completed 5m bars for {len(positions)} positions')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('signals', 'prices'))
    args = p.parse_args()
    if args.command == 'signals':
        gather_signals()
    else:
        gather_prices()


if __name__ == '__main__':
    main()
