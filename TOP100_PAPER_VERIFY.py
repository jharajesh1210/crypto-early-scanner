"""Read-only CoinDCX futures paper-trade verifier.
Reads TOP100_CRIPTO_AMC_POC_state.json; writes TOP100_PAPER_VERIFY_results.csv.
Does not modify the original state, live scanner or trades CSV.
"""
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

STATE = Path('TOP100_CRIPTO_AMC_POC_state.json')
OUTPUT = Path('TOP100_PAPER_VERIFY_results.csv')
API = 'https://public.coindcx.com/market_data/candlesticks'
SLIPPAGE = 0.0002
FEE_PER_SIDE = 0.0005


def fetch_bars(pair, entry_time):
    now = int(time.time())
    start = int(pd.Timestamp(entry_time).timestamp()) - 600
    if start >= now:
        return pd.DataFrame(), 'ENTRY_IN_FUTURE'
    # Query in chunks to avoid API candle-count limits; overlapping chunks are deduplicated.
    rows = []
    chunk_seconds = 24 * 3600
    for begin in range(start, now + 1, chunk_seconds):
        end = min(begin + chunk_seconds + 600, now)
        for attempt in range(3):
            try:
                response = requests.get(API, params={
                    'pair': pair, 'pcode': 'f', 'resolution': '5',
                    'from': begin, 'to': end
                }, timeout=30)
                response.raise_for_status()
                payload = response.json()
                batch = payload.get('data') if isinstance(payload, dict) else None
                if not isinstance(batch, list):
                    raise ValueError('Invalid candle response')
                rows.extend(batch)
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == 2:
                    return pd.DataFrame(), f'API_ERROR: {exc}'
                time.sleep(2 * (attempt + 1))
    if not rows:
        return pd.DataFrame(), 'NO_CANDLES'
    df = pd.DataFrame(rows)
    required = ('time', 'high', 'low')
    if any(k not in df for k in required):
        return pd.DataFrame(), 'MISSING_CANDLE_COLUMNS'
    for k in required:
        df[k] = pd.to_numeric(df[k], errors='coerce')
    df = df.dropna(subset=list(required))
    if df.empty:
        return pd.DataFrame(), 'NO_VALID_CANDLES'
    unit = 'ms' if df['time'].median() > 1e11 else 's'
    df['bar_start'] = pd.to_datetime(df['time'], unit=unit, utc=True)
    df = df.drop_duplicates('bar_start').sort_values('bar_start')
    entry = pd.Timestamp(entry_time)
    now_ts = pd.Timestamp.now(tz='UTC')
    df = df[(df['bar_start'] >= entry) &
            (df['bar_start'] + pd.Timedelta(minutes=5) <= now_ts)]
    return df.reset_index(drop=True), ''


def verify(trade, df, error):
    entry_time = pd.Timestamp(trade['entry_time'])
    result = {
        'trade_id': trade['id'], 'pair': trade['pair'],
        'direction': trade['direction'], 'rr': trade['rr'],
        'entry_time_utc': trade['entry_time'],
        'entry': trade['entry'], 'stop': trade['stop'],
        'target': trade['target'], 'result': 'UNKNOWN',
        'exit_candle_utc': '', 'net_return_pct': '',
        'candles_checked': len(df), 'first_candle_utc': '',
        'last_candle_utc': '', 'notes': error,
    }
    if df.empty:
        return result
    result['first_candle_utc'] = df.iloc[0]['bar_start'].isoformat()
    result['last_candle_utc'] = df.iloc[-1]['bar_start'].isoformat()
    # Missing first or intermediate candle means an earlier exit might be unknown.
    if df.iloc[0]['bar_start'] > entry_time:
        result['notes'] = 'INCOMPLETE: missing candles immediately after entry'
        return result
    differences = df['bar_start'].diff().dropna()
    if (differences > pd.Timedelta(minutes=5)).any():
        result['notes'] = 'INCOMPLETE: 5m gaps in history; outcome uncertain'
        return result
    direction = trade['direction']
    for _, candle in df.iterrows():
        if direction == 'LONG':
            sl = candle['low'] <= trade['stop']
            tp = candle['high'] >= trade['target']
        else:
            sl = candle['high'] >= trade['stop']
            tp = candle['low'] <= trade['target']
        if not (sl or tp):
            continue
        outcome = 'SL' if sl else 'TP'  # SL-first on ambiguous same bar
        raw_exit = trade['stop'] if sl else trade['target']
        exit_price = raw_exit * (1 - SLIPPAGE if direction == 'LONG' else 1 + SLIPPAGE)
        gross = (exit_price / trade['entry'] - 1) * 100
        if direction == 'SHORT':
            gross = -gross
        net = gross - 2 * FEE_PER_SIDE * 100
        result.update(result=outcome,
                      exit_candle_utc=candle['bar_start'].isoformat(),
                      net_return_pct=round(net, 4),
                      notes='SL first if TP and SL in same 5m candle' if sl and tp else '5m OHLC paper estimate')
        return result
    result['result'] = 'STILL_OPEN'
    result['notes'] = 'No SL or TP in retrieved completed 5m candles'
    return result


def main():
    if not STATE.exists():
        sys.exit(f'MISSING {STATE}; upload state file to repository first')
    state = json.loads(STATE.read_text(encoding='utf-8'))
    trades = state.get('open_trades', [])
    if not trades:
        print('No open paper trades in state. Nothing to verify.')
        return
    by_pair = {}
    for trade in trades:
        by_pair.setdefault(trade['pair'], []).append(trade)
    results = []
    for pair, items in by_pair.items():
        earliest = min(t['entry_time'] for t in items)
        print(f'Checking {pair} from {earliest}', flush=True)
        bars, error = fetch_bars(pair, earliest)
        for trade in items:
            # If a pair has multiple entries, filter candles by individual entry.
            subset = bars[bars['bar_start'] >= pd.Timestamp(trade['entry_time'])] if not bars.empty else bars
            result = verify(trade, subset, error)
            results.append(result)
            print(f"  RR{trade['rr']}: {result['result']} {result['notes']}", flush=True)
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f'RESULT FILE: {OUTPUT} | {len(results)} legs', flush=True)
    print('READ ONLY: live state and trade files were not modified.', flush=True)


if __name__ == '__main__':
    main()
