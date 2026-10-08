"""V20 corrected fast directional study. Research only; no SL, TP, RR or orders.
Requires crypto_futures_historical_v19.py and crypto_futures_historical_v20.py.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import crypto_futures_historical_v19 as v19
import crypto_futures_historical_v20 as v20

DAYS = 180
WORKERS = 4
MIN_SCORE = 6
COOLDOWN_BARS = 12
HORIZONS = {'15m': 3, '1h': 12, '4h': 48, '12h': 144, '24h': 288}
COST_PCT = 0.10  # illustrative round-trip cost; not SL/TP


def valid(x):
    return bool(pd.notna(x) and x)


def corrected_signal(df, i):
    row = df.iloc[i]
    # Expensive profile computation only after an actual directional trigger.
    bull_trigger = valid(row['bull_breakout']) or valid(row['bull_rejection']) or valid(row['weak_hammer'])
    bear_trigger = valid(row['bear_breakdown']) or valid(row['bear_rejection']) or valid(row['shooting_star'])
    if not (bull_trigger or bear_trigger):
        return None
    profile = v20.volume_profile_at(df, i)
    if not profile:
        return None
    close = float(row['close'])
    poc, vah, val = profile['POC'], profile['VAH'], profile['VAL']
    shape = profile['SHAPE']
    # Profile shape is descriptive, not automatically bullish/bearish.
    # Direction is confirmed by price relative to value area and POC.
    volume = valid(row['volume_surge'])
    bb = valid(row['bb_expansion']) or valid(row['bb_squeeze'])
    macd = float(row['macd_hist']) if pd.notna(row['macd_hist']) else 0
    rsi = float(row['rsi']) if pd.notna(row['rsi']) else 50
    candidates = []
    if bull_trigger and close > poc and (close >= val):
        checks = {
            'BULL_TREND': valid(row['bull_trend']),
            'ABOVE_POC': True,
            'BB': bb,
            'VOLUME_1_5X': volume,
            'BULL_PRICE_ACTION': bull_trigger,
            'RSI': (30 <= rsi <= 65) and valid(row['rsi_rising']),
            'MACD': macd > 0,
            'BREAKOUT': valid(row['bull_breakout']),
        }
        score = sum(checks.values()) + int(checks['BREAKOUT'])
        if score >= MIN_SCORE and (volume or checks['BREAKOUT']):
            candidates.append(('LONG', score, checks))
    if bear_trigger and close < poc and (close <= vah):
        checks = {
            'BEAR_TREND': valid(row['bear_trend']),
            'BELOW_POC': True,
            'BB': bb,
            'VOLUME_1_5X': volume,
            'BEAR_PRICE_ACTION': bear_trigger,
            'RSI': (35 <= rsi <= 75) and valid(row['rsi_falling']),
            'MACD': macd < 0,
            'BREAKDOWN': valid(row['bear_breakdown']),
        }
        score = sum(checks.values()) + int(checks['BREAKDOWN'])
        if score >= MIN_SCORE and (volume or checks['BREAKDOWN']):
            candidates.append(('SHORT', score, checks))
    if len(candidates) != 1:
        return None
    side, score, checks = candidates[0]
    return dict(side=side, score=score, reasons='|'.join(k for k, v in checks.items() if v),
                profile_shape=shape, poc=poc, vah=vah, val=val)


def run_pair(pair):
    raw = v19.fetch_history(pair, DAYS)
    if raw is None or raw.empty or len(raw) < 3000:
        return pair, [], 'Insufficient candles'
    df = v20.prepare_indicators(raw).copy()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    for c in ('open', 'high', 'low', 'close'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Discard potentially incomplete last candle.
    df = df.iloc[:-1].reset_index(drop=True)
    rows = []
    last = -COOLDOWN_BARS
    max_h = max(HORIZONS.values())
    # Do not inspect any candle without full 24h follow-up.
    for i in range(250, len(df) - max_h - 1):
        if i - last < COOLDOWN_BARS:
            continue
        s = corrected_signal(df, i)
        if s is None:
            continue
        entry_i = i + 1  # trade cannot enter on the signal candle close retroactively
        entry = float(df.at[entry_i, 'open'])
        if not np.isfinite(entry) or entry <= 0:
            continue
        last = i
        sign = 1 if s['side'] == 'LONG' else -1
        rec = {'pair': pair, 'signal_time': df.at[i, 'datetime'],
               'entry_time': df.at[entry_i, 'datetime'], 'side': s['side'],
               'entry_price': entry, 'score': s['score'], 'reasons': s['reasons'],
               'profile_shape': s['profile_shape'], 'poc': s['poc'],
               'vah': s['vah'], 'val': s['val'],
               'rsi': rowval(df, i, 'rsi'), 'macd_hist': rowval(df, i, 'macd_hist'),
               'bb_width_pct': rowval(df, i, 'bb_width_pct'),
               'volume_ratio': rowval(df, i, 'volume_ratio')}
        for label, n in HORIZONS.items():
            # Close of nth completed 5m candle starting from entry candle.
            end_i = entry_i + n - 1
            exit_close = float(df.at[end_i, 'close'])
            ret = sign * (exit_close / entry - 1) * 100
            rec[f'{label}_direction_pct'] = round(ret, 5)
            rec[f'{label}_after_cost_pct'] = round(ret - COST_PCT, 5)
        rows.append(rec)
    return pair, rows, None


def rowval(df, i, name):
    x = df.at[i, name]
    return float(x) if pd.notna(x) else np.nan


def main():
    print('V20 CORRECTED FAST | 50 candidates | 180d 5m | NO SL / TP / RR', flush=True)
    pairs = v20.get_top50_futures()
    print('Eligible pairs:', len(pairs), flush=True)
    all_rows, errors = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        tasks = {pool.submit(run_pair, p): p for p in pairs}
        for done, fut in enumerate(as_completed(tasks), 1):
            pair = tasks[fut]
            try:
                _, rows, err = fut.result()
                all_rows.extend(rows)
                if err:
                    errors.append((pair, err))
                print(f'[{done}/{len(pairs)}] {pair}: {len(rows)} signals' + (f' | {err}' if err else ''), flush=True)
            except Exception as exc:
                errors.append((pair, str(exc)))
                print(f'[{done}/{len(pairs)}] {pair}: ERROR {exc}', flush=True)
    trades = pd.DataFrame(all_rows)
    trades.to_csv('v20_corrected_signals.csv', index=False)
    summaries = []
    if not trades.empty:
        for (pair, side), g in trades.groupby(['pair', 'side']):
            for h in HORIZONS:
                ret = g[f'{h}_direction_pct']
                net = g[f'{h}_after_cost_pct']
                summaries.append({'pair': pair, 'side': side, 'horizon': h,
                                  'signals': len(g), 'direction_accuracy_pct': round(100 * (ret > 0).mean(), 2),
                                  'after_cost_positive_pct': round(100 * (net > 0).mean(), 2),
                                  'avg_direction_pct': round(ret.mean(), 5),
                                  'median_direction_pct': round(ret.median(), 5),
                                  'avg_after_cost_pct': round(net.mean(), 5)})
    result = pd.DataFrame(summaries)
    if not result.empty:
        result = result.sort_values(['horizon', 'avg_after_cost_pct'], ascending=[True, False])
    result.to_csv('v20_corrected_summary.csv', index=False)
    pd.DataFrame(errors, columns=['pair', 'error']).to_csv('v20_corrected_errors.csv', index=False)
    print('Done. Signals:', len(trades), 'Pairs with errors:', len(errors), flush=True)
    print('Saved v20_corrected_signals.csv, v20_corrected_summary.csv, v20_corrected_errors.csv', flush=True)
    print('Directional horizon analysis only. No real orders. Not an executable portfolio return.', flush=True)


if __name__ == '__main__':
    main()
