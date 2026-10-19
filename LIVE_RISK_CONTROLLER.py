"""DRY-RUN ONLY: USDT futures signal/risk simulator. NO exchange order endpoints.

Usage:
  python LIVE_RISK_CONTROLLER.py --signals LIVE_SIGNALS.jsonl
  python LIVE_RISK_CONTROLLER.py --prices LIVE_PRICES.jsonl

Signal JSONL example:
 {"signal_id":"unique-123","scanner":"BTC_SMC_Paper","symbol":"BTCUSDT","side":"BUY","entry":100000,"stop_loss":99500,"timestamp_utc":"2026-10-19T02:00:00Z"}

Price JSONL example (COMPLETED bar only, one per line):
 {"symbol":"BTCUSDT","high":101000,"low":100100,"close":100800,"timestamp_utc":"2026-10-19T02:05:00Z"}

Signal and price files must be produced by a separate, validated adapter.
No automatic connection to existing scanners or market data is implemented here.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'LIVE_RISK_CONFIG.json'
STATE = ROOT / 'LIVE_RISK_STATE.json'
DECISIONS = ROOT / 'LIVE_RISK_DECISIONS.csv'
TRADES = ROOT / 'LIVE_RISK_TRADES.csv'
APPROVED = {'BTC_SMC_Paper', 'TOP100_CRIPTO_AMC_POC', 'ASIAN_LONDON_NY_PAPER'}
SIGNAL_FIELDS = {'signal_id', 'scanner', 'symbol', 'side', 'entry', 'stop_loss', 'timestamp_utc'}


def now():
    return datetime.now(timezone.utc)


def parse_time(value):
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timestamp requires timezone')
    return dt.astimezone(timezone.utc)


def num(value, name, zero=False):
    n = float(value)
    if not math.isfinite(n) or (n < 0 if zero else n <= 0):
        raise ValueError(f'{name} invalid')
    return n


def config():
    c = json.loads(CONFIG.read_text(encoding='utf-8'))
    if c.get('mode') != 'DRY_RUN' or c.get('live_orders_enabled') is not False:
        raise RuntimeError('SAFETY LOCK: DRY_RUN and live_orders_enabled=false required')
    if c.get('exchange') != 'COINDCX' or c.get('market') != 'FUTURES' or c.get('margin_currency') != 'USDT':
        raise RuntimeError('Unsupported exchange/market/currency')
    if c.get('margin_mode') != 'isolated' or set(c.get('allowed_scanners', [])) != APPROVED:
        raise RuntimeError('Invalid margin mode or approved scanner list')
    if num(c['margin_per_trade_usdt'], 'margin') != 5 or num(c['leverage'], 'leverage') != 25:
        raise RuntimeError('Expected 5 USDT margin and 25x leverage')
    if int(c['max_open_positions']) != 20 or num(c['max_total_margin_usdt'], 'margin cap') > 100:
        raise RuntimeError('20-position cap / max 100 USDT total margin required')
    if c.get('partial_exit_enabled') is not False or c.get('trailing_stop_enabled') is not True:
        raise RuntimeError('Trailing enabled; partial exit disabled')
    for key, expected in [('move_stop_at_2r', 2), ('new_stop_at_2r', 2), ('move_stop_at_3r', 3), ('new_stop_at_3r', 3), ('full_exit_at_4r', 4)]:
        if float(c.get(key, -999)) != expected:
            raise RuntimeError('2R/3R/4R trailing rules must not change')
    for key in ('starting_capital_usdt', 'max_risk_per_trade_usdt', 'max_total_open_risk_usdt', 'max_daily_loss_usdt'):
        num(c[key], key)
    num(c.get('min_free_balance_buffer_usdt', 0), 'buffer', zero=True)
    for key in ('estimated_fee_rate_per_side', 'estimated_slippage_rate_per_side'):
        if num(c[key], key, zero=True) > 0.01:
            raise RuntimeError('Excessive fee/slippage assumption')
    return c


def load_state():
    s = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    for k, default in [('positions', {}), ('seen_ids', []), ('realized_by_day', {}), ('last_price_time', {})]:
        s.setdefault(k, default)
    return s


def save_state(s):
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(s, indent=2), encoding='utf-8')
    temp.replace(STATE)


def append_csv(path, row):
    columns = list(row)
    if path.exists() and path.stat().st_size:
        with path.open(newline='', encoding='utf-8-sig') as f:
            existing = csv.DictReader(f)
            old_columns = existing.fieldnames or []
            if set(columns) - set(old_columns):
                raise RuntimeError(f'{path.name} has incompatible columns; refusing to overwrite history')
            columns = old_columns
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        if new_file:
            w.writeheader()
        w.writerow(row)


def open_risk(s):
    return sum(float(p['estimated_risk_usdt']) for p in s['positions'].values())


def margin_used(s, c):
    return len(s['positions']) * float(c['margin_per_trade_usdt'])


def available_sim_balance(s, c):
    realized = sum(float(v) for v in s['realized_by_day'].values())
    return float(c['starting_capital_usdt']) + realized - margin_used(s, c)


def evaluate(sig, c, s):
    row = {'checked_utc': now().isoformat(), 'signal_id': str(sig.get('signal_id', '')), 'scanner': str(sig.get('scanner', '')), 'symbol': str(sig.get('symbol', '')).upper(), 'side': str(sig.get('side', '')).upper(), 'status': 'REJECTED', 'reason': '', 'entry': '', 'stop_loss': '', 'target_2r': '', 'target_3r': '', 'target_4r': '', 'quantity': '', 'margin_usdt': '', 'notional_usdt': '', 'estimated_risk_usdt': ''}
    try:
        missing = SIGNAL_FIELDS - set(sig)
        if missing:
            raise ValueError('Missing: ' + ', '.join(sorted(missing)))
        sid, symbol = row['signal_id'], row['symbol']
        if not sid or len(sid) > 150 or sid in s['seen_ids']:
            raise ValueError('Invalid or duplicate signal_id')
        if row['scanner'] not in APPROVED:
            raise ValueError('Scanner not approved')
        if not symbol.endswith('USDT') or not symbol[:-4].isalnum():
            raise ValueError('Invalid USDT futures symbol')
        side = row['side']
        if side not in ('BUY', 'SELL', 'LONG', 'SHORT'):
            raise ValueError('Invalid side')
        side = 'LONG' if side in ('BUY', 'LONG') else 'SHORT'
        entry = num(sig['entry'], 'entry')
        stop = num(sig['stop_loss'], 'stop_loss')
        direction = 1 if side == 'LONG' else -1
        distance = direction * (entry - stop)
        if distance <= 0:
            raise ValueError('Stop on wrong side')
        age = (now() - parse_time(sig['timestamp_utc'])).total_seconds()
        if age < -30 or age > int(c['max_signal_age_seconds']):
            raise ValueError(f'Stale/future signal ({age:.0f}s)')
        if symbol in s['positions']:
            raise ValueError('Position already open on this symbol')
        if len(s['positions']) >= int(c['max_open_positions']):
            raise ValueError('20-position limit reached')
        if float(s['realized_by_day'].get(now().date().isoformat(), 0)) <= -float(c['max_daily_loss_usdt']):
            raise ValueError('Daily loss limit reached')
        margin = float(c['margin_per_trade_usdt'])
        notional = margin * float(c['leverage'])
        qty = notional / entry
        fee_slip = float(c['estimated_fee_rate_per_side']) + float(c['estimated_slippage_rate_per_side'])
        risk = qty * distance + 2 * notional * fee_slip
        if risk > float(c['max_risk_per_trade_usdt']) + 1e-9:
            raise ValueError(f'Risk {risk:.4f} > per-trade limit')
        if open_risk(s) + risk > float(c['max_total_open_risk_usdt']) + 1e-9:
            raise ValueError('Combined open risk exceeded')
        if margin_used(s, c) + margin > float(c['max_total_margin_usdt']) + 1e-9:
            raise ValueError('Combined margin cap exceeded')
        if available_sim_balance(s, c) - margin < float(c['min_free_balance_buffer_usdt']):
            raise ValueError('Insufficient free simulated margin + reserve buffer')
        levels = {f'target_{r}r': entry + direction * r * distance for r in (2, 3, 4)}
        if min(levels.values()) <= 0:
            raise ValueError('Nonpositive target price')
        row.update(status='ACCEPTED_DRY_RUN', reason='SIMULATION ONLY; ZERO LIVE ORDERS', side=side, entry=entry, stop_loss=stop, quantity=qty, margin_usdt=margin, notional_usdt=notional, estimated_risk_usdt=round(risk, 6), **levels)
        s['positions'][symbol] = {'signal_id': sid, 'scanner': row['scanner'], 'symbol': symbol, 'side': side, 'entry': entry, 'initial_stop': stop, 'current_stop': stop, 'risk_distance': distance, 'quantity': qty, 'estimated_risk_usdt': risk, 'locked_r': 0, 'opened_utc': now().isoformat(), **levels}
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        row['reason'] = str(exc)
    if row['signal_id'] and row['signal_id'] not in s['seen_ids']:
        s['seen_ids'].append(row['signal_id'])
    return row


def process_price(bar, c, s):
    symbol = str(bar['symbol']).upper()
    if symbol not in s['positions']:
        return None
    ts = parse_time(bar['timestamp_utc'])
    last = s['last_price_time'].get(symbol)
    if last and ts <= parse_time(last):
        return None
    p = s['positions'][symbol]
    if ts <= parse_time(p['opened_utc']):
        return None
    hi, lo = num(bar['high'], 'high'), num(bar['low'], 'low')
    if hi < lo:
        raise ValueError('Price high below low')
    sign = 1 if p['side'] == 'LONG' else -1
    stop, target = float(p['current_stop']), float(p['target_4r'])
    stop_hit = lo <= stop if sign == 1 else hi >= stop
    target_hit = hi >= target if sign == 1 else lo <= target
    s['last_price_time'][symbol] = ts.isoformat()
    if stop_hit or target_hit:
        # Conservative assumption when both hit in the same OHLC bar: stop wins.
        price = stop if stop_hit else target
        reason = ('TRAIL_STOP' if p['locked_r'] else 'STOP') if stop_hit else 'TARGET_4R'
        slip = float(c['estimated_slippage_rate_per_side'])
        fee = float(c['estimated_fee_rate_per_side'])
        effective_exit = price * (1 - sign * slip)
        pnl = sign * (effective_exit - float(p['entry'])) * float(p['quantity'])
        pnl -= (float(p['entry']) + effective_exit) * float(p['quantity']) * fee
        day = ts.date().isoformat()
        s['realized_by_day'][day] = round(float(s['realized_by_day'].get(day, 0)) + pnl, 8)
        row = {'closed_utc': ts.isoformat(), 'signal_id': p['signal_id'], 'scanner': p['scanner'], 'symbol': symbol, 'side': p['side'], 'entry': p['entry'], 'exit': effective_exit, 'exit_reason': reason, 'locked_r': p['locked_r'], 'pnl_usdt': round(pnl, 6)}
        del s['positions'][symbol]
        return row
    best = hi if sign == 1 else lo
    progress = sign * (best - float(p['entry'])) / float(p['risk_distance'])
    new_lock = 3 if progress >= 3 else 2 if progress >= 2 else int(p['locked_r'])
    if new_lock > int(p['locked_r']):
        p['locked_r'] = new_lock
        p['current_stop'] = float(p['entry']) + sign * new_lock * float(p['risk_distance'])
        # Updated stop applies only to the NEXT completed bar.
        print(f'TRAIL SIMULATED: {symbol} {new_lock}R -> SL {p["current_stop"]:.8f} (next bar)')
    return None


def main():
    parser = argparse.ArgumentParser(description='DRY RUN ONLY; no CoinDCX order calls')
    parser.add_argument('--signals', help='Fresh normalized JSONL signal file')
    parser.add_argument('--prices', help='Completed OHLC JSONL bar file for trailing simulation')
    args = parser.parse_args()
    c = config()
    s = load_state()
    if not args.signals and not args.prices:
        print('DRY RUN READY | supply --signals or --prices | ZERO LIVE ORDERS')
        return
    if args.signals:
        path = Path(args.signals)
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError('Expected JSON object')
                result = evaluate(obj, c, s)
            except (ValueError, TypeError) as exc:
                result = {'checked_utc': now().isoformat(), 'signal_id': '', 'scanner': '', 'symbol': '', 'side': '', 'status': 'REJECTED', 'reason': str(exc), 'entry': '', 'stop_loss': '', 'target_2r': '', 'target_3r': '', 'target_4r': '', 'quantity': '', 'margin_usdt': '', 'notional_usdt': '', 'estimated_risk_usdt': ''}
            append_csv(DECISIONS, result)
            save_state(s)
            print(f'{result["status"]}: {result["signal_id"]} | {result["reason"]}')
    if args.prices:
        path = Path(args.prices)
        if not path.exists():
            raise FileNotFoundError(path)
        bars = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        bars.sort(key=lambda b: parse_time(b['timestamp_utc']))
        for bar in bars:
            result = process_price(bar, c, s)
            if result:
                append_csv(TRADES, result)
                print(f'CLOSED SIMULATED: {result["symbol"]} {result["exit_reason"]} PNL {result["pnl_usdt"]:+.4f}')
            save_state(s)
    print(f'DRY RUN COMPLETE | OPEN {len(s["positions"])} / 20 | MARGIN {margin_used(s,c):.2f} USDT | LIVE ORDERS ZERO')


if __name__ == '__main__':
    main()
