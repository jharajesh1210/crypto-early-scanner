"""Cumulative paper P/L for six scanners, with independent HYBRID breakdown.
Run from repository root. Python standard library only.
Existing history is read-only; this script only writes the summary report.
"""
import csv
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

MARGIN = Decimal('100')
LEVERAGE = Decimal('30')
OUTPUT = Path('ALL_4_SCANNERS_PROFIT_LOSS.csv')
HYBRID_FILE = 'HYBRID_SMC_AMC_SESSION_PAPER_trades.csv'
SCANNERS = [
    ('BTC_SMC_Paper', 'BTC_SMC_Paper_trades.csv', 'net_pct'),
    ('TOP100 CRIPTO AMC POC Paper Scanner', 'TOP100_CRIPTO_AMC_POC_trades.csv', 'net_return_pct_per_leg'),
    ('ASIAN LONDON NY PAPER', 'ASIAN_LONDON_NY_PAPER_trades.csv', 'net_return_pct_per_leg'),
    ('HYBRID SMC AMC SESSION Paper', HYBRID_FILE, 'net_return_pct_per_leg'),
    ('CRYPTO SMC V2 POC Paper', 'CRYPTO_SMC_V2_POC_PAPER_trades.csv', 'net_return_pct_per_leg'),
    ('NEW ASIA LONDO NY Paper', 'NEW_ASIA_LONDO_NY_trades.csv', 'net_return_pct_per_leg'),
]
HYBRID_STRATEGIES = ('BTC_SMC', 'AMC_POC', 'ASIAN_LONDON_NY')
FIELDS = [
    'Scanner', 'Status', 'Closed_Legs', 'Winning_Legs', 'Losing_Legs',
    'Win_Rate_%', 'Total_Profit_USDT', 'Total_Loss_USDT', 'Net_Profit_USDT',
    'Margin_Per_Leg_USDT', 'Leverage', 'Source_CSV', 'Updated_UTC'
]


def dec(value, label):
    try:
        result = Decimal(str(value).strip())
        if not result.is_finite():
            raise ValueError('non-finite')
        return result
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f'Invalid {label}: {value!r}') from exc


def money(value):
    return str(value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def read_trades(filename, percent_col):
    path = Path(filename)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or percent_col not in reader.fieldnames:
            raise ValueError(f'{filename}: missing column {percent_col}')
        trades = []
        for line, row in enumerate(reader, start=2):
            raw = (row.get(percent_col) or '').strip()
            if not raw:
                # A mixed-schema CSV may contain a blank field only when
                # older rows used another supported return column.
                fallback = (row.get('net_pct') or '').strip()
                if not fallback:
                    raise ValueError(f'{filename} line {line}: missing net return')
                raw = fallback
            pct = dec(raw, f'{filename} line {line} return')
            margin_raw = (row.get('margin_usdt') or '').strip()
            margin = dec(margin_raw, 'margin_usdt') if margin_raw else MARGIN
            leverage_raw = (row.get('leverage') or '').strip()
            leverage = dec(leverage_raw, 'leverage') if leverage_raw else LEVERAGE
            if margin <= 0 or leverage <= 0:
                raise ValueError(f'{filename} line {line}: margin/leverage must be positive')
            # Use raw net percentage to avoid rounding a precomputed dollar P/L.
            pnl = pct * margin * leverage / Decimal('100')
            strategy = (row.get('strategy') or 'LEGACY_HYBRID').strip().upper()
            trades.append((pnl, strategy, margin, leverage))
    return trades


def summarize(name, filename, trades, stamp, detail=False):
    row = dict.fromkeys(FIELDS, '')
    row.update(Scanner=name, Status='NO DATA', Closed_Legs=0,
               Winning_Legs=0, Losing_Legs=0,
               Margin_Per_Leg_USDT=str(MARGIN), Leverage=str(LEVERAGE),
               Source_CSV=filename, Updated_UTC=stamp)
    if not trades:
        return row, None
    pnls = [t[0] for t in trades]
    profits = sum((v for v in pnls if v > 0), Decimal('0'))
    losses = sum((-v for v in pnls if v < 0), Decimal('0'))
    wins = sum(v > 0 for v in pnls)
    n = len(pnls)
    margins = {t[2] for t in trades}
    leverages = {t[3] for t in trades}
    row.update(
        Status='DETAIL' if detail else 'OK', Closed_Legs=n,
        Winning_Legs=wins, Losing_Legs=n - wins,
        **{'Win_Rate_%': str((Decimal(wins) * 100 / n).quantize(Decimal('0.01')))},
        Total_Profit_USDT=money(profits), Total_Loss_USDT=money(losses),
