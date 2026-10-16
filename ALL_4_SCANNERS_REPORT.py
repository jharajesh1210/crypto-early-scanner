"""Combine four independent paper-trade CSV histories into ONE cumulative P/L report.
Run from the GitHub repository root. Python standard library only.
Assumption: $100 margin and 30x leverage PER CLOSED TRADE LEG.
"""
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

MARGIN_USDT = Decimal('100')
LEVERAGE = Decimal('30')
NOTIONAL = MARGIN_USDT * LEVERAGE
OUTPUT = Path('ALL_4_SCANNERS_PROFIT_LOSS.csv')

SCANNERS = [
    ('BTC_SMC_Paper', 'BTC_SMC_Paper_trades.csv', 'net_pct'),
    ('100AND BTC SMC Paper Scanner', '100AND_BTC_paper_trades.csv', 'net_pct'),
    ('TOP100 CRIPTO AMC POC Paper Scanner', 'TOP100_CRIPTO_AMC_POC_trades.csv', 'net_return_pct_per_leg'),
    ('NEW TOP100 AMD POC 1H Paper', 'TOP100_AMD_POC_1H_PAPER_trades.csv', 'net_return_pct_per_leg'),
]

FIELDS = [
    'Scanner', 'Status', 'Closed_Legs', 'Winning_Legs', 'Losing_Legs',
    'Win_Rate_%', 'Total_Profit_USDT', 'Total_Loss_USDT', 'Net_Profit_USDT',
    'Margin_Per_Leg_USDT', 'Leverage', 'Source_CSV', 'Updated_UTC'
]


def money(value):
    return str(value.quantize(Decimal('0.01')))


def scan(name, filename, percent_col, timestamp):
    path = Path(filename)
    base = dict(Scanner=name, Status='NO DATA', Closed_Legs=0,
                Winning_Legs=0, Losing_Legs=0, **{'Win_Rate_%': ''},
                Total_Profit_USDT='', Total_Loss_USDT='', Net_Profit_USDT='',
                Margin_Per_Leg_USDT=str(MARGIN_USDT), Leverage=str(LEVERAGE),
                Source_CSV=filename, Updated_UTC=timestamp)
    if not path.exists() or path.stat().st_size == 0:
        return base, None
    try:
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or percent_col not in reader.fieldnames:
                raise ValueError(f'Missing required column: {percent_col}')
            values = []
            for row_num, row in enumerate(reader, start=2):
                raw = (row.get(percent_col) or '').strip()
                if not raw:
                    raise ValueError(f'Empty {percent_col} at row {row_num}')
                try:
                    values.append(Decimal(raw) * NOTIONAL / Decimal('100'))
                except InvalidOperation as exc:
                    raise ValueError(f'Invalid return at row {row_num}: {raw}') from exc
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        base['Status'] = 'ERROR: ' + str(exc)[:140]
        return base, None
    if not values:
        return base, None
    profit = sum((v for v in values if v > 0), Decimal('0'))
    loss = sum((-v for v in values if v < 0), Decimal('0'))
    wins = sum(v > 0 for v in values)
    losses = sum(v <= 0 for v in values)
    base.update(Status='OK', Closed_Legs=len(values), Winning_Legs=wins,
                Losing_Legs=losses,
                **{'Win_Rate_%': str((Decimal(wins) * 100 / len(values)).quantize(Decimal('0.01')))},
                Total_Profit_USDT=money(profit), Total_Loss_USDT=money(loss),
                Net_Profit_USDT=money(profit - loss))
    return base, (len(values), wins, losses, profit, loss)


def main():
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows, totals = [], []
    for name, filename, col in SCANNERS:
        row, figures = scan(name, filename, col, timestamp)
        rows.append(row)
        if figures is not None:
            totals.append(figures)
        print(f"{name}: {row['Status']} | net={row['Net_Profit_USDT']}")
    count = sum(v[0] for v in totals)
    wins = sum(v[1] for v in totals)
    losses = sum(v[2] for v in totals)
    profit = sum((v[3] for v in totals), Decimal('0'))
    loss = sum((v[4] for v in totals), Decimal('0'))
    complete = len(totals) == len(SCANNERS)
    rows.append(dict(
        Scanner='TOTAL (available histories)',
        Status='ALL 4 AVAILABLE' if complete else f'PARTIAL: {len(totals)}/4 available',
        Closed_Legs=count, Winning_Legs=wins, Losing_Legs=losses,
        **{'Win_Rate_%': str((Decimal(wins) * 100 / count).quantize(Decimal('0.01'))) if count else ''},
        Total_Profit_USDT=money(profit) if count else '',
        Total_Loss_USDT=money(loss) if count else '',
        Net_Profit_USDT=money(profit - loss) if count else '',
        Margin_Per_Leg_USDT=str(MARGIN_USDT), Leverage=str(LEVERAGE),
        Source_CSV='Cumulative closed legs; no unrealized P/L', Updated_UTC=timestamp
    ))
    with OUTPUT.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'CREATED {OUTPUT} | {rows[-1]["Status"]} | NET={rows[-1]["Net_Profit_USDT"]} USDT')


if __name__ == '__main__':
    main()
