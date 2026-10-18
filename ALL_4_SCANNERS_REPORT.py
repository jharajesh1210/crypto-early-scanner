
"""Paper trading P&L report for 5 scanners.

Includes HYBRID strategy-wise breakdown.
HYBRID detail rows are excluded from grand totals.
Python standard library only.
"""

import csv
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

MARGIN = Decimal("100")
LEVERAGE = Decimal("30")

OUTPUT = Path("ALL_4_SCANNERS_PROFIT_LOSS.csv")

HYBRID_FILE = "HYBRID_SMC_AMC_SESSION_PAPER_trades.csv"

SCANNERS = [
    (
        "BTC_SMC_Paper",
        "BTC_SMC_Paper_trades.csv",
        "net_pct",
    ),
    (
        "TOP100 CRIPTO AMC POC Paper Scanner",
        "TOP100_CRIPTO_AMC_POC_trades.csv",
        "net_return_pct_per_leg",
    ),
    (
        "ASIAN LONDON NY PAPER",
        "ASIAN_LONDON_NY_PAPER_trades.csv",
        "net_return_pct_per_leg",
    ),
    (
        "HYBRID SMC AMC SESSION Paper",
        HYBRID_FILE,
        "net_return_pct_per_leg",
    ),
    (
        "CRYPTO SMC V2 POC Paper",
        "CRYPTO_SMC_V2_POC_PAPER_trades.csv",
        "net_return_pct_per_leg",
    ),
]

HYBRID_STRATEGIES = (
    "BTC_SMC",
    "AMC_POC",
    "ASIAN_LONDON_NY",
)

FIELDS = [
    "Scanner",
    "Status",
    "Closed_Legs",
    "Winning_Legs",
    "Losing_Legs",
    "Win_Rate_%",
    "Total_Profit_USDT",
    "Total_Loss_USDT",
    "Net_Profit_USDT",
    "Margin_Per_Leg_USDT",
    "Leverage",
    "Source_CSV",
    "Updated_UTC",
]


def dec(value, label):
    try:
        result = Decimal(str(value).strip())

        if not result.is_finite():
            raise ValueError("non-finite")

        return result

    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid {label}: {value!r}"
        ) from exc


def money(value):
    return str(
        value.quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    )


def read_trades(filename, percent_col):
    path = Path(filename)

    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(handle)

        if (
            not reader.fieldnames
            or percent_col not in reader.fieldnames
        ):
            raise ValueError(
                f"{filename}: missing column {percent_col}"
            )

        trades = []

        for line, row in enumerate(reader, start=2):
            raw = (
                row.get(percent_col) or ""
            ).strip()

            if not raw:
                fallback = (
                    row.get("net_pct") or ""
                ).strip()

                if not fallback:
                    raise ValueError(
                        f"{filename} line {line}: "
                        "missing net return"
                    )

                raw = fallback

            pct = dec(
                raw,
                f"{filename} line {line} return",
            )

            margin_raw = (
                row.get("margin_usdt") or ""
            ).strip()

            margin = (
                dec(margin_raw, "margin_usdt")
                if margin_raw
                else MARGIN
            )

            leverage_raw = (
                row.get("leverage") or ""
            ).strip()

            leverage = (
                dec(leverage_raw, "leverage")
                if leverage_raw
                else LEVERAGE
            )

            if margin <= 0 or leverage <= 0:
                raise ValueError(
                    f"{filename} line {line}: "
                    "margin/leverage must be positive"
                )

            pnl = (
                pct * margin * leverage
                / Decimal("100")
            )

            strategy = (
                row.get("strategy")
                or "LEGACY_HYBRID"
            ).strip().upper()

            trades.append(
                (pnl, strategy, margin, leverage)
            )

    return trades


def summarize(
    name,
    filename,
    trades,
    stamp,
    detail=False,
):
    row = dict.fromkeys(FIELDS, "")

    row.update(
        Scanner=name,
        Status="NO DATA",
        Closed_Legs=0,
        Winning_Legs=0,
        Losing_Legs=0,
        Margin_Per_Leg_USDT=str(MARGIN),
        Leverage=str(LEVERAGE),
        Source_CSV=filename,
        Updated_UTC=stamp,
    )

    if not trades:
        return row, None

    pnls = [trade[0] for trade in trades]

    profits = sum(
        (value for value in pnls if value > 0),
        Decimal("0"),
    )

    losses = sum(
        (-value for value in pnls if value < 0),
        Decimal("0"),
    )

    wins = sum(value > 0 for value in pnls)
    count = len(pnls)

    margins = {trade[2] for trade in trades}
    leverages = {trade[3] for trade in trades}

    win_rate = (
        Decimal(wins) * 100 / count
    ).quantize(Decimal("0.01"))

    row.update(
        Status="DETAIL" if detail else "OK",
        Closed_Legs=count,
        Winning_Legs=wins,
        Losing_Legs=count - wins,
        **{"Win_Rate_%": str(win_rate)},
        Total_Profit_USDT=money(profits),
        Total_Loss_USDT=money(losses),
        Net_Profit_USDT=money(
            profits - losses
        ),
        Margin_Per_Leg_USDT=(
            str(next(iter(margins)))
            if len(margins) == 1
            else "MIXED"
        ),
        Leverage=(
            str(next(iter(leverages)))
            if len(leverages) == 1
            else "MIXED"
        ),
    )

    return row, (
        count,
        wins,
        count - wins,
        profits,
        losses,
    )


def main():
    stamp = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    rows = []
    totals = []
    hybrid_trades = None

    for name, filename, col in SCANNERS:
        try:
            trades = read_trades(
                filename,
                col,
            )

            row, figures = summarize(
                name,
                filename,
                trades,
                stamp,
            )

        except (
            OSError,
            UnicodeError,
            ValueError,
            csv.Error,
        ) as exc:

            row, figures = summarize(
                name,
                filename,
                None,
                stamp,
            )

            row["Status"] = (
                "ERROR: " + str(exc)[:130]
            )

            trades = None

        rows.append(row)

        if figures:
            totals.append(figures)

        if filename == HYBRID_FILE:
            hybrid_trades = trades

        print(
            f"{name}: {row['Status']} | "
            f"net={row['Net_Profit_USDT']}"
        )

    # HYBRID strategy details:
    # These rows are NOT counted twice in TOTAL.

    for strategy in HYBRID_STRATEGIES:
        subset = [
            trade
            for trade in (hybrid_trades or [])
            if trade[1] == strategy
        ]

        detail, _ = summarize(
            "  HYBRID / " + strategy,
            HYBRID_FILE,
            subset,
            stamp,
            detail=True,
        )

        if not subset:
            detail["Status"] = "NO DATA (detail)"

        rows.append(detail)

    legacy = [
        trade
        for trade in (hybrid_trades or [])
        if trade[1] not in HYBRID_STRATEGIES
    ]

    if legacy:
        detail, _ = summarize(
            "  HYBRID / LEGACY_OR_OTHER",
            HYBRID_FILE,
            legacy,
            stamp,
            detail=True,
        )

        rows.append(detail)

    count = sum(value[0] for value in totals)
    wins = sum(value[1] for value in totals)
    losses = sum(value[2] for value in totals)

    profit = sum(
        (value[3] for value in totals),
        Decimal("0"),
    )

    loss = sum(
        (value[4] for value in totals),
        Decimal("0"),
    )

    available = len(totals)

    total = dict.fromkeys(FIELDS, "")

    if count:
        win_rate = str(
            (
                Decimal(wins) * 100 / count
            ).quantize(Decimal("0.01"))
        )
    else:
        win_rate = ""

    status = (
        f"ALL {len(SCANNERS)} AVAILABLE"
        if available == len(SCANNERS)
        else (
            f"PARTIAL: {available}/"
            f"{len(SCANNERS)} available"
        )
    )

    total.update(
        Scanner="TOTAL (available histories)",
        Status=status,
        Closed_Legs=count,
        Winning_Legs=wins,
        Losing_Legs=losses,
        **{"Win_Rate_%": win_rate},
        Total_Profit_USDT=(
            money(profit) if count else ""
        ),
        Total_Loss_USDT=(
            money(loss) if count else ""
        ),
        Net_Profit_USDT=(
            money(profit - loss)
            if count
            else ""
        ),
        Margin_Per_Leg_USDT=(
            "MIXED / see scanner rows"
        ),
        Leverage=str(LEVERAGE),
        Source_CSV=(
            "Closed trades only; "
            "HYBRID detail rows excluded "
            "from totals"
        ),
        Updated_UTC=stamp,
    )

    rows.append(total)

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"CREATED {OUTPUT} | "
        f"{total['Status']} | "
        f"NET={total['Net_Profit_USDT']} USDT"
    )


if __name__ == "__main__":
    main()
