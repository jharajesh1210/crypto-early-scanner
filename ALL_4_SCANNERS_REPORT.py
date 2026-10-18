
import csv
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

OUTPUT = Path("ALL_4_SCANNERS_PROFIT_LOSS.csv")
MARGIN = Decimal("100")
LEVERAGE = Decimal("30")

HYBRID_FILE = "HYBRID_SMC_AMC_SESSION_PAPER_trades.csv"

SCANNERS = [
    ("BTC_SMC_Paper",
     "BTC_SMC_Paper_trades.csv", "net_pct"),

    ("TOP100 CRIPTO AMC POC Paper Scanner",
     "TOP100_CRIPTO_AMC_POC_trades.csv",
     "net_return_pct_per_leg"),

    ("ASIAN LONDON NY PAPER",
     "ASIAN_LONDON_NY_PAPER_trades.csv",
     "net_return_pct_per_leg"),

    ("HYBRID SMC AMC SESSION Paper",
     HYBRID_FILE, "net_return_pct_per_leg"),

    ("CRYPTO SMC V2 POC Paper",
     "CRYPTO_SMC_V2_POC_PAPER_trades.csv",
     "net_return_pct_per_leg"),

    ("NEW ASIA LONDO NY Paper",
     "NEW_ASIA_LONDO_NY_trades.csv",
     "net_return_pct_per_leg"),
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


def decimal_value(value):
    try:
        result = Decimal(str(value).strip())
        if not result.is_finite():
            raise ValueError("Invalid number")
        return result
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Invalid number: {value}")


def fmt(value):
    return str(
        value.quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    )


def read_trades(filename, pct_column):
    path = Path(filename)

    if not path.exists() or path.stat().st_size == 0:
        return None

    result = []

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            return None

        if pct_column not in reader.fieldnames:
            raise ValueError(
                f"Missing column {pct_column}"
            )

        for line_number, row in enumerate(
            reader, start=2
        ):
            raw = (row.get(pct_column) or "").strip()

            if not raw:
                raw = (row.get("net_pct") or "").strip()

            if not raw:
                raise ValueError(
                    f"Missing return on line {line_number}"
                )

            pct = decimal_value(raw)

            margin = decimal_value(
                row.get("margin_usdt") or MARGIN
            )

            leverage = decimal_value(
                row.get("leverage") or LEVERAGE
            )

            if margin <= 0 or leverage <= 0:
                raise ValueError(
                    f"Invalid margin/leverage on line {line_number}"
                )

            pnl = pct * margin * leverage / Decimal("100")

            strategy = (
                row.get("strategy") or "LEGACY_HYBRID"
            ).strip().upper()

            result.append({
                "pnl": pnl,
                "strategy": strategy,
                "margin": margin,
                "leverage": leverage,
            })

    return result


def summary(name, filename, trades, now, detail=False):
    result = {field: "" for field in FIELDS}

    result.update({
        "Scanner": name,
        "Status": "NO DATA",
        "Closed_Legs": 0,
        "Winning_Legs": 0,
        "Losing_Legs": 0,
        "Margin_Per_Leg_USDT": str(MARGIN),
        "Leverage": str(LEVERAGE),
        "Source_CSV": filename,
        "Updated_UTC": now,
    })

    if not trades:
        return result

    values = [trade["pnl"] for trade in trades]

    wins = sum(pnl > 0 for pnl in values)
    losses = sum(pnl <= 0 for pnl in values)
    count = len(values)

    profit = sum(
        (pnl for pnl in values if pnl > 0),
        Decimal("0"),
    )

    loss = sum(
        (-pnl for pnl in values if pnl < 0),
        Decimal("0"),
    )

    margins = {trade["margin"] for trade in trades}
    leverages = {trade["leverage"] for trade in trades}

    win_rate = Decimal(wins) * 100 / count

    result.update({
        "Status": "DETAIL" if detail else "OK",
        "Closed_Legs": count,
        "Winning_Legs": wins,
        "Losing_Legs": losses,
        "Win_Rate_%": fmt(win_rate),
        "Total_Profit_USDT": fmt(profit),
        "Total_Loss_USDT": fmt(loss),
        "Net_Profit_USDT": fmt(profit - loss),
        "Margin_Per_Leg_USDT": (
            str(next(iter(margins)))
            if len(margins) == 1 else "MIXED"
        ),
        "Leverage": (
            str(next(iter(leverages)))
            if len(leverages) == 1 else "MIXED"
        ),
    })

    return result


def main():
    now = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    rows = []
    all_trades = []
    hybrid_trades = None
    available = 0

    for name, filename, pct_column in SCANNERS:
        try:
            trades = read_trades(filename, pct_column)

            row = summary(
                name, filename, trades, now
            )

            if trades:
                all_trades.extend(trades)
                available += 1

            if filename == HYBRID_FILE:
                hybrid_trades = trades

        except (OSError, ValueError, UnicodeError, csv.Error) as error:
            row = summary(name, filename, None, now)
            row["Status"] = "ERROR: " + str(error)[:130]

        rows.append(row)

        print(
            name,
            "|",
            row["Status"],
            "| P/L:",
            row["Net_Profit_USDT"],
        )

    # HYBRID DETAILS - NOT ADDED TWICE

    for strategy in HYBRID_STRATEGIES:
        selected = [
            trade
            for trade in (hybrid_trades or [])
            if trade["strategy"] == strategy
        ]

        detail = summary(
            "HYBRID / " + strategy,
            HYBRID_FILE,
            selected,
            now,
            detail=True,
        )

        if not selected:
            detail["Status"] = "NO DATA (detail)"

        rows.append(detail)

    other_hybrid = [
        trade
        for trade in (hybrid_trades or [])
        if trade["strategy"] not in HYBRID_STRATEGIES
    ]

    if other_hybrid:
        rows.append(
            summary(
                "HYBRID / LEGACY_OR_OTHER",
                HYBRID_FILE,
                other_hybrid,
                now,
                detail=True,
            )
        )

    # TOTAL: ONLY MAIN SIX SCANNERS

    values = [
        trade["pnl"] for trade in all_trades
    ]

    count = len(values)
    wins = sum(pnl > 0 for pnl in values)
    losses = sum(pnl <= 0 for pnl in values)

    profit = sum(
        (pnl for pnl in values if pnl > 0),
        Decimal("0"),
    )

    loss = sum(
        (-pnl for pnl in values if pnl < 0),
        Decimal("0"),
    )

    total = {field: "" for field in FIELDS}

    total.update({
        "Scanner": "TOTAL (available histories)",
        "Status": (
            "ALL 6 AVAILABLE"
            if available == 6
            else f"PARTIAL: {available}/6 available"
        ),
        "Closed_Legs": count,
        "Winning_Legs": wins,
        "Losing_Legs": losses,
        "Win_Rate_%": (
            fmt(Decimal(wins) * 100 / count)
            if count else ""
        ),
        "Total_Profit_USDT": (
            fmt(profit) if count else ""
        ),
        "Total_Loss_USDT": (
            fmt(loss) if count else ""
        ),
        "Net_Profit_USDT": (
            fmt(profit - loss) if count else ""
        ),
        "Margin_Per_Leg_USDT": "MIXED",
        "Leverage": str(LEVERAGE),
        "Source_CSV": (
            "HYBRID detail excluded from total"
        ),
        "Updated_UTC": now,
    })

    rows.append(total)

    with OUTPUT.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 55)
    print("REPORT CREATED:", OUTPUT)
    print("STATUS:", total["Status"])
    print("CLOSED TRADES:", count)
    print("WINS:", wins)
    print("LOSSES:", losses)
    print("NET P/L:", total["Net_Profit_USDT"], "USDT")
    print("=" * 55)


if __name__ == "__main__":
    main()
