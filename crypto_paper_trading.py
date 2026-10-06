import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ============================================================
# COINDCX CRYPTO PAPER TRADING PERFORMANCE TRACKER
# ============================================================

BASE_URL = "https://api.coindcx.com"

SIGNAL_FILE = "crypto_5m_signals.csv"
PAPER_FILE = "crypto_paper_trading.csv"

# Forward result periods
CHECK_PERIODS = {
    "15M": 15,
    "30M": 30,
    "1H": 60,
    "4H": 240,
}

PAPER_COLUMNS = [
    "SYMBOL",
    "SIGNAL",
    "ENTRY_TIME",
    "ENTRY_PRICE",
    "HISTORICAL_MATCH",
    "CONFIRMATIONS",
    "PASSED",
    "PRICE_15M",
    "RETURN_15M_%",
    "PRICE_30M",
    "RETURN_30M_%",
    "PRICE_1H",
    "RETURN_1H_%",
    "PRICE_4H",
    "RETURN_4H_%",
    "MAX_PROFIT_4H_%",
    "MAX_LOSS_4H_%",
    "RESULT_4H",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def normalize_time(value):
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts
    except Exception:
        return pd.NaT


def calculate_return(signal, entry_price, exit_price):
    if (
        pd.isna(entry_price)
        or pd.isna(exit_price)
        or entry_price <= 0
    ):
        return np.nan

    if str(signal).upper() == "BUY":
        result = ((exit_price - entry_price) / entry_price) * 100
    else:
        result = ((entry_price - exit_price) / entry_price) * 100

    return round(result, 4)


# ============================================================
# COINDCX MARKET DETAILS
# ============================================================

def get_market_details():
    url = BASE_URL + "/exchange/v1/markets_details"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        mapping = {}

        for item in data:
            symbol = str(
                item.get("symbol", "")
            ).upper()

            pair = item.get("pair")

            if symbol and pair:
                mapping[symbol] = pair

        return mapping

    except Exception as e:
        print("Market details error:", e)
        return {}


# ============================================================
# DOWNLOAD 1-MINUTE CANDLES
# ============================================================

def get_1m_candles(pair, start_ms, end_ms):
    url = BASE_URL + "/market_data/candles"

    params = {
        "pair": pair,
        "interval": "1m",
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": 1000,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        required = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        raw_time = pd.to_numeric(
            df["time"],
            errors="coerce"
        )

        # Detect timestamp unit
        median_time = raw_time.dropna().median()

        if pd.isna(median_time):
            return pd.DataFrame()

        if median_time > 10_000_000_000:
            df["datetime"] = pd.to_datetime(
                raw_time,
                unit="ms",
                utc=True,
                errors="coerce"
            )
        else:
            df["datetime"] = pd.to_datetime(
                raw_time,
                unit="s",
                utc=True,
                errors="coerce"
            )

        df = (
            df
            .dropna(
                subset=[
                    "datetime",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            )
            .sort_values("datetime")
            .drop_duplicates(
                subset=["datetime"]
            )
            .reset_index(drop=True)
        )

        return df

    except Exception as e:
        print("Candle error:", pair, e)
        return pd.DataFrame()


# ============================================================
# ADD NEW LIVE SIGNALS TO PAPER TRADING
# ============================================================

def add_new_signals():
    if not os.path.exists(SIGNAL_FILE):
        print(
            "Signal file not found:",
            SIGNAL_FILE
        )
        return

    try:
        signals = pd.read_csv(SIGNAL_FILE)
    except Exception as e:
        print("Signal CSV read error:", e)
        return

    if signals.empty:
        print("Signal CSV is empty.")
        return

    if os.path.exists(PAPER_FILE):
        try:
            paper = pd.read_csv(PAPER_FILE)
        except Exception:
            paper = pd.DataFrame(
                columns=PAPER_COLUMNS
            )
    else:
        paper = pd.DataFrame(
            columns=PAPER_COLUMNS
        )

    for col in PAPER_COLUMNS:
        if col not in paper.columns:
            paper[col] = np.nan

    added = 0

    for _, row in signals.iterrows():

        symbol = str(
            row.get("SYMBOL", "")
        ).upper().strip()

        signal = str(
            row.get("SIGNAL", "")
        ).upper().strip()

        entry_time_raw = row.get("TIME")

        entry_price = safe_float(
            row.get("PRICE")
        )

        if (
            not symbol
            or signal not in ["BUY", "SELL"]
            or pd.isna(entry_price)
        ):
            continue

        entry_time = normalize_time(
            entry_time_raw
        )

        if pd.isna(entry_time):
            continue

        entry_iso = entry_time.isoformat()

        duplicate = False

        if not paper.empty:
            duplicate = (
                (
                    paper["SYMBOL"]
                    .astype(str)
                    .str.upper()
                    == symbol
                )
                &
                (
                    paper["SIGNAL"]
                    .astype(str)
                    .str.upper()
                    == signal
                )
                &
                (
                    paper["ENTRY_TIME"]
                    .astype(str)
                    == entry_iso
                )
            ).any()

        if duplicate:
            continue

        new_row = {
            "SYMBOL": symbol,
            "SIGNAL": signal,
            "ENTRY_TIME": entry_iso,
            "ENTRY_PRICE": entry_price,
            "HISTORICAL_MATCH": row.get(
                "HISTORICAL_MATCH",
                np.nan
            ),
            "CONFIRMATIONS": row.get(
                "CONFIRMATIONS",
                np.nan
            ),
            "PASSED": row.get(
                "PASSED",
                ""
            ),
            "PRICE_15M": np.nan,
            "RETURN_15M_%": np.nan,
            "PRICE_30M": np.nan,
            "RETURN_30M_%": np.nan,
            "PRICE_1H": np.nan,
            "RETURN_1H_%": np.nan,
            "PRICE_4H": np.nan,
            "RETURN_4H_%": np.nan,
            "MAX_PROFIT_4H_%": np.nan,
            "MAX_LOSS_4H_%": np.nan,
            "RESULT_4H": "PENDING",
        }

        paper = pd.concat(
            [
                paper,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

        added += 1

    paper.to_csv(
        PAPER_FILE,
        index=False
    )

    print("New paper signals added:", added)
    print("Total paper signals:", len(paper))


# ============================================================
# UPDATE FORWARD PERFORMANCE
# ============================================================

def update_performance():

    if not os.path.exists(PAPER_FILE):
        print("Paper trading file not found.")
        return

    paper = pd.read_csv(PAPER_FILE)

    if paper.empty:
        print("No paper trades yet.")
        return

    markets = get_market_details()

    if not markets:
        print("Could not load market details.")
        return

    now = pd.Timestamp.now(tz="UTC")

    updated = 0

    for idx, row in paper.iterrows():

        symbol = str(
            row["SYMBOL"]
        ).upper().strip()

        signal = str(
            row["SIGNAL"]
        ).upper().strip()

        entry_price = safe_float(
            row["ENTRY_PRICE"]
        )

        entry_time = normalize_time(
            row["ENTRY_TIME"]
        )

        if (
            pd.isna(entry_time)
            or pd.isna(entry_price)
            or entry_price <= 0
        ):
            continue

        pair = markets.get(symbol)

        if not pair:
            print(
                "Pair not found:",
                symbol
            )
            continue

        age_minutes = (
            now - entry_time
        ).total_seconds() / 60

        # Nothing to check yet
        if age_minutes < 15:
            continue

        # Download only required forward period
        forward_end = min(
            now,
            entry_time
            + pd.Timedelta(minutes=245)
        )

        start_ms = int(
            entry_time.timestamp() * 1000
        )

        end_ms = int(
            forward_end.timestamp() * 1000
        )

        candles = get_1m_candles(
            pair,
            start_ms,
            end_ms
        )

        if candles.empty:
            continue

        changed = False

        # ----------------------------------------
        # 15M / 30M / 1H / 4H returns
        # ----------------------------------------

        for label, minutes in CHECK_PERIODS.items():

            price_col = "PRICE_" + label
            return_col = "RETURN_" + label + "_%"

            existing = row.get(
                return_col,
                np.nan
            )

            if pd.notna(existing):
                continue

            target_time = (
                entry_time
                + pd.Timedelta(
                    minutes=minutes
                )
            )

            if now < target_time:
                continue

            eligible = candles[
                candles["datetime"]
                >= target_time
            ]

            if eligible.empty:
                continue

            exit_price = safe_float(
                eligible.iloc[0]["close"]
            )

            trade_return = calculate_return(
                signal,
                entry_price,
                exit_price
            )

            paper.at[
                idx,
                price_col
            ] = round(exit_price, 8)

            paper.at[
                idx,
                return_col
            ] = trade_return

            changed = True

        # ----------------------------------------
        # MAX PROFIT / MAX LOSS during first 4H
        # ----------------------------------------

        four_hour_end = (
            entry_time
            + pd.Timedelta(hours=4)
        )

        available_end = min(
            now,
            four_hour_end
        )

        window = candles[
            (candles["datetime"] >= entry_time)
            &
            (candles["datetime"] <= available_end)
        ]

        if not window.empty:

            highest = safe_float(
                window["high"].max()
            )

            lowest = safe_float(
                window["low"].min()
            )

            if signal == "BUY":

                max_profit = (
                    (highest - entry_price)
                    / entry_price
                    * 100
                )

                max_loss = (
                    (lowest - entry_price)
                    / entry_price
                    * 100
                )

            else:

                max_profit = (
                    (entry_price - lowest)
                    / entry_price
                    * 100
                )

                max_loss = (
                    (entry_price - highest)
                    / entry_price
                    * 100
                )

            paper.at[
                idx,
                "MAX_PROFIT_4H_%"
            ] = round(max_profit, 4)

            paper.at[
                idx,
                "MAX_LOSS_4H_%"
            ] = round(max_loss, 4)

            changed = True

        # ----------------------------------------
        # FINAL RESULT after 4 hours
        # ----------------------------------------

        if age_minutes >= 240:

            final_return = safe_float(
                paper.at[
                    idx,
                    "RETURN_4H_%"
                ]
            )

            if not pd.isna(final_return):

                if final_return > 0:
                    result = "WIN"

                elif final_return < 0:
                    result = "LOSS"

                else:
                    result = "FLAT"

                paper.at[
                    idx,
                    "RESULT_4H"
                ] = result

                changed = True

        if changed:
            updated += 1

        time.sleep(0.10)

    paper.to_csv(
        PAPER_FILE,
        index=False
    )

    print(
        "Paper trades updated:",
        updated
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary():

    if not os.path.exists(PAPER_FILE):
        return

    df = pd.read_csv(PAPER_FILE)

    if df.empty:
        return

    print()
    print("=" * 60)
    print("PAPER TRADING SUMMARY")
    print("=" * 60)

    print("Total signals:", len(df))

    completed = df[
        df["RESULT_4H"].isin(
            ["WIN", "LOSS", "FLAT"]
        )
    ]

    print(
        "Completed 4H:",
        len(completed)
    )

    if len(completed) > 0:

        wins = (
            completed["RESULT_4H"]
            == "WIN"
        ).sum()

        losses = (
            completed["RESULT_4H"]
            == "LOSS"
        ).sum()

        flats = (
            completed["RESULT_4H"]
            == "FLAT"
        ).sum()

        win_rate = (
            wins
            / len(completed)
            * 100
        )

        print("Wins:", wins)
        print("Losses:", losses)
        print("Flat:", flats)

        print(
            "4H Win Rate:",
            round(win_rate, 2),
            "%"
        )

        returns = pd.to_numeric(
            completed["RETURN_4H_%"],
            errors="coerce"
        )

        print(
            "Average 4H Return:",
            round(
                returns.mean(),
                4
            ),
            "%"
        )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("COINDCX PAPER TRADING TRACKER")
    print("=" * 60)

    print(
        "Input:",
        SIGNAL_FILE
    )

    print(
        "Output:",
        PAPER_FILE
    )

    add_new_signals()

    update_performance()

    print_summary()

    print()
    print(
        "Paper trading update complete."
    )


if __name__ == "__main__":
    main()
