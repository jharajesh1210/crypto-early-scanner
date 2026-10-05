import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


# ============================================================
# COINDCX 6-MONTH HISTORICAL LEARNING
# ============================================================

BASE_URL = "https://api.coindcx.com"

MIN_24H_VOLUME = 5_000_000

HISTORY_DAYS = 180

# 15-minute historical candles
HIST_INTERVAL = "15m"

# Next 8 x 15m = next 2 hours
FORWARD_BARS = 8

# Historical move considered meaningful
SPIKE_PCT = 3.0

MIN_EVENTS = 5

# IMPORTANT:
# Only this many NEW coins are trained in one GitHub run.
COINS_PER_RUN = 3

MODEL_FILE = "crypto_6month_patterns.json"


# ============================================================
# GET $5M+ USDT MARKETS
# ============================================================

def get_symbols():

    print("Getting CoinDCX markets...")

    try:
        markets_response = requests.get(
            f"{BASE_URL}/exchange/v1/markets_details",
            timeout=30
        )
        markets_response.raise_for_status()
        markets = markets_response.json()

        ticker_response = requests.get(
            f"{BASE_URL}/exchange/ticker",
            timeout=30
        )
        ticker_response.raise_for_status()
        tickers = ticker_response.json()

    except Exception as e:
        print("CoinDCX market API error:", e)
        return []

    ticker_map = {}

    for ticker in tickers:

        try:
            market_name = str(
                ticker.get("market") or ""
            ).upper().strip()

            volume = float(
                ticker.get("volume") or 0
            )

            last_price = float(
                ticker.get("last_price") or 0
            )

            if not market_name:
                continue

            if last_price <= 0:
                continue

            volume_usdt = volume * last_price

            ticker_map[market_name] = {
                "volume_usdt": volume_usdt,
                "last_price": last_price
            }

        except Exception:
            continue

    print(
        "Ticker markets received:",
        len(ticker_map)
    )

    result = []
    seen = set()

    for market in markets:

        try:
            status = str(
                market.get("status") or ""
            ).lower().strip()

            base_currency = str(
                market.get(
                    "base_currency_short_name"
                ) or ""
            ).upper().strip()

            symbol = str(
                market.get("coindcx_name")
                or market.get("symbol")
                or ""
            ).upper().strip()

            pair = str(
                market.get("pair") or ""
            ).strip()

            if status != "active":
                continue

            if base_currency != "USDT":
                continue

            if not symbol or not pair:
                continue

            ticker = ticker_map.get(symbol)

            if ticker is None:
                continue

            volume_usdt = float(
                ticker["volume_usdt"]
            )

            if volume_usdt < MIN_24H_VOLUME:
                continue

            if symbol in seen:
                continue

            seen.add(symbol)

            result.append({
                "symbol": symbol,
                "pair": pair,
                "volume_24h": volume_usdt
            })

        except Exception:
            continue

    result.sort(
        key=lambda x: x["volume_24h"],
        reverse=True
    )

    print(
        "USDT markets >= $5M:",
        len(result)
    )

    for market in result[:10]:

        print(
            market["symbol"],
            "| Volume: $"
            f"{market['volume_24h']/1_000_000:.2f}M"
        )

    return result


# ============================================================
# DOWNLOAD CANDLES
# ============================================================

def request_candles(
    pair,
    start_ms,
    end_ms
):

    params = {
        "pair": pair,
        "interval": HIST_INTERVAL,
        "limit": 1000,
        "startTime": int(start_ms),
        "endTime": int(end_ms)
    }

    try:

        response = requests.get(
            f"{BASE_URL}/market_data/candles",
            params=params,
            timeout=30
        )

        if response.status_code != 200:

            print(
                "Candle API status:",
                response.status_code
            )

            return None

        data = response.json()

        if not isinstance(data, list):
            return None

        if not data:
            return None

        df = pd.DataFrame(data)

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "time"
        ]

        if not all(
            column in df.columns
            for column in required
        ):
            return None

        for column in required:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = (
            df
            .dropna(subset=required)
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

        return df

    except Exception as e:

        print(
            "Candle request error:",
            e
        )

        return None


# ============================================================
# DOWNLOAD 6 MONTHS
# ============================================================

def get_six_month_history(pair):

    now = datetime.now(
        timezone.utc
    )

    start = now - timedelta(
        days=HISTORY_DAYS
    )

    start_ms = int(
        start.timestamp() * 1000
    )

    end_ms = int(
        now.timestamp() * 1000
    )

    # 1000 x 15 minute candles
    chunk_ms = (
        1000 *
        15 *
        60 *
        1000
    )

    cursor = start_ms

    frames = []

    chunk_number = 0

    while cursor < end_ms:

        chunk_number += 1

        chunk_end = min(
            cursor + chunk_ms,
            end_ms
        )

        print(
            f"  Download chunk {chunk_number}",
            end="\r"
        )

        df = request_candles(
            pair,
            cursor,
            chunk_end
        )

        if df is not None and not df.empty:
            frames.append(df)

        cursor = chunk_end + 1

        time.sleep(0.10)

    print()

    if not frames:
        return None

    final_df = pd.concat(
        frames,
        ignore_index=True
    )

    final_df = (
        final_df
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    return final_df


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    return (
        100 -
        (
            100 /
            (1 + rs)
        )
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA 20
    df["EMA20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # EMA 50
    df["EMA50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # RSI 14
    df["RSI14"] = calculate_rsi(
        df["close"],
        14
    )

    # MACD 12,26,9
    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_HIST"] = (
        df["MACD"] -
        df["MACD_SIGNAL"]
    )

    # Bollinger Bands 20,2
    df["BB_MIDDLE"] = (
        df["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        df["close"]
        .rolling(20)
        .std()
    )

    df["BB_UPPER"] = (
        df["BB_MIDDLE"] +
        2 * bb_std
    )

    df["BB_LOWER"] = (
        df["BB_MIDDLE"] -
        2 * bb_std
    )

    df["BB_WIDTH"] = (
        (
            df["BB_UPPER"] -
            df["BB_LOWER"]
        )
        /
        df["BB_MIDDLE"]
    )

    # Volume ratio
    df["VOLUME_AVG20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["VOLUME_RATIO"] = (
        df["volume"] /
        df["VOLUME_AVG20"]
    )

    return df


# ============================================================
# VOLUME PROFILE POC
# ============================================================

def calculate_poc(
    df,
    bins=40
):

    if df is None:
        return np.nan

    if len(df) < 20:
        return np.nan

    low = float(
        df["low"].min()
    )

    high = float(
        df["high"].max()
    )

    if not np.isfinite(low):
        return np.nan

    if not np.isfinite(high):
        return np.nan

    if high <= low:
        return np.nan

    edges = np.linspace(
        low,
        high,
        bins + 1
    )

    profile = np.zeros(
        bins
    )

    for _, candle in df.iterrows():

        candle_low = float(
            candle["low"]
        )

        candle_high = float(
            candle["high"]
        )

        candle_volume = float(
            candle["volume"]
        )

        if candle_high <= candle_low:
            continue

        total_range = (
            candle_high -
            candle_low
        )

        for i in range(bins):

            overlap = max(
                0,
                min(
                    candle_high,
                    edges[i + 1]
                )
                -
                max(
                    candle_low,
                    edges[i]
                )
            )

            if overlap > 0:

                profile[i] += (
                    candle_volume *
                    overlap /
                    total_range
                )

    if profile.sum() <= 0:
        return np.nan

    index = int(
        np.argmax(profile)
    )

    poc = (
        edges[index] +
        edges[index + 1]
    ) / 2

    return float(poc)


# ============================================================
# GET TECHNICAL CONDITION AT HISTORICAL EVENT
# ============================================================

def get_features(
    df,
    index
):

    if index < 55:
        return None

    row = df.iloc[index]

    prev = df.iloc[
        index - 1
    ]

    price = float(
        row["close"]
    )

    open_price = float(
        row["open"]
    )

    if price <= 0:
        return None

    if open_price <= 0:
        return None

    if not np.isfinite(
        row["RSI14"]
    ):
        return None

    if not np.isfinite(
        row["VOLUME_RATIO"]
    ):
        return None

    poc_start = max(
        0,
        index - 100
    )

    poc = calculate_poc(
        df.iloc[
            poc_start:
            index + 1
        ]
    )

    if not np.isfinite(poc):
        return None

    candle_move = (
        (
            price -
            open_price
        )
        /
        open_price
        *
        100
    )

    return {

        "rsi":
            float(
                row["RSI14"]
            ),

        "macd_hist_pct":
            float(
                row["MACD_HIST"]
            )
            /
            price
            *
            100,

        "macd_rising":
            bool(
                row["MACD_HIST"]
                >
                prev["MACD_HIST"]
            ),

        "ema_gap_pct":
            (
                float(
                    row["EMA20"]
                )
                -
                float(
                    row["EMA50"]
                )
            )
            /
            price
            *
            100,

        "price_vs_bb_middle_pct":
            (
                price
                -
                float(
                    row["BB_MIDDLE"]
                )
            )
            /
            price
            *
            100,

        "bb_width_pct":
            float(
                row["BB_WIDTH"]
            )
            *
            100,

        "volume_ratio":
            float(
                row["VOLUME_RATIO"]
            ),

        "price_vs_poc_pct":
            (
                price - poc
            )
            /
            poc
            *
            100,

        "candle_move_pct":
            float(
                candle_move
            )
    }


# ============================================================
# FIND BUY / SELL HISTORICAL STARTS
# ============================================================

def find_events(df):

    df = add_indicators(df)

    buy_events = []
    sell_events = []

    last_buy = -999
    last_sell = -999

    end_index = (
        len(df) -
        FORWARD_BARS -
        1
    )

    for i in range(
        55,
        end_index
    ):

        current_price = float(
            df.iloc[i]["close"]
        )

        if current_price <= 0:
            continue

        future = df.iloc[
            i + 1:
            i + 1 + FORWARD_BARS
        ]

        if future.empty:
            continue

        future_high = float(
            future["high"].max()
        )

        future_low = float(
            future["low"].min()
        )

        future_up_pct = (
            (
                future_high -
                current_price
            )
            /
            current_price
            *
            100
        )

        future_down_pct = (
            (
                future_low -
                current_price
            )
            /
            current_price
            *
            100
        )

        features = get_features(
            df,
            i
        )

        if features is None:
            continue

        # ---------------------------
        # Historical BUY start
        # ---------------------------

        if (
            future_up_pct
            >=
            SPIKE_PCT
        ):

            if (
                i - last_buy
                >=
                FORWARD_BARS
            ):

                # Avoid learning after huge candle
                if (
                    features[
                        "candle_move_pct"
                    ]
                    <=
                    2.5
                ):

                    buy_events.append(
                        features
                    )

                    last_buy = i

        # ---------------------------
        # Historical SELL start
        # ---------------------------

        if (
            future_down_pct
            <=
            -SPIKE_PCT
        ):

            if (
                i - last_sell
                >=
                FORWARD_BARS
            ):

                if (
                    features[
                        "candle_move_pct"
                    ]
                    >=
                    -2.5
                ):

                    sell_events.append(
                        features
                    )

                    last_sell = i

    return (
        buy_events,
        sell_events
    )


# ============================================================
# SUMMARIZE LEARNED CONDITIONS
# ============================================================

FEATURE_NAMES = [

    "rsi",
    "macd_hist_pct",
    "ema_gap_pct",
    "price_vs_bb_middle_pct",
    "bb_width_pct",
    "volume_ratio",
    "price_vs_poc_pct",
    "candle_move_pct"

]


def summarize_events(
    events
):

    if len(events) < MIN_EVENTS:
        return None

    result = {
        "event_count":
            len(events)
    }

    for feature in FEATURE_NAMES:

        values = []

        for event in events:

            value = event.get(
                feature
            )

            if (
                value is not None
                and
                np.isfinite(value)
            ):

                values.append(
                    float(value)
                )

        if len(values) < MIN_EVENTS:
            continue

        array = np.array(
            values,
            dtype=float
        )

        median = float(
            np.median(array)
        )

        mad = float(
            np.median(
                np.abs(
                    array -
                    median
                )
            )
        )

        minimum_scale = {

            "rsi":
                5.0,

            "macd_hist_pct":
                0.02,

            "ema_gap_pct":
                0.15,

            "price_vs_bb_middle_pct":
                0.20,

            "bb_width_pct":
                0.30,

            "volume_ratio":
                0.25,

            "price_vs_poc_pct":
                0.50,

            "candle_move_pct":
                0.20

        }.get(
            feature,
            0.1
        )

        scale = max(
            mad * 1.4826,
            minimum_scale
        )

        result[feature] = {

            "median":
                round(
                    median,
                    6
                ),

            "scale":
                round(
                    scale,
                    6
                )
        }

    return result


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

def load_existing_model():

    if not os.path.exists(
        MODEL_FILE
    ):

        return {

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "updated_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "history_days":
                HISTORY_DAYS,

            "historical_interval":
                HIST_INTERVAL,

            "spike_pct":
                SPIKE_PCT,

            "symbols":
                {}
        }

    try:

        with open(
            MODEL_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            model = json.load(
                file
            )

        if "symbols" not in model:
            model["symbols"] = {}

        return model

    except Exception as e:

        print(
            "Existing model read error:",
            e
        )

        return {

            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "symbols":
                {}
        }


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    model["updated_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    model["history_days"] = (
        HISTORY_DAYS
    )

    model[
        "historical_interval"
    ] = HIST_INTERVAL

    model["spike_pct"] = (
        SPIKE_PCT
    )

    with open(
        MODEL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            model,
            file,
            indent=2
        )

    print(
        "\nModel saved:",
        MODEL_FILE
    )


# ============================================================
# TRAIN ONE COIN
# ============================================================

def train_coin(
    market
):

    symbol = market[
        "symbol"
    ]

    pair = market[
        "pair"
    ]

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "LEARNING:",
        symbol
    )

    print(
        "PAIR:",
        pair
    )

    print(
        "24H VOLUME: $"
        f"{market['volume_24h']/1_000_000:.2f}M"
    )

    print(
        "=" * 60
    )

    history = (
        get_six_month_history(
            pair
        )
    )

    if history is None:

        print(
            symbol,
            "historical data unavailable."
        )

        return None

    print(
        symbol,
        "historical candles:",
        len(history)
    )

    if len(history) < 500:

        print(
            symbol,
            "not enough historical data."
        )

        return None

    buy_events, sell_events = (
        find_events(
            history
        )
    )

    print(
        symbol,
        "BUY events:",
        len(buy_events)
    )

    print(
        symbol,
        "SELL events:",
        len(sell_events)
    )

    buy_model = (
        summarize_events(
            buy_events
        )
    )

    sell_model = (
        summarize_events(
            sell_events
        )
    )

    result = {

        "pair":
            pair,

        "volume_24h_usdt":
            round(
                float(
                    market[
                        "volume_24h"
                    ]
                ),
                2
            ),

        "trained_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "buy":
            buy_model,

        "sell":
            sell_model
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "COINDCX 6-MONTH "
        "HISTORICAL LEARNING"
    )

    print(
        "=" * 70
    )

    print(
        "Minimum 24H volume: "
        "$5,000,000"
    )

    print(
        "History:",
        HISTORY_DAYS,
        "days"
    )

    print(
        "Historical timeframe:",
        HIST_INTERVAL
    )

    print(
        "Move definition:",
        f"{SPIKE_PCT}%"
    )

    print(
        "Coins per run:",
        COINS_PER_RUN
    )

    markets = get_symbols()

    if not markets:

        print(
            "\nNo qualifying markets found."
        )

        return

    print(
        "\nQualifying markets:",
        len(markets)
    )

    model = (
        load_existing_model()
    )

    learned_symbols = (
        model.get(
            "symbols",
            {}
        )
    )

    print(
        "Already learned:",
        len(learned_symbols)
    )

    # Find coins not yet trained
    pending = []

    for market in markets:

        symbol = market[
            "symbol"
        ]

        if (
            symbol
            not in
            learned_symbols
        ):

            pending.append(
                market
            )

    print(
        "Still pending:",
        len(pending)
    )

    if not pending:

        print(
            "\nAll current $5M+ "
            "markets are already learned."
        )

        save_model(
            model
        )

        return

    # Only a small batch each run
    batch = pending[
        :COINS_PER_RUN
    ]

    print(
        "\nThis run will learn:"
    )

    for market in batch:

        print(
            "-",
            market["symbol"]
        )

    successful = 0

    for market in batch:

        symbol = market[
            "symbol"
        ]

        try:

            result = train_coin(
                market
            )

            if result is not None:

                model[
                    "symbols"
                ][symbol] = result

                successful += 1

                # Save after every coin.
                # If GitHub run stops later,
                # completed work is still in file.
                save_model(
                    model
                )

                print(
                    symbol,
                    "learning completed."
                )

            else:

                print(
                    symbol,
                    "learning failed/skipped."
                )

        except Exception as e:

            print(
                symbol,
                "ERROR:",
                e
            )

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "HISTORICAL LEARNING RUN COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Coins completed this run:",
        successful
    )

    print(
        "Total models saved:",
        len(
            model.get(
                "symbols",
                {}
            )
        )
    )

    remaining = max(
        0,
        len(pending)
        -
        len(batch)
    )

    print(
        "Approx. remaining:",
        remaining
    )

    print(
        "Output file:",
        MODEL_FILE
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nHistorical learning stopped."
        )

    except Exception as e:

        print(
            "\nFATAL ERROR:",
            e
        )

        raise
