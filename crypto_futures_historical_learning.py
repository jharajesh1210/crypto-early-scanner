import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ============================================================
# COINDCX FUTURES HISTORICAL LEARNING - VERSION 1
# ============================================================
#
# Main idea:
#   1H chart = market direction / filter
#   5M chart = trade setup / entry
#
# Historical setups tested:
#
# 1. Volume Profile:
#       P -> P -> D
#       P -> P -> P -> D
#       + bearish confirmation
#       => SHORT candidate
#
# 2. Bollinger Band Squeeze:
#       BB Upper + Lower close / narrow / parallel
#       then 2 green candles => LONG
#       then 2 red candles   => SHORT
#
# 3. Big candle + rejection wick:
#       Upper rejection + next bearish candle => SHORT
#       Lower rejection near lower BB
#       + next bullish candle => LONG
#
# 4. Weak hammer / long lower wick:
#       next candle sweeps below hammer low
#       and recovers => LONG
#
# IMPORTANT:
# This is a research / paper-testing script.
# It does NOT place real orders.
# ============================================================


BASE_URL = "https://public.coindcx.com/market_data/candlesticks"

# Start with BTC only.
# Later we will automatically scan multiple Futures pairs.
PAIR = "B-BTC_USDT"

DAYS_TO_TEST = 180

OUTPUT_FILE = "crypto_futures_historical_results.csv"

REQUEST_TIMEOUT = 30

# Number of days fetched in one API request.
# Keeping chunks smaller makes the API easier to handle.
CHUNK_DAYS_5M = 5
CHUNK_DAYS_1H = 30


# ============================================================
# HISTORICAL RESULT SETTINGS
# ============================================================

# We will check what happened after each setup.
FORWARD_BARS = {
    "15M": 3,
    "30M": 6,
    "1H": 12,
    "4H": 48,
}

# Temporary V1 classification threshold.
# Later this will be optimized from real results.
WIN_THRESHOLD_PCT = 0.50


# ============================================================
# API
# ============================================================

def fetch_candles(pair, resolution, start_ts, end_ts):
    params = {
        "pair": pair,
        "from": int(start_ts),
        "to": int(end_ts),
        "resolution": str(resolution),
        "pcode": "f",
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    raw = response.json()

    # CoinDCX can return candle data inside a dictionary.
    if isinstance(raw, dict):
        if "data" in raw:
            raw = raw["data"]
        elif "candles" in raw:
            raw = raw["candles"]
        else:
            # Find first list value if response structure changes.
            found = None

            for value in raw.values():
                if isinstance(value, list):
                    found = value
                    break

            raw = found if found is not None else []

    if not isinstance(raw, list):
        return pd.DataFrame()

    if len(raw) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(raw)

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "time",
    ]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].copy()

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "time",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna()

    if df.empty:
        return df

    # CoinDCX time is milliseconds in our verified Futures response.
    df["datetime"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True,
    )

    df = (
        df.sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def fetch_history(pair, resolution, days, chunk_days):
    now = int(time.time())

    start = now - (days * 24 * 60 * 60)

    chunk_seconds = chunk_days * 24 * 60 * 60

    frames = []

    current = start

    chunk_no = 0

    while current < now:
        chunk_no += 1

        chunk_end = min(
            current + chunk_seconds,
            now,
        )

        print(
            f"Fetching resolution={resolution} "
            f"chunk {chunk_no}: "
            f"{datetime.fromtimestamp(current, tz=timezone.utc)} "
            f"to "
            f"{datetime.fromtimestamp(chunk_end, tz=timezone.utc)}"
        )

        try:
            df = fetch_candles(
                pair,
                resolution,
                current,
                chunk_end,
            )

            if not df.empty:
                frames.append(df)

                print(
                    "Candles received:",
                    len(df),
                )
            else:
                print("No candles in this chunk.")

        except Exception as exc:
            print(
                "API ERROR:",
                exc,
            )

        current = chunk_end

        time.sleep(0.20)

    if not frames:
        return pd.DataFrame()

    final = pd.concat(
        frames,
        ignore_index=True,
    )

    final = (
        final.sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return final


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    df = df.copy()

    # -------------------------
    # EMA
    # -------------------------

    df["ema20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    # -------------------------
    # RSI 14
    # -------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    df["rsi"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # -------------------------
    # MACD
    # -------------------------

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["macd_hist"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # -------------------------
    # Bollinger Bands 20, 2
    # -------------------------

    df["bb_middle"] = (
        df["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        df["close"]
        .rolling(20)
        .std()
    )

    df["bb_upper"] = (
        df["bb_middle"]
        + (2 * bb_std)
    )

    df["bb_lower"] = (
        df["bb_middle"]
        - (2 * bb_std)
    )

    df["bb_width_pct"] = (
        (
            df["bb_upper"]
            - df["bb_lower"]
        )
        / df["bb_middle"]
        * 100
    )

    # -------------------------
    # Volume
    # -------------------------

    df["volume_ma20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        / df["volume_ma20"]
    )

    # -------------------------
    # Candle calculations
    # -------------------------

    df["body"] = (
        df["close"]
        - df["open"]
    ).abs()

    df["range"] = (
        df["high"]
        - df["low"]
    )

    df["upper_wick"] = (
        df["high"]
        - df[
            ["open", "close"]
        ].max(axis=1)
    )

    df["lower_wick"] = (
        df[
            ["open", "close"]
        ].min(axis=1)
        - df["low"]
    )

    df["body_pct"] = (
        df["body"]
        / df["close"]
        * 100
    )

    df["range_pct"] = (
        df["range"]
        / df["close"]
        * 100
    )

    df["green"] = (
        df["close"]
        > df["open"]
    )

    df["red"] = (
        df["close"]
        < df["open"]
    )

    return df


# ============================================================
# 1-HOUR TREND
# ============================================================

def calculate_1h_trend(df_1h):
    df = calculate_indicators(df_1h)

    conditions = [
        (
            (df["close"] > df["ema20"])
            & (df["ema20"] > df["ema50"])
        ),
        (
            (df["close"] < df["ema20"])
            & (df["ema20"] < df["ema50"])
        ),
    ]

    choices = [
        "BULLISH",
        "BEARISH",
    ]

    df["trend_1h"] = np.select(
        conditions,
        choices,
        default="NEUTRAL",
    )

    return df[
        [
            "datetime",
            "close",
            "ema20",
            "ema50",
            "rsi",
            "macd_hist",
            "trend_1h",
        ]
    ].copy()


def attach_1h_trend(df_5m, df_1h):
    trend = calculate_1h_trend(
        df_1h
    )

    df_5m = df_5m.sort_values(
        "datetime"
    ).copy()

    trend = trend.sort_values(
        "datetime"
    ).copy()

    merged = pd.merge_asof(
        df_5m,
        trend[
            [
                "datetime",
                "trend_1h",
            ]
        ],
        on="datetime",
        direction="backward",
    )

    return merged


# ============================================================
# APPROXIMATE VOLUME PROFILE
# ============================================================

def profile_shape(df, end_index, lookback=24):
    """
    V1 approximation of P / D / b / UNKNOWN profile.

    We do not have tick-by-tick volume-at-price from normal
    OHLCV candles, so V1 distributes each candle's volume
    around its typical price.

    Later versions can improve this if better Futures
    volume-at-price data is available.
    """

    start = max(
        0,
        end_index - lookback + 1,
    )

    window = df.iloc[
        start:end_index + 1
    ]

    if len(window) < 12:
        return "UNKNOWN", np.nan

    prices = (
        (
            window["high"]
            + window["low"]
            + window["close"]
        )
        / 3
    ).to_numpy()

    volumes = (
        window["volume"]
        .to_numpy()
    )

    low_price = float(
        window["low"].min()
    )

    high_price = float(
        window["high"].max()
    )

    if (
        not np.isfinite(low_price)
        or not np.isfinite(high_price)
        or high_price <= low_price
    ):
        return "UNKNOWN", np.nan

    bins = np.linspace(
        low_price,
        high_price,
        21,
    )

    hist, edges = np.histogram(
        prices,
        bins=bins,
        weights=volumes,
    )

    if hist.sum() <= 0:
        return "UNKNOWN", np.nan

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2

    poc_index = int(
        np.argmax(hist)
    )

    poc = float(
        centers[poc_index]
    )

    total_volume = float(
        hist.sum()
    )

    lower_volume = float(
        hist[:7].sum()
    )

    middle_volume = float(
        hist[7:13].sum()
    )

    upper_volume = float(
        hist[13:].sum()
    )

    lower_share = (
        lower_volume
        / total_volume
    )

    middle_share = (
        middle_volume
        / total_volume
    )

    upper_share = (
        upper_volume
        / total_volume
    )

    price_range = (
        high_price
        - low_price
    )

    poc_position = (
        (poc - low_price)
        / price_range
    )

    # P-shape:
    # majority volume / POC in upper portion.
    if (
        poc_position >= 0.60
        and upper_share
        > lower_share * 1.15
    ):
        return "P", poc

    # b-shape:
    # majority volume / POC in lower portion.
    if (
        poc_position <= 0.40
        and lower_share
        > upper_share * 1.15
    ):
        return "b", poc

    # D-shape:
    # relatively balanced distribution around middle.
    balance = abs(
        upper_share
        - lower_share
    )

    if (
        balance <= 0.12
        and middle_share >= 0.25
    ):
        return "D", poc

    return "UNKNOWN", poc


def add_volume_profile(df):
    shapes = []

    pocs = []

    for i in range(len(df)):
        shape, poc = profile_shape(
            df,
            i,
            lookback=24,
        )

        shapes.append(shape)

        pocs.append(poc)

    df = df.copy()

    df["vp_shape"] = shapes

    df["poc"] = pocs

    return df


# ============================================================
# SETUP 1
# P -> P -> D / P -> P -> P -> D
# ============================================================

def setup_volume_profile(df, i):
    if i < 3:
        return None

    current = df.iloc[i]

    last3 = list(
        df["vp_shape"]
        .iloc[i - 2:i + 1]
    )

    last4 = list(
        df["vp_shape"]
        .iloc[i - 3:i + 1]
    )

    pattern_3 = (
        last3
        == ["P", "P", "D"]
    )

    pattern_4 = (
        last4
        == ["P", "P", "P", "D"]
    )

    if not (
        pattern_3
        or pattern_4
    ):
        return None

    bearish_confirmation = (
        current["red"]
        and current["macd_hist"] < 0
        and current["close"]
        < current["ema20"]
    )

    if (
        bearish_confirmation
        and current["trend_1h"]
        in ["BEARISH", "NEUTRAL"]
    ):
        return "SHORT"

    return None


# ============================================================
# SETUP 2
# BB SQUEEZE + 2 GREEN / 2 RED
# ============================================================

def setup_bb_squeeze(df, i):
    if i < 25:
        return None

    current = df.iloc[i]

    previous = df.iloc[i - 1]

    recent_width = (
        df["bb_width_pct"]
        .iloc[i - 20:i]
        .dropna()
    )

    if len(recent_width) < 10:
        return None

    # Squeeze threshold based on recent market behavior.
    squeeze_threshold = float(
        recent_width.quantile(0.25)
    )

    # Look for squeeze just before breakout candles.
    squeeze_window = (
        df["bb_width_pct"]
        .iloc[max(0, i - 5):i - 1]
        .dropna()
    )

    if squeeze_window.empty:
        return None

    squeeze = (
        squeeze_window.min()
        <= squeeze_threshold
    )

    if not squeeze:
        return None

    # Approximate "parallel" BB:
    # middle band has not moved strongly.
    bb_middle_now = current["bb_middle"]

    bb_middle_old = df.iloc[
        max(0, i - 3)
    ]["bb_middle"]

    if (
        pd.isna(bb_middle_now)
        or pd.isna(bb_middle_old)
        or bb_middle_old == 0
    ):
        return None

    bb_middle_slope_pct = (
        (
            bb_middle_now
            - bb_middle_old
        )
        / bb_middle_old
        * 100
    )

    parallel = (
        abs(bb_middle_slope_pct)
        <= 0.40
    )

    if not parallel:
        return None

    two_green = (
        previous["green"]
        and current["green"]
    )

    two_red = (
        previous["red"]
        and current["red"]
    )

    long_breakout = (
        two_green
        and current["close"]
        > current["bb_middle"]
        and current["macd_hist"] > 0
    )

    short_breakdown = (
        two_red
        and current["close"]
        < current["bb_middle"]
        and current["macd_hist"] < 0
    )

    if (
        long_breakout
        and current["trend_1h"]
        in ["BULLISH", "NEUTRAL"]
    ):
        return "LONG"

    if (
        short_breakdown
        and current["trend_1h"]
        in ["BEARISH", "NEUTRAL"]
    ):
        return "SHORT"

    return None


# ============================================================
# SETUP 3
# BIG CANDLE + INJECTION / REJECTION WICK
# ============================================================

def setup_rejection(df, i):
    if i < 25:
        return None

    rejection = df.iloc[i - 1]

    confirm = df.iloc[i]

    recent_ranges = (
        df["range_pct"]
        .iloc[i - 20:i - 1]
        .dropna()
    )

    if recent_ranges.empty:
        return None

    median_range = float(
        recent_ranges.median()
    )

    big_candle = (
        rejection["range_pct"]
        >= median_range * 1.5
    )

    if not big_candle:
        return None

    body = max(
        float(rejection["body"]),
        1e-12,
    )

    upper_rejection = (
        rejection["upper_wick"]
        >= body * 1.5
    )

    lower_rejection = (
        rejection["lower_wick"]
        >= body * 1.5
    )

    # Upper rejection + next red candle.
    short_condition = (
        upper_rejection
        and confirm["red"]
        and confirm["close"]
        < rejection["close"]
        and confirm["trend_1h"]
        in ["BEARISH", "NEUTRAL"]
    )

    # Lower wick touches / moves around lower BB,
    # then next candle is green.
    touches_lower_bb = (
        pd.notna(
            rejection["bb_lower"]
        )
        and rejection["low"]
        <= rejection["bb_lower"] * 1.002
    )

    long_condition = (
        lower_rejection
        and touches_lower_bb
        and confirm["green"]
        and confirm["close"]
        > rejection["close"]
        and confirm["trend_1h"]
        in ["BULLISH", "NEUTRAL"]
    )

    if short_condition:
        return "SHORT"

    if long_condition:
        return "LONG"

    return None


# ============================================================
# SETUP 4
# WEAK HAMMER + LOWER-WICK SWEEP + RECOVERY BUY
# ============================================================

def setup_weak_hammer_sweep(df, i):
    if i < 2:
        return None

    hammer = df.iloc[i - 1]

    sweep = df.iloc[i]

    body = max(
        float(hammer["body"]),
        1e-12,
    )

    candle_range = max(
        float(hammer["range"]),
        1e-12,
    )

    # Long lower wick.
    long_lower_wick = (
        hammer["lower_wick"]
        >= body * 1.8
    )

    # Small / weak body compared with whole candle.
    weak_body = (
        hammer["body"]
        / candle_range
        <= 0.40
    )

    if not (
        long_lower_wick
        and weak_body
    ):
        return None

    # Second candle must go below the hammer low.
    sweeps_low = (
        sweep["low"]
        < hammer["low"]
    )

    # It then recovers from the sweep.
    recovery_level = (
        hammer["low"]
        + hammer["lower_wick"] * 0.50
    )

    recovery = (
        sweep["close"]
        > recovery_level
    )

    # Prefer bullish close, but the main concept
    # is the sweep and recovery.
    bullish_recovery = (
        sweep["close"]
        > sweep["open"]
    )

    if (
        sweeps_low
        and recovery
        and bullish_recovery
        and sweep["trend_1h"]
        in ["BULLISH", "NEUTRAL"]
    ):
        return "LONG"

    return None


# ============================================================
# FORWARD PERFORMANCE
# ============================================================

def directional_return(
    signal,
    entry,
    exit_price,
):
    if entry == 0:
        return np.nan

    if signal == "LONG":
        return (
            (exit_price - entry)
            / entry
            * 100
        )

    return (
        (entry - exit_price)
        / entry
        * 100
    )


def calculate_forward_results(
    df,
    i,
    signal,
):
    entry_price = float(
        df.iloc[i]["close"]
    )

    result = {}

    for label, bars in FORWARD_BARS.items():
        target = i + bars

        if target >= len(df):
            result[
                f"RETURN_{label}_%"
            ] = np.nan

            continue

        exit_price = float(
            df.iloc[target]["close"]
        )

        result[
            f"RETURN_{label}_%"
        ] = round(
            directional_return(
                signal,
                entry_price,
                exit_price,
            ),
            4,
        )

    end = min(
        i + FORWARD_BARS["4H"],
        len(df) - 1,
    )

    future = df.iloc[
        i + 1:end + 1
    ]

    if future.empty:
        result[
            "MAX_PROFIT_4H_%"
        ] = np.nan

        result[
            "MAX_LOSS_4H_%"
        ] = np.nan

        return result

    if signal == "LONG":
        max_profit = (
            (
                future["high"].max()
                - entry_price
            )
            / entry_price
            * 100
        )

        max_loss = (
            (
                future["low"].min()
                - entry_price
            )
            / entry_price
            * 100
        )

    else:
        max_profit = (
            (
                entry_price
                - future["low"].min()
            )
            / entry_price
            * 100
        )

        max_loss = (
            (
                entry_price
                - future["high"].max()
            )
            / entry_price
            * 100
        )

    result[
        "MAX_PROFIT_4H_%"
    ] = round(
        float(max_profit),
        4,
    )

    result[
        "MAX_LOSS_4H_%"
    ] = round(
        float(max_loss),
        4,
    )

    return result


# ============================================================
# SCAN HISTORICAL SETUPS
# ============================================================

def scan_history(df):
    events = []

    setup_functions = [
        (
            "VOLUME_PROFILE_PPD",
            setup_volume_profile,
        ),
        (
            "BB_SQUEEZE",
            setup_bb_squeeze,
        ),
        (
            "REJECTION_WICK",
            setup_rejection,
        ),
        (
            "WEAK_HAMMER_SWEEP",
            setup_weak_hammer_sweep,
        ),
    ]

    for i in range(
        60,
        len(df) - FORWARD_BARS["4H"] - 1,
    ):
        row = df.iloc[i]

        for setup_name, setup_function in setup_functions:
            try:
                signal = setup_function(
                    df,
                    i,
                )
            except Exception as exc:
                print(
                    "SETUP ERROR:",
                    setup_name,
                    i,
                    exc,
                )

                continue

            if signal is None:
                continue

            forward = calculate_forward_results(
                df,
                i,
                signal,
            )

            event = {
                "PAIR": PAIR,
                "TIME": row[
                    "datetime"
                ].isoformat(),
                "SETUP": setup_name,
                "SIGNAL": signal,
                "ENTRY_PRICE": round(
                    float(row["close"]),
                    8,
                ),
                "TREND_1H": row[
                    "trend_1h"
                ],
                "RSI": round(
                    float(row["rsi"]),
                    2,
                )
                if pd.notna(
                    row["rsi"]
                )
                else np.nan,
                "MACD_HIST": round(
                    float(
                        row["macd_hist"]
                    ),
                    8,
                )
                if pd.notna(
                    row["macd_hist"]
                )
                else np.nan,
                "BB_WIDTH_%": round(
                    float(
                        row["bb_width_pct"]
                    ),
                    4,
                )
                if pd.notna(
                    row["bb_width_pct"]
                )
                else np.nan,
                "VOLUME_RATIO": round(
                    float(
                        row["volume_ratio"]
                    ),
                    3,
                )
                if pd.notna(
                    row["volume_ratio"]
                )
                else np.nan,
                "VP_SHAPE": row[
                    "vp_shape"
                ],
                "POC": round(
                    float(row["poc"]),
                    8,
                )
                if pd.notna(
                    row["poc"]
                )
                else np.nan,
            }

            event.update(
                forward
            )

            events.append(
                event
            )

    return pd.DataFrame(
        events
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):
    print()
    print("=" * 70)
    print("HISTORICAL TEST SUMMARY")
    print("=" * 70)

    if results.empty:
        print(
            "No historical setups found."
        )

        return

    print(
        "Total signals:",
        len(results),
    )

    print()

    for (
        setup,
        signal,
    ), group in results.groupby(
        [
            "SETUP",
            "SIGNAL",
        ]
    ):
        returns = (
            group["RETURN_4H_%"]
            .dropna()
        )

        if returns.empty:
            continue

        wins = int(
            (
                returns
                >= WIN_THRESHOLD_PCT
            ).sum()
        )

        losses = int(
            (
                returns
                <= -WIN_THRESHOLD_PCT
            ).sum()
        )

        neutral = (
            len(returns)
            - wins
            - losses
        )

        win_rate = (
            wins
            / len(returns)
            * 100
        )

        avg_return = float(
            returns.mean()
        )

        print("-" * 70)

        print(
            "SETUP:",
            setup,
        )

        print(
            "SIDE:",
            signal,
        )

        print(
            "SIGNALS:",
            len(returns),
        )

        print(
            "WINS:",
            wins,
        )

        print(
            "LOSSES:",
            losses,
        )

        print(
            "NEUTRAL:",
            neutral,
        )

        print(
            "WIN RATE:",
            round(
                win_rate,
                2,
            ),
            "%",
        )

        print(
            "AVG 4H RETURN:",
            round(
                avg_return,
                4,
            ),
            "%",
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print(
        "COINDCX FUTURES HISTORICAL LEARNING V1"
    )
    print("=" * 70)

    print(
        "PAIR:",
        PAIR,
    )

    print(
        "HISTORY:",
        DAYS_TO_TEST,
        "days",
    )

    print()
    print(
        "Downloading native 5-minute Futures candles..."
    )

    df_5m = fetch_history(
        PAIR,
        "5",
        DAYS_TO_TEST,
        CHUNK_DAYS_5M,
    )

    if df_5m.empty:
        raise RuntimeError(
            "No 5-minute Futures data received."
        )

    print()
    print(
        "Total 5M candles:",
        len(df_5m),
    )

    print()
    print(
        "Downloading 1-hour Futures candles..."
    )

    df_1h = fetch_history(
        PAIR,
        "60",
        DAYS_TO_TEST,
        CHUNK_DAYS_1H,
    )

    if df_1h.empty:
        raise RuntimeError(
            "No 1-hour Futures data received."
        )

    print()
    print(
        "Total 1H candles:",
        len(df_1h),
    )

    print()
    print(
        "Calculating 5M indicators..."
    )

    df_5m = calculate_indicators(
        df_5m
    )

    print(
        "Calculating Volume Profile..."
    )

    df_5m = add_volume_profile(
        df_5m
    )

    print(
        "Attaching 1H market direction..."
    )

    df_5m = attach_1h_trend(
        df_5m,
        df_1h,
    )

    df_5m = df_5m.dropna(
        subset=[
            "ema20",
            "ema50",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "rsi",
            "macd_hist",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        "Scanning historical setups..."
    )

    results = scan_history(
        df_5m
    )

    if results.empty:
        print(
            "No setup events detected."
        )

        # Still create CSV so GitHub workflow
        # can confirm script completed.
        results.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print(
            "CSV saved:",
            OUTPUT_FILE,
        )

        return

    results = results.sort_values(
        "TIME"
    ).reset_index(
        drop=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "CSV saved:",
        OUTPUT_FILE,
    )

    print_summary(
        results
    )

    print()
    print("=" * 70)
    print(
        "HISTORICAL LEARNING V1 COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
