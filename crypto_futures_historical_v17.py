# ============================================================
# crypto_futures_historical_v17.py
#
# COINDCX FUTURES HISTORICAL V17
#
# REGIME + VOLATILITY + ATR EXPECTANCY ENGINE
#
# Historical research only.
# NO Telegram.
# NO real orders.
# ============================================================

import time
import math
import itertools
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


# ============================================================
# BASIC SETTINGS
# ============================================================

PAIR = "B-BTC_USDT"

FUTURES_CANDLES_URL = (
    "https://public.coindcx.com/market_data/candlesticks"
)

HISTORY_ATTEMPTS = [
    365,
    270,
    180,
]

ENTRY_RESOLUTION = "5"
TREND_RESOLUTION = "60"

CHUNK_DAYS_5M = 5
CHUNK_DAYS_1H = 30


# ============================================================
# DATA SPLIT
# ============================================================

DISCOVERY_RATIO = 0.60
SELECTION_RATIO = 0.20
HOLDOUT_RATIO = 0.20

FINAL_BLOCKS = 4


# ============================================================
# COST / TRADE SETTINGS
# ============================================================

ROUND_TRIP_COST_PCT = 0.10

MAX_HOLD_BARS = 48

SAME_BAR_POLICY = "SL_FIRST"


# ============================================================
# ATR TARGET GRID
# ============================================================
#
# Small grid deliberately.
# We do NOT test hundreds of combinations.
# ============================================================

ATR_TP_MULTIPLIERS = [
    1.0,
    1.25,
    1.50,
    1.75,
    2.0,
]

ATR_SL_MULTIPLIERS = [
    0.75,
    1.0,
    1.25,
    1.50,
]


# ============================================================
# MINIMUM SAMPLE
# ============================================================

MIN_DISCOVERY_TRADES = 35
MIN_SELECTION_TRADES = 18
MIN_HOLDOUT_TRADES = 15


# ============================================================
# SELECTION REQUIREMENTS
# ============================================================

SELECTION_MIN_NET_AVG = 0.0
SELECTION_MIN_PF = 1.05
SELECTION_MIN_TRADES = 18


# ============================================================
# FINAL PASS
# ============================================================

PASS_MIN_TRADES = 20
PASS_MIN_NET_AVG = 0.0
PASS_MIN_PF = 1.15

PASS_MIN_POSITIVE_BLOCKS = 3
PASS_MIN_VALID_BLOCKS = 3

PASS_MAX_DRAWDOWN_PCT = 6.0


# ============================================================
# FINAL WATCH
# ============================================================

WATCH_MIN_TRADES = 15
WATCH_MIN_NET_AVG = 0.0
WATCH_MIN_PF = 1.00

WATCH_MIN_POSITIVE_BLOCKS = 2
WATCH_MIN_VALID_BLOCKS = 2


# ============================================================
# OUTPUT FILES
# ============================================================

EVENT_FILE = (
    "crypto_futures_v17_events.csv"
)

DISCOVERY_FILE = (
    "crypto_futures_v17_discovery.csv"
)

SELECTION_FILE = (
    "crypto_futures_v17_selection.csv"
)

FINAL_BLOCK_FILE = (
    "crypto_futures_v17_final_blocks.csv"
)

SUMMARY_FILE = (
    "crypto_futures_v17_summary.csv"
)


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value, default=np.nan):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


# ============================================================
# COINDCX CANDLE PARSER
# ============================================================

def parse_candles(payload):

    if payload is None:
        return pd.DataFrame()

    if isinstance(payload, dict):

        if "data" in payload:
            payload = payload["data"]

        elif "candles" in payload:
            payload = payload["candles"]

        elif "result" in payload:
            payload = payload["result"]

    if not isinstance(payload, list):
        return pd.DataFrame()

    rows = []

    for candle in payload:

        try:

            # ------------------------------------------------
            # Dictionary response
            # ------------------------------------------------

            if isinstance(candle, dict):

                timestamp = (
                    candle.get("time")
                    or
                    candle.get("timestamp")
                    or
                    candle.get("t")
                    or
                    candle.get("startTime")
                )

                open_price = (
                    candle.get("open")
                    if "open" in candle
                    else candle.get("o")
                )

                high = (
                    candle.get("high")
                    if "high" in candle
                    else candle.get("h")
                )

                low = (
                    candle.get("low")
                    if "low" in candle
                    else candle.get("l")
                )

                close = (
                    candle.get("close")
                    if "close" in candle
                    else candle.get("c")
                )

                volume = (
                    candle.get("volume")
                    if "volume" in candle
                    else candle.get("v")
                )

            # ------------------------------------------------
            # Array response
            # ------------------------------------------------

            elif isinstance(
                candle,
                (list, tuple)
            ):

                if len(candle) < 5:
                    continue

                timestamp = candle[0]
                open_price = candle[1]
                high = candle[2]
                low = candle[3]
                close = candle[4]

                if len(candle) > 5:
                    volume = candle[5]
                else:
                    volume = np.nan

            else:
                continue

            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Timestamp conversion
    # --------------------------------------------------------

    numeric_time = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    median_time = numeric_time.median()

    if pd.notna(median_time):

        if median_time > 1e12:

            df["datetime"] = pd.to_datetime(
                numeric_time,
                unit="ms",
                utc=True,
                errors="coerce",
            )

        else:

            df["datetime"] = pd.to_datetime(
                numeric_time,
                unit="s",
                utc=True,
                errors="coerce",
            )

    else:

        df["datetime"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce",
        )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
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

    return df[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


# ============================================================
# DOWNLOAD ONE CHUNK
# ============================================================

def fetch_chunk(
    pair,
    resolution,
    start_time,
    end_time,
):

    start_ms = int(
        start_time.timestamp()
        *
        1000
    )

    end_ms = int(
        end_time.timestamp()
        *
        1000
    )

    params = {
        "pair": pair,
        "from": start_ms,
        "to": end_ms,
        "resolution": resolution,
        "pcode": "f",
    }

    response = requests.get(
        FUTURES_CANDLES_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return parse_candles(
        response.json()
    )


# ============================================================
# DOWNLOAD FULL HISTORY
# ============================================================

def fetch_history(
    pair,
    resolution,
    days,
    chunk_days,
):

    end_time = datetime.now(
        timezone.utc
    )

    start_time = (
        end_time
        -
        timedelta(days=days)
    )

    frames = []

    cursor = start_time

    chunk_number = 0

    while cursor < end_time:

        chunk_number += 1

        chunk_end = min(
            cursor
            +
            timedelta(
                days=chunk_days
            ),
            end_time,
        )

        print(
            f"Chunk {chunk_number}: "
            f"{cursor} -> {chunk_end} "
            f"resolution={resolution}"
        )

        success = False

        for attempt in range(1, 4):

            try:

                chunk = fetch_chunk(
                    pair,
                    resolution,
                    cursor,
                    chunk_end,
                )

                print(
                    "Candles received:",
                    len(chunk),
                )

                if not chunk.empty:

                    frames.append(
                        chunk
                    )

                success = True
                break

            except Exception as exc:

                print(
                    f"Attempt {attempt} failed:",
                    str(exc),
                )

                time.sleep(
                    attempt * 2
                )

        if not success:

            raise RuntimeError(
                "CoinDCX history chunk failed: "
                f"{cursor} -> {chunk_end}"
            )

        cursor = chunk_end

        time.sleep(0.05)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = (
        result
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"]
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# HISTORY FALLBACK
# ============================================================

def download_history():

    last_error = None

    for days in HISTORY_ATTEMPTS:

        try:

            print()
            print("=" * 100)

            print(
                "TRYING HISTORY:",
                days,
                "DAYS"
            )

            print("=" * 100)

            print()
            print(
                "Downloading native 5-minute data..."
            )

            df_5m = fetch_history(
                PAIR,
                ENTRY_RESOLUTION,
                days,
                CHUNK_DAYS_5M,
            )

            if df_5m.empty:

                raise RuntimeError(
                    "5-minute data empty."
                )

            print()
            print(
                "Downloading native 1-hour data..."
            )

            df_1h = fetch_history(
                PAIR,
                TREND_RESOLUTION,
                days,
                CHUNK_DAYS_1H,
            )

            if df_1h.empty:

                raise RuntimeError(
                    "1-hour data empty."
                )

            print()
            print(
                "HISTORY DOWNLOAD SUCCESS"
            )

            print(
                "5M CANDLES:",
                len(df_5m),
            )

            print(
                "1H CANDLES:",
                len(df_1h),
            )

            return (
                df_5m,
                df_1h,
                days,
            )

        except Exception as exc:

            last_error = exc

            print()
            print(
                "History attempt failed:",
                days
            )

            print(
                "ERROR:",
                str(exc)
            )

    raise RuntimeError(
        "All history attempts failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# EMA
# ============================================================

def ema(series, length):

    return (
        series
        .ewm(
            span=length,
            adjust=False,
        )
        .mean()
    )


# ============================================================
# RSI
# ============================================================

def rsi(series, length=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = gain.ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan,
        )
    )

    result = (
        100.0
        -
        (
            100.0
            /
            (
                1.0
                +
                rs
            )
        )
    )

    return result


# ============================================================
# ATR
# ============================================================

def atr(df, length=14):

    previous_close = (
        df["close"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            (
                df["high"]
                -
                df["low"]
            ),

            (
                df["high"]
                -
                previous_close
            ).abs(),

            (
                df["low"]
                -
                previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean()


# ============================================================
# ADX
# ============================================================

def adx(df, length=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()

    down_move = (
        -low.diff()
    )

    plus_dm = pd.Series(
        np.where(
            (
                up_move > down_move
            )
            &
            (
                up_move > 0
            ),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (
                down_move > up_move
            )
            &
            (
                down_move > 0
            ),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,

            (
                high
                -
                previous_close
            ).abs(),

            (
                low
                -
                previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_smoothed = tr.ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean()

    plus_di = (
        100.0
        *
        plus_dm.ewm(
            alpha=1.0 / length,
            adjust=False,
        ).mean()
        /
        atr_smoothed.replace(
            0,
            np.nan,
        )
    )

    minus_di = (
        100.0
        *
        minus_dm.ewm(
            alpha=1.0 / length,
            adjust=False,
        ).mean()
        /
        atr_smoothed.replace(
            0,
            np.nan,
        )
    )

    dx = (
        100.0
        *
        (
            plus_di
            -
            minus_di
        ).abs()
        /
        (
            plus_di
            +
            minus_di
        ).replace(
            0,
            np.nan,
        )
    )

    adx_value = dx.ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean()

    return (
        adx_value,
        plus_di,
        minus_di,
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["ema20"] = ema(
        df["close"],
        20,
    )

    df["ema50"] = ema(
        df["close"],
        50,
    )

    df["ema200"] = ema(
        df["close"],
        200,
    )

    df["rsi"] = rsi(
        df["close"],
        14,
    )

    macd_fast = ema(
        df["close"],
        12,
    )

    macd_slow = ema(
        df["close"],
        26,
    )

    df["macd"] = (
        macd_fast
        -
        macd_slow
    )

    df["macd_signal"] = ema(
        df["macd"],
        9,
    )

    df["macd_hist"] = (
        df["macd"]
        -
        df["macd_signal"]
    )

    df["atr"] = atr(
        df,
        14,
    )

    df["atr_pct"] = (
        df["atr"]
        /
        df["close"]
        *
        100.0
    )

    (
        df["adx"],
        df["plus_di"],
        df["minus_di"],
    ) = adx(
        df,
        14,
    )

    # --------------------------------------------------------
    # Bollinger
    # --------------------------------------------------------

    df["bb_mid"] = (
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
        df["bb_mid"]
        +
        2.0 * bb_std
    )

    df["bb_lower"] = (
        df["bb_mid"]
        -
        2.0 * bb_std
    )

    df["bb_width_pct"] = (
        (
            df["bb_upper"]
            -
            df["bb_lower"]
        )
        /
        df["bb_mid"].replace(
            0,
            np.nan,
        )
        *
        100.0
    )

    df["bb_position"] = (
        (
            df["close"]
            -
            df["bb_lower"]
        )
        /
        (
            df["bb_upper"]
            -
            df["bb_lower"]
        ).replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["volume_ma20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        /
        df["volume_ma20"].replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Candle anatomy
    # --------------------------------------------------------

    candle_range = (
        df["high"]
        -
        df["low"]
    )

    body = (
        df["close"]
        -
        df["open"]
    ).abs()

    df["range_pct"] = (
        candle_range
        /
        df["close"]
        *
        100.0
    )

    df["body_ratio"] = (
        body
        /
        candle_range.replace(
            0,
            np.nan,
        )
    )

    df["close_position"] = (
        (
            df["close"]
            -
            df["low"]
        )
        /
        candle_range.replace(
            0,
            np.nan,
        )
    )

    df["lower_wick_ratio"] = (
        (
            np.minimum(
                df["open"],
                df["close"],
            )
            -
            df["low"]
        )
        /
        candle_range.replace(
            0,
            np.nan,
        )
    )

    df["upper_wick_ratio"] = (
        (
            df["high"]
            -
            np.maximum(
                df["open"],
                df["close"],
            )
        )
        /
        candle_range.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Slopes / momentum changes
    # --------------------------------------------------------

    df["ema20_slope_3"] = (
        df["ema20"]
        .pct_change(3)
        *
        100.0
    )

    df["ema50_slope_3"] = (
        df["ema50"]
        .pct_change(3)
        *
        100.0
    )

    df["rsi_change_3"] = (
        df["rsi"]
        -
        df["rsi"].shift(3)
    )

    df["macd_change_3"] = (
        df["macd_hist"]
        -
        df["macd_hist"].shift(3)
    )

    df["ema_distance_pct"] = (
        (
            df["ema20"]
            -
            df["ema50"]
        )
        /
        df["close"]
        *
        100.0
    )

    return df


# ============================================================
# PREPARE 1H
# ============================================================

def prepare_1h(df):

    df = add_indicators(
        df
    )

    df = df[
        [
            "datetime",
            "close",
            "ema20",
            "ema50",
            "ema200",
            "rsi",
            "macd_hist",
            "adx",
            "atr_pct",
        ]
    ].copy()

    df = df.rename(
        columns={
            "close":
                "close_1h",

            "ema20":
                "ema20_1h",

            "ema50":
                "ema50_1h",

            "ema200":
                "ema200_1h",

            "rsi":
                "rsi_1h",

            "macd_hist":
                "macd_hist_1h",

            "adx":
                "adx_1h",

            "atr_pct":
                "atr_pct_1h",
        }
    )

    return df


# ============================================================
# PREPARE FULL DATA
# ============================================================

def prepare_data(
    df_5m,
    df_1h,
):

    print()
    print(
        "Calculating V17 indicators..."
    )

    df_5m = add_indicators(
        df_5m
    )

    df_1h = prepare_1h(
        df_1h
    )

    df_5m = (
        df_5m
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    df_1h = (
        df_1h
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # ========================================================
    # IMPORTANT:
    # Shift 1H values by one completed 1H candle
    # to avoid using unfinished higher-timeframe information.
    # ========================================================

    h1_columns = [
        column
        for column in df_1h.columns
        if column != "datetime"
    ]

    df_1h[
        h1_columns
    ] = (
        df_1h[
            h1_columns
        ]
        .shift(1)
    )

    data = pd.merge_asof(
        df_5m,
        df_1h,
        on="datetime",
        direction="backward",
    )

    # --------------------------------------------------------
    # H1 EMA distance
    # --------------------------------------------------------

    data["h1_ema_distance_pct"] = (
        (
            data["ema20_1h"]
            -
            data["ema50_1h"]
        )
        /
        data["close_1h"]
        *
        100.0
    )

    # --------------------------------------------------------
    # ATR percentile
    #
    # Rolling percentile-like rank using trailing 20 days.
    # 20 days x 288 5m bars = 5760 bars.
    # --------------------------------------------------------

    atr_window = 5760

    data["atr_pct_median"] = (
        data["atr_pct"]
        .rolling(
            atr_window,
            min_periods=500,
        )
        .median()
    )

    data["atr_pct_q75"] = (
        data["atr_pct"]
        .rolling(
            atr_window,
            min_periods=500,
        )
        .quantile(0.75)
    )

    # --------------------------------------------------------
    # Volatility regime
    # --------------------------------------------------------

    data["VOL_REGIME"] = np.select(
        [
            (
                data["atr_pct"]
                <
                data["atr_pct_median"]
            ),

            (
                data["atr_pct"]
                >=
                data["atr_pct_q75"]
            ),
        ],
        [
            "LOW_VOL",
            "HIGH_VOL",
        ],
        default="NORMAL_VOL",
    )

    # --------------------------------------------------------
    # Trend regime
    # --------------------------------------------------------

    uptrend = (
        (
            data["ema20"]
            >
            data["ema50"]
        )
        &
        (
            data["ema50"]
            >
            data["ema200"]
        )
        &
        (
            data["ema20_1h"]
            >
            data["ema50_1h"]
        )
        &
        (
            data["close_1h"]
            >
            data["ema200_1h"]
        )
    )

    downtrend = (
        (
            data["ema20"]
            <
            data["ema50"]
        )
        &
        (
            data["ema50"]
            <
            data["ema200"]
        )
        &
        (
            data["ema20_1h"]
            <
            data["ema50_1h"]
        )
        &
        (
            data["close_1h"]
            <
            data["ema200_1h"]
        )
    )

    data["TREND_REGIME"] = np.select(
        [
            uptrend,
            downtrend,
        ],
        [
            "UPTREND",
            "DOWNTREND",
        ],
        default="RANGE",
    )

    # --------------------------------------------------------
    # Trend strength
    # --------------------------------------------------------

    data["TREND_STRENGTH"] = np.select(
        [
            (
                data["adx"] >= 25
            )
            &
            (
                data["adx_1h"] >= 20
            ),

            (
                data["adx"] >= 18
            ),
        ],
        [
            "STRONG",
            "MEDIUM",
        ],
        default="WEAK",
    )

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "macd_hist",
        "atr",
        "atr_pct",
        "adx",
        "plus_di",
        "minus_di",
        "bb_width_pct",
        "bb_position",
        "volume_ratio",
        "range_pct",
        "body_ratio",
        "close_position",
        "lower_wick_ratio",
        "upper_wick_ratio",
        "ema20_slope_3",
        "rsi_change_3",
        "macd_change_3",
        "ema_distance_pct",
        "close_1h",
        "ema20_1h",
        "ema50_1h",
        "ema200_1h",
        "rsi_1h",
        "macd_hist_1h",
        "adx_1h",
        "h1_ema_distance_pct",
        "TREND_REGIME",
        "VOL_REGIME",
        "TREND_STRENGTH",
    ]

    data = (
        data
        .dropna(
            subset=required
        )
        .reset_index(drop=True)
    )

    print(
        "USABLE 5M CANDLES:",
        len(data),
    )

    return data


# ============================================================
# BASE LONG SETUP
# ============================================================

def long_setup(df, i):

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    if (
        row["TREND_REGIME"]
        !=
        "UPTREND"
    ):
        return False

    # Pullback/rejection inside uptrend

    pullback = (
        row["low"]
        <=
        row["ema20"] * 1.003
    )

    reclaim = (
        row["close"]
        >=
        row["ema20"]
    )

    momentum = (
        row["macd_hist"]
        >
        prev["macd_hist"]
    )

    rsi_ok = (
        40
        <=
        row["rsi"]
        <=
        70
    )

    candle_ok = (
        row["close_position"]
        >=
        0.55
    )

    trend_di = (
        row["plus_di"]
        >
        row["minus_di"]
    )

    return bool(
        pullback
        and
        reclaim
        and
        momentum
        and
        rsi_ok
        and
        candle_ok
        and
        trend_di
    )


# ============================================================
# BASE SHORT SETUP
# ============================================================

def short_setup(df, i):

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    if (
        row["TREND_REGIME"]
        !=
        "DOWNTREND"
    ):
        return False

    pullback = (
        row["high"]
        >=
        row["ema20"] * 0.997
    )

    rejection = (
        row["close"]
        <=
        row["ema20"]
    )

    momentum = (
        row["macd_hist"]
        <
        prev["macd_hist"]
    )

    rsi_ok = (
        30
        <=
        row["rsi"]
        <=
        60
    )

    candle_ok = (
        row["close_position"]
        <=
        0.45
    )

    trend_di = (
        row["minus_di"]
        >
        row["plus_di"]
    )

    return bool(
        pullback
        and
        rejection
        and
        momentum
        and
        rsi_ok
        and
        candle_ok
        and
        trend_di
    )


# ============================================================
# ATR TRADE SIMULATOR
# ============================================================

def simulate_atr_trade(
    df,
    index,
    side,
    tp_atr,
    sl_atr,
):

    row = df.iloc[index]

    entry = safe_float(
        row["close"]
    )

    atr_value = safe_float(
        row["atr"]
    )

    if (
        pd.isna(entry)
        or
        pd.isna(atr_value)
        or
        entry <= 0
        or
        atr_value <= 0
    ):
        return None

    tp_distance = (
        atr_value
        *
        tp_atr
    )

    sl_distance = (
        atr_value
        *
        sl_atr
    )

    if side == "LONG":

        tp_price = (
            entry
            +
            tp_distance
        )

        sl_price = (
            entry
            -
            sl_distance
        )

    else:

        tp_price = (
            entry
            -
            tp_distance
        )

        sl_price = (
            entry
            +
            sl_distance
        )

    end_index = min(
        index + MAX_HOLD_BARS,
        len(df) - 1,
    )

    if end_index <= index:
        return None

    future = (
        df
        .iloc[
            index + 1:
            end_index + 1
        ]
        .copy()
    )

    if future.empty:
        return None

    outcome = "TIME_EXIT"

    exit_price = safe_float(
        future.iloc[-1]["close"]
    )

    exit_bars = len(future)

    for offset, (_, bar) in enumerate(
        future.iterrows(),
        start=1,
    ):

        high = safe_float(
            bar["high"]
        )

        low = safe_float(
            bar["low"]
        )

        if side == "LONG":

            hit_tp = (
                high >= tp_price
            )

            hit_sl = (
                low <= sl_price
            )

        else:

            hit_tp = (
                low <= tp_price
            )

            hit_sl = (
                high >= sl_price
            )

        if hit_tp and hit_sl:

            if SAME_BAR_POLICY == "SL_FIRST":

                outcome = "LOSS"
                exit_price = sl_price

            else:

                outcome = "WIN"
                exit_price = tp_price

            exit_bars = offset
            break

        if hit_tp:

            outcome = "WIN"
            exit_price = tp_price
            exit_bars = offset
            break

        if hit_sl:

            outcome = "LOSS"
            exit_price = sl_price
            exit_bars = offset
            break

    observed = (
        future
        .iloc[:exit_bars]
        .copy()
    )

    max_high = safe_float(
        observed["high"].max()
    )

    min_low = safe_float(
        observed["low"].min()
    )

    if side == "LONG":

        raw_return = (
            (
                exit_price / entry
            )
            -
            1.0
        ) * 100.0

        mfe = (
            (
                max_high / entry
            )
            -
            1.0
        ) * 100.0

        mae = (
            (
                min_low / entry
            )
            -
            1.0
        ) * 100.0

    else:

        raw_return = (
            (
                entry / exit_price
            )
            -
            1.0
        ) * 100.0

        mfe = (
            (
                entry / min_low
            )
            -
            1.0
        ) * 100.0

        mae = -(
            (
                max_high / entry
            )
            -
            1.0
        ) * 100.0

    net_return = (
        raw_return
        -
        ROUND_TRIP_COST_PCT
    )

    return {
        "OUTCOME": outcome,

        "EXIT_BARS":
            int(exit_bars),

        "RAW_RETURN_%":
            round(
                raw_return,
                4,
            ),

        "NET_RETURN_%":
            round(
                net_return,
                4,
            ),

        "MFE_%":
            round(
                mfe,
                4,
            ),

        "MAE_%":
            round(
                mae,
                4,
            ),
    }


# ============================================================
# COLLECT SIGNAL EVENTS
# ============================================================

def collect_events(df):

    events = []

    last_index = (
        len(df)
        -
        MAX_HOLD_BARS
        -
        1
    )

    for i in range(
        200,
        last_index,
    ):

        row = df.iloc[i]

        side = None

        if long_setup(
            df,
            i,
        ):
            side = "LONG"

        elif short_setup(
            df,
            i,
        ):
            side = "SHORT"

        if side is None:
            continue

        events.append(
            {
                "DATA_INDEX": i,

                "TIME":
                    row["datetime"],

                "SIDE":
                    side,

                "TREND_REGIME":
                    row[
                        "TREND_REGIME"
                    ],

                "VOL_REGIME":
                    row[
                        "VOL_REGIME"
                    ],

                "TREND_STRENGTH":
                    row[
                        "TREND_STRENGTH"
                    ],

                "ENTRY_PRICE":
                    safe_float(
                        row["close"]
                    ),

                "ATR":
                    safe_float(
                        row["atr"]
                    ),

                "ATR_PCT":
                    safe_float(
                        row["atr_pct"]
                    ),

                "ADX_5M":
                    safe_float(
                        row["adx"]
                    ),

                "ADX_1H":
                    safe_float(
                        row["adx_1h"]
                    ),

                "RSI_5M":
                    safe_float(
                        row["rsi"]
                    ),

                "RSI_1H":
                    safe_float(
                        row["rsi_1h"]
                    ),

                "MACD_HIST_5M":
                    safe_float(
                        row["macd_hist"]
                    ),

                "MACD_HIST_1H":
                    safe_float(
                        row["macd_hist_1h"]
                    ),

                "MACD_CHANGE_3":
                    safe_float(
                        row["macd_change_3"]
                    ),

                "RSI_CHANGE_3":
                    safe_float(
                        row["rsi_change_3"]
                    ),

                "EMA_DISTANCE_PCT":
                    safe_float(
                        row["ema_distance_pct"]
                    ),

                "H1_EMA_DISTANCE_PCT":
                    safe_float(
                        row[
                            "h1_ema_distance_pct"
                        ]
                    ),

                "EMA20_SLOPE_3":
                    safe_float(
                        row["ema20_slope_3"]
                    ),

                "BB_WIDTH_PCT":
                    safe_float(
                        row["bb_width_pct"]
                    ),

                "BB_POSITION":
                    safe_float(
                        row["bb_position"]
                    ),

                "VOLUME_RATIO":
                    safe_float(
                        row["volume_ratio"]
                    ),

                "RANGE_PCT":
                    safe_float(
                        row["range_pct"]
                    ),

                "BODY_RATIO":
                    safe_float(
                        row["body_ratio"]
                    ),

                "CLOSE_POSITION":
                    safe_float(
                        row["close_position"]
                    ),

                "LOWER_WICK_RATIO":
                    safe_float(
                        row["lower_wick_ratio"]
                    ),

                "UPPER_WICK_RATIO":
                    safe_float(
                        row["upper_wick_ratio"]
                    ),
            }
        )

    events = pd.DataFrame(
        events
    )

    if not events.empty:

        events["TIME"] = pd.to_datetime(
            events["TIME"],
            utc=True,
            errors="coerce",
        )

        events = (
            events
            .sort_values("TIME")
            .reset_index(drop=True)
        )

    return events


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_events(events):

    events = (
        events
        .sort_values("TIME")
        .reset_index(drop=True)
        .copy()
    )

    total = len(events)

    discovery_end = int(
        total
        *
        DISCOVERY_RATIO
    )

    selection_end = int(
        total
        *
        (
            DISCOVERY_RATIO
            +
            SELECTION_RATIO
        )
    )

    discovery = (
        events
        .iloc[:discovery_end]
        .copy()
        .reset_index(drop=True)
    )

    selection = (
        events
        .iloc[
            discovery_end:
            selection_end
        ]
        .copy()
        .reset_index(drop=True)
    )

    holdout = (
        events
        .iloc[
            selection_end:
        ]
        .copy()
        .reset_index(drop=True)
    )

    discovery["DATASET"] = (
        "DISCOVERY"
    )

    selection["DATASET"] = (
        "SELECTION"
    )

    holdout["DATASET"] = (
        "FINAL_HOLDOUT"
    )

    return (
        discovery,
        selection,
        holdout,
    )


# ============================================================
# CANDIDATE DEFINITIONS
# ============================================================

def build_candidates():

    candidates = []

    for side in [
        "LONG",
        "SHORT",
    ]:

        candidates.extend(
            [
                {
                    "NAME":
                        f"V17_{side}_BASE",

                    "SIDE":
                        side,

                    "VOL":
                        None,

                    "STRENGTH":
                        None,
                },

                {
                    "NAME":
                        f"V17_{side}_LOW_VOL",

                    "SIDE":
                        side,

                    "VOL":
                        "LOW_VOL",

                    "STRENGTH":
                        None,
                },

                {
                    "NAME":
                        f"V17_{side}_NORMAL_VOL",

                    "SIDE":
                        side,

                    "VOL":
                        "NORMAL_VOL",

                    "STRENGTH":
                        None,
                },

                {
                    "NAME":
                        f"V17_{side}_HIGH_VOL",

                    "SIDE":
                        side,

                    "VOL":
                        "HIGH_VOL",

                    "STRENGTH":
                        None,
                },

                {
                    "NAME":
                        f"V17_{side}_MEDIUM_TREND",

                    "SIDE":
                        side,

                    "VOL":
                        None,

                    "STRENGTH":
                        "MEDIUM",
                },

                {
                    "NAME":
                        f"V17_{side}_STRONG_TREND",

                    "SIDE":
                        side,

                    "VOL":
                        None,

                    "STRENGTH":
                        "STRONG",
                },

                {
                    "NAME":
                        f"V17_{side}_NORMAL_STRONG",

                    "SIDE":
                        side,

                    "VOL":
                        "NORMAL_VOL",

                    "STRENGTH":
                        "STRONG",
                },

                {
                    "NAME":
                        f"V17_{side}_HIGH_STRONG",

                    "SIDE":
                        side,

                    "VOL":
                        "HIGH_VOL",

                    "STRENGTH":
                        "STRONG",
                },
            ]
        )

    return candidates


# ============================================================
# FILTER CANDIDATE
# ============================================================

def filter_candidate(
    events,
    candidate,
):

    filtered = (
        events[
            events["SIDE"]
            ==
            candidate["SIDE"]
        ]
        .copy()
    )

    if candidate[
        "VOL"
    ] is not None:

        filtered = (
            filtered[
                filtered["VOL_REGIME"]
                ==
                candidate["VOL"]
            ]
            .copy()
        )

    if candidate[
        "STRENGTH"
    ] is not None:

        filtered = (
            filtered[
                filtered[
                    "TREND_STRENGTH"
                ]
                ==
                candidate[
                    "STRENGTH"
                ]
            ]
            .copy()
        )

    return filtered


# ============================================================
# RESIMULATE
# ============================================================

def resimulate(
    market,
    events,
    side,
    tp_atr,
    sl_atr,
):

    rows = []

    for _, event in events.iterrows():

        index = int(
            event[
                "DATA_INDEX"
            ]
        )

        result = simulate_atr_trade(
            market,
            index,
            side,
            tp_atr,
            sl_atr,
        )

        if result is None:
            continue

        row = event.to_dict()

        row.update(
            result
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAX DRAWDOWN
# ============================================================

def max_drawdown(
    returns,
):

    if len(returns) == 0:
        return np.nan

    equity = (
        pd.Series(returns)
        .fillna(0.0)
        .cumsum()
    )

    peak = (
        equity
        .cummax()
    )

    drawdown = (
        equity
        -
        peak
    )

    return abs(
        float(
            drawdown.min()
        )
    )


# ============================================================
# PERFORMANCE
# ============================================================

def performance(df):

    if df.empty:
        return None

    returns = (
        pd.to_numeric(
            df["NET_RETURN_%"],
            errors="coerce",
        )
        .dropna()
    )

    if returns.empty:
        return None

    wins = int(
        (
            df["OUTCOME"]
            ==
            "WIN"
        ).sum()
    )

    losses = int(
        (
            df["OUTCOME"]
            ==
            "LOSS"
        ).sum()
    )

    time_exits = int(
        (
            df["OUTCOME"]
            ==
            "TIME_EXIT"
        ).sum()
    )

    decisive = (
        wins
        +
        losses
    )

    if decisive > 0:

        win_rate = (
            wins
            /
            decisive
            *
            100.0
        )

    else:
        win_rate = np.nan

    positive = (
        returns[
            returns > 0
        ]
    )

    negative = (
        returns[
            returns < 0
        ]
    )

    gross_profit = float(
        positive.sum()
    )

    gross_loss = abs(
        float(
            negative.sum()
        )
    )

    if gross_loss > 0:

        pf = (
            gross_profit
            /
            gross_loss
        )

    elif gross_profit > 0:

        pf = 999.0

    else:

        pf = 0.0

    average_win = (
        positive.mean()
        if not positive.empty
        else 0.0
    )

    average_loss = (
        abs(
            negative.mean()
        )
        if not negative.empty
        else 0.0
    )

    return {
        "TRADES":
            len(df),

        "DECISIVE":
            decisive,

        "WINS":
            wins,

        "LOSSES":
            losses,

        "TIME_EXITS":
            time_exits,

        "WIN_RATE_%":
            round(
                win_rate,
                2,
            )
            if pd.notna(
                win_rate
            )
            else np.nan,

        "NET_AVG_RETURN_%":
            round(
                returns.mean(),
                4,
            ),

        "NET_MEDIAN_RETURN_%":
            round(
                returns.median(),
                4,
            ),

        "TOTAL_NET_RETURN_%":
            round(
                returns.sum(),
                4,
            ),

        "PROFIT_FACTOR":
            round(
                pf,
                4,
            ),

        "AVG_WIN_%":
            round(
                average_win,
                4,
            ),

        "AVG_LOSS_%":
            round(
                average_loss,
                4,
            ),

        "MAX_DRAWDOWN_%":
            round(
                max_drawdown(
                    returns.tolist()
                ),
                4,
            ),

        "AVG_MFE_%":
            round(
                pd.to_numeric(
                    df["MFE_%"],
                    errors="coerce",
                ).mean(),
                4,
            ),

        "AVG_MAE_%":
            round(
                pd.to_numeric(
                    df["MAE_%"],
                    errors="coerce",
                ).mean(),
                4,
            ),
    }


# ============================================================
# DISCOVERY TEST
# ============================================================

def discovery_test(
    market,
    discovery,
    candidates,
):

    rows = []

    for candidate in candidates:

        filtered = filter_candidate(
            discovery,
            candidate,
        )

        if (
            len(filtered)
            <
            MIN_DISCOVERY_TRADES
        ):
            continue

        for tp_atr in (
            ATR_TP_MULTIPLIERS
        ):

            for sl_atr in (
                ATR_SL_MULTIPLIERS
            ):

                simulated = resimulate(
                    market,
                    filtered,
                    candidate[
                        "SIDE"
                    ],
                    tp_atr,
                    sl_atr,
                )

                if (
                    len(simulated)
                    <
                    MIN_DISCOVERY_TRADES
                ):
                    continue

                stats = performance(
                    simulated
                )

                if stats is None:
                    continue

                rows.append(
                    {
                        "CANDIDATE":
                            candidate[
                                "NAME"
                            ],

                        "SIDE":
                            candidate[
                                "SIDE"
                            ],

                        "VOL_REGIME":
                            candidate[
                                "VOL"
                            ],

                        "TREND_STRENGTH":
                            candidate[
                                "STRENGTH"
                            ],

                        "TP_ATR":
                            tp_atr,

                        "SL_ATR":
                            sl_atr,

                        **stats,
                    }
                )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = (
            result
            .sort_values(
                by=[
                    "NET_AVG_RETURN_%",
                    "PROFIT_FACTOR",
                    "TRADES",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

    return result


# ============================================================
# FIND CANDIDATE
# ============================================================

def find_candidate(
    candidates,
    name,
):

    for candidate in candidates:

        if (
            candidate["NAME"]
            ==
            name
        ):
            return candidate

    return None


# ============================================================
# SELECTION TEST
# ============================================================

def selection_test(
    market,
    selection,
    discovery_results,
    candidates,
):

    if discovery_results.empty:
        return pd.DataFrame()

    # ========================================================
    # Limit selection exposure.
    #
    # Top 10 discovery configs per side only.
    # ========================================================

    shortlist_frames = []

    for side in [
        "LONG",
        "SHORT",
    ]:

        side_results = (
            discovery_results[
                discovery_results[
                    "SIDE"
                ]
                ==
                side
            ]
            .head(10)
            .copy()
        )

        shortlist_frames.append(
            side_results
        )

    shortlist = pd.concat(
        shortlist_frames,
        ignore_index=True,
    )

    rows = []

    for _, config in shortlist.iterrows():

        candidate = find_candidate(
            candidates,
            config[
                "CANDIDATE"
            ],
        )

        if candidate is None:
            continue

        filtered = filter_candidate(
            selection,
            candidate,
        )

        if (
            len(filtered)
            <
            MIN_SELECTION_TRADES
        ):
            continue

        simulated = resimulate(
            market,
            filtered,
            candidate[
                "SIDE"
            ],
            float(
                config[
                    "TP_ATR"
                ]
            ),
            float(
                config[
                    "SL_ATR"
                ]
            ),
        )

        stats = performance(
            simulated
        )

        if stats is None:
            continue

        rows.append(
            {
                "CANDIDATE":
                    candidate[
                        "NAME"
                    ],

                "SIDE":
                    candidate[
                        "SIDE"
                    ],

                "VOL_REGIME":
                    candidate[
                        "VOL"
                    ],

                "TREND_STRENGTH":
                    candidate[
                        "STRENGTH"
                    ],

                "TP_ATR":
                    float(
                        config[
                            "TP_ATR"
                        ]
                    ),

                "SL_ATR":
                    float(
                        config[
                            "SL_ATR"
                        ]
                    ),

                "DISCOVERY_TRADES":
                    int(
                        config[
                            "TRADES"
                        ]
                    ),

                "DISCOVERY_WIN_RATE_%":
                    config[
                        "WIN_RATE_%"
                    ],

                "DISCOVERY_NET_AVG_%":
                    config[
                        "NET_AVG_RETURN_%"
                    ],

                "DISCOVERY_PF":
                    config[
                        "PROFIT_FACTOR"
                    ],

                "SELECTION_TRADES":
                    stats[
                        "TRADES"
                    ],

                "SELECTION_WIN_RATE_%":
                    stats[
                        "WIN_RATE_%"
                    ],

                "SELECTION_NET_AVG_%":
                    stats[
                        "NET_AVG_RETURN_%"
                    ],

                "SELECTION_PF":
                    stats[
                        "PROFIT_FACTOR"
                    ],

                "SELECTION_MAX_DD_%":
                    stats[
                        "MAX_DRAWDOWN_%"
                    ],
            }
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = (
            result
            .sort_values(
                by=[
                    "SELECTION_NET_AVG_%",
                    "SELECTION_PF",
                    "SELECTION_TRADES",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

    return result


# ============================================================
# LOCK ONE STRATEGY PER SIDE
# ============================================================

def lock_candidates(
    selection_results,
):

    locked = []

    if selection_results.empty:
        return locked

    for side in [
        "LONG",
        "SHORT",
    ]:

        side_results = (
            selection_results[
                selection_results[
                    "SIDE"
                ]
                ==
                side
            ]
            .copy()
        )

        if side_results.empty:
            continue

        eligible = (
            side_results[
                (
                    side_results[
                        "SELECTION_TRADES"
                    ]
                    >=
                    SELECTION_MIN_TRADES
                )
                &
                (
                    side_results[
                        "SELECTION_NET_AVG_%"
                    ]
                    >
                    SELECTION_MIN_NET_AVG
                )
                &
                (
                    side_results[
                        "SELECTION_PF"
                    ]
                    >=
                    SELECTION_MIN_PF
                )
            ]
            .copy()
        )

        if eligible.empty:
            continue

        # ----------------------------------------------------
        # Combined robustness score.
        #
        # Expectancy first.
        # Profit factor second.
        # Sample size adds modest support.
        # ----------------------------------------------------

        eligible[
            "LOCK_SCORE"
        ] = (
            eligible[
                "SELECTION_NET_AVG_%"
            ]
            *
            100.0
            +
            (
                eligible[
                    "SELECTION_PF"
                ]
                -
                1.0
            )
            *
            10.0
            +
            np.log1p(
                eligible[
                    "SELECTION_TRADES"
                ]
            )
        )

        eligible = (
            eligible
            .sort_values(
                by=[
                    "LOCK_SCORE",
                    "SELECTION_NET_AVG_%",
                    "SELECTION_PF",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

        locked.append(
            eligible.iloc[0].to_dict()
        )

    return locked


# ============================================================
# CALENDAR BLOCKS
# ============================================================

def calendar_blocks(
    events,
    number_of_blocks,
):

    if events.empty:
        return []

    times = pd.to_datetime(
        events["TIME"],
        utc=True,
        errors="coerce",
    )

    start = times.min()
    end = times.max()

    if (
        pd.isna(start)
        or
        pd.isna(end)
    ):
        return []

    if start == end:

        return [
            (
                1,
                events.copy(),
            )
        ]

    edges = pd.date_range(
        start=start,
        end=end,
        periods=number_of_blocks + 1,
    )

    blocks = []

    for i in range(
        number_of_blocks
    ):

        if (
            i
            ==
            number_of_blocks - 1
        ):

            mask = (
                (times >= edges[i])
                &
                (
                    times
                    <=
                    edges[i + 1]
                )
            )

        else:

            mask = (
                (times >= edges[i])
                &
                (
                    times
                    <
                    edges[i + 1]
                )
            )

        blocks.append(
            (
                i + 1,
                events[
                    mask
                ].copy(),
            )
        )

    return blocks


# ============================================================
# FINAL CLASSIFICATION
# ============================================================

def classify_final(row):

    trades = int(
        row["HOLDOUT_TRADES"]
    )

    net_avg = safe_float(
        row[
            "HOLDOUT_NET_AVG_%"
        ]
    )

    pf = safe_float(
        row[
            "HOLDOUT_PF"
        ]
    )

    max_dd = safe_float(
        row[
            "HOLDOUT_MAX_DD_%"
        ]
    )

    positive_blocks = int(
        row[
            "POSITIVE_BLOCKS"
        ]
    )

    valid_blocks = int(
        row[
            "VALID_BLOCKS"
        ]
    )

    if (
        trades
        >=
        PASS_MIN_TRADES

        and
        net_avg
        >
        PASS_MIN_NET_AVG

        and
        pf
        >=
        PASS_MIN_PF

        and
        positive_blocks
        >=
        PASS_MIN_POSITIVE_BLOCKS

        and
        valid_blocks
        >=
        PASS_MIN_VALID_BLOCKS

        and
        max_dd
        <=
        PASS_MAX_DRAWDOWN_PCT
    ):

        return "PASS"

    if (
        trades
        >=
        WATCH_MIN_TRADES

        and
        net_avg
        >
        WATCH_MIN_NET_AVG

        and
        pf
        >=
        WATCH_MIN_PF

        and
        positive_blocks
        >=
        WATCH_MIN_POSITIVE_BLOCKS

        and
        valid_blocks
        >=
        WATCH_MIN_VALID_BLOCKS
    ):

        return "WATCH"

    return "REJECT"


# ============================================================
# FINAL HOLDOUT TEST
# ============================================================

def final_test(
    market,
    holdout,
    locked,
    candidates,
):

    summary_rows = []
    block_rows = []

    for config in locked:

        candidate = find_candidate(
            candidates,
            config[
                "CANDIDATE"
            ],
        )

        if candidate is None:
            continue

        filtered = filter_candidate(
            holdout,
            candidate,
        )

        simulated = resimulate(
            market,
            filtered,
            candidate[
                "SIDE"
            ],
            float(
                config[
                    "TP_ATR"
                ]
            ),
            float(
                config[
                    "SL_ATR"
                ]
            ),
        )

        stats = performance(
            simulated
        )

        if stats is None:
            continue

        positive_blocks = 0
        valid_blocks = 0

        for (
            block_number,
            block_events,
        ) in calendar_blocks(
            holdout,
            FINAL_BLOCKS,
        ):

            block_filtered = (
                filter_candidate(
                    block_events,
                    candidate,
                )
            )

            block_simulated = (
                resimulate(
                    market,
                    block_filtered,
                    candidate[
                        "SIDE"
                    ],
                    float(
                        config[
                            "TP_ATR"
                        ]
                    ),
                    float(
                        config[
                            "SL_ATR"
                        ]
                    ),
                )
            )

            block_stats = performance(
                block_simulated
            )

            if block_stats is None:

                block_rows.append(
                    {
                        "CANDIDATE":
                            candidate[
                                "NAME"
                            ],

                        "SIDE":
                            candidate[
                                "SIDE"
                            ],

                        "BLOCK":
                            block_number,

                        "TRADES":
                            0,

                        "NET_AVG_RETURN_%":
                            np.nan,

                        "PROFIT_FACTOR":
                            np.nan,
                    }
                )

                continue

            if (
                block_stats[
                    "TRADES"
                ]
                >
                0
            ):

                valid_blocks += 1

            if (
                block_stats[
                    "NET_AVG_RETURN_%"
                ]
                >
                0
            ):

                positive_blocks += 1

            block_rows.append(
                {
                    "CANDIDATE":
                        candidate[
                            "NAME"
                        ],

                    "SIDE":
                        candidate[
                            "SIDE"
                        ],

                    "BLOCK":
                        block_number,

                    "TRADES":
                        block_stats[
                            "TRADES"
                        ],

                    "WIN_RATE_%":
                        block_stats[
                            "WIN_RATE_%"
                        ],

                    "NET_AVG_RETURN_%":
                        block_stats[
                            "NET_AVG_RETURN_%"
                        ],

                    "PROFIT_FACTOR":
                        block_stats[
                            "PROFIT_FACTOR"
                        ],

                    "MAX_DRAWDOWN_%":
                        block_stats[
                            "MAX_DRAWDOWN_%"
                        ],
                }
            )

        row = {
            "CANDIDATE":
                candidate[
                    "NAME"
                ],

            "SIDE":
                candidate[
                    "SIDE"
                ],

            "VOL_REGIME":
                candidate[
                    "VOL"
                ],

            "TREND_STRENGTH":
                candidate[
                    "STRENGTH"
                ],

            "TP_ATR":
                config[
                    "TP_ATR"
                ],

            "SL_ATR":
                config[
                    "SL_ATR"
                ],

            "DISCOVERY_TRADES":
                config[
                    "DISCOVERY_TRADES"
                ],

            "DISCOVERY_NET_AVG_%":
                config[
                    "DISCOVERY_NET_AVG_%"
                ],

            "DISCOVERY_PF":
                config[
                    "DISCOVERY_PF"
                ],

            "SELECTION_TRADES":
                config[
                    "SELECTION_TRADES"
                ],

            "SELECTION_NET_AVG_%":
                config[
                    "SELECTION_NET_AVG_%"
                ],

            "SELECTION_PF":
                config[
                    "SELECTION_PF"
                ],

            "HOLDOUT_TRADES":
                stats[
                    "TRADES"
                ],

            "HOLDOUT_DECISIVE":
                stats[
                    "DECISIVE"
                ],

            "HOLDOUT_WINS":
                stats[
                    "WINS"
                ],

            "HOLDOUT_LOSSES":
                stats[
                    "LOSSES"
                ],

            "HOLDOUT_TIME_EXITS":
                stats[
                    "TIME_EXITS"
                ],

            "HOLDOUT_WIN_RATE_%":
                stats[
                    "WIN_RATE_%"
                ],

            "HOLDOUT_NET_AVG_%":
                stats[
                    "NET_AVG_RETURN_%"
                ],

            "HOLDOUT_TOTAL_NET_%":
                stats[
                    "TOTAL_NET_RETURN_%"
                ],

            "HOLDOUT_PF":
                stats[
                    "PROFIT_FACTOR"
                ],

            "HOLDOUT_MAX_DD_%":
                stats[
                    "MAX_DRAWDOWN_%"
                ],

            "HOLDOUT_AVG_MFE_%":
                stats[
                    "AVG_MFE_%"
                ],

            "HOLDOUT_AVG_MAE_%":
                stats[
                    "AVG_MAE_%"
                ],

            "POSITIVE_BLOCKS":
                positive_blocks,

            "VALID_BLOCKS":
                valid_blocks,
        }

        row["STATUS"] = (
            classify_final(
                row
            )
        )

        summary_rows.append(
            row
        )

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.DataFrame(
            block_rows
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 120)

    print(
        "COINDCX FUTURES HISTORICAL V17"
    )

    print(
        "REGIME + VOLATILITY + ATR EXPECTANCY ENGINE"
    )

    print("=" * 120)

    print(
        "PAIR:",
        PAIR
    )

    print(
        "ENTRY TIMEFRAME: 5 MINUTES"
    )

    print(
        "TREND TIMEFRAME: 1 HOUR"
    )

    print(
        "HISTORY TARGET: 365 DAYS"
    )

    print()

    print(
        "DISCOVERY: 60%"
    )

    print(
        "SELECTION: 20%"
    )

    print(
        "FINAL UNTOUCHED HOLDOUT: 20%"
    )

    print()

    print(
        "ROUND-TRIP COST:",
        ROUND_TRIP_COST_PCT,
        "%"
    )

    print(
        "MAX HOLD:",
        MAX_HOLD_BARS,
        "x 5m"
    )

    print()

    print(
        "V17 DOES NOT OPTIMIZE"
    )

    print(
        "FOR HIGH WIN RATE ALONE."
    )

    print()

    print(
        "PRIMARY GOAL:"
    )

    print(
        "POSITIVE NET EXPECTANCY"
    )

    print(
        "+ PROFIT FACTOR"
    )

    print(
        "+ TIME-BLOCK CONSISTENCY"
    )

    print()

    print(
        "NO REAL ORDERS WILL BE PLACED."
    )

    # ========================================================
    # HISTORY
    # ========================================================

    (
        df_5m,
        df_1h,
        actual_days,
    ) = download_history()

    print()
    print(
        "ACTUAL HISTORY:",
        actual_days,
        "DAYS"
    )

    # ========================================================
    # PREPARE
    # ========================================================

    market = prepare_data(
        df_5m,
        df_1h,
    )

    print()
    print(
        "TREND REGIME COUNTS:"
    )

    print(
        market[
            "TREND_REGIME"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "VOLATILITY REGIME COUNTS:"
    )

    print(
        market[
            "VOL_REGIME"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "TREND STRENGTH COUNTS:"
    )

    print(
        market[
            "TREND_STRENGTH"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # EVENTS
    # ========================================================

    print()
    print(
        "Collecting V17 base signals..."
    )

    events = collect_events(
        market
    )

    if events.empty:

        print()
        print(
            "NO V17 SIGNAL EVENTS FOUND."
        )

        return

    print()
    print(
        "TOTAL EVENTS:",
        len(events)
    )

    print(
        "LONG:",
        int(
            (
                events["SIDE"]
                ==
                "LONG"
            ).sum()
        )
    )

    print(
        "SHORT:",
        int(
            (
                events["SIDE"]
                ==
                "SHORT"
            ).sum()
        )
    )

    # ========================================================
    # SPLIT
    # ========================================================

    (
        discovery,
        selection,
        holdout,
    ) = split_events(
        events
    )

    print()
    print("=" * 120)

    print(
        "V17 CHRONOLOGICAL SPLIT"
    )

    print("=" * 120)

    print(
        "DISCOVERY EVENTS:",
        len(discovery)
    )

    print(
        "SELECTION EVENTS:",
        len(selection)
    )

    print(
        "FINAL HOLDOUT EVENTS:",
        len(holdout)
    )

    print()

    print(
        "DISCOVERY RANGE:",
        discovery["TIME"].min(),
        "->",
        discovery["TIME"].max(),
    )

    print(
        "SELECTION RANGE:",
        selection["TIME"].min(),
        "->",
        selection["TIME"].max(),
    )

    print(
        "FINAL HOLDOUT RANGE:",
        holdout["TIME"].min(),
        "->",
        holdout["TIME"].max(),
    )

    # ========================================================
    # CANDIDATES
    # ========================================================

    candidates = build_candidates()

    print()
    print(
        "REGIME CANDIDATES:",
        len(candidates)
    )

    # ========================================================
    # DISCOVERY
    # ========================================================

    print()
    print(
        "Testing ATR TP/SL"
    )

    print(
        "on DISCOVERY data only..."
    )

    discovery_results = (
        discovery_test(
            market,
            discovery,
            candidates,
        )
    )

    print()
    print(
        "DISCOVERY CONFIGURATIONS:",
        len(
            discovery_results
        )
    )

    if not discovery_results.empty:

        print()
        print(
            "TOP 20 DISCOVERY RESULTS:"
        )

        print()

        print(
            discovery_results
            .head(20)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # SELECTION
    # ========================================================

    print()
    print(
        "Testing shortlist"
    )

    print(
        "on SEPARATE SELECTION data..."
    )

    selection_results = (
        selection_test(
            market,
            selection,
            discovery_results,
            candidates,
        )
    )

    if not selection_results.empty:

        print()
        print(
            "SELECTION RESULTS:"
        )

        print()

        print(
            selection_results
            .to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "NO candidate had enough"
        )

        print(
            "selection sample."
        )

    # ========================================================
    # LOCK
    # ========================================================

    locked = lock_candidates(
        selection_results
    )

    print()
    print("=" * 120)

    print(
        "LOCKED BEFORE FINAL HOLDOUT"
    )

    print("=" * 120)

    if not locked:

        print()
        print(
            "NO STRATEGY SURVIVED SELECTION."
        )

        print()
        print(
            "FINAL HOLDOUT WILL NOT"
        )

        print(
            "BE USED TO RESCUE IT."
        )

    else:

        for config in locked:

            print()

            print(
                "CANDIDATE:",
                config[
                    "CANDIDATE"
                ]
            )

            print(
                "SIDE:",
                config[
                    "SIDE"
                ]
            )

            print(
                "VOL:",
                config[
                    "VOL_REGIME"
                ]
            )

            print(
                "TREND STRENGTH:",
                config[
                    "TREND_STRENGTH"
                ]
            )

            print(
                "TP ATR:",
                config[
                    "TP_ATR"
                ]
            )

            print(
                "SL ATR:",
                config[
                    "SL_ATR"
                ]
            )

            print(
                "SELECTION TRADES:",
                config[
                    "SELECTION_TRADES"
                ]
            )

            print(
                "SELECTION NET AVG:",
                config[
                    "SELECTION_NET_AVG_%"
                ]
            )

            print(
                "SELECTION PF:",
                config[
                    "SELECTION_PF"
                ]
            )

    # ========================================================
    # FINAL HOLDOUT
    # ========================================================

    if locked:

        print()
        print(
            "Opening FINAL UNTOUCHED HOLDOUT..."
        )

        (
            final_results,
            final_blocks,
        ) = final_test(
            market,
            holdout,
            locked,
            candidates,
        )

    else:

        final_results = (
            pd.DataFrame()
        )

        final_blocks = (
            pd.DataFrame()
        )

    # ========================================================
    # SAVE
    # ========================================================

    combined_events = pd.concat(
        [
            discovery,
            selection,
            holdout,
        ],
        ignore_index=True,
    )

    combined_events.to_csv(
        EVENT_FILE,
        index=False,
    )

    discovery_results.to_csv(
        DISCOVERY_FILE,
        index=False,
    )

    selection_results.to_csv(
        SELECTION_FILE,
        index=False,
    )

    final_blocks.to_csv(
        FINAL_BLOCK_FILE,
        index=False,
    )

    final_results.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 120)

    print(
        "V17 FILES SAVED"
    )

    print("=" * 120)

    print(EVENT_FILE)
    print(DISCOVERY_FILE)
    print(SELECTION_FILE)
    print(FINAL_BLOCK_FILE)
    print(SUMMARY_FILE)

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 160)

    print(
        "V17 FINAL UNTOUCHED HOLDOUT RESULTS"
    )

    print("=" * 160)

    if final_results.empty:

        print()
        print(
            "NO FINAL HOLDOUT CANDIDATE."
        )

        print()
        print(
            "V17 RESULT = NO STRATEGY"
        )

    else:

        print()

        print(
            final_results
            .to_string(
                index=False
            )
        )

        print()
        print("=" * 160)

        pass_count = int(
            (
                final_results[
                    "STATUS"
                ]
                ==
                "PASS"
            ).sum()
        )

        watch_count = int(
            (
                final_results[
                    "STATUS"
                ]
                ==
                "WATCH"
            ).sum()
        )

        reject_count = int(
            (
                final_results[
                    "STATUS"
                ]
                ==
                "REJECT"
            ).sum()
        )

        print(
            "PASS:",
            pass_count
        )

        print(
            "WATCH:",
            watch_count
        )

        print(
            "REJECT:",
            reject_count
        )

        for _, row in (
            final_results.iterrows()
        ):

            print()
            print("-" * 100)

            print(
                "CANDIDATE:",
                row[
                    "CANDIDATE"
                ]
            )

            print(
                "STATUS:",
                row[
                    "STATUS"
                ]
            )

            print(
                "SIDE:",
                row[
                    "SIDE"
                ]
            )

            print(
                "TP ATR:",
                row[
                    "TP_ATR"
                ]
            )

            print(
                "SL ATR:",
                row[
                    "SL_ATR"
                ]
            )

            print(
                "HOLDOUT TRADES:",
                row[
                    "HOLDOUT_TRADES"
                ]
            )

            print(
                "HOLDOUT WIN RATE:",
                row[
                    "HOLDOUT_WIN_RATE_%"
                ],
                "%"
            )

            print(
                "HOLDOUT NET AVG:",
                row[
                    "HOLDOUT_NET_AVG_%"
                ],
                "%"
            )

            print(
                "HOLDOUT PROFIT FACTOR:",
                row[
                    "HOLDOUT_PF"
                ]
            )

            print(
                "HOLDOUT MAX DRAWDOWN:",
                row[
                    "HOLDOUT_MAX_DD_%"
                ],
                "%"
            )

            print(
                "POSITIVE BLOCKS:",
                row[
                    "POSITIVE_BLOCKS"
                ],
                "/",
                FINAL_BLOCKS,
            )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 120)

    print(
        "V17 INTERPRETATION"
    )

    print("=" * 120)

    print()

    print(
        "PASS:"
    )

    print(
        "Positive expectancy survived"
    )

    print(
        "the untouched final holdout."
    )

    print(
        "PASS is PAPER-TRADING candidate only."
    )

    print()

    print(
        "WATCH:"
    )

    print(
        "Promising but sample or"
    )

    print(
        "consistency is not strong enough."
    )

    print()

    print(
        "REJECT:"
    )

    print(
        "Do not use this strategy live."
    )

    print()

    print(
        "If V17 has no PASS,"
    )

    print(
        "we will NOT force another"
    )

    print(
        "high win-rate filter."
    )

    print()

    print(
        "NO REAL ORDERS WERE PLACED."
    )

    print()
    print("=" * 120)

    print(
        "FUTURES HISTORICAL V17 COMPLETE"
    )

    print("=" * 120)


if __name__ == "__main__":

    main()
