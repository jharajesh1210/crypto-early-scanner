# ============================================================
# crypto_futures_historical_v17.py
#
# COINDCX FUTURES HISTORICAL V17
# REGIME + VOLATILITY + ATR EXPECTANCY ENGINE
#
# Uses the already-proven CoinDCX Futures downloader from:
# crypto_futures_historical_learning.py
#
# Historical research only.
# NO Telegram.
# NO real orders.
# ============================================================

import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1


# ============================================================
# SETTINGS
# ============================================================

PAIR = "B-BTC_USDT"

HISTORY_ATTEMPTS = [365, 270, 180]

DISCOVERY_RATIO = 0.60
SELECTION_RATIO = 0.20
HOLDOUT_RATIO = 0.20

FINAL_BLOCKS = 4

ROUND_TRIP_COST_PCT = 0.10

MAX_HOLD_BARS = 48

SAME_BAR_POLICY = "SL_FIRST"


# Small TP/SL grid.
# We deliberately avoid hundreds of combinations.

ATR_TP_MULTIPLIERS = [
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]

ATR_SL_MULTIPLIERS = [
    0.75,
    1.00,
    1.25,
    1.50,
]


MIN_DISCOVERY_TRADES = 35
MIN_SELECTION_TRADES = 18
MIN_HOLDOUT_TRADES = 15


# Final PASS criteria

PASS_MIN_TRADES = 20
PASS_MIN_NET_AVG = 0.0
PASS_MIN_PF = 1.15
PASS_MIN_POSITIVE_BLOCKS = 3
PASS_MIN_VALID_BLOCKS = 3
PASS_MAX_DRAWDOWN_PCT = 6.0


# Final WATCH criteria

WATCH_MIN_TRADES = 15
WATCH_MIN_NET_AVG = 0.0
WATCH_MIN_PF = 1.00
WATCH_MIN_POSITIVE_BLOCKS = 2
WATCH_MIN_VALID_BLOCKS = 2


# ============================================================
# OUTPUT FILES
# ============================================================

EVENT_FILE = "crypto_futures_v17_events.csv"

DISCOVERY_FILE = "crypto_futures_v17_discovery.csv"

SELECTION_FILE = "crypto_futures_v17_selection.csv"

FINAL_BLOCK_FILE = "crypto_futures_v17_final_blocks.csv"

SUMMARY_FILE = "crypto_futures_v17_summary.csv"


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value, default=np.nan):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

        return default

    except Exception:

        return default


# ============================================================
# DOWNLOAD USING PROVEN V1 DOWNLOADER
# ============================================================

def download_history():

    last_error = None

    for days in HISTORY_ATTEMPTS:

        print()
        print("=" * 110)

        print(
            f"V17 TRYING HISTORY: {days} DAYS"
        )

        print("=" * 110)

        try:

            print()
            print(
                "Downloading native 5-minute Futures data..."
            )

            df_5m = v1.fetch_history(
                PAIR,
                "5",
                days,
                v1.CHUNK_DAYS_5M,
            )

            if df_5m is None or df_5m.empty:

                raise RuntimeError(
                    "5-minute data empty."
                )

            print()
            print(
                "5M CANDLES:",
                len(df_5m)
            )

            print()
            print(
                "Downloading native 1-hour Futures data..."
            )

            df_1h = v1.fetch_history(
                PAIR,
                "60",
                days,
                v1.CHUNK_DAYS_1H,
            )

            if df_1h is None or df_1h.empty:

                raise RuntimeError(
                    "1-hour data empty."
                )

            print()
            print(
                "1H CANDLES:",
                len(df_1h)
            )

            return (
                df_5m.copy(),
                df_1h.copy(),
                days,
            )

        except Exception as exc:

            last_error = exc

            print()
            print(
                "HISTORY ATTEMPT FAILED:"
            )

            print(exc)

    raise RuntimeError(
        "All V17 history attempts failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# EMA
# ============================================================

def ema(series, length):

    return series.ewm(
        span=length,
        adjust=False,
    ).mean()


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, length=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
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

    return (
        100.0
        -
        (
            100.0
            /
            (1.0 + rs)
        )
    )


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, length=14):

    previous_close = (
        df["close"]
        .shift(1)
    )

    tr1 = (
        df["high"]
        -
        df["low"]
    )

    tr2 = (
        df["high"]
        -
        previous_close
    ).abs()

    tr3 = (
        df["low"]
        -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
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

def calculate_adx(df, length=14):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    up_move = high.diff()

    down_move = -low.diff()

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

    data = df.copy()

    data = (
        data
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    data["ema20"] = ema(
        data["close"],
        20,
    )

    data["ema50"] = ema(
        data["close"],
        50,
    )

    data["ema200"] = ema(
        data["close"],
        200,
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    data["rsi"] = calculate_rsi(
        data["close"],
        14,
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = ema(
        data["close"],
        12,
    )

    ema26 = ema(
        data["close"],
        26,
    )

    data["macd"] = (
        ema12
        -
        ema26
    )

    data["macd_signal"] = ema(
        data["macd"],
        9,
    )

    data["macd_hist"] = (
        data["macd"]
        -
        data["macd_signal"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    data["atr"] = calculate_atr(
        data,
        14,
    )

    data["atr_pct"] = (
        data["atr"]
        /
        data["close"]
        *
        100.0
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    (
        data["adx"],
        data["plus_di"],
        data["minus_di"],
    ) = calculate_adx(
        data,
        14,
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    data["bb_middle"] = (
        data["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        data["close"]
        .rolling(20)
        .std()
    )

    data["bb_upper"] = (
        data["bb_middle"]
        +
        2.0 * bb_std
    )

    data["bb_lower"] = (
        data["bb_middle"]
        -
        2.0 * bb_std
    )

    data["bb_width_pct"] = (
        (
            data["bb_upper"]
            -
            data["bb_lower"]
        )
        /
        data["bb_middle"].replace(
            0,
            np.nan,
        )
        *
        100.0
    )

    data["bb_position"] = (
        (
            data["close"]
            -
            data["bb_lower"]
        )
        /
        (
            data["bb_upper"]
            -
            data["bb_lower"]
        ).replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    data["volume_ma20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"]
        /
        data["volume_ma20"].replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Candle features
    # --------------------------------------------------------

    candle_range = (
        data["high"]
        -
        data["low"]
    )

    body = (
        data["close"]
        -
        data["open"]
    ).abs()

    data["range_pct"] = (
        candle_range
        /
        data["close"]
        *
        100.0
    )

    data["body_ratio"] = (
        body
        /
        candle_range.replace(
            0,
            np.nan,
        )
    )

    data["close_position"] = (
        (
            data["close"]
            -
            data["low"]
        )
        /
        candle_range.replace(
            0,
            np.nan,
        )
    )

    data["lower_wick_ratio"] = (
        (
            np.minimum(
                data["open"],
                data["close"],
            )
            -
            data["low"]
        )
        /
        candle_range.replace(
            0,
            np.nan,
        )
    )

    data["upper_wick_ratio"] = (
        (
            data["high"]
            -
            np.maximum(
                data["open"],
                data["close"],
            )
        )
        /
        candle_range.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Momentum / slope
    # --------------------------------------------------------

    data["ema20_slope_3"] = (
        data["ema20"]
        .pct_change(3)
        *
        100.0
    )

    data["rsi_change_3"] = (
        data["rsi"]
        -
        data["rsi"].shift(3)
    )

    data["macd_change_3"] = (
        data["macd_hist"]
        -
        data["macd_hist"].shift(3)
    )

    data["ema_distance_pct"] = (
        (
            data["ema20"]
            -
            data["ema50"]
        )
        /
        data["close"]
        *
        100.0
    )

    return data


# ============================================================
# PREPARE 1H DATA
# ============================================================

def prepare_1h(df):

    data = add_indicators(
        df
    )

    keep = [
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

    data = data[
        keep
    ].copy()

    data = data.rename(
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

    # Use completed previous 1H candle only.

    columns = [
        c
        for c in data.columns
        if c != "datetime"
    ]

    data[
        columns
    ] = data[
        columns
    ].shift(1)

    return data


# ============================================================
# PREPARE MARKET
# ============================================================

def prepare_market(
    df_5m,
    df_1h,
):

    print()
    print(
        "Calculating V17 indicators..."
    )

    five = add_indicators(
        df_5m
    )

    hour = prepare_1h(
        df_1h
    )

    five = (
        five
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    hour = (
        hour
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    data = pd.merge_asof(
        five,
        hour,
        on="datetime",
        direction="backward",
    )

    # --------------------------------------------------------
    # 1H EMA distance
    # --------------------------------------------------------

    data[
        "h1_ema_distance_pct"
    ] = (
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
    # Rolling volatility reference
    # Approx 20 days of 5m candles
    # --------------------------------------------------------

    data[
        "atr_median"
    ] = (
        data["atr_pct"]
        .rolling(
            5760,
            min_periods=500,
        )
        .median()
    )

    data[
        "atr_q75"
    ] = (
        data["atr_pct"]
        .rolling(
            5760,
            min_periods=500,
        )
        .quantile(0.75)
    )

    # --------------------------------------------------------
    # Volatility regime
    # --------------------------------------------------------

    data[
        "VOL_REGIME"
    ] = np.select(
        [
            (
                data["atr_pct"]
                <
                data["atr_median"]
            ),

            (
                data["atr_pct"]
                >=
                data["atr_q75"]
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

    data[
        "TREND_REGIME"
    ] = np.select(
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

    data[
        "TREND_STRENGTH"
    ] = np.select(
        [
            (
                data["adx"]
                >=
                25
            )
            &
            (
                data["adx_1h"]
                >=
                20
            ),

            (
                data["adx"]
                >=
                18
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
    ]

    data = (
        data
        .dropna(
            subset=required
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "USABLE 5M CANDLES:",
        len(data)
    )

    return data


# ============================================================
# LONG ENTRY
# ============================================================

def long_signal(
    data,
    i,
):

    row = data.iloc[i]

    previous = data.iloc[
        i - 1
    ]

    if (
        row["TREND_REGIME"]
        !=
        "UPTREND"
    ):
        return False

    pullback = (
        row["low"]
        <=
        row["ema20"]
        *
        1.003
    )

    reclaim = (
        row["close"]
        >=
        row["ema20"]
    )

    momentum = (
        row["macd_hist"]
        >
        previous["macd_hist"]
    )

    rsi_ok = (
        row["rsi"]
        >=
        40
        and
        row["rsi"]
        <=
        70
    )

    candle_ok = (
        row["close_position"]
        >=
        0.55
    )

    di_ok = (
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
        di_ok
    )


# ============================================================
# SHORT ENTRY
# ============================================================

def short_signal(
    data,
    i,
):

    row = data.iloc[i]

    previous = data.iloc[
        i - 1
    ]

    if (
        row["TREND_REGIME"]
        !=
        "DOWNTREND"
    ):
        return False

    pullback = (
        row["high"]
        >=
        row["ema20"]
        *
        0.997
    )

    rejection = (
        row["close"]
        <=
        row["ema20"]
    )

    momentum = (
        row["macd_hist"]
        <
        previous["macd_hist"]
    )

    rsi_ok = (
        row["rsi"]
        >=
        30
        and
        row["rsi"]
        <=
        60
    )

    candle_ok = (
        row["close_position"]
        <=
        0.45
    )

    di_ok = (
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
        di_ok
    )


# ============================================================
# COLLECT BASE EVENTS
# ============================================================

def collect_events(
    data
):

    rows = []

    last_index = (
        len(data)
        -
        MAX_HOLD_BARS
        -
        1
    )

    for i in range(
        200,
        last_index,
    ):

        side = None

        if long_signal(
            data,
            i,
        ):

            side = "LONG"

        elif short_signal(
            data,
            i,
        ):

            side = "SHORT"

        if side is None:
            continue

        row = data.iloc[i]

        rows.append(
            {
                "DATA_INDEX":
                    i,

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

                "CLOSE_POSITION":
                    safe_float(
                        row["close_position"]
                    ),

                "LOWER_WICK_RATIO":
                    safe_float(
                        row[
                            "lower_wick_ratio"
                        ]
                    ),

                "UPPER_WICK_RATIO":
                    safe_float(
                        row[
                            "upper_wick_ratio"
                        ]
                    ),
            }
        )

    events = pd.DataFrame(
        rows
    )

    if not events.empty:

        events["TIME"] = (
            pd.to_datetime(
                events["TIME"],
                utc=True,
                errors="coerce",
            )
        )

        events = (
            events
            .sort_values("TIME")
            .reset_index(drop=True)
        )

    return events


# ============================================================
# TRADE SIMULATOR
# ============================================================

def simulate_trade(
    market,
    index,
    side,
    tp_atr,
    sl_atr,
):

    row = market.iloc[
        index
    ]

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
        index
        +
        MAX_HOLD_BARS,
        len(market) - 1,
    )

    future = market.iloc[
        index + 1:
        end_index + 1
    ]

    if future.empty:
        return None

    outcome = "TIME_EXIT"

    exit_price = safe_float(
        future.iloc[-1][
            "close"
        ]
    )

    bars_held = len(
        future
    )

    for number, (_, bar) in enumerate(
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

            tp_hit = (
                high >= tp_price
            )

            sl_hit = (
                low <= sl_price
            )

        else:

            tp_hit = (
                low <= tp_price
            )

            sl_hit = (
                high >= sl_price
            )

        if tp_hit and sl_hit:

            if (
                SAME_BAR_POLICY
                ==
                "SL_FIRST"
            ):

                outcome = "LOSS"

                exit_price = (
                    sl_price
                )

            else:

                outcome = "WIN"

                exit_price = (
                    tp_price
                )

            bars_held = number

            break

        if tp_hit:

            outcome = "WIN"

            exit_price = (
                tp_price
            )

            bars_held = number

            break

        if sl_hit:

            outcome = "LOSS"

            exit_price = (
                sl_price
            )

            bars_held = number

            break

    observed = future.iloc[
        :bars_held
    ]

    max_high = safe_float(
        observed["high"].max()
    )

    min_low = safe_float(
        observed["low"].min()
    )

    if side == "LONG":

        raw_return = (
            (
                exit_price
                /
                entry
            )
            -
            1.0
        ) * 100.0

        mfe = (
            (
                max_high
                /
                entry
            )
            -
            1.0
        ) * 100.0

        mae = (
            (
                min_low
                /
                entry
            )
            -
            1.0
        ) * 100.0

    else:

        raw_return = (
            (
                entry
                /
                exit_price
            )
            -
            1.0
        ) * 100.0

        mfe = (
            (
                entry
                /
                min_low
            )
            -
            1.0
        ) * 100.0

        mae = -(
            (
                max_high
                /
                entry
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
        "OUTCOME":
            outcome,

        "EXIT_BARS":
            bars_held,

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
# RESIMULATE EVENTS
# ============================================================

def resimulate(
    market,
    events,
    side,
    tp_atr,
    sl_atr,
):

    rows = []

    for _, event in (
        events.iterrows()
    ):

        result = simulate_trade(
            market,
            int(
                event[
                    "DATA_INDEX"
                ]
            ),
            side,
            tp_atr,
            sl_atr,
        )

        if result is None:
            continue

        output = (
            event.to_dict()
        )

        output.update(
            result
        )

        rows.append(
            output
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PERFORMANCE
# ============================================================

def performance(
    trades
):

    if trades.empty:
        return None

    returns = pd.to_numeric(
        trades[
            "NET_RETURN_%"
        ],
        errors="coerce",
    ).dropna()

    if returns.empty:
        return None

    wins = int(
        (
            trades["OUTCOME"]
            ==
            "WIN"
        ).sum()
    )

    losses = int(
        (
            trades["OUTCOME"]
            ==
            "LOSS"
        ).sum()
    )

    time_exits = int(
        (
            trades["OUTCOME"]
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

    positive = returns[
        returns > 0
    ]

    negative = returns[
        returns < 0
    ]

    gross_profit = float(
        positive.sum()
    )

    gross_loss = abs(
        float(
            negative.sum()
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    elif gross_profit > 0:

        profit_factor = (
            999.0
        )

    else:

        profit_factor = (
            0.0
        )

    equity = (
        returns
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

    max_drawdown = abs(
        float(
            drawdown.min()
        )
    )

    return {
        "TRADES":
            len(trades),

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

        "TOTAL_NET_RETURN_%":
            round(
                returns.sum(),
                4,
            ),

        "PROFIT_FACTOR":
            round(
                profit_factor,
                4,
            ),

        "MAX_DRAWDOWN_%":
            round(
                max_drawdown,
                4,
            ),

        "AVG_MFE_%":
            round(
                pd.to_numeric(
                    trades["MFE_%"],
                    errors="coerce",
                ).mean(),
                4,
            ),

        "AVG_MAE_%":
            round(
                pd.to_numeric(
                    trades["MAE_%"],
                    errors="coerce",
                ).mean(),
                4,
            ),
    }


# ============================================================
# SPLIT EVENTS
# ============================================================

def split_events(
    events
):

    events = (
        events
        .sort_values("TIME")
        .reset_index(drop=True)
        .copy()
    )

    total = len(
        events
    )

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
        .iloc[
            :discovery_end
        ]
        .copy()
    )

    selection = (
        events
        .iloc[
            discovery_end:
            selection_end
        ]
        .copy()
    )

    holdout = (
        events
        .iloc[
            selection_end:
        ]
        .copy()
    )

    discovery[
        "DATASET"
    ] = "DISCOVERY"

    selection[
        "DATASET"
    ] = "SELECTION"

    holdout[
        "DATASET"
    ] = "FINAL_HOLDOUT"

    return (
        discovery.reset_index(
            drop=True
        ),

        selection.reset_index(
            drop=True
        ),

        holdout.reset_index(
            drop=True
        ),
    )


# ============================================================
# CANDIDATES
# ============================================================

def build_candidates():

    candidates = []

    for side in [
        "LONG",
        "SHORT",
    ]:

        definitions = [
            (
                "BASE",
                None,
                None,
            ),

            (
                "LOW_VOL",
                "LOW_VOL",
                None,
            ),

            (
                "NORMAL_VOL",
                "NORMAL_VOL",
                None,
            ),

            (
                "HIGH_VOL",
                "HIGH_VOL",
                None,
            ),

            (
                "MEDIUM_TREND",
                None,
                "MEDIUM",
            ),

            (
                "STRONG_TREND",
                None,
                "STRONG",
            ),

            (
                "NORMAL_STRONG",
                "NORMAL_VOL",
                "STRONG",
            ),

            (
                "HIGH_STRONG",
                "HIGH_VOL",
                "STRONG",
            ),
        ]

        for (
            name,
            volatility,
            strength,
        ) in definitions:

            candidates.append(
                {
                    "NAME":
                        f"V17_{side}_{name}",

                    "SIDE":
                        side,

                    "VOL":
                        volatility,

                    "STRENGTH":
                        strength,
                }
            )

    return candidates


# ============================================================
# FILTER CANDIDATE
# ============================================================

def filter_candidate(
    events,
    candidate,
):

    output = events[
        events["SIDE"]
        ==
        candidate["SIDE"]
    ].copy()

    if (
        candidate["VOL"]
        is not None
    ):

        output = output[
            output["VOL_REGIME"]
            ==
            candidate["VOL"]
        ].copy()

    if (
        candidate["STRENGTH"]
        is not None
    ):

        output = output[
            output["TREND_STRENGTH"]
            ==
            candidate[
                "STRENGTH"
            ]
        ].copy()

    return output


# ============================================================
# DISCOVERY
# ============================================================

def run_discovery(
    market,
    discovery,
    candidates,
):

    rows = []

    for candidate in candidates:

        base = filter_candidate(
            discovery,
            candidate,
        )

        if (
            len(base)
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

                trades = resimulate(
                    market,
                    base,
                    candidate[
                        "SIDE"
                    ],
                    tp_atr,
                    sl_atr,
                )

                if (
                    len(trades)
                    <
                    MIN_DISCOVERY_TRADES
                ):

                    continue

                stats = performance(
                    trades
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
                [
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
# SELECTION
# ============================================================

def run_selection(
    market,
    selection,
    discovery_results,
    candidates,
):

    if discovery_results.empty:

        return pd.DataFrame()

    shortlists = []

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

        shortlists.append(
            side_results
        )

    shortlist = pd.concat(
        shortlists,
        ignore_index=True,
    )

    rows = []

    for _, config in (
        shortlist.iterrows()
    ):

        candidate = find_candidate(
            candidates,
            config[
                "CANDIDATE"
            ],
        )

        if candidate is None:
            continue

        base = filter_candidate(
            selection,
            candidate,
        )

        if (
            len(base)
            <
            MIN_SELECTION_TRADES
        ):

            continue

        trades = resimulate(
            market,
            base,
            candidate["SIDE"],
            float(
                config["TP_ATR"]
            ),
            float(
                config["SL_ATR"]
            ),
        )

        stats = performance(
            trades
        )

        if stats is None:
            continue

        rows.append(
            {
                "CANDIDATE":
                    candidate["NAME"],

                "SIDE":
                    candidate["SIDE"],

                "VOL_REGIME":
                    candidate["VOL"],

                "TREND_STRENGTH":
                    candidate[
                        "STRENGTH"
                    ],

                "TP_ATR":
                    config["TP_ATR"],

                "SL_ATR":
                    config["SL_ATR"],

                "DISCOVERY_TRADES":
                    config["TRADES"],

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
                    stats["TRADES"],

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
                [
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
# LOCK BEST ONE PER SIDE
# ============================================================

def lock_strategies(
    selection_results
):

    locked = []

    if selection_results.empty:

        return locked

    for side in [
        "LONG",
        "SHORT",
    ]:

        data = selection_results[
            selection_results[
                "SIDE"
            ]
            ==
            side
        ].copy()

        data = data[
            (
                data[
                    "SELECTION_TRADES"
                ]
                >=
                MIN_SELECTION_TRADES
            )
            &
            (
                data[
                    "SELECTION_NET_AVG_%"
                ]
                >
                0
            )
            &
            (
                data[
                    "SELECTION_PF"
                ]
                >=
                1.05
            )
        ].copy()

        if data.empty:
            continue

        data["LOCK_SCORE"] = (
            data[
                "SELECTION_NET_AVG_%"
            ]
            *
            100.0
            +
            (
                data[
                    "SELECTION_PF"
                ]
                -
                1.0
            )
            *
            10.0
            +
            np.log1p(
                data[
                    "SELECTION_TRADES"
                ]
            )
        )

        data = (
            data
            .sort_values(
                [
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
            data.iloc[0].to_dict()
        )

    return locked


# ============================================================
# FINAL CALENDAR BLOCKS
# ============================================================

def make_calendar_blocks(
    events
):

    if events.empty:
        return []

    start = events[
        "TIME"
    ].min()

    end = events[
        "TIME"
    ].max()

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
        periods=FINAL_BLOCKS + 1,
    )

    output = []

    for i in range(
        FINAL_BLOCKS
    ):

        if (
            i
            ==
            FINAL_BLOCKS - 1
        ):

            mask = (
                (
                    events["TIME"]
                    >=
                    edges[i]
                )
                &
                (
                    events["TIME"]
                    <=
                    edges[i + 1]
                )
            )

        else:

            mask = (
                (
                    events["TIME"]
                    >=
                    edges[i]
                )
                &
                (
                    events["TIME"]
                    <
                    edges[i + 1]
                )
            )

        output.append(
            (
                i + 1,
                events[
                    mask
                ].copy(),
            )
        )

    return output


# ============================================================
# CLASSIFY FINAL
# ============================================================

def classify(
    trades,
    net_avg,
    pf,
    positive_blocks,
    valid_blocks,
    max_dd,
):

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
# FINAL HOLDOUT
# ============================================================

def run_final(
    market,
    holdout,
    locked,
    candidates,
):

    summaries = []

    blocks_output = []

    for config in locked:

        candidate = find_candidate(
            candidates,
            config[
                "CANDIDATE"
            ],
        )

        if candidate is None:
            continue

        base = filter_candidate(
            holdout,
            candidate,
        )

        trades = resimulate(
            market,
            base,
            candidate["SIDE"],
            float(
                config["TP_ATR"]
            ),
            float(
                config["SL_ATR"]
            ),
        )

        stats = performance(
            trades
        )

        if stats is None:
            continue

        positive_blocks = 0

        valid_blocks = 0

        for (
            block_number,
            block_events,
        ) in make_calendar_blocks(
            holdout
        ):

            block_base = (
                filter_candidate(
                    block_events,
                    candidate,
                )
            )

            block_trades = (
                resimulate(
                    market,
                    block_base,
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

            block_stats = (
                performance(
                    block_trades
                )
            )

            if block_stats is None:

                blocks_output.append(
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
                    }
                )

                continue

            valid_blocks += 1

            if (
                block_stats[
                    "NET_AVG_RETURN_%"
                ]
                >
                0
            ):

                positive_blocks += 1

            blocks_output.append(
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

                    **block_stats,
                }
            )

        status = classify(
            stats["TRADES"],
            stats[
                "NET_AVG_RETURN_%"
            ],
            stats[
                "PROFIT_FACTOR"
            ],
            positive_blocks,
            valid_blocks,
            stats[
                "MAX_DRAWDOWN_%"
            ],
        )

        summaries.append(
            {
                "STATUS":
                    status,

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
        )

    return (
        pd.DataFrame(
            summaries
        ),
        pd.DataFrame(
            blocks_output
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
        "IMPORTANT:"
    )

    print(
        "V17 DOES NOT OPTIMIZE FOR"
    )

    print(
        "HIGH WIN RATE ALONE."
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

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Market preparation
    # --------------------------------------------------------

    market = prepare_market(
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

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    print()
    print(
        "Collecting V17 base signals..."
    )

    events = collect_events(
        market
    )

    if events.empty:

        raise RuntimeError(
            "No V17 base events found."
        )

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

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

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
        discovery[
            "TIME"
        ].min(),
        "->",
        discovery[
            "TIME"
        ].max(),
    )

    print(
        "SELECTION RANGE:",
        selection[
            "TIME"
        ].min(),
        "->",
        selection[
            "TIME"
        ].max(),
    )

    print(
        "FINAL HOLDOUT RANGE:",
        holdout[
            "TIME"
        ].min(),
        "->",
        holdout[
            "TIME"
        ].max(),
    )

    # --------------------------------------------------------
    # Discovery
    # --------------------------------------------------------

    candidates = (
        build_candidates()
    )

    print()
    print(
        "REGIME CANDIDATES:",
        len(candidates)
    )

    print()
    print(
        "Running DISCOVERY..."
    )

    discovery_results = (
        run_discovery(
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

    if (
        not
        discovery_results.empty
    ):

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

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    print()
    print("=" * 120)

    print(
        "SELECTION TEST"
    )

    print("=" * 120)

    selection_results = (
        run_selection(
            market,
            selection,
            discovery_results,
            candidates,
        )
    )

    if (
        not
        selection_results.empty
    ):

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
            "NO STRATEGY SURVIVED"
        )

        print(
            "TO SELECTION SAMPLE."
        )

    # --------------------------------------------------------
    # Lock
    # --------------------------------------------------------

    locked = lock_strategies(
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
            "BE USED TO RESCUE A FAILED STRATEGY."
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

    # --------------------------------------------------------
    # Final untouched holdout
    # --------------------------------------------------------

    if locked:

        print()
        print("=" * 120)

        print(
            "OPENING FINAL UNTOUCHED HOLDOUT"
        )

        print("=" * 120)

        (
            final_results,
            final_blocks,
        ) = run_final(
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

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    all_events = pd.concat(
        [
            discovery,
            selection,
            holdout,
        ],
        ignore_index=True,
    )

    all_events.to_csv(
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

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

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

        print()
        print("=" * 120)

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

        print("=" * 120)

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
                "HOLDOUT MAX DD:",
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

    print()
    print("=" * 120)

    print(
        "V17 INTERPRETATION"
    )

    print("=" * 120)

    print()

    print(
        "PASS = candidate may proceed"
    )

    print(
        "to PAPER TRADING only."
    )

    print()

    print(
        "WATCH = promising but"
    )

    print(
        "not strong enough."
    )

    print()

    print(
        "REJECT = do not use live."
    )

    print()

    print(
        "If V17 has no PASS,"
    )

    print(
        "we will not force"
    )

    print(
        "a high-win-rate strategy."
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


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
