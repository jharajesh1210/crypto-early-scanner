import math
import itertools
import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v12 as v12


# ============================================================
# COINDCX FUTURES HISTORICAL V16
# WIN / LOSS PATTERN MINING + EXPECTANCY STRATEGY BUILDER
# ============================================================
#
# PURPOSE
# -------
# V15 showed that rules which looked good in discovery
# did not survive unseen validation.
#
# V16 therefore uses THREE chronological zones:
#
#   1. DISCOVERY      = first 60%
#   2. SELECTION      = next 20%
#   3. FINAL HOLDOUT  = last 20%
#
# FINAL HOLDOUT is NEVER used for:
#
#   - threshold discovery
#   - feature selection
#   - TP/SL selection
#   - candidate ranking
#
# V16:
#
#   - separates LONG and SHORT
#   - studies WIN / LOSS / TIME_EXIT patterns
#   - discovers feature thresholds only on DISCOVERY
#   - tests limited 1-feature and 2-feature rules
#   - tests limited TP/SL combinations
#   - ranks on DISCOVERY + SELECTION
#   - locks ONE candidate per side
#   - evaluates locked candidate on FINAL HOLDOUT
#
# Historical research only.
# NO Telegram.
# NO real orders.
# ============================================================


# ============================================================
# OUTPUT FILES
# ============================================================

EVENT_FILE = "crypto_futures_v16_events.csv"

PROFILE_FILE = "crypto_futures_v16_win_loss_profile.csv"

DISCOVERY_FILE = "crypto_futures_v16_discovery.csv"

SELECTION_FILE = "crypto_futures_v16_selection.csv"

FINAL_FILE = "crypto_futures_v16_final_holdout.csv"

SUMMARY_FILE = "crypto_futures_v16_summary.csv"


# ============================================================
# HISTORY
# ============================================================

HISTORY_ATTEMPTS = [
    365,
    270,
    180,
]


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

DISCOVERY_RATIO = 0.60

SELECTION_RATIO = 0.20

HOLDOUT_RATIO = 0.20


# ============================================================
# TRADE MODEL
# ============================================================

MAX_HOLD_BARS = 48

ROUND_TRIP_COST_PCT = 0.10

SAME_BAR_POLICY = "SL_FIRST"


# ============================================================
# TP / SL GRID
# ============================================================
#
# Limited grid.
# We intentionally do NOT test hundreds of combinations.
# ============================================================

TP_GRID = [
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]

SL_GRID = [
    0.40,
    0.50,
    0.60,
    0.70,
]


# ============================================================
# DISCOVERY QUANTILES
# ============================================================

QUANTILES = [
    0.20,
    0.30,
    0.40,
    0.60,
    0.70,
    0.80,
]


# ============================================================
# SAMPLE REQUIREMENTS
# ============================================================

MIN_DISCOVERY_TRADES = 35

MIN_SELECTION_TRADES = 15

MIN_HOLDOUT_TRADES = 15

MAX_FEATURES_PER_SIDE = 4

MAX_LOCKED_CANDIDATES_PER_SIDE = 1


# ============================================================
# SELECTION REQUIREMENTS
# ============================================================

SELECTION_MIN_NET_AVG = 0.0

SELECTION_MIN_PF = 1.00

SELECTION_MIN_WIN_RATE = 50.0


# ============================================================
# FINAL PASS
# ============================================================

PASS_MIN_HOLDOUT_TRADES = 20

PASS_MIN_NET_AVG = 0.0

PASS_MIN_PF = 1.10

PASS_MIN_WIN_RATE = 52.0

PASS_MIN_POSITIVE_BLOCKS = 2

PASS_MIN_VALID_BLOCKS = 3


# ============================================================
# FINAL WATCH
# ============================================================

WATCH_MIN_HOLDOUT_TRADES = 15

WATCH_MIN_NET_AVG = 0.0

WATCH_MIN_PF = 1.00

WATCH_MIN_WIN_RATE = 50.0

WATCH_MIN_POSITIVE_BLOCKS = 2

WATCH_MIN_VALID_BLOCKS = 2


# ============================================================
# FINAL HOLDOUT BLOCKS
# ============================================================

FINAL_BLOCKS = 4


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    "RSI_5M",

    "RSI_CHANGE_1",
    "RSI_CHANGE_3",

    "MACD_HIST_5M",

    "MACD_CHANGE_1",
    "MACD_CHANGE_3",

    "EMA_DISTANCE_PCT",

    "EMA20_SLOPE_1",
    "EMA20_SLOPE_3",

    "H1_EMA_DISTANCE_PCT",

    "MACD_HIST_1H",

    "BB_POSITION",

    "BB_WIDTH_PCT",

    "VOLUME_RATIO",

    "POC_DISTANCE_PCT",

    "LOWER_WICK_RATIO",

    "UPPER_WICK_RATIO",

    "BODY_RATIO",

    "CLOSE_POSITION",

    "RANGE_PCT",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=np.nan):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def ensure_dataframe(data):

    if isinstance(data, pd.DataFrame):

        return data.copy()

    return pd.DataFrame(data)


def numeric_series(df, column):

    if column not in df.columns:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(
    wins,
    total,
    z=1.96,
):

    if total <= 0:

        return (
            np.nan,
            np.nan,
        )

    p = wins / total

    denominator = (
        1.0
        +
        z ** 2 / total
    )

    centre = (
        p
        +
        z ** 2 / (
            2.0 * total
        )
    )

    margin = (
        z
        *
        math.sqrt(
            (
                p * (1.0 - p)
                +
                z ** 2 / (
                    4.0 * total
                )
            )
            /
            total
        )
    )

    lower = (
        centre - margin
    ) / denominator

    upper = (
        centre + margin
    ) / denominator

    return (
        lower * 100.0,
        upper * 100.0,
    )


# ============================================================
# DOWNLOAD HISTORY
# ============================================================

def download_history():

    last_error = None

    for days in HISTORY_ATTEMPTS:

        try:

            print()
            print("=" * 120)

            print(
                "V16 TRYING HISTORY:",
                days,
                "DAYS"
            )

            print("=" * 120)

            print()
            print(
                "Downloading native 5-minute Futures data..."
            )

            df_5m = v1.fetch_history(
                v1.PAIR,
                "5",
                days,
                v1.CHUNK_DAYS_5M,
            )

            df_5m = ensure_dataframe(
                df_5m
            )

            if df_5m.empty:

                raise RuntimeError(
                    "No 5-minute history returned."
                )

            print()
            print(
                "Downloading native 1-hour Futures data..."
            )

            df_1h = v1.fetch_history(
                v1.PAIR,
                "60",
                days,
                v1.CHUNK_DAYS_1H,
            )

            df_1h = ensure_dataframe(
                df_1h
            )

            if df_1h.empty:

                raise RuntimeError(
                    "No 1-hour history returned."
                )

            print()
            print(
                "HISTORY DOWNLOAD SUCCESS"
            )

            print(
                "REQUESTED DAYS:",
                days,
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
                days,
            )

            print(
                "ERROR:",
                str(exc),
            )

    raise RuntimeError(
        "All history attempts failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# PREPARE INDICATORS
# ============================================================

def prepare_data(
    df_5m,
    df_1h,
):

    print()
    print(
        "Preparing base indicator stack..."
    )

    df = v12.prepare_data(
        df_5m,
        df_1h,
    )

    df = ensure_dataframe(
        df
    )

    print(
        "Adding market regime..."
    )

    df = v12.add_v12_market_regime(
        df
    )

    print(
        "Adding candle features..."
    )

    df = v12.add_v12_candle_features(
        df
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    # ========================================================
    # RSI
    # ========================================================

    df["V16_RSI_CHANGE_1"] = (

        df["rsi"]

        -

        df["rsi"].shift(1)
    )

    df["V16_RSI_CHANGE_3"] = (

        df["rsi"]

        -

        df["rsi"].shift(3)
    )

    # ========================================================
    # MACD
    # ========================================================

    df["V16_MACD_CHANGE_1"] = (

        df["macd_hist"]

        -

        df["macd_hist"].shift(1)
    )

    df["V16_MACD_CHANGE_3"] = (

        df["macd_hist"]

        -

        df["macd_hist"].shift(3)
    )

    # ========================================================
    # EMA
    # ========================================================

    df["V16_EMA_DISTANCE_PCT"] = (

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

    df["V16_EMA20_SLOPE_1"] = (

        df["ema20"]
        .pct_change(1)

        *

        100.0
    )

    df["V16_EMA20_SLOPE_3"] = (

        df["ema20"]
        .pct_change(3)

        *

        100.0
    )

    # ========================================================
    # 1H TREND
    # ========================================================

    df["V16_H1_EMA_DISTANCE_PCT"] = (

        (
            df["ema20_1h"]
            -
            df["ema50_1h"]
        )

        /

        df["close"]

        *

        100.0
    )

    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

    bb_mid = (
        df["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        df["close"]
        .rolling(20)
        .std()
    )

    bb_upper = (
        bb_mid
        +
        2.0 * bb_std
    )

    bb_lower = (
        bb_mid
        -
        2.0 * bb_std
    )

    bb_range = (
        bb_upper
        -
        bb_lower
    )

    df["V16_BB_POSITION"] = (

        (
            df["close"]
            -
            bb_lower
        )

        /

        bb_range.replace(
            0,
            np.nan,
        )
    )

    df["V16_BB_WIDTH_PCT"] = (

        bb_range

        /

        bb_mid.replace(
            0,
            np.nan,
        )

        *

        100.0
    )

    # ========================================================
    # VOLUME
    # ========================================================

    if "volume" in df.columns:

        volume_ma = (

            df["volume"]
            .rolling(20)
            .mean()
        )

        df["V16_VOLUME_RATIO"] = (

            df["volume"]

            /

            volume_ma.replace(
                0,
                np.nan,
            )
        )

    else:

        df["V16_VOLUME_RATIO"] = np.nan

    # ========================================================
    # POC DISTANCE
    # ========================================================

    if "poc" in df.columns:

        df["V16_POC_DISTANCE_PCT"] = (

            (
                df["close"]
                -
                df["poc"]
            )
            .abs()

            /

            df["close"]

            *

            100.0
        )

    elif "v10_poc_distance_pct" in df.columns:

        df["V16_POC_DISTANCE_PCT"] = (

            pd.to_numeric(
                df[
                    "v10_poc_distance_pct"
                ],
                errors="coerce",
            )
        )

    else:

        df["V16_POC_DISTANCE_PCT"] = np.nan

    # ========================================================
    # BODY / RANGE
    # ========================================================

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

    df["V16_BODY_RATIO"] = (

        body

        /

        candle_range.replace(
            0,
            np.nan,
        )
    )

    df["V16_RANGE_PCT"] = (

        candle_range

        /

        df["close"]

        *

        100.0
    )

    required = [

        "datetime",

        "open",
        "high",
        "low",
        "close",

        "rsi",

        "macd_hist",

        "ema20",
        "ema50",

        "ema20_1h",
        "ema50_1h",

        "macd_hist_1h",

        "V12_REGIME",

        "V12_CLOSE_POSITION",

        "V12_LOWER_WICK_RATIO",

        "V12_UPPER_WICK_RATIO",

        "V16_RSI_CHANGE_1",

        "V16_RSI_CHANGE_3",

        "V16_MACD_CHANGE_1",

        "V16_MACD_CHANGE_3",

        "V16_EMA_DISTANCE_PCT",

        "V16_EMA20_SLOPE_1",

        "V16_EMA20_SLOPE_3",

        "V16_H1_EMA_DISTANCE_PCT",

        "V16_BB_POSITION",

        "V16_BB_WIDTH_PCT",

        "V16_BODY_RATIO",

        "V16_RANGE_PCT",
    ]

    missing = [

        column

        for column in required

        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(

            "V16 missing columns: "

            +

            ", ".join(missing)
        )

    numeric_required = [

        column

        for column in required

        if column
        not in [
            "datetime",
            "V12_REGIME",
        ]
    ]

    df = (

        df
        .dropna(
            subset=[
                "datetime"
            ]
            +
            numeric_required
        )
        .reset_index(drop=True)
    )

    print(
        "USABLE 5M CANDLES:",
        len(df),
    )

    return df


# ============================================================
# TRADE SIMULATOR
# ============================================================

def simulate_trade(
    data,
    index,
    side,
    tp_pct,
    sl_pct,
):

    entry = safe_float(
        data.iloc[index]["close"]
    )

    if (
        pd.isna(entry)
        or
        entry <= 0
    ):

        return None

    if side == "LONG":

        tp_price = (

            entry

            *

            (
                1.0
                +
                tp_pct / 100.0
            )
        )

        sl_price = (

            entry

            *

            (
                1.0
                -
                sl_pct / 100.0
            )
        )

    else:

        tp_price = (

            entry

            *

            (
                1.0
                -
                tp_pct / 100.0
            )
        )

        sl_price = (

            entry

            *

            (
                1.0
                +
                sl_pct / 100.0
            )
        )

    end_index = min(

        index
        +
        MAX_HOLD_BARS,

        len(data) - 1,
    )

    if end_index <= index:

        return None

    future = (

        data
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
# COLLECT BASE SIGNALS
# ============================================================
#
# We initially simulate 0.50 / 0.50 simply to create
# WIN / LOSS / TIME_EXIT labels for pattern mining.
#
# Later TP/SL optimization is done separately.
# ============================================================

def collect_base_events(df):

    events = []

    last_index = (

        len(df)

        -

        MAX_HOLD_BARS

        -

        1
    )

    for i in range(
        100,
        last_index,
    ):

        row = df.iloc[i]

        regime = str(
            row["V12_REGIME"]
        )

        side = None

        if (

            regime == "UPTREND"

            and

            v12.long_setup(
                df,
                i,
            )
        ):

            side = "LONG"

        elif (

            regime == "DOWNTREND"

            and

            v12.short_setup(
                df,
                i,
            )
        ):

            side = "SHORT"

        if side is None:

            continue

        trade = simulate_trade(

            df,

            i,

            side,

            0.50,

            0.50,
        )

        if trade is None:

            continue

        event = {

            "DATA_INDEX":
                i,

            "TIME":
                row[
                    "datetime"
                ],

            "SIDE":
                side,

            "REGIME":
                regime,

            "ENTRY_PRICE":
                safe_float(
                    row["close"]
                ),

            # RSI

            "RSI_5M":
                safe_float(
                    row["rsi"]
                ),

            "RSI_CHANGE_1":
                safe_float(
                    row[
                        "V16_RSI_CHANGE_1"
                    ]
                ),

            "RSI_CHANGE_3":
                safe_float(
                    row[
                        "V16_RSI_CHANGE_3"
                    ]
                ),

            # MACD

            "MACD_HIST_5M":
                safe_float(
                    row["macd_hist"]
                ),

            "MACD_CHANGE_1":
                safe_float(
                    row[
                        "V16_MACD_CHANGE_1"
                    ]
                ),

            "MACD_CHANGE_3":
                safe_float(
                    row[
                        "V16_MACD_CHANGE_3"
                    ]
                ),

            # EMA

            "EMA_DISTANCE_PCT":
                safe_float(
                    row[
                        "V16_EMA_DISTANCE_PCT"
                    ]
                ),

            "EMA20_SLOPE_1":
                safe_float(
                    row[
                        "V16_EMA20_SLOPE_1"
                    ]
                ),

            "EMA20_SLOPE_3":
                safe_float(
                    row[
                        "V16_EMA20_SLOPE_3"
                    ]
                ),

            # 1H

            "H1_EMA_DISTANCE_PCT":
                safe_float(
                    row[
                        "V16_H1_EMA_DISTANCE_PCT"
                    ]
                ),

            "MACD_HIST_1H":
                safe_float(
                    row[
                        "macd_hist_1h"
                    ]
                ),

            # BB

            "BB_POSITION":
                safe_float(
                    row[
                        "V16_BB_POSITION"
                    ]
                ),

            "BB_WIDTH_PCT":
                safe_float(
                    row[
                        "V16_BB_WIDTH_PCT"
                    ]
                ),

            # VOLUME

            "VOLUME_RATIO":
                safe_float(
                    row[
                        "V16_VOLUME_RATIO"
                    ]
                ),

            # POC

            "POC_DISTANCE_PCT":
                safe_float(
                    row[
                        "V16_POC_DISTANCE_PCT"
                    ]
                ),

            # CANDLE

            "LOWER_WICK_RATIO":
                safe_float(
                    row[
                        "V12_LOWER_WICK_RATIO"
                    ]
                ),

            "UPPER_WICK_RATIO":
                safe_float(
                    row[
                        "V12_UPPER_WICK_RATIO"
                    ]
                ),

            "BODY_RATIO":
                safe_float(
                    row[
                        "V16_BODY_RATIO"
                    ]
                ),

            "CLOSE_POSITION":
                safe_float(
                    row[
                        "V12_CLOSE_POSITION"
                    ]
                ),

            "RANGE_PCT":
                safe_float(
                    row[
                        "V16_RANGE_PCT"
                    ]
                ),
        }

        event.update(
            trade
        )

        events.append(
            event
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
        .iloc[
            :discovery_end
        ]
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
        discovery,
        selection,
        holdout,
    )


# ============================================================
# WIN / LOSS PROFILE
# ============================================================

def build_win_loss_profile(
    discovery,
):

    rows = []

    for side in [
        "LONG",
        "SHORT",
    ]:

        side_data = (

            discovery[
                discovery["SIDE"]
                ==
                side
            ]
            .copy()
        )

        for feature in FEATURES:

            if feature not in side_data.columns:

                continue

            for outcome in [
                "WIN",
                "LOSS",
                "TIME_EXIT",
            ]:

                values = (

                    pd.to_numeric(
                        side_data.loc[
                            side_data[
                                "OUTCOME"
                            ]
                            ==
                            outcome,
                            feature,
                        ],
                        errors="coerce",
                    )
                    .dropna()
                )

                if values.empty:

                    continue

                rows.append({

                    "SIDE":
                        side,

                    "FEATURE":
                        feature,

                    "OUTCOME":
                        outcome,

                    "COUNT":
                        len(values),

                    "MEAN":
                        values.mean(),

                    "MEDIAN":
                        values.median(),

                    "Q20":
                        values.quantile(
                            0.20
                        ),

                    "Q40":
                        values.quantile(
                            0.40
                        ),

                    "Q60":
                        values.quantile(
                            0.60
                        ),

                    "Q80":
                        values.quantile(
                            0.80
                        ),
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# PERFORMANCE
# ============================================================

def performance(df):

    if df.empty:

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

        wilson_low, wilson_high = (
            wilson_interval(
                wins,
                decisive,
            )
        )

    else:

        win_rate = np.nan
        wilson_low = np.nan
        wilson_high = np.nan

    returns = (

        pd.to_numeric(
            df[
                "NET_RETURN_%"
            ],
            errors="coerce",
        )
        .dropna()
    )

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
            )
            if not returns.empty
            else np.nan,

        "NET_MEDIAN_RETURN_%":
            round(
                returns.median(),
                4,
            )
            if not returns.empty
            else np.nan,

        "PROFIT_FACTOR":
            round(
                pf,
                4,
            ),

        "WILSON_LOW_%":
            round(
                wilson_low,
                2,
            )
            if pd.notna(
                wilson_low
            )
            else np.nan,

        "WILSON_HIGH_%":
            round(
                wilson_high,
                2,
            )
            if pd.notna(
                wilson_high
            )
            else np.nan,
    }


# ============================================================
# APPLY FEATURE RULE
# ============================================================

def apply_rule(
    df,
    feature,
    operator,
    threshold,
):

    values = numeric_series(
        df,
        feature,
    )

    if operator == ">=":

        return (
            values
            >=
            threshold
        ).fillna(False)

    return (
        values
        <=
        threshold
    ).fillna(False)


# ============================================================
# DISCOVER SINGLE FEATURE RULES
# ============================================================

def discover_rules(
    discovery,
    side,
):

    side_data = (

        discovery[
            discovery["SIDE"]
            ==
            side
        ]
        .copy()
    )

    rows = []

    for feature in FEATURES:

        if feature not in side_data.columns:

            continue

        values = (

            pd.to_numeric(
                side_data[
                    feature
                ],
                errors="coerce",
            )
            .dropna()
        )

        if (
            len(values)
            <
            MIN_DISCOVERY_TRADES
        ):

            continue

        for quantile in QUANTILES:

            threshold = float(

                values.quantile(
                    quantile
                )
            )

            for operator in [
                ">=",
                "<=",
            ]:

                mask = apply_rule(

                    side_data,

                    feature,

                    operator,

                    threshold,
                )

                sample = (

                    side_data[
                        mask
                    ]
                    .copy()
                )

                if (
                    len(sample)
                    <
                    MIN_DISCOVERY_TRADES
                ):

                    continue

                stats = performance(
                    sample
                )

                if stats is None:

                    continue

                rows.append({

                    "SIDE":
                        side,

                    "FEATURE":
                        feature,

                    "OPERATOR":
                        operator,

                    "QUANTILE":
                        quantile,

                    "THRESHOLD":
                        threshold,

                    "TRADES":
                        stats[
                            "TRADES"
                        ],

                    "WIN_RATE_%":
                        stats[
                            "WIN_RATE_%"
                        ],

                    "NET_AVG_RETURN_%":
                        stats[
                            "NET_AVG_RETURN_%"
                        ],

                    "PROFIT_FACTOR":
                        stats[
                            "PROFIT_FACTOR"
                        ],

                    "WILSON_LOW_%":
                        stats[
                            "WILSON_LOW_%"
                        ],
                })

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    result = (

        result
        .sort_values(
            by=[
                "NET_AVG_RETURN_%",
                "PROFIT_FACTOR",
                "WIN_RATE_%",
                "TRADES",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# SELECT TOP DIFFERENT FEATURES
# ============================================================

def select_top_features(
    rules,
):

    if rules.empty:

        return []

    selected = []

    used = set()

    for _, row in rules.iterrows():

        feature = str(
            row["FEATURE"]
        )

        if feature in used:

            continue

        selected.append({

            "FEATURE":
                feature,

            "OPERATOR":
                str(
                    row[
                        "OPERATOR"
                    ]
                ),

            "THRESHOLD":
                float(
                    row[
                        "THRESHOLD"
                    ]
                ),
        })

        used.add(
            feature
        )

        if (
            len(selected)
            >=
            MAX_FEATURES_PER_SIDE
        ):

            break

    return selected


# ============================================================
# BUILD LIMITED CANDIDATES
# ============================================================

def build_candidates(
    side,
    selected_rules,
):

    candidates = []

    # BASE
    candidates.append({

        "NAME":
            f"V16_{side}_BASE",

        "SIDE":
            side,

        "RULES":
            [],
    })

    # SINGLE RULES
    for i, rule in enumerate(
        selected_rules,
        start=1,
    ):

        candidates.append({

            "NAME":
                f"V16_{side}_R{i}",

            "SIDE":
                side,

            "RULES":
                [
                    rule
                ],
        })

    # TWO-FEATURE RULES ONLY
    for (
        first_index,
        second_index,
    ) in itertools.combinations(
        range(
            len(selected_rules)
        ),
        2,
    ):

        candidates.append({

            "NAME":
                (
                    f"V16_{side}_"
                    f"R{first_index + 1}_"
                    f"R{second_index + 1}"
                ),

            "SIDE":
                side,

            "RULES":
                [
                    selected_rules[
                        first_index
                    ],
                    selected_rules[
                        second_index
                    ],
                ],
        })

    return candidates


# ============================================================
# FILTER EVENTS BY CANDIDATE
# ============================================================

def filter_events(
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

    for rule in candidate[
        "RULES"
    ]:

        mask = apply_rule(

            filtered,

            rule[
                "FEATURE"
            ],

            rule[
                "OPERATOR"
            ],

            rule[
                "THRESHOLD"
            ],
        )

        filtered = (

            filtered[
                mask
            ]
            .copy()
        )

        if filtered.empty:

            break

    return filtered


# ============================================================
# RULE DESCRIPTION
# ============================================================

def describe_rules(
    candidate,
):

    if not candidate[
        "RULES"
    ]:

        return "BASE"

    return " AND ".join(

        [

            (
                f'{rule["FEATURE"]} '
                f'{rule["OPERATOR"]} '
                f'{rule["THRESHOLD"]:.6f}'
            )

            for rule
            in candidate[
                "RULES"
            ]
        ]
    )


# ============================================================
# RE-SIMULATE FILTERED EVENTS
# ============================================================

def resimulate_events(
    market_data,
    events,
    side,
    tp,
    sl,
):

    rows = []

    for _, event in events.iterrows():

        index = int(
            event[
                "DATA_INDEX"
            ]
        )

        trade = simulate_trade(

            market_data,

            index,

            side,

            tp,

            sl,
        )

        if trade is None:

            continue

        row = event.to_dict()

        row.update(
            trade
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DISCOVERY + TP/SL TEST
# ============================================================

def test_candidates_on_discovery(
    market_data,
    discovery,
    candidates,
):

    rows = []

    for candidate in candidates:

        filtered = filter_events(
            discovery,
            candidate,
        )

        if (
            len(filtered)
            <
            MIN_DISCOVERY_TRADES
        ):

            continue

        for tp in TP_GRID:

            for sl in SL_GRID:

                simulated = resimulate_events(

                    market_data,

                    filtered,

                    candidate["SIDE"],

                    tp,

                    sl,
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

                rows.append({

                    "CANDIDATE":
                        candidate[
                            "NAME"
                        ],

                    "SIDE":
                        candidate[
                            "SIDE"
                        ],

                    "RULES":
                        describe_rules(
                            candidate
                        ),

                    "TP_%":
                        tp,

                    "SL_%":
                        sl,

                    "TRADES":
                        stats[
                            "TRADES"
                        ],

                    "DECISIVE":
                        stats[
                            "DECISIVE"
                        ],

                    "WIN_RATE_%":
                        stats[
                            "WIN_RATE_%"
                        ],

                    "NET_AVG_RETURN_%":
                        stats[
                            "NET_AVG_RETURN_%"
                        ],

                    "NET_MEDIAN_RETURN_%":
                        stats[
                            "NET_MEDIAN_RETURN_%"
                        ],

                    "PROFIT_FACTOR":
                        stats[
                            "PROFIT_FACTOR"
                        ],

                    "WILSON_LOW_%":
                        stats[
                            "WILSON_LOW_%"
                        ],
                })

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
                    "WIN_RATE_%",
                    "TRADES",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(drop=True)
        )

    return result


# ============================================================
# GET CANDIDATE OBJECT
# ============================================================

def find_candidate(
    candidates,
    name,
):

    for candidate in candidates:

        if (
            candidate[
                "NAME"
            ]
            ==
            name
        ):

            return candidate

    return None


# ============================================================
# TEST DISCOVERY WINNERS ON SELECTION
# ============================================================

def selection_test(
    market_data,
    selection,
    discovery_results,
    candidates,
):

    if discovery_results.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Only a limited number of discovery configurations
    # are allowed to reach selection.
    # --------------------------------------------------------

    shortlist = (

        discovery_results
        .groupby(
            "SIDE",
            group_keys=False,
        )
        .head(15)
        .reset_index(drop=True)
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

        filtered = filter_events(

            selection,

            candidate,
        )

        if (
            len(filtered)
            <
            MIN_SELECTION_TRADES
        ):

            continue

        simulated = resimulate_events(

            market_data,

            filtered,

            candidate[
                "SIDE"
            ],

            float(
                config[
                    "TP_%"
                ]
            ),

            float(
                config[
                    "SL_%"
                ]
            ),
        )

        stats = performance(
            simulated
        )

        if stats is None:

            continue

        rows.append({

            "CANDIDATE":
                candidate[
                    "NAME"
                ],

            "SIDE":
                candidate[
                    "SIDE"
                ],

            "RULES":
                describe_rules(
                    candidate
                ),

            "TP_%":
                float(
                    config[
                        "TP_%"
                    ]
                ),

            "SL_%":
                float(
                    config[
                        "SL_%"
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

            "DISCOVERY_NET_AVG_RETURN_%":
                config[
                    "NET_AVG_RETURN_%"
                ],

            "DISCOVERY_PROFIT_FACTOR":
                config[
                    "PROFIT_FACTOR"
                ],

            "SELECTION_TRADES":
                stats[
                    "TRADES"
                ],

            "SELECTION_DECISIVE":
                stats[
                    "DECISIVE"
                ],

            "SELECTION_WIN_RATE_%":
                stats[
                    "WIN_RATE_%"
                ],

            "SELECTION_NET_AVG_RETURN_%":
                stats[
                    "NET_AVG_RETURN_%"
                ],

            "SELECTION_PROFIT_FACTOR":
                stats[
                    "PROFIT_FACTOR"
                ],

            "SELECTION_WILSON_LOW_%":
                stats[
                    "WILSON_LOW_%"
                ],
        })

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = (

            result
            .sort_values(
                by=[
                    "SELECTION_NET_AVG_RETURN_%",
                    "SELECTION_PROFIT_FACTOR",
                    "SELECTION_WIN_RATE_%",
                    "SELECTION_TRADES",
                ],
                ascending=[
                    False,
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

def lock_final_candidates(
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
                    MIN_SELECTION_TRADES
                )
                &
                (
                    side_results[
                        "SELECTION_NET_AVG_RETURN_%"
                    ]
                    >
                    SELECTION_MIN_NET_AVG
                )
                &
                (
                    side_results[
                        "SELECTION_PROFIT_FACTOR"
                    ]
                    >=
                    SELECTION_MIN_PF
                )
                &
                (
                    side_results[
                        "SELECTION_WIN_RATE_%"
                    ]
                    >=
                    SELECTION_MIN_WIN_RATE
                )
            ]
            .copy()
        )

        if eligible.empty:

            # No profitable candidate reaches holdout.
            continue

        eligible = (

            eligible
            .sort_values(
                by=[
                    "SELECTION_NET_AVG_RETURN_%",
                    "SELECTION_PROFIT_FACTOR",
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

        best = eligible.iloc[0]

        locked.append(
            best.to_dict()
        )

    return locked


# ============================================================
# CALENDAR BLOCKS
# ============================================================

def calendar_blocks(
    data,
    number_of_blocks,
):

    if data.empty:

        return []

    data = (

        data
        .sort_values("TIME")
        .copy()
    )

    times = pd.to_datetime(
        data["TIME"],
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
                data,
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

        if i == (
            number_of_blocks - 1
        ):

            mask = (

                (times >= edges[i])

                &

                (times <= edges[i + 1])
            )

        else:

            mask = (

                (times >= edges[i])

                &

                (times < edges[i + 1])
            )

        blocks.append(

            (
                i + 1,

                data[
                    mask
                ].copy(),
            )
        )

    return blocks


# ============================================================
# FINAL HOLDOUT TEST
# ============================================================

def final_holdout_test(
    market_data,
    holdout,
    locked,
    candidates,
):

    rows = []

    block_rows = []

    for locked_config in locked:

        candidate = find_candidate(

            candidates,

            locked_config[
                "CANDIDATE"
            ],
        )

        if candidate is None:

            continue

        filtered = filter_events(

            holdout,

            candidate,
        )

        tp = float(
            locked_config[
                "TP_%"
            ]
        )

        sl = float(
            locked_config[
                "SL_%"
            ]
        )

        simulated = resimulate_events(

            market_data,

            filtered,

            candidate[
                "SIDE"
            ],

            tp,

            sl,
        )

        stats = performance(
            simulated
        )

        if stats is None:

            continue

        positive_blocks = 0

        valid_blocks = 0

        block_wrs = []

        for (
            block_number,
            block_data,
        ) in calendar_blocks(
            holdout,
            FINAL_BLOCKS,
        ):

            block_filtered = (
                filter_events(
                    block_data,
                    candidate,
                )
            )

            block_simulated = (
                resimulate_events(

                    market_data,

                    block_filtered,

                    candidate[
                        "SIDE"
                    ],

                    tp,

                    sl,
                )
            )

            block_stats = performance(
                block_simulated
            )

            if block_stats is None:

                block_rows.append({

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

                    "WIN_RATE_%":
                        np.nan,

                    "NET_AVG_RETURN_%":
                        np.nan,

                    "PROFIT_FACTOR":
                        np.nan,
                })

                continue

            if (
                block_stats[
                    "DECISIVE"
                ]
                >
                0
            ):

                valid_blocks += 1

            if (
                pd.notna(
                    block_stats[
                        "NET_AVG_RETURN_%"
                    ]
                )

                and

                block_stats[
                    "NET_AVG_RETURN_%"
                ]
                >
                0
            ):

                positive_blocks += 1

            if pd.notna(
                block_stats[
                    "WIN_RATE_%"
                ]
            ):

                block_wrs.append(

                    block_stats[
                        "WIN_RATE_%"
                    ]
                )

            block_rows.append({

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
            })

        if block_wrs:

            min_block_wr = float(
                np.min(
                    block_wrs
                )
            )

            median_block_wr = float(
                np.median(
                    block_wrs
                )
            )

            block_std = float(
                np.std(
                    block_wrs
                )
            )

        else:

            min_block_wr = np.nan

            median_block_wr = np.nan

            block_std = np.nan

        row = {

            "CANDIDATE":
                candidate[
                    "NAME"
                ],

            "SIDE":
                candidate[
                    "SIDE"
                ],

            "RULES":
                describe_rules(
                    candidate
                ),

            "TP_%":
                tp,

            "SL_%":
                sl,

            "DISCOVERY_TRADES":
                locked_config[
                    "DISCOVERY_TRADES"
                ],

            "DISCOVERY_WIN_RATE_%":
                locked_config[
                    "DISCOVERY_WIN_RATE_%"
                ],

            "DISCOVERY_NET_AVG_RETURN_%":
                locked_config[
                    "DISCOVERY_NET_AVG_RETURN_%"
                ],

            "DISCOVERY_PROFIT_FACTOR":
                locked_config[
                    "DISCOVERY_PROFIT_FACTOR"
                ],

            "SELECTION_TRADES":
                locked_config[
                    "SELECTION_TRADES"
                ],

            "SELECTION_WIN_RATE_%":
                locked_config[
                    "SELECTION_WIN_RATE_%"
                ],

            "SELECTION_NET_AVG_RETURN_%":
                locked_config[
                    "SELECTION_NET_AVG_RETURN_%"
                ],

            "SELECTION_PROFIT_FACTOR":
                locked_config[
                    "SELECTION_PROFIT_FACTOR"
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

            "HOLDOUT_NET_AVG_RETURN_%":
                stats[
                    "NET_AVG_RETURN_%"
                ],

            "HOLDOUT_NET_MEDIAN_RETURN_%":
                stats[
                    "NET_MEDIAN_RETURN_%"
                ],

            "HOLDOUT_PROFIT_FACTOR":
                stats[
                    "PROFIT_FACTOR"
                ],

            "HOLDOUT_WILSON_LOW_%":
                stats[
                    "WILSON_LOW_%"
                ],

            "HOLDOUT_WILSON_HIGH_%":
                stats[
                    "WILSON_HIGH_%"
                ],

            "VALID_BLOCKS":
                valid_blocks,

            "POSITIVE_BLOCKS":
                positive_blocks,

            "MIN_BLOCK_WIN_RATE_%":
                round(
                    min_block_wr,
                    2,
                )
                if pd.notna(
                    min_block_wr
                )
                else np.nan,

            "MEDIAN_BLOCK_WIN_RATE_%":
                round(
                    median_block_wr,
                    2,
                )
                if pd.notna(
                    median_block_wr
                )
                else np.nan,

            "BLOCK_WIN_RATE_STD_%":
                round(
                    block_std,
                    2,
                )
                if pd.notna(
                    block_std
                )
                else np.nan,
        }

        row[
            "STATUS"
        ] = classify_final(
            row
        )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            block_rows
        ),
    )


# ============================================================
# FINAL CLASSIFICATION
# ============================================================

def classify_final(row):

    trades = int(
        row[
            "HOLDOUT_TRADES"
        ]
    )

    wr = safe_float(
        row[
            "HOLDOUT_WIN_RATE_%"
        ]
    )

    net_avg = safe_float(
        row[
            "HOLDOUT_NET_AVG_RETURN_%"
        ]
    )

    pf = safe_float(
        row[
            "HOLDOUT_PROFIT_FACTOR"
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
        PASS_MIN_HOLDOUT_TRADES

        and

        pd.notna(wr)

        and

        wr
        >=
        PASS_MIN_WIN_RATE

        and

        pd.notna(net_avg)

        and

        net_avg
        >
        PASS_MIN_NET_AVG

        and

        pd.notna(pf)

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
    ):

        return "PASS"

    if (

        trades
        >=
        WATCH_MIN_HOLDOUT_TRADES

        and

        pd.notna(wr)

        and

        wr
        >=
        WATCH_MIN_WIN_RATE

        and

        pd.notna(net_avg)

        and

        net_avg
        >
        WATCH_MIN_NET_AVG

        and

        pd.notna(pf)

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
# PRINT LOCKED CONFIGURATION
# ============================================================

def print_locked(
    locked,
):

    print()
    print("=" * 150)

    print(
        "V16 LOCKED CANDIDATES BEFORE FINAL HOLDOUT"
    )

    print("=" * 150)

    if not locked:

        print()
        print(
            "NO candidate survived SELECTION."
        )

        print()
        print(
            "FINAL HOLDOUT WILL NOT BE USED"
        )

        print(
            "TO RESCUE A FAILED STRATEGY."
        )

        return

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
            "RULES:",
            config[
                "RULES"
            ]
        )

        print(
            "TP:",
            config[
                "TP_%"
            ],
            "%"
        )

        print(
            "SL:",
            config[
                "SL_%"
            ],
            "%"
        )

        print(
            "SELECTION TRADES:",
            config[
                "SELECTION_TRADES"
            ]
        )

        print(
            "SELECTION WIN RATE:",
            config[
                "SELECTION_WIN_RATE_%"
            ],
            "%"
        )

        print(
            "SELECTION NET AVG:",
            config[
                "SELECTION_NET_AVG_RETURN_%"
            ],
            "%"
        )

        print(
            "SELECTION PF:",
            config[
                "SELECTION_PROFIT_FACTOR"
            ]
        )


# ============================================================
# PRINT FINAL
# ============================================================

def print_final(
    final_results,
):

    print()
    print("=" * 180)

    print(
        "V16 FINAL UNTOUCHED HOLDOUT RESULTS"
    )

    print("=" * 180)

    if final_results.empty:

        print()
        print(
            "NO FINAL HOLDOUT CANDIDATE."
        )

        print()
        print(
            "V16 RESULT: NO STRATEGY"
        )

        return

    columns = [

        "CANDIDATE",

        "STATUS",

        "SIDE",

        "RULES",

        "TP_%",

        "SL_%",

        "SELECTION_TRADES",

        "SELECTION_WIN_RATE_%",

        "SELECTION_NET_AVG_RETURN_%",

        "SELECTION_PROFIT_FACTOR",

        "HOLDOUT_TRADES",

        "HOLDOUT_WIN_RATE_%",

        "HOLDOUT_NET_AVG_RETURN_%",

        "HOLDOUT_PROFIT_FACTOR",

        "HOLDOUT_WILSON_LOW_%",

        "POSITIVE_BLOCKS",

        "VALID_BLOCKS",

        "MIN_BLOCK_WIN_RATE_%",

        "BLOCK_WIN_RATE_STD_%",
    ]

    print()

    print(
        final_results[
            columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 180)

    for _, row in final_results.iterrows():

        print()

        print(
            "CANDIDATE:",
            row[
                "CANDIDATE"
            ]
        )

        print(
            "FINAL STATUS:",
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
            "RULES:",
            row[
                "RULES"
            ]
        )

        print(
            "TP / SL:",
            row[
                "TP_%"
            ],
            "/",
            row[
                "SL_%"
            ]
        )

        print()

        print(
            "FINAL HOLDOUT TRADES:",
            row[
                "HOLDOUT_TRADES"
            ]
        )

        print(
            "FINAL HOLDOUT WIN RATE:",
            row[
                "HOLDOUT_WIN_RATE_%"
            ],
            "%"
        )

        print(
            "FINAL HOLDOUT NET AVG:",
            row[
                "HOLDOUT_NET_AVG_RETURN_%"
            ],
            "%"
        )

        print(
            "FINAL HOLDOUT PROFIT FACTOR:",
            row[
                "HOLDOUT_PROFIT_FACTOR"
            ]
        )

        print(
            "95% WILSON LOW:",
            row[
                "HOLDOUT_WILSON_LOW_%"
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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 140)

    print(
        "COINDCX FUTURES HISTORICAL V16"
    )

    print(
        "WIN / LOSS PATTERN MINING"
    )

    print(
        "+ EXPECTANCY STRATEGY BUILDER"
    )

    print("=" * 140)

    print(
        "PAIR:",
        v1.PAIR,
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
        "DISCOVERY:",
        "60%"
    )

    print(
        "SELECTION:",
        "20%"
    )

    print(
        "FINAL HOLDOUT:",
        "20%"
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
        "x 5-minute candles"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "FINAL HOLDOUT WILL NOT BE USED"
    )

    print(
        "FOR RULE OR TP/SL DISCOVERY."
    )

    print()
    print(
        "NO REAL ORDERS WILL BE PLACED."
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    (
        df_5m,
        df_1h,
        history_days,
    ) = download_history()

    # ========================================================
    # PREPARE
    # ========================================================

    market_data = prepare_data(
        df_5m,
        df_1h,
    )

    print()
    print(
        "REGIME COUNTS:"
    )

    print(
        market_data[
            "V12_REGIME"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # BASE EVENTS
    # ========================================================

    print()
    print(
        "Collecting V16 base events..."
    )

    events = collect_base_events(
        market_data
    )

    if events.empty:

        print(
            "No V16 events found."
        )

        return

    print()
    print(
        "TOTAL EVENTS:",
        len(events),
    )

    print(
        "LONG:",
        (
            events["SIDE"]
            ==
            "LONG"
        ).sum(),
    )

    print(
        "SHORT:",
        (
            events["SIDE"]
            ==
            "SHORT"
        ).sum(),
    )

    print()
    print(
        "BASE OUTCOMES:"
    )

    print(
        events[
            "OUTCOME"
        ]
        .value_counts()
        .to_string()
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
    print("=" * 140)

    print(
        "V16 CHRONOLOGICAL SPLIT"
    )

    print("=" * 140)

    print(
        "DISCOVERY EVENTS:",
        len(discovery),
    )

    print(
        "SELECTION EVENTS:",
        len(selection),
    )

    print(
        "FINAL HOLDOUT EVENTS:",
        len(holdout),
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

    # ========================================================
    # WIN LOSS PROFILE
    # ========================================================

    print()
    print(
        "Building WIN / LOSS / TIME_EXIT profile"
    )

    print(
        "using DISCOVERY data only..."
    )

    profile = build_win_loss_profile(
        discovery
    )

    # ========================================================
    # DISCOVER LONG
    # ========================================================

    print()
    print(
        "Discovering LONG thresholds..."
    )

    long_rules = discover_rules(

        discovery,

        "LONG",
    )

    long_selected = select_top_features(
        long_rules
    )

    # ========================================================
    # DISCOVER SHORT
    # ========================================================

    print()
    print(
        "Discovering SHORT thresholds..."
    )

    short_rules = discover_rules(

        discovery,

        "SHORT",
    )

    short_selected = select_top_features(
        short_rules
    )

    print()
    print("=" * 140)

    print(
        "V16 SELECTED DISCOVERY FEATURES"
    )

    print("=" * 140)

    print()
    print(
        "LONG:"
    )

    for i, rule in enumerate(
        long_selected,
        start=1,
    ):

        print(

            f'R{i}: '
            f'{rule["FEATURE"]} '
            f'{rule["OPERATOR"]} '
            f'{rule["THRESHOLD"]:.6f}'
        )

    print()
    print(
        "SHORT:"
    )

    for i, rule in enumerate(
        short_selected,
        start=1,
    ):

        print(

            f'R{i}: '
            f'{rule["FEATURE"]} '
            f'{rule["OPERATOR"]} '
            f'{rule["THRESHOLD"]:.6f}'
        )

    # ========================================================
    # CANDIDATES
    # ========================================================

    long_candidates = build_candidates(

        "LONG",

        long_selected,
    )

    short_candidates = build_candidates(

        "SHORT",

        short_selected,
    )

    candidates = (

        long_candidates

        +

        short_candidates
    )

    print()
    print(
        "TOTAL LIMITED CANDIDATES:",
        len(candidates),
    )

    # ========================================================
    # DISCOVERY TP/SL
    # ========================================================

    print()
    print(
        "Testing limited TP/SL grid"
    )

    print(
        "on DISCOVERY data only..."
    )

    discovery_results = (
        test_candidates_on_discovery(

            market_data,

            discovery,

            candidates,
        )
    )

    print()
    print(
        "DISCOVERY CONFIGURATIONS TESTED:",
        len(
            discovery_results
        ),
    )

    if not discovery_results.empty:

        print()
        print(
            "TOP DISCOVERY CONFIGURATIONS:"
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
        "Testing discovery shortlist"
    )

    print(
        "on SEPARATE SELECTION data..."
    )

    selection_results = selection_test(

        market_data,

        selection,

        discovery_results,

        candidates,
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
            "No candidate had enough"
        )

        print(
            "SELECTION trades."
        )

    # ========================================================
    # LOCK
    # ========================================================

    locked = lock_final_candidates(
        selection_results
    )

    print_locked(
        locked
    )

    # ========================================================
    # FINAL HOLDOUT
    # ========================================================

    if locked:

        print()
        print(
            "NOW opening FINAL UNTOUCHED HOLDOUT..."
        )

        (
            final_results,
            final_blocks,
        ) = final_holdout_test(

            market_data,

            holdout,

            locked,

            candidates,
        )

    else:

        final_results = pd.DataFrame()

        final_blocks = pd.DataFrame()

    # ========================================================
    # SAVE EVENTS
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

    profile.to_csv(
        PROFILE_FILE,
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
        FINAL_FILE,
        index=False,
    )

    final_results.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 140)

    print(
        "V16 FILES SAVED"
    )

    print("=" * 140)

    print(
        EVENT_FILE
    )

    print(
        PROFILE_FILE
    )

    print(
        DISCOVERY_FILE
    )

    print(
        SELECTION_FILE
    )

    print(
        FINAL_FILE
    )

    print(
        SUMMARY_FILE
    )

    # ========================================================
    # FINAL PRINT
    # ========================================================

    print_final(
        final_results
    )

    print()
    print("=" * 140)

    print(
        "V16 INTERPRETATION"
    )

    print("=" * 140)

    print()

    print(
        "DISCOVERY was used to find"
    )

    print(
        "possible patterns and TP/SL."
    )

    print()

    print(
        "SELECTION was used to decide"
    )

    print(
        "whether a discovered setup"
    )

    print(
        "deserved final testing."
    )

    print()

    print(
        "FINAL HOLDOUT was untouched"
    )

    print(
        "until the strategy was locked."
    )

    print()

    print(
        "If FINAL STATUS = REJECT,"
    )

    print(
        "we do NOT change the rule"
    )

    print(
        "after looking at holdout."
    )

    print()

    print(
        "If FINAL STATUS = WATCH,"
    )

    print(
        "the setup needs more data"
    )

    print(
        "and paper testing."
    )

    print()

    print(
        "If FINAL STATUS = PASS,"
    )

    print(
        "it is only a candidate"
    )

    print(
        "for paper trading."
    )

    print()

    print(
        "PASS does NOT mean guaranteed"
    )

    print(
        "future profitability."
    )

    print()

    print(
        "NO REAL ORDERS WERE PLACED."
    )

    print()
    print("=" * 140)

    print(
        "FUTURES HISTORICAL V16 COMPLETE"
    )

    print("=" * 140)


if __name__ == "__main__":

    main()
