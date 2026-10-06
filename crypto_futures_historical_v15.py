import math
import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v12 as v12


# ============================================================
# COINDCX FUTURES HISTORICAL V15
# DATA-DRIVEN STRATEGY DISCOVERY ENGINE
# ============================================================
#
# V15 PURPOSE
# -----------
# V14 showed that manually selected filters did not create
# a profitable robust edge.
#
# V15 changes the method:
#
# 1. Download 365 days of BTC Futures data
# 2. Build LONG and SHORT base signals
# 3. Split data chronologically:
#
#       DISCOVERY = first 70%
#       VALIDATION = last 30%
#
# 4. Learn useful feature thresholds ONLY from DISCOVERY
# 5. Lock those thresholds
# 6. Apply them unchanged to VALIDATION
# 7. Measure:
#
#       Win Rate
#       Net Average Return
#       Profit Factor
#       MFE
#       MAE
#       Wilson Confidence
#       Validation consistency
#
# IMPORTANT:
#
# Validation data is NOT used for threshold discovery.
#
# Historical research only.
# No Telegram.
# No real orders.
# ============================================================


EVENT_FILE = "crypto_futures_v15_events.csv"
DISCOVERY_FILE = "crypto_futures_v15_discovery.csv"
RESULT_FILE = "crypto_futures_v15_results.csv"
SUMMARY_FILE = "crypto_futures_v15_summary.csv"


# ============================================================
# HISTORY
# ============================================================

HISTORY_ATTEMPTS = [
    365,
    270,
    180,
]

DISCOVERY_RATIO = 0.70

VALIDATION_BLOCKS = 4


# ============================================================
# FIXED TRADE MODEL
# ============================================================
#
# V15 is ENTRY discovery.
# We therefore keep ONE fixed exit model.
# ============================================================

TP_PCT = 0.50
SL_PCT = 0.50

MAX_HOLD_BARS = 48

ROUND_TRIP_COST_PCT = 0.10

SAME_BAR_POLICY = "SL_FIRST"


# ============================================================
# DISCOVERY SETTINGS
# ============================================================

QUANTILES = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]

MIN_DISCOVERY_TRADES = 40

MIN_VALIDATION_TRADES = 20

MAX_SELECTED_FEATURES = 3

MIN_DISCOVERY_PF = 1.00

MIN_DISCOVERY_NET_RETURN = 0.0


# ============================================================
# FINAL STATUS RULES
# ============================================================

PASS_MIN_VALIDATION_TRADES = 30
PASS_MIN_WIN_RATE = 55.0
PASS_MIN_NET_AVG_RETURN = 0.0
PASS_MIN_PROFIT_FACTOR = 1.10
PASS_MIN_POSITIVE_BLOCKS = 3
PASS_MIN_VALID_BLOCKS = 3

WATCH_MIN_VALIDATION_TRADES = 20
WATCH_MIN_WIN_RATE = 52.0
WATCH_MIN_NET_AVG_RETURN = 0.0
WATCH_MIN_PROFIT_FACTOR = 1.00
WATCH_MIN_POSITIVE_BLOCKS = 2
WATCH_MIN_VALID_BLOCKS = 2

WILSON_Z = 1.96


# ============================================================
# FEATURES USED FOR DISCOVERY
# ============================================================

FEATURES = [

    "RSI_5M",
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
    "VOLUME_RATIO",
    "POC_DISTANCE_PCT",

    "LOWER_WICK_RATIO",
    "UPPER_WICK_RATIO",
    "CLOSE_POSITION",
]


# ============================================================
# HELPERS
# ============================================================

def ensure_dataframe(data, name="data"):

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:

        return pd.DataFrame(data)

    except Exception as exc:

        raise TypeError(
            f"{name} could not be converted to DataFrame. "
            f"Received: {type(data)}"
        ) from exc


def safe_float(value, default=np.nan):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


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
    z=WILSON_Z,
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
                "V15 TRYING HISTORY:",
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
                df_5m,
                "5-minute history",
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
                df_1h,
                "1-hour history",
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
                "days"
            )

            print(
                "ERROR:",
                str(exc),
            )

            print(
                "Trying shorter history..."
            )

    raise RuntimeError(
        "All V15 history attempts failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df_5m,
    df_1h,
):

    print()
    print(
        "Preparing V12/V10 indicator stack..."
    )

    df = v12.prepare_data(
        df_5m,
        df_1h,
    )

    df = ensure_dataframe(
        df,
        "prepared data",
    )

    print(
        "Adding V12 market regime..."
    )

    df = v12.add_v12_market_regime(
        df
    )

    print(
        "Adding V12 candle features..."
    )

    df = v12.add_v12_candle_features(
        df
    )

    df = ensure_dataframe(
        df,
        "V15 data",
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    print(
        "Calculating V15 entry features..."
    )

    # ========================================================
    # EMA DISTANCE
    # ========================================================

    df["V15_EMA_DISTANCE_PCT"] = (

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

    # ========================================================
    # EMA SLOPES
    # ========================================================

    df["V15_EMA20_SLOPE_1"] = (

        df["ema20"]
        .pct_change(1)

        *

        100.0
    )

    df["V15_EMA20_SLOPE_3"] = (

        df["ema20"]
        .pct_change(3)

        *

        100.0
    )

    # ========================================================
    # MACD CHANGES
    # ========================================================

    df["V15_MACD_CHANGE_1"] = (

        df["macd_hist"]

        -

        df["macd_hist"].shift(1)
    )

    df["V15_MACD_CHANGE_3"] = (

        df["macd_hist"]

        -

        df["macd_hist"].shift(3)
    )

    # ========================================================
    # RSI MOMENTUM
    # ========================================================

    df["V15_RSI_CHANGE_3"] = (

        df["rsi"]

        -

        df["rsi"].shift(3)
    )

    # ========================================================
    # 1H EMA DISTANCE
    # ========================================================

    df["V15_H1_EMA_DISTANCE_PCT"] = (

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
    # BOLLINGER POSITION
    # ========================================================

    if "v10_bb_position" in df.columns:

        df["V15_BB_POSITION"] = (
            pd.to_numeric(
                df["v10_bb_position"],
                errors="coerce",
            )
        )

    else:

        df["V15_BB_POSITION"] = np.nan

    # ========================================================
    # VOLUME RATIO
    # ========================================================

    if "v10_volume_ratio" in df.columns:

        df["V15_VOLUME_RATIO"] = (
            pd.to_numeric(
                df["v10_volume_ratio"],
                errors="coerce",
            )
        )

    elif "volume" in df.columns:

        volume_ma = (
            df["volume"]
            .rolling(20)
            .mean()
        )

        df["V15_VOLUME_RATIO"] = (

            df["volume"]

            /

            volume_ma
        )

    else:

        df["V15_VOLUME_RATIO"] = np.nan

    # ========================================================
    # POC DISTANCE
    # ========================================================

    if "v10_poc_distance_pct" in df.columns:

        df["V15_POC_DISTANCE_PCT"] = (
            pd.to_numeric(
                df[
                    "v10_poc_distance_pct"
                ],
                errors="coerce",
            )
        )

    elif "poc" in df.columns:

        df["V15_POC_DISTANCE_PCT"] = (

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

    else:

        df["V15_POC_DISTANCE_PCT"] = np.nan

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required = [

        "datetime",

        "open",
        "high",
        "low",
        "close",

        "ema20",
        "ema50",

        "rsi",
        "macd_hist",

        "ema20_1h",
        "ema50_1h",
        "macd_hist_1h",

        "V12_REGIME",

        "V12_CLOSE_POSITION",
        "V12_LOWER_WICK_RATIO",
        "V12_UPPER_WICK_RATIO",

        "V15_EMA_DISTANCE_PCT",

        "V15_EMA20_SLOPE_1",
        "V15_EMA20_SLOPE_3",

        "V15_MACD_CHANGE_1",
        "V15_MACD_CHANGE_3",

        "V15_RSI_CHANGE_3",

        "V15_H1_EMA_DISTANCE_PCT",
    ]

    missing = [

        column

        for column in required

        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(

            "V15 missing required columns: "

            +

            ", ".join(missing)
        )

    required_numeric = [

        column

        for column in required

        if column != "V12_REGIME"
    ]

    df = (

        df
        .dropna(
            subset=required_numeric
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# FIXED TRADE SIMULATOR
# ============================================================

def simulate_trade(
    data,
    index,
    side,
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
                TP_PCT / 100.0
            )
        )

        sl_price = (

            entry

            *

            (
                1.0
                -
                SL_PCT / 100.0
            )
        )

    else:

        tp_price = (

            entry

            *

            (
                1.0
                -
                TP_PCT / 100.0
            )
        )

        sl_price = (

            entry

            *

            (
                1.0
                +
                SL_PCT / 100.0
            )
        )

    end_index = min(

        index + MAX_HOLD_BARS,

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

    # ========================================================
    # PATH BASED TP / SL
    # ========================================================

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
                pd.notna(high)
                and
                high >= tp_price
            )

            hit_sl = (
                pd.notna(low)
                and
                low <= sl_price
            )

        else:

            hit_tp = (
                pd.notna(low)
                and
                low <= tp_price
            )

            hit_sl = (
                pd.notna(high)
                and
                high >= sl_price
            )

        # Conservative same-candle rule
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

        mfe_pct = (

            (
                max_high
                /
                entry
            )

            -

            1.0

        ) * 100.0

        mae_pct = (

            (
                min_low
                /
                entry
            )

            -

            1.0

        ) * 100.0

        raw_return_pct = (

            (
                exit_price
                /
                entry
            )

            -

            1.0

        ) * 100.0

    else:

        mfe_pct = (

            (
                entry
                /
                min_low
            )

            -

            1.0

        ) * 100.0

        mae_pct = -(

            (
                max_high
                /
                entry
            )

            -

            1.0

        ) * 100.0

        raw_return_pct = (

            (
                entry
                /
                exit_price
            )

            -

            1.0

        ) * 100.0

    net_return_pct = (

        raw_return_pct

        -

        ROUND_TRIP_COST_PCT
    )

    return {

        "OUTCOME":
            outcome,

        "EXIT_PRICE":
            round(
                exit_price,
                8,
            ),

        "EXIT_BARS":
            int(
                exit_bars
            ),

        "RAW_RETURN_%":
            round(
                raw_return_pct,
                4,
            ),

        "NET_RETURN_%":
            round(
                net_return_pct,
                4,
            ),

        "MFE_%":
            round(
                mfe_pct,
                4,
            ),

        "MAE_%":
            round(
                mae_pct,
                4,
            ),
    }


# ============================================================
# COLLECT BASE SIGNALS
# ============================================================

def collect_events(df):

    data = ensure_dataframe(
        df,
        "V15 event data",
    )

    events = []

    last_index = (

        len(data)

        -

        MAX_HOLD_BARS

        -

        1
    )

    for i in range(
        100,
        last_index,
    ):

        row = data.iloc[i]

        regime = str(
            row["V12_REGIME"]
        )

        side = None

        # ====================================================
        # LONG BASE SIGNAL
        # ====================================================

        if (

            regime == "UPTREND"

            and

            v12.long_setup(
                data,
                i,
            )
        ):

            side = "LONG"

        # ====================================================
        # SHORT BASE SIGNAL
        # ====================================================

        elif (

            regime == "DOWNTREND"

            and

            v12.short_setup(
                data,
                i,
            )
        ):

            side = "SHORT"

        if side is None:

            continue

        trade = simulate_trade(
            data,
            i,
            side,
        )

        if trade is None:

            continue

        event = {

            "DATA_INDEX":
                i,

            "PAIR":
                v1.PAIR,

            "TIME":
                row[
                    "datetime"
                ].isoformat(),

            "SIDE":
                side,

            "REGIME":
                regime,

            "ENTRY_PRICE":
                safe_float(
                    row["close"]
                ),

            # ================================================
            # RSI
            # ================================================

            "RSI_5M":
                safe_float(
                    row["rsi"]
                ),

            "RSI_CHANGE_3":
                safe_float(
                    row[
                        "V15_RSI_CHANGE_3"
                    ]
                ),

            # ================================================
            # MACD
            # ================================================

            "MACD_HIST_5M":
                safe_float(
                    row["macd_hist"]
                ),

            "MACD_CHANGE_1":
                safe_float(
                    row[
                        "V15_MACD_CHANGE_1"
                    ]
                ),

            "MACD_CHANGE_3":
                safe_float(
                    row[
                        "V15_MACD_CHANGE_3"
                    ]
                ),

            # ================================================
            # EMA
            # ================================================

            "EMA_DISTANCE_PCT":
                safe_float(
                    row[
                        "V15_EMA_DISTANCE_PCT"
                    ]
                ),

            "EMA20_SLOPE_1":
                safe_float(
                    row[
                        "V15_EMA20_SLOPE_1"
                    ]
                ),

            "EMA20_SLOPE_3":
                safe_float(
                    row[
                        "V15_EMA20_SLOPE_3"
                    ]
                ),

            # ================================================
            # 1H
            # ================================================

            "H1_EMA_DISTANCE_PCT":
                safe_float(
                    row[
                        "V15_H1_EMA_DISTANCE_PCT"
                    ]
                ),

            "MACD_HIST_1H":
                safe_float(
                    row[
                        "macd_hist_1h"
                    ]
                ),

            # ================================================
            # BB / VOLUME / POC
            # ================================================

            "BB_POSITION":
                safe_float(
                    row[
                        "V15_BB_POSITION"
                    ]
                ),

            "VOLUME_RATIO":
                safe_float(
                    row[
                        "V15_VOLUME_RATIO"
                    ]
                ),

            "POC_DISTANCE_PCT":
                safe_float(
                    row[
                        "V15_POC_DISTANCE_PCT"
                    ]
                ),

            # ================================================
            # CANDLE QUALITY
            # ================================================

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

            "CLOSE_POSITION":
                safe_float(
                    row[
                        "V12_CLOSE_POSITION"
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
# PERFORMANCE
# ============================================================

def performance(data):

    df = ensure_dataframe(
        data,
        "performance data",
    )

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

    total = len(df)

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

        (
            wilson_low,
            wilson_high,
        ) = wilson_interval(
            wins,
            decisive,
        )

    else:

        win_rate = np.nan

        wilson_low = np.nan

        wilson_high = np.nan

    net_returns = (

        pd.to_numeric(
            df["NET_RETURN_%"],
            errors="coerce",
        )
        .dropna()
    )

    raw_returns = (

        pd.to_numeric(
            df["RAW_RETURN_%"],
            errors="coerce",
        )
        .dropna()
    )

    mfe = (

        pd.to_numeric(
            df["MFE_%"],
            errors="coerce",
        )
        .dropna()
    )

    mae = (

        pd.to_numeric(
            df["MAE_%"],
            errors="coerce",
        )
        .dropna()
    )

    positive_returns = (
        net_returns[
            net_returns > 0
        ]
    )

    negative_returns = (
        net_returns[
            net_returns < 0
        ]
    )

    gross_profit = float(
        positive_returns.sum()
    )

    gross_loss = abs(
        float(
            negative_returns.sum()
        )
    )

    if gross_loss > 0:

        profit_factor = (

            gross_profit

            /

            gross_loss
        )

    elif gross_profit > 0:

        profit_factor = 999.0

    else:

        profit_factor = 0.0

    return {

        "TRADES":
            int(total),

        "DECISIVE":
            int(decisive),

        "WINS":
            wins,

        "LOSSES":
            losses,

        "TIME_EXITS":
            time_exits,

        "WIN_RATE_%":
            round(
                float(win_rate),
                2,
            )
            if pd.notna(win_rate)
            else np.nan,

        "RAW_AVG_RETURN_%":
            round(
                float(
                    raw_returns.mean()
                ),
                4,
            )
            if not raw_returns.empty
            else np.nan,

        "NET_AVG_RETURN_%":
            round(
                float(
                    net_returns.mean()
                ),
                4,
            )
            if not net_returns.empty
            else np.nan,

        "NET_MEDIAN_RETURN_%":
            round(
                float(
                    net_returns.median()
                ),
                4,
            )
            if not net_returns.empty
            else np.nan,

        "PROFIT_FACTOR":
            round(
                float(
                    profit_factor
                ),
                4,
            ),

        "AVG_MFE_%":
            round(
                float(
                    mfe.mean()
                ),
                4,
            )
            if not mfe.empty
            else np.nan,

        "AVG_MAE_%":
            round(
                float(
                    mae.mean()
                ),
                4,
            )
            if not mae.empty
            else np.nan,

        "WILSON_LOW_95_%":
            round(
                float(
                    wilson_low
                ),
                2,
            )
            if pd.notna(wilson_low)
            else np.nan,

        "WILSON_HIGH_95_%":
            round(
                float(
                    wilson_high
                ),
                2,
            )
            if pd.notna(wilson_high)
            else np.nan,
    }


# ============================================================
# CHRONOLOGICAL DISCOVERY / VALIDATION SPLIT
# ============================================================

def split_discovery_validation(events):

    data = (

        events
        .sort_values("TIME")
        .reset_index(drop=True)
        .copy()
    )

    if len(data) < 2:

        raise RuntimeError(
            "Not enough events for V15 split."
        )

    split_index = int(
        len(data)
        *
        DISCOVERY_RATIO
    )

    split_index = max(
        1,
        min(
            split_index,
            len(data) - 1,
        ),
    )

    discovery = (

        data
        .iloc[:split_index]
        .copy()
        .reset_index(drop=True)
    )

    validation = (

        data
        .iloc[split_index:]
        .copy()
        .reset_index(drop=True)
    )

    validation_start = (
        validation.iloc[0]["TIME"]
    )

    discovery["DATASET"] = "DISCOVERY"

    validation["DATASET"] = "VALIDATION"

    return (
        discovery,
        validation,
        validation_start,
    )


# ============================================================
# APPLY RULE
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

        mask = (
            values
            >=
            threshold
        )

    elif operator == "<=":

        mask = (
            values
            <=
            threshold
        )

    else:

        raise ValueError(
            f"Unsupported operator: {operator}"
        )

    return (
        mask
        .fillna(False)
    )


# ============================================================
# DISCOVER SINGLE FEATURE RULES
# ============================================================

def discover_single_rules(
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
                side_data[feature],
                errors="coerce",
            )
            .dropna()
        )

        if len(values) < MIN_DISCOVERY_TRADES:

            continue

        for quantile in QUANTILES:

            try:

                threshold = float(
                    values.quantile(
                        quantile
                    )
                )

            except Exception:

                continue

            if not np.isfinite(
                threshold
            ):

                continue

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

                filtered = (

                    side_data[
                        mask
                    ]
                    .copy()
                )

                if (
                    len(filtered)
                    <
                    MIN_DISCOVERY_TRADES
                ):

                    continue

                stats = performance(
                    filtered
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

                    "DISCOVERY_TRADES":
                        stats["TRADES"],

                    "DISCOVERY_DECISIVE":
                        stats["DECISIVE"],

                    "DISCOVERY_WIN_RATE_%":
                        stats["WIN_RATE_%"],

                    "DISCOVERY_NET_AVG_RETURN_%":
                        stats[
                            "NET_AVG_RETURN_%"
                        ],

                    "DISCOVERY_PROFIT_FACTOR":
                        stats[
                            "PROFIT_FACTOR"
                        ],

                    "DISCOVERY_WILSON_LOW_%":
                        stats[
                            "WILSON_LOW_95_%"
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
                "DISCOVERY_NET_AVG_RETURN_%",
                "DISCOVERY_PROFIT_FACTOR",
                "DISCOVERY_WIN_RATE_%",
                "DISCOVERY_TRADES",
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
# SELECT NON-DUPLICATE BEST RULES
# ============================================================

def select_best_rules(
    discovery_rules,
):

    if discovery_rules.empty:

        return []

    eligible = (

        discovery_rules[
            (
                discovery_rules[
                    "DISCOVERY_TRADES"
                ]
                >=
                MIN_DISCOVERY_TRADES
            )
            &
            (
                discovery_rules[
                    "DISCOVERY_NET_AVG_RETURN_%"
                ]
                >
                MIN_DISCOVERY_NET_RETURN
            )
            &
            (
                discovery_rules[
                    "DISCOVERY_PROFIT_FACTOR"
                ]
                >=
                MIN_DISCOVERY_PF
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # If no profitable rule exists, still retain the strongest
    # discovery rules for diagnostic validation.
    # --------------------------------------------------------

    if eligible.empty:

        eligible = (
            discovery_rules
            .head(20)
            .copy()
        )

    selected = []

    used_features = set()

    for _, row in eligible.iterrows():

        feature = str(
            row["FEATURE"]
        )

        if feature in used_features:

            continue

        selected.append({

            "FEATURE":
                feature,

            "OPERATOR":
                str(
                    row["OPERATOR"]
                ),

            "THRESHOLD":
                float(
                    row["THRESHOLD"]
                ),

            "DISCOVERY_TRADES":
                int(
                    row[
                        "DISCOVERY_TRADES"
                    ]
                ),

            "DISCOVERY_WIN_RATE_%":
                safe_float(
                    row[
                        "DISCOVERY_WIN_RATE_%"
                    ]
                ),

            "DISCOVERY_NET_AVG_RETURN_%":
                safe_float(
                    row[
                        "DISCOVERY_NET_AVG_RETURN_%"
                    ]
                ),

            "DISCOVERY_PROFIT_FACTOR":
                safe_float(
                    row[
                        "DISCOVERY_PROFIT_FACTOR"
                    ]
                ),
        })

        used_features.add(
            feature
        )

        if (
            len(selected)
            >=
            MAX_SELECTED_FEATURES
        ):

            break

    return selected


# ============================================================
# CREATE LOCKED CANDIDATES
# ============================================================

def build_locked_candidates(
    discovery,
    side,
    selected_rules,
):

    side_data = (

        discovery[
            discovery["SIDE"]
            ==
            side
        ]
        .copy()
    )

    candidates = []

    # BASE
    candidates.append({

        "NAME":
            f"V15_{side}_BASE",

        "SIDE":
            side,

        "RULES":
            [],
    })

    # SINGLE RULES
    for number, rule in enumerate(
        selected_rules,
        start=1,
    ):

        candidates.append({

            "NAME":
                f"V15_{side}_R{number}",

            "SIDE":
                side,

            "RULES":
                [
                    rule
                ],
        })

    # TWO RULE COMBINATIONS
    if len(selected_rules) >= 2:

        for i in range(
            len(selected_rules)
        ):

            for j in range(
                i + 1,
                len(selected_rules),
            ):

                candidates.append({

                    "NAME":
                        (
                            f"V15_{side}_"
                            f"R{i + 1}_R{j + 1}"
                        ),

                    "SIDE":
                        side,

                    "RULES":
                        [
                            selected_rules[i],
                            selected_rules[j],
                        ],
                })

    # THREE RULE COMBINATION
    if len(selected_rules) >= 3:

        candidates.append({

            "NAME":
                f"V15_{side}_R1_R2_R3",

            "SIDE":
                side,

            "RULES":
                selected_rules[:3],
        })

    # --------------------------------------------------------
    # Evaluate on discovery.
    # This is diagnostic only.
    # Final decision is based on VALIDATION.
    # --------------------------------------------------------

    for candidate in candidates:

        filtered = filter_candidate(
            side_data,
            candidate,
        )

        stats = performance(
            filtered
        )

        candidate[
            "DISCOVERY_STATS"
        ] = stats

    return candidates


# ============================================================
# FILTER CANDIDATE
# ============================================================

def filter_candidate(
    data,
    candidate,
):

    df = ensure_dataframe(
        data,
        "candidate data",
    )

    side = candidate[
        "SIDE"
    ]

    filtered = (

        df[
            df["SIDE"]
            ==
            side
        ]
        .copy()
    )

    for rule in candidate["RULES"]:

        mask = apply_rule(

            filtered,

            rule["FEATURE"],

            rule["OPERATOR"],

            rule["THRESHOLD"],
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

def rule_description(candidate):

    rules = candidate[
        "RULES"
    ]

    if not rules:

        return "BASE"

    descriptions = []

    for rule in rules:

        descriptions.append(

            f'{rule["FEATURE"]} '
            f'{rule["OPERATOR"]} '
            f'{rule["THRESHOLD"]:.6f}'
        )

    return " AND ".join(
        descriptions
    )


# ============================================================
# VALIDATION BLOCKS
# ============================================================

def create_validation_blocks(
    validation,
):

    data = (

        validation
        .sort_values("TIME")
        .reset_index(drop=True)
        .copy()
    )

    if data.empty:

        return []

    time_values = pd.to_datetime(
        data["TIME"],
        utc=True,
        errors="coerce",
    )

    start_time = time_values.min()
    end_time = time_values.max()

    if (
        pd.isna(start_time)
        or
        pd.isna(end_time)
    ):

        return []

    if start_time == end_time:

        return [
            (
                1,
                data.copy(),
            )
        ]

    edges = pd.date_range(

        start=start_time,

        end=end_time,

        periods=VALIDATION_BLOCKS + 1,
    )

    blocks = []

    for i in range(
        VALIDATION_BLOCKS
    ):

        block_number = i + 1

        block_start = edges[i]
        block_end = edges[i + 1]

        if i == VALIDATION_BLOCKS - 1:

            mask = (

                (time_values >= block_start)

                &

                (time_values <= block_end)
            )

        else:

            mask = (

                (time_values >= block_start)

                &

                (time_values < block_end)
            )

        block = (

            data[
                mask
            ]
            .copy()
        )

        blocks.append(
            (
                block_number,
                block,
            )
        )

    return blocks


# ============================================================
# STATUS
# ============================================================

def classify_result(row):

    validation_trades = int(
        row[
            "VALIDATION_TRADES"
        ]
    )

    win_rate = safe_float(
        row[
            "VALIDATION_WIN_RATE_%"
        ]
    )

    net_avg = safe_float(
        row[
            "VALIDATION_NET_AVG_RETURN_%"
        ]
    )

    profit_factor = safe_float(
        row[
            "VALIDATION_PROFIT_FACTOR"
        ]
    )

    valid_blocks = int(
        row[
            "VALID_BLOCKS"
        ]
    )

    positive_blocks = int(
        row[
            "POSITIVE_BLOCKS"
        ]
    )

    # ========================================================
    # PASS
    # ========================================================

    if (

        validation_trades
        >=
        PASS_MIN_VALIDATION_TRADES

        and

        pd.notna(win_rate)

        and

        win_rate
        >=
        PASS_MIN_WIN_RATE

        and

        pd.notna(net_avg)

        and

        net_avg
        >
        PASS_MIN_NET_AVG_RETURN

        and

        pd.notna(profit_factor)

        and

        profit_factor
        >=
        PASS_MIN_PROFIT_FACTOR

        and

        valid_blocks
        >=
        PASS_MIN_VALID_BLOCKS

        and

        positive_blocks
        >=
        PASS_MIN_POSITIVE_BLOCKS
    ):

        return "PASS"

    # ========================================================
    # WATCH
    # ========================================================

    if (

        validation_trades
        >=
        WATCH_MIN_VALIDATION_TRADES

        and

        pd.notna(win_rate)

        and

        win_rate
        >=
        WATCH_MIN_WIN_RATE

        and

        pd.notna(net_avg)

        and

        net_avg
        >
        WATCH_MIN_NET_AVG_RETURN

        and

        pd.notna(profit_factor)

        and

        profit_factor
        >=
        WATCH_MIN_PROFIT_FACTOR

        and

        valid_blocks
        >=
        WATCH_MIN_VALID_BLOCKS

        and

        positive_blocks
        >=
        WATCH_MIN_POSITIVE_BLOCKS
    ):

        return "WATCH"

    return "REJECT"


# ============================================================
# VALIDATE LOCKED CANDIDATES
# ============================================================

def validate_candidates(
    validation,
    candidates,
):

    rows = []

    detail_rows = []

    validation_blocks = (
        create_validation_blocks(
            validation
        )
    )

    for candidate in candidates:

        candidate_name = (
            candidate["NAME"]
        )

        side = (
            candidate["SIDE"]
        )

        rules_text = (
            rule_description(
                candidate
            )
        )

        # ====================================================
        # DISCOVERY PERFORMANCE
        # ====================================================

        discovery_stats = (
            candidate.get(
                "DISCOVERY_STATS"
            )
        )

        if discovery_stats is None:

            discovery_stats = {

                "TRADES": 0,

                "DECISIVE": 0,

                "WIN_RATE_%": np.nan,

                "NET_AVG_RETURN_%": np.nan,

                "PROFIT_FACTOR": np.nan,
            }

        # ====================================================
        # VALIDATION PERFORMANCE
        # ====================================================

        validation_filtered = (
            filter_candidate(
                validation,
                candidate,
            )
        )

        validation_stats = (
            performance(
                validation_filtered
            )
        )

        if validation_stats is None:

            validation_stats = {

                "TRADES": 0,

                "DECISIVE": 0,

                "WINS": 0,

                "LOSSES": 0,

                "TIME_EXITS": 0,

                "WIN_RATE_%": np.nan,

                "RAW_AVG_RETURN_%": np.nan,

                "NET_AVG_RETURN_%": np.nan,

                "NET_MEDIAN_RETURN_%": np.nan,

                "PROFIT_FACTOR": np.nan,

                "AVG_MFE_%": np.nan,

                "AVG_MAE_%": np.nan,

                "WILSON_LOW_95_%": np.nan,

                "WILSON_HIGH_95_%": np.nan,
            }

        # ====================================================
        # VALIDATION BLOCKS
        # ====================================================

        block_win_rates = []

        positive_blocks = 0
        valid_blocks = 0

        for (
            block_number,
            block_data,
        ) in validation_blocks:

            block_filtered = (
                filter_candidate(
                    block_data,
                    candidate,
                )
            )

            block_stats = performance(
                block_filtered
            )

            if block_stats is None:

                detail_rows.append({

                    "CANDIDATE":
                        candidate_name,

                    "SIDE":
                        side,

                    "BLOCK":
                        block_number,

                    "TRADES":
                        0,

                    "DECISIVE":
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

                block_win_rates.append(

                    block_stats[
                        "WIN_RATE_%"
                    ]
                )

            detail_rows.append({

                "CANDIDATE":
                    candidate_name,

                "SIDE":
                    side,

                "BLOCK":
                    block_number,

                "TRADES":
                    block_stats[
                        "TRADES"
                    ],

                "DECISIVE":
                    block_stats[
                        "DECISIVE"
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

        if block_win_rates:

            min_block_wr = float(
                np.min(
                    block_win_rates
                )
            )

            median_block_wr = float(
                np.median(
                    block_win_rates
                )
            )

            max_block_wr = float(
                np.max(
                    block_win_rates
                )
            )

            std_block_wr = float(
                np.std(
                    block_win_rates
                )
            )

        else:

            min_block_wr = np.nan
            median_block_wr = np.nan
            max_block_wr = np.nan
            std_block_wr = np.nan

        row = {

            "CANDIDATE":
                candidate_name,

            "SIDE":
                side,

            "RULES":
                rules_text,

            # ================================================
            # DISCOVERY
            # ================================================

            "DISCOVERY_TRADES":
                discovery_stats[
                    "TRADES"
                ],

            "DISCOVERY_DECISIVE":
                discovery_stats[
                    "DECISIVE"
                ],

            "DISCOVERY_WIN_RATE_%":
                discovery_stats[
                    "WIN_RATE_%"
                ],

            "DISCOVERY_NET_AVG_RETURN_%":
                discovery_stats[
                    "NET_AVG_RETURN_%"
                ],

            "DISCOVERY_PROFIT_FACTOR":
                discovery_stats[
                    "PROFIT_FACTOR"
                ],

            # ================================================
            # VALIDATION
            # ================================================

            "VALIDATION_TRADES":
                validation_stats[
                    "TRADES"
                ],

            "VALIDATION_DECISIVE":
                validation_stats[
                    "DECISIVE"
                ],

            "VALIDATION_WINS":
                validation_stats[
                    "WINS"
                ],

            "VALIDATION_LOSSES":
                validation_stats[
                    "LOSSES"
                ],

            "VALIDATION_TIME_EXITS":
                validation_stats[
                    "TIME_EXITS"
                ],

            "VALIDATION_WIN_RATE_%":
                validation_stats[
                    "WIN_RATE_%"
                ],

            "VALIDATION_NET_AVG_RETURN_%":
                validation_stats[
                    "NET_AVG_RETURN_%"
                ],

            "VALIDATION_PROFIT_FACTOR":
                validation_stats[
                    "PROFIT_FACTOR"
                ],

            "VALIDATION_WILSON_LOW_%":
                validation_stats[
                    "WILSON_LOW_95_%"
                ],

            "VALIDATION_WILSON_HIGH_%":
                validation_stats[
                    "WILSON_HIGH_95_%"
                ],

            "VALIDATION_AVG_MFE_%":
                validation_stats[
                    "AVG_MFE_%"
                ],

            "VALIDATION_AVG_MAE_%":
                validation_stats[
                    "AVG_MAE_%"
                ],

            # ================================================
            # BLOCKS
            # ================================================

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

            "MAX_BLOCK_WIN_RATE_%":
                round(
                    max_block_wr,
                    2,
                )
                if pd.notna(
                    max_block_wr
                )
                else np.nan,

            "BLOCK_WIN_RATE_STD_%":
                round(
                    std_block_wr,
                    2,
                )
                if pd.notna(
                    std_block_wr
                )
                else np.nan,
        }

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    detail = pd.DataFrame(
        detail_rows
    )

    if not summary.empty:

        summary["STATUS"] = (
            summary.apply(
                classify_result,
                axis=1,
            )
        )

        status_rank = {

            "PASS": 3,

            "WATCH": 2,

            "REJECT": 1,
        }

        summary["_STATUS_RANK"] = (

            summary["STATUS"]
            .map(
                status_rank
            )
            .fillna(0)
        )

        summary = (

            summary
            .sort_values(
                by=[
                    "_STATUS_RANK",
                    "VALIDATION_NET_AVG_RETURN_%",
                    "VALIDATION_PROFIT_FACTOR",
                    "VALIDATION_WIN_RATE_%",
                    "VALIDATION_TRADES",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .drop(
                columns=[
                    "_STATUS_RANK"
                ]
            )
            .reset_index(drop=True)
        )

    return (
        summary,
        detail,
    )


# ============================================================
# PRINT DISCOVERY RULES
# ============================================================

def print_discovery_rules(
    side,
    rules_df,
    selected_rules,
):

    print()
    print("=" * 150)

    print(
        "V15",
        side,
        "DISCOVERY RESULTS"
    )

    print("=" * 150)

    if rules_df.empty:

        print(
            "No discovery rules found."
        )

        return

    display_columns = [

        "SIDE",

        "FEATURE",

        "OPERATOR",

        "QUANTILE",

        "THRESHOLD",

        "DISCOVERY_TRADES",

        "DISCOVERY_WIN_RATE_%",

        "DISCOVERY_NET_AVG_RETURN_%",

        "DISCOVERY_PROFIT_FACTOR",

        "DISCOVERY_WILSON_LOW_%",
    ]

    print()
    print(
        "TOP DISCOVERY RULES:"
    )

    print()

    print(
        rules_df[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "LOCKED FEATURES FOR VALIDATION:"
    )

    if not selected_rules:

        print(
            "No locked features."
        )

    else:

        for number, rule in enumerate(
            selected_rules,
            start=1,
        ):

            print(

                f'R{number}: '

                f'{rule["FEATURE"]} '

                f'{rule["OPERATOR"]} '

                f'{rule["THRESHOLD"]:.6f}'
            )


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_final_summary(
    summary,
):

    print()
    print("=" * 190)

    print(
        "V15 LOCKED UNSEEN VALIDATION RESULTS"
    )

    print("=" * 190)

    if summary.empty:

        print(
            "No V15 summary results."
        )

        return

    display = [

        "CANDIDATE",

        "STATUS",

        "SIDE",

        "DISCOVERY_TRADES",

        "DISCOVERY_WIN_RATE_%",

        "DISCOVERY_NET_AVG_RETURN_%",

        "DISCOVERY_PROFIT_FACTOR",

        "VALIDATION_TRADES",

        "VALIDATION_WIN_RATE_%",

        "VALIDATION_NET_AVG_RETURN_%",

        "VALIDATION_PROFIT_FACTOR",

        "VALIDATION_WILSON_LOW_%",

        "POSITIVE_BLOCKS",

        "VALID_BLOCKS",

        "MIN_BLOCK_WIN_RATE_%",

        "MEDIAN_BLOCK_WIN_RATE_%",

        "BLOCK_WIN_RATE_STD_%",
    ]

    print(
        summary[
            display
        ]
        .to_string(
            index=False
        )
    )

    passed = (

        summary[
            summary["STATUS"]
            ==
            "PASS"
        ]
        .copy()
    )

    watch = (

        summary[
            summary["STATUS"]
            ==
            "WATCH"
        ]
        .copy()
    )

    rejected = (

        summary[
            summary["STATUS"]
            ==
            "REJECT"
        ]
        .copy()
    )

    print()
    print("=" * 190)

    print(
        "V15 FINAL PASS CANDIDATES:",
        len(passed),
    )

    print("=" * 190)

    if passed.empty:

        print(
            "No V15 candidate passed strict unseen validation."
        )

    else:

        print(
            passed[
                display
            ]
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 190)

    print(
        "V15 WATCH CANDIDATES:",
        len(watch),
    )

    print("=" * 190)

    if watch.empty:

        print(
            "No V15 WATCH candidate."
        )

    else:

        print(
            watch[
                display
            ]
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 190)

    print(
        "V15 REJECT CANDIDATES:",
        len(rejected),
    )

    print("=" * 190)

    if rejected.empty:

        print(
            "No rejected candidates."
        )

    else:

        print(
            rejected[
                display
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # BEST OVERALL
    # ========================================================

    best = summary.iloc[0]

    print()
    print("=" * 190)

    print(
        "V15 BEST OVERALL LOCKED CANDIDATE"
    )

    print("=" * 190)

    print(
        "CANDIDATE:",
        best["CANDIDATE"],
    )

    print(
        "STATUS:",
        best["STATUS"],
    )

    print(
        "SIDE:",
        best["SIDE"],
    )

    print(
        "RULES:",
        best["RULES"],
    )

    print()

    print(
        "DISCOVERY TRADES:",
        int(
            best[
                "DISCOVERY_TRADES"
            ]
        ),
    )

    print(
        "DISCOVERY WIN RATE:",
        f'{best["DISCOVERY_WIN_RATE_%"]:.2f}%',
    )

    print(
        "DISCOVERY NET AVG:",
        f'{best["DISCOVERY_NET_AVG_RETURN_%"]:.4f}%',
    )

    print(
        "DISCOVERY PROFIT FACTOR:",
        f'{best["DISCOVERY_PROFIT_FACTOR"]:.4f}',
    )

    print()

    print(
        "UNSEEN VALIDATION TRADES:",
        int(
            best[
                "VALIDATION_TRADES"
            ]
        ),
    )

    print(
        "UNSEEN VALIDATION WIN RATE:",
        f'{best["VALIDATION_WIN_RATE_%"]:.2f}%',
    )

    print(
        "UNSEEN VALIDATION NET AVG:",
        f'{best["VALIDATION_NET_AVG_RETURN_%"]:.4f}%',
    )

    print(
        "UNSEEN VALIDATION PROFIT FACTOR:",
        f'{best["VALIDATION_PROFIT_FACTOR"]:.4f}',
    )

    print(
        "VALIDATION 95% WILSON LOW:",
        f'{best["VALIDATION_WILSON_LOW_%"]:.2f}%',
    )

    print(
        "POSITIVE VALIDATION BLOCKS:",
        int(
            best[
                "POSITIVE_BLOCKS"
            ]
        ),
        "/",
        VALIDATION_BLOCKS,
    )

    print(
        "MIN VALIDATION BLOCK WIN RATE:",
        f'{best["MIN_BLOCK_WIN_RATE_%"]:.2f}%',
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 130)

    print(
        "COINDCX FUTURES HISTORICAL V15"
    )

    print(
        "DATA-DRIVEN STRATEGY DISCOVERY ENGINE"
    )

    print("=" * 130)

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
        "HISTORY TARGET:",
        HISTORY_ATTEMPTS[0],
        "DAYS"
    )

    print(
        "DISCOVERY DATA:",
        f"{DISCOVERY_RATIO * 100:.0f}%"
    )

    print(
        "UNSEEN VALIDATION DATA:",
        f"{(1.0 - DISCOVERY_RATIO) * 100:.0f}%"
    )

    print(
        "VALIDATION BLOCKS:",
        VALIDATION_BLOCKS,
    )

    print(
        "FIXED TP:",
        f"{TP_PCT:.2f}%"
    )

    print(
        "FIXED SL:",
        f"{SL_PCT:.2f}%"
    )

    print(
        "MAX HOLD:",
        MAX_HOLD_BARS,
        "x 5-minute candles"
    )

    print(
        "ROUND-TRIP COST:",
        f"{ROUND_TRIP_COST_PCT:.2f}%"
    )

    print(
        "SAME-BAR POLICY:",
        SAME_BAR_POLICY,
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "VALIDATION DATA WILL NOT BE USED"
    )

    print(
        "FOR THRESHOLD DISCOVERY."
    )

    print()
    print(
        "HISTORICAL RESEARCH ONLY."
    )

    print(
        "NO REAL ORDERS WILL BE PLACED."
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    (
        df_5m,
        df_1h,
        requested_days,
    ) = download_history()

    # ========================================================
    # PREPARE
    # ========================================================

    data = prepare_data(
        df_5m,
        df_1h,
    )

    print()
    print(
        "USABLE 5M CANDLES:",
        len(data),
    )

    print()
    print(
        "REGIME COUNTS:"
    )

    print(
        data[
            "V12_REGIME"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # EVENTS
    # ========================================================

    print()
    print(
        "Collecting V15 base signals..."
    )

    events = collect_events(
        data
    )

    if events.empty:

        print(
            "No V15 events found."
        )

        return

    print()
    print(
        "TOTAL V15 EVENTS:",
        len(events),
    )

    print(
        "LONG EVENTS:",
        int(
            (
                events["SIDE"]
                ==
                "LONG"
            ).sum()
        ),
    )

    print(
        "SHORT EVENTS:",
        int(
            (
                events["SIDE"]
                ==
                "SHORT"
            ).sum()
        ),
    )

    print()
    print(
        "OUTCOME COUNTS:"
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
        validation,
        validation_start,
    ) = split_discovery_validation(
        events
    )

    print()
    print("=" * 130)

    print(
        "V15 CHRONOLOGICAL SPLIT"
    )

    print("=" * 130)

    print(
        "TOTAL EVENTS:",
        len(events),
    )

    print(
        "DISCOVERY EVENTS:",
        len(discovery),
    )

    print(
        "VALIDATION EVENTS:",
        len(validation),
    )

    print(
        "VALIDATION START:",
        validation_start,
    )

    print()

    print(
        "DISCOVERY LONG:",
        int(
            (
                discovery["SIDE"]
                ==
                "LONG"
            ).sum()
        ),
    )

    print(
        "DISCOVERY SHORT:",
        int(
            (
                discovery["SIDE"]
                ==
                "SHORT"
            ).sum()
        ),
    )

    print(
        "VALIDATION LONG:",
        int(
            (
                validation["SIDE"]
                ==
                "LONG"
            ).sum()
        ),
    )

    print(
        "VALIDATION SHORT:",
        int(
            (
                validation["SIDE"]
                ==
                "SHORT"
            ).sum()
        ),
    )

    # ========================================================
    # DISCOVERY
    # ========================================================

    print()
    print(
        "Discovering LONG rules using DISCOVERY data only..."
    )

    long_discovery = (
        discover_single_rules(
            discovery,
            "LONG",
        )
    )

    long_selected = (
        select_best_rules(
            long_discovery
        )
    )

    print_discovery_rules(
        "LONG",
        long_discovery,
        long_selected,
    )

    print()
    print(
        "Discovering SHORT rules using DISCOVERY data only..."
    )

    short_discovery = (
        discover_single_rules(
            discovery,
            "SHORT",
        )
    )

    short_selected = (
        select_best_rules(
            short_discovery
        )
    )

    print_discovery_rules(
        "SHORT",
        short_discovery,
        short_selected,
    )

    # ========================================================
    # LOCK CANDIDATES
    # ========================================================

    long_candidates = (
        build_locked_candidates(
            discovery,
            "LONG",
            long_selected,
        )
    )

    short_candidates = (
        build_locked_candidates(
            discovery,
            "SHORT",
            short_selected,
        )
    )

    candidates = (
        long_candidates
        +
        short_candidates
    )

    print()
    print("=" * 130)

    print(
        "V15 LOCKED CANDIDATES"
    )

    print("=" * 130)

    for candidate in candidates:

        print(
            candidate["NAME"],
            "->",
            rule_description(
                candidate
            )
        )

    # ========================================================
    # VALIDATE
    # ========================================================

    print()
    print(
        "Applying LOCKED rules to UNSEEN validation data..."
    )

    (
        summary,
        detail,
    ) = validate_candidates(
        validation,
        candidates,
    )

    # ========================================================
    # DISCOVERY CSV
    # ========================================================

    discovery_tables = []

    if not long_discovery.empty:

        discovery_tables.append(
            long_discovery
        )

    if not short_discovery.empty:

        discovery_tables.append(
            short_discovery
        )

    if discovery_tables:

        discovery_output = (
            pd.concat(
                discovery_tables,
                ignore_index=True,
            )
        )

    else:

        discovery_output = pd.DataFrame()

    # ========================================================
    # SAVE
    # ========================================================

    all_events = pd.concat(
        [
            discovery,
            validation,
        ],
        ignore_index=True,
    )

    all_events.to_csv(
        EVENT_FILE,
        index=False,
    )

    discovery_output.to_csv(
        DISCOVERY_FILE,
        index=False,
    )

    detail.to_csv(
        RESULT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        "V15 EVENTS SAVED:",
        EVENT_FILE,
    )

    print(
        "V15 DISCOVERY SAVED:",
        DISCOVERY_FILE,
    )

    print(
        "V15 BLOCK RESULTS SAVED:",
        RESULT_FILE,
    )

    print(
        "V15 SUMMARY SAVED:",
        SUMMARY_FILE,
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print_final_summary(
        summary
    )

    print()
    print("=" * 130)

    print(
        "V15 INTERPRETATION"
    )

    print("=" * 130)

    print()
    print(
        "V15 DOES NOT choose a strategy"
    )

    print(
        "because it looks good on the same data"
    )

    print(
        "used to discover it."
    )

    print()
    print(
        "The first 70% of historical signals"
    )

    print(
        "are used for DISCOVERY."
    )

    print()
    print(
        "The final 30% are kept UNSEEN"
    )

    print(
        "until the rules are locked."
    )

    print()
    print(
        "The final decision is based mainly on"
    )

    print(
        "UNSEEN VALIDATION performance."
    )

    print()
    print(
        "PASS requires:"
    )

    print(
        "- Positive validation net expectancy"
    )

    print(
        "- Profit Factor >= 1.10"
    )

    print(
        "- Validation Win Rate >= 55%"
    )

    print(
        "- Minimum validation sample"
    )

    print(
        "- Positive performance across"
    )

    print(
        "  multiple validation time blocks"
    )

    print()
    print(
        "WATCH means promising but"
    )

    print(
        "not ready for live use."
    )

    print()
    print(
        "REJECT means the discovered edge"
    )

    print(
        "did not survive unseen data."
    )

    print()
    print(
        "Even a PASS is NOT automatically"
    )

    print(
        "approved for real-money trading."
    )

    print(
        "It should next be paper-tested."
    )

    print()
    print(
        "Backtests do not guarantee"
    )

    print(
        "future performance."
    )

    print()
    print(
        "NO REAL ORDERS WERE PLACED."
    )

    print()
    print("=" * 130)

    print(
        "FUTURES HISTORICAL V15 COMPLETE"
    )

    print("=" * 130)


if __name__ == "__main__":
    main()
