import math
import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v12 as v12


# ============================================================
# COINDCX FUTURES HISTORICAL V14
# WINNING VS LOSING SIGNAL ANALYSIS
# + ENTRY FILTER DISCOVERY
# ============================================================
#
# PURPOSE
# -------
# V13 showed that simply changing TP / SL was not enough.
#
# V14 changes the question:
#
# "What was different at ENTRY between winning and losing
#  trades?"
#
# We analyse:
#
# - LONG and SHORT separately
# - RSI
# - MACD histogram strength / direction
# - EMA20 / EMA50 distance
# - EMA20 slope
# - 1H EMA trend strength
# - 1H MACD
# - Bollinger Band position
# - Volume ratio
# - POC distance
# - Candle wick quality
# - Candle close position
#
# Then we test a LIMITED set of logical entry filters.
#
# IMPORTANT:
# - Historical research only
# - No Telegram
# - No live orders
# - No real-money execution
# ============================================================


EVENT_FILE = "crypto_futures_v14_events.csv"
PROFILE_FILE = "crypto_futures_v14_win_loss_profile.csv"
DETAIL_FILE = "crypto_futures_v14_results.csv"
SUMMARY_FILE = "crypto_futures_v14_summary.csv"


# ============================================================
# HISTORY
# ============================================================

HISTORY_ATTEMPTS = [
    365,
    270,
    180,
]

WALK_FORWARD_BLOCKS = 8


# ============================================================
# TRADE MODEL
# ============================================================
#
# We use ONE fixed trade model in V14.
#
# This prevents V14 from becoming another TP/SL optimizer.
# ============================================================

TP_PCT = 0.50
SL_PCT = 0.50

MAX_HOLD_BARS = 48

ROUND_TRIP_COST_PCT = 0.10

SAME_BAR_POLICY = "SL_FIRST"


# ============================================================
# ROBUSTNESS RULES
# ============================================================

MIN_TOTAL_TRADES_PASS = 40
MIN_TOTAL_TRADES_WATCH = 25

MIN_TRADES_PER_BLOCK = 3

PASS_MIN_WIN_RATE = 58.0
PASS_MIN_NET_AVG_RETURN = 0.0
PASS_MIN_PROFIT_FACTOR = 1.15

PASS_MIN_VALID_BLOCKS = 6
PASS_MIN_POSITIVE_BLOCKS = 6

PASS_MAX_BLOCK_STD = 22.0


WATCH_MIN_WIN_RATE = 54.0
WATCH_MIN_NET_AVG_RETURN = 0.0
WATCH_MIN_PROFIT_FACTOR = 1.00

WATCH_MIN_VALID_BLOCKS = 5
WATCH_MIN_POSITIVE_BLOCKS = 5


WILSON_Z = 1.96


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


def safe_divide(a, b, default=np.nan):

    try:

        if pd.isna(a) or pd.isna(b):
            return default

        if float(b) == 0:
            return default

        return float(a) / float(b)

    except Exception:
        return default


# ============================================================
# WILSON INTERVAL
# ============================================================

def wilson_interval(wins, total, z=WILSON_Z):

    if total <= 0:
        return np.nan, np.nan

    p = wins / total

    denominator = (
        1.0
        +
        z ** 2 / total
    )

    centre = (
        p
        +
        z ** 2 / (2.0 * total)
    )

    margin = (
        z
        *
        math.sqrt(
            (
                p * (1.0 - p)
                +
                z ** 2 / (4.0 * total)
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
                "V14 TRYING HISTORY:",
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
        "All V14 history attempts failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# HISTORY RANGE
# ============================================================

def print_history_range(
    df_5m,
    df_1h,
    requested_days,
):

    print()
    print("=" * 120)
    print("V14 DATA RANGE")
    print("=" * 120)

    print(
        "REQUESTED HISTORY:",
        requested_days,
        "days",
    )

    if (
        "datetime" in df_5m.columns
        and
        not df_5m.empty
    ):

        dt = pd.to_datetime(
            df_5m["datetime"],
            utc=True,
            errors="coerce",
        )

        start = dt.min()
        end = dt.max()

        print(
            "5M START:",
            start,
        )

        print(
            "5M END:",
            end,
        )

        if (
            pd.notna(start)
            and
            pd.notna(end)
        ):

            actual_days = (
                end - start
            ).total_seconds() / 86400.0

            print(
                "ACTUAL 5M HISTORY DAYS:",
                round(actual_days, 2),
            )

    if (
        "datetime" in df_1h.columns
        and
        not df_1h.empty
    ):

        dt = pd.to_datetime(
            df_1h["datetime"],
            utc=True,
            errors="coerce",
        )

        print(
            "1H START:",
            dt.min(),
        )

        print(
            "1H END:",
            dt.max(),
        )


# ============================================================
# PREPARE INDICATOR STACK
# ============================================================

def prepare_data(df_5m, df_1h):

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
        "V12 prepared data",
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

    df = ensure_dataframe(
        df,
        "V14 prepared data",
    )

    # --------------------------------------------------------
    # Make sure datetime is proper Timestamp
    # --------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # V14 EXTRA ENTRY FEATURES
    # --------------------------------------------------------

    print(
        "Calculating V14 entry-quality features..."
    )

    # 5M EMA distance
    df["V14_EMA_DISTANCE_PCT"] = (
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

    # EMA20 slope
    df["V14_EMA20_SLOPE_1"] = (
        df["ema20"]
        .pct_change(1)
        *
        100.0
    )

    df["V14_EMA20_SLOPE_3"] = (
        df["ema20"]
        .pct_change(3)
        *
        100.0
    )

    # MACD histogram change
    df["V14_MACD_CHANGE_1"] = (
        df["macd_hist"]
        -
        df["macd_hist"].shift(1)
    )

    df["V14_MACD_CHANGE_3"] = (
        df["macd_hist"]
        -
        df["macd_hist"].shift(3)
    )

    # RSI momentum
    df["V14_RSI_CHANGE_3"] = (
        df["rsi"]
        -
        df["rsi"].shift(3)
    )

    # 1H EMA distance
    df["V14_H1_EMA_DISTANCE_PCT"] = (
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

    # --------------------------------------------------------
    # Existing optional indicators
    # --------------------------------------------------------

    if "v10_bb_position" in df.columns:

        df["V14_BB_POSITION"] = pd.to_numeric(
            df["v10_bb_position"],
            errors="coerce",
        )

    else:

        df["V14_BB_POSITION"] = np.nan

    if "v10_volume_ratio" in df.columns:

        df["V14_VOLUME_RATIO"] = pd.to_numeric(
            df["v10_volume_ratio"],
            errors="coerce",
        )

    else:

        # fallback volume ratio
        if "volume" in df.columns:

            volume_ma = (
                df["volume"]
                .rolling(20)
                .mean()
            )

            df["V14_VOLUME_RATIO"] = (
                df["volume"]
                /
                volume_ma
            )

        else:

            df["V14_VOLUME_RATIO"] = np.nan

    if "v10_poc_distance_pct" in df.columns:

        df["V14_POC_DISTANCE_PCT"] = pd.to_numeric(
            df["v10_poc_distance_pct"],
            errors="coerce",
        )

    elif (
        "poc" in df.columns
    ):

        df["V14_POC_DISTANCE_PCT"] = (
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

        df["V14_POC_DISTANCE_PCT"] = np.nan

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

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
        "V14_EMA_DISTANCE_PCT",
        "V14_EMA20_SLOPE_1",
        "V14_EMA20_SLOPE_3",
        "V14_MACD_CHANGE_1",
        "V14_MACD_CHANGE_3",
        "V14_RSI_CHANGE_3",
        "V14_H1_EMA_DISTANCE_PCT",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "V14 missing required columns: "
            +
            ", ".join(missing)
        )

    drop_columns = [
        col
        for col in required
        if col != "V12_REGIME"
    ]

    df = (
        df
        .dropna(
            subset=drop_columns
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# FIXED TP/SL TRADE SIMULATOR
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

    # --------------------------------------------------------
    # Path-based TP / SL
    # --------------------------------------------------------

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

        if hit_tp and hit_sl:

            if SAME_BAR_POLICY == "SL_FIRST":

                outcome = "LOSS"
                exit_price = sl_price
                exit_bars = offset

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
                max_high / entry
            )
            -
            1.0
        ) * 100.0

        mae_pct = (
            (
                min_low / entry
            )
            -
            1.0
        ) * 100.0

        raw_return_pct = (
            (
                exit_price / entry
            )
            -
            1.0
        ) * 100.0

    else:

        mfe_pct = (
            (
                entry / min_low
            )
            -
            1.0
        ) * 100.0

        mae_pct = -(
            (
                max_high / entry
            )
            -
            1.0
        ) * 100.0

        raw_return_pct = (
            (
                entry / exit_price
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
# COLLECT BASE EVENTS
# ============================================================

def collect_base_events(df):

    data = ensure_dataframe(
        df,
        "base event input",
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
        setup = None

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if (
            regime == "UPTREND"
            and
            v12.long_setup(
                data,
                i,
            )
        ):

            side = "LONG"
            setup = "LOWER_WICK_SWEEP"

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        elif (
            regime == "DOWNTREND"
            and
            v12.short_setup(
                data,
                i,
            )
        ):

            side = "SHORT"
            setup = "UPPER_WICK_SWEEP"

        if side is None:

            continue

        result = simulate_trade(
            data,
            i,
            side,
        )

        if result is None:

            continue

        close = safe_float(
            row["close"]
        )

        ema20 = safe_float(
            row["ema20"]
        )

        ema50 = safe_float(
            row["ema50"]
        )

        ema20_1h = safe_float(
            row["ema20_1h"]
        )

        ema50_1h = safe_float(
            row["ema50_1h"]
        )

        event = {

            "DATA_INDEX":
                i,

            "PAIR":
                v1.PAIR,

            "TIME":
                row["datetime"].isoformat(),

            "SIDE":
                side,

            "REGIME":
                regime,

            "SETUP":
                setup,

            "ENTRY_PRICE":
                close,

            "RSI_5M":
                safe_float(
                    row["rsi"]
                ),

            "RSI_CHANGE_3":
                safe_float(
                    row["V14_RSI_CHANGE_3"]
                ),

            "MACD_HIST_5M":
                safe_float(
                    row["macd_hist"]
                ),

            "MACD_CHANGE_1":
                safe_float(
                    row["V14_MACD_CHANGE_1"]
                ),

            "MACD_CHANGE_3":
                safe_float(
                    row["V14_MACD_CHANGE_3"]
                ),

            "EMA20":
                ema20,

            "EMA50":
                ema50,

            "EMA_DISTANCE_PCT":
                safe_float(
                    row["V14_EMA_DISTANCE_PCT"]
                ),

            "EMA20_SLOPE_1":
                safe_float(
                    row["V14_EMA20_SLOPE_1"]
                ),

            "EMA20_SLOPE_3":
                safe_float(
                    row["V14_EMA20_SLOPE_3"]
                ),

            "EMA20_1H":
                ema20_1h,

            "EMA50_1H":
                ema50_1h,

            "H1_EMA_DISTANCE_PCT":
                safe_float(
                    row[
                        "V14_H1_EMA_DISTANCE_PCT"
                    ]
                ),

            "MACD_HIST_1H":
                safe_float(
                    row["macd_hist_1h"]
                ),

            "BB_POSITION":
                safe_float(
                    row["V14_BB_POSITION"]
                ),

            "VOLUME_RATIO":
                safe_float(
                    row["V14_VOLUME_RATIO"]
                ),

            "POC_DISTANCE_PCT":
                safe_float(
                    row[
                        "V14_POC_DISTANCE_PCT"
                    ]
                ),

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
            result
        )

        events.append(
            event
        )

    return pd.DataFrame(
        events
    )


# ============================================================
# WIN / LOSS PROFILE
# ============================================================

PROFILE_FEATURES = [

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


def build_win_loss_profile(events):

    data = ensure_dataframe(
        events,
        "profile events",
    )

    rows = []

    for side in [
        "LONG",
        "SHORT",
    ]:

        side_data = (
            data[
                data["SIDE"]
                ==
                side
            ]
            .copy()
        )

        winners = (
            side_data[
                side_data["OUTCOME"]
                ==
                "WIN"
            ]
            .copy()
        )

        losers = (
            side_data[
                side_data["OUTCOME"]
                ==
                "LOSS"
            ]
            .copy()
        )

        for feature in PROFILE_FEATURES:

            if feature not in side_data.columns:
                continue

            win_values = pd.to_numeric(
                winners[feature],
                errors="coerce",
            ).dropna()

            loss_values = pd.to_numeric(
                losers[feature],
                errors="coerce",
            ).dropna()

            if (
                win_values.empty
                and
                loss_values.empty
            ):

                continue

            win_mean = (
                float(
                    win_values.mean()
                )
                if not win_values.empty
                else np.nan
            )

            loss_mean = (
                float(
                    loss_values.mean()
                )
                if not loss_values.empty
                else np.nan
            )

            win_median = (
                float(
                    win_values.median()
                )
                if not win_values.empty
                else np.nan
            )

            loss_median = (
                float(
                    loss_values.median()
                )
                if not loss_values.empty
                else np.nan
            )

            difference = (
                win_mean
                -
                loss_mean
                if (
                    pd.notna(win_mean)
                    and
                    pd.notna(loss_mean)
                )
                else np.nan
            )

            rows.append({

                "SIDE":
                    side,

                "FEATURE":
                    feature,

                "WIN_COUNT":
                    len(win_values),

                "LOSS_COUNT":
                    len(loss_values),

                "WIN_MEAN":
                    round(
                        win_mean,
                        6,
                    )
                    if pd.notna(win_mean)
                    else np.nan,

                "LOSS_MEAN":
                    round(
                        loss_mean,
                        6,
                    )
                    if pd.notna(loss_mean)
                    else np.nan,

                "MEAN_DIFFERENCE":
                    round(
                        difference,
                        6,
                    )
                    if pd.notna(difference)
                    else np.nan,

                "WIN_MEDIAN":
                    round(
                        win_median,
                        6,
                    )
                    if pd.notna(win_median)
                    else np.nan,

                "LOSS_MEDIAN":
                    round(
                        loss_median,
                        6,
                    )
                    if pd.notna(loss_median)
                    else np.nan,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# FILTER HELPERS
# ============================================================

def numeric_series(df, column):

    if column not in df.columns:

        return pd.Series(
            np.nan,
            index=df.index,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def long_filter_masks(df):

    rsi = numeric_series(
        df,
        "RSI_5M",
    )

    rsi_change = numeric_series(
        df,
        "RSI_CHANGE_3",
    )

    macd_change_1 = numeric_series(
        df,
        "MACD_CHANGE_1",
    )

    macd_change_3 = numeric_series(
        df,
        "MACD_CHANGE_3",
    )

    ema_distance = numeric_series(
        df,
        "EMA_DISTANCE_PCT",
    )

    ema_slope_3 = numeric_series(
        df,
        "EMA20_SLOPE_3",
    )

    h1_distance = numeric_series(
        df,
        "H1_EMA_DISTANCE_PCT",
    )

    h1_macd = numeric_series(
        df,
        "MACD_HIST_1H",
    )

    volume_ratio = numeric_series(
        df,
        "VOLUME_RATIO",
    )

    poc_distance = numeric_series(
        df,
        "POC_DISTANCE_PCT",
    )

    lower_wick = numeric_series(
        df,
        "LOWER_WICK_RATIO",
    )

    close_position = numeric_series(
        df,
        "CLOSE_POSITION",
    )

    masks = {}

    masks["V14_LONG_BASE"] = (
        df["SIDE"] == "LONG"
    )

    masks["V14_LONG_RSI45_65"] = (
        (df["SIDE"] == "LONG")
        &
        rsi.between(
            45,
            65,
            inclusive="both",
        )
    )

    masks["V14_LONG_RSI_RISING"] = (
        (df["SIDE"] == "LONG")
        &
        (rsi_change > 0)
    )

    masks["V14_LONG_MACD_ACCEL"] = (
        (df["SIDE"] == "LONG")
        &
        (macd_change_1 > 0)
        &
        (macd_change_3 > 0)
    )

    masks["V14_LONG_EMA_SLOPE"] = (
        (df["SIDE"] == "LONG")
        &
        (ema_slope_3 > 0)
    )

    masks["V14_LONG_EMA_STRONG"] = (
        (df["SIDE"] == "LONG")
        &
        (ema_distance > 0.05)
    )

    masks["V14_LONG_H1_STRONG"] = (
        (df["SIDE"] == "LONG")
        &
        (h1_distance > 0.05)
        &
        (h1_macd >= 0)
    )

    masks["V14_LONG_VOLUME12"] = (
        (df["SIDE"] == "LONG")
        &
        (volume_ratio >= 1.20)
    )

    masks["V14_LONG_POC075"] = (
        (df["SIDE"] == "LONG")
        &
        (poc_distance <= 0.75)
    )

    masks["V14_LONG_WICK45"] = (
        (df["SIDE"] == "LONG")
        &
        (lower_wick >= 0.45)
    )

    masks["V14_LONG_CLOSE65"] = (
        (df["SIDE"] == "LONG")
        &
        (close_position >= 0.65)
    )

    masks["V14_LONG_QUALITY1"] = (
        (df["SIDE"] == "LONG")
        &
        rsi.between(
            45,
            65,
            inclusive="both",
        )
        &
        (macd_change_1 > 0)
        &
        (ema_slope_3 > 0)
    )

    masks["V14_LONG_QUALITY2"] = (
        (df["SIDE"] == "LONG")
        &
        (macd_change_1 > 0)
        &
        (ema_slope_3 > 0)
        &
        (close_position >= 0.65)
    )

    masks["V14_LONG_QUALITY3"] = (
        (df["SIDE"] == "LONG")
        &
        (macd_change_1 > 0)
        &
        (h1_distance > 0.05)
        &
        (close_position >= 0.65)
    )

    masks["V14_LONG_QUALITY4"] = (
        (df["SIDE"] == "LONG")
        &
        rsi.between(
            45,
            65,
            inclusive="both",
        )
        &
        (macd_change_1 > 0)
        &
        (ema_slope_3 > 0)
        &
        (poc_distance <= 0.75)
    )

    return masks


def short_filter_masks(df):

    rsi = numeric_series(
        df,
        "RSI_5M",
    )

    rsi_change = numeric_series(
        df,
        "RSI_CHANGE_3",
    )

    macd_change_1 = numeric_series(
        df,
        "MACD_CHANGE_1",
    )

    macd_change_3 = numeric_series(
        df,
        "MACD_CHANGE_3",
    )

    ema_distance = numeric_series(
        df,
        "EMA_DISTANCE_PCT",
    )

    ema_slope_3 = numeric_series(
        df,
        "EMA20_SLOPE_3",
    )

    h1_distance = numeric_series(
        df,
        "H1_EMA_DISTANCE_PCT",
    )

    h1_macd = numeric_series(
        df,
        "MACD_HIST_1H",
    )

    volume_ratio = numeric_series(
        df,
        "VOLUME_RATIO",
    )

    poc_distance = numeric_series(
        df,
        "POC_DISTANCE_PCT",
    )

    upper_wick = numeric_series(
        df,
        "UPPER_WICK_RATIO",
    )

    close_position = numeric_series(
        df,
        "CLOSE_POSITION",
    )

    masks = {}

    masks["V14_SHORT_BASE"] = (
        df["SIDE"] == "SHORT"
    )

    masks["V14_SHORT_RSI35_55"] = (
        (df["SIDE"] == "SHORT")
        &
        rsi.between(
            35,
            55,
            inclusive="both",
        )
    )

    masks["V14_SHORT_RSI_FALLING"] = (
        (df["SIDE"] == "SHORT")
        &
        (rsi_change < 0)
    )

    masks["V14_SHORT_MACD_ACCEL"] = (
        (df["SIDE"] == "SHORT")
        &
        (macd_change_1 < 0)
        &
        (macd_change_3 < 0)
    )

    masks["V14_SHORT_EMA_SLOPE"] = (
        (df["SIDE"] == "SHORT")
        &
        (ema_slope_3 < 0)
    )

    masks["V14_SHORT_EMA_STRONG"] = (
        (df["SIDE"] == "SHORT")
        &
        (ema_distance < -0.05)
    )

    masks["V14_SHORT_H1_STRONG"] = (
        (df["SIDE"] == "SHORT")
        &
        (h1_distance < -0.05)
        &
        (h1_macd <= 0)
    )

    masks["V14_SHORT_VOLUME12"] = (
        (df["SIDE"] == "SHORT")
        &
        (volume_ratio >= 1.20)
    )

    masks["V14_SHORT_POC075"] = (
        (df["SIDE"] == "SHORT")
        &
        (poc_distance <= 0.75)
    )

    masks["V14_SHORT_WICK45"] = (
        (df["SIDE"] == "SHORT")
        &
        (upper_wick >= 0.45)
    )

    masks["V14_SHORT_CLOSE35"] = (
        (df["SIDE"] == "SHORT")
        &
        (close_position <= 0.35)
    )

    masks["V14_SHORT_QUALITY1"] = (
        (df["SIDE"] == "SHORT")
        &
        rsi.between(
            35,
            55,
            inclusive="both",
        )
        &
        (macd_change_1 < 0)
        &
        (ema_slope_3 < 0)
    )

    masks["V14_SHORT_QUALITY2"] = (
        (df["SIDE"] == "SHORT")
        &
        (macd_change_1 < 0)
        &
        (ema_slope_3 < 0)
        &
        (close_position <= 0.35)
    )

    masks["V14_SHORT_QUALITY3"] = (
        (df["SIDE"] == "SHORT")
        &
        (macd_change_1 < 0)
        &
        (h1_distance < -0.05)
        &
        (close_position <= 0.35)
    )

    masks["V14_SHORT_QUALITY4"] = (
        (df["SIDE"] == "SHORT")
        &
        rsi.between(
            35,
            55,
            inclusive="both",
        )
        &
        (macd_change_1 < 0)
        &
        (ema_slope_3 < 0)
        &
        (poc_distance <= 0.75)
    )

    return masks


# ============================================================
# BUILD CANDIDATE DATASETS
# ============================================================

def build_candidates(events):

    data = ensure_dataframe(
        events,
        "candidate events",
    )

    masks = {}

    masks.update(
        long_filter_masks(
            data
        )
    )

    masks.update(
        short_filter_masks(
            data
        )
    )

    candidates = {}

    for name, mask in masks.items():

        filtered = (
            data[
                mask.fillna(False)
            ]
            .copy()
            .reset_index(drop=True)
        )

        candidates[name] = filtered

    return candidates


# ============================================================
# PERFORMANCE
# ============================================================

def performance(df):

    data = ensure_dataframe(
        df,
        "performance input",
    )

    if data.empty:

        return None

    wins = int(
        (
            data["OUTCOME"]
            ==
            "WIN"
        ).sum()
    )

    losses = int(
        (
            data["OUTCOME"]
            ==
            "LOSS"
        ).sum()
    )

    time_exits = int(
        (
            data["OUTCOME"]
            ==
            "TIME_EXIT"
        ).sum()
    )

    total = len(data)

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

    net_returns = pd.to_numeric(
        data["NET_RETURN_%"],
        errors="coerce",
    ).dropna()

    raw_returns = pd.to_numeric(
        data["RAW_RETURN_%"],
        errors="coerce",
    ).dropna()

    mfe = pd.to_numeric(
        data["MFE_%"],
        errors="coerce",
    ).dropna()

    mae = pd.to_numeric(
        data["MAE_%"],
        errors="coerce",
    ).dropna()

    gross_profit = float(
        net_returns[
            net_returns > 0
        ].sum()
    )

    gross_loss = abs(
        float(
            net_returns[
                net_returns < 0
            ].sum()
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
# CREATE CHRONOLOGICAL BLOCKS
# ============================================================

def create_base_blocks(events):

    data = ensure_dataframe(
        events,
        "block events",
    )

    if data.empty:

        return []

    data = (
        data
        .sort_values("TIME")
        .reset_index(drop=True)
    )

    indexes = np.array_split(
        np.arange(
            len(data)
        ),
        WALK_FORWARD_BLOCKS,
    )

    blocks = []

    for block_number, idx in enumerate(
        indexes,
        start=1,
    ):

        if len(idx) == 0:

            continue

        block = (
            data
            .iloc[idx]
            .copy()
        )

        blocks.append(
            (
                block_number,
                block,
            )
        )

    return blocks


def create_time_block_map(events):

    blocks = create_base_blocks(
        events
    )

    mapping = {}

    for block_number, block in blocks:

        for time_value in block["TIME"]:

            mapping[
                str(time_value)
            ] = block_number

    return mapping


# ============================================================
# WALK-FORWARD TEST
# ============================================================

def walk_forward_test(
    events,
    candidates,
):

    block_map = create_time_block_map(
        events
    )

    rows = []

    for candidate_name, candidate_data in candidates.items():

        candidate_data = (
            candidate_data
            .copy()
        )

        candidate_data["BLOCK"] = (
            candidate_data["TIME"]
            .astype(str)
            .map(block_map)
        )

        for block_number in range(
            1,
            WALK_FORWARD_BLOCKS + 1,
        ):

            block = (
                candidate_data[
                    candidate_data["BLOCK"]
                    ==
                    block_number
                ]
                .copy()
            )

            stats = performance(
                block
            )

            if stats is None:

                rows.append({

                    "BLOCK":
                        block_number,

                    "CANDIDATE":
                        candidate_name,

                    "TRADES":
                        0,

                    "DECISIVE":
                        0,

                    "WINS":
                        0,

                    "LOSSES":
                        0,

                    "TIME_EXITS":
                        0,

                    "WIN_RATE_%":
                        np.nan,

                    "RAW_AVG_RETURN_%":
                        np.nan,

                    "NET_AVG_RETURN_%":
                        np.nan,

                    "NET_MEDIAN_RETURN_%":
                        np.nan,

                    "PROFIT_FACTOR":
                        np.nan,

                    "AVG_MFE_%":
                        np.nan,

                    "AVG_MAE_%":
                        np.nan,

                    "WILSON_LOW_95_%":
                        np.nan,

                    "WILSON_HIGH_95_%":
                        np.nan,
                })

            else:

                row = {

                    "BLOCK":
                        block_number,

                    "CANDIDATE":
                        candidate_name,
                }

                row.update(
                    stats
                )

                rows.append(
                    row
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_candidate(row):

    trades = int(
        row["TOTAL_TRADES"]
    )

    win_rate = safe_float(
        row["OVERALL_WIN_RATE_%"]
    )

    net_avg = safe_float(
        row["OVERALL_NET_AVG_RETURN_%"]
    )

    profit_factor = safe_float(
        row["OVERALL_PROFIT_FACTOR"]
    )

    valid_blocks = int(
        row["VALID_BLOCKS"]
    )

    positive_blocks = int(
        row["POSITIVE_NET_BLOCKS"]
    )

    block_std = safe_float(
        row["BLOCK_RATE_STD_%"]
    )

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    pass_test = (

        trades
        >=
        MIN_TOTAL_TRADES_PASS

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

        and

        pd.notna(block_std)

        and

        block_std
        <=
        PASS_MAX_BLOCK_STD
    )

    if pass_test:

        return "PASS"

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    watch_test = (

        trades
        >=
        MIN_TOTAL_TRADES_WATCH

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
    )

    if watch_test:

        return "WATCH"

    return "REJECT"


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    candidates,
    block_results,
):

    rows = []

    for candidate_name, data in candidates.items():

        overall = performance(
            data
        )

        if overall is None:

            continue

        blocks = (
            block_results[
                block_results["CANDIDATE"]
                ==
                candidate_name
            ]
            .copy()
        )

        rates = (
            blocks["WIN_RATE_%"]
            .dropna()
        )

        valid_blocks = int(
            (
                blocks["DECISIVE"]
                >
                0
            ).sum()
        )

        positive_blocks = int(
            (
                blocks[
                    "NET_AVG_RETURN_%"
                ]
                >
                0
            ).sum()
        )

        blocks_with_min_sample = int(
            (
                blocks["TRADES"]
                >=
                MIN_TRADES_PER_BLOCK
            ).sum()
        )

        blocks_55 = int(
            (
                rates >= 55
            ).sum()
        )

        blocks_60 = int(
            (
                rates >= 60
            ).sum()
        )

        blocks_65 = int(
            (
                rates >= 65
            ).sum()
        )

        blocks_70 = int(
            (
                rates >= 70
            ).sum()
        )

        if rates.empty:

            min_rate = np.nan
            median_rate = np.nan
            max_rate = np.nan
            std_rate = np.nan

        else:

            min_rate = float(
                rates.min()
            )

            median_rate = float(
                rates.median()
            )

            max_rate = float(
                rates.max()
            )

            std_rate = float(
                rates.std(
                    ddof=0
                )
            )

        side = "UNKNOWN"

        if "LONG" in candidate_name:

            side = "LONG"

        elif "SHORT" in candidate_name:

            side = "SHORT"

        row = {

            "CANDIDATE":
                candidate_name,

            "SIDE":
                side,

            "TOTAL_TRADES":
                overall["TRADES"],

            "TOTAL_DECISIVE":
                overall["DECISIVE"],

            "WINS":
                overall["WINS"],

            "LOSSES":
                overall["LOSSES"],

            "TIME_EXITS":
                overall["TIME_EXITS"],

            "OVERALL_WIN_RATE_%":
                overall["WIN_RATE_%"],

            "OVERALL_RAW_AVG_RETURN_%":
                overall[
                    "RAW_AVG_RETURN_%"
                ],

            "OVERALL_NET_AVG_RETURN_%":
                overall[
                    "NET_AVG_RETURN_%"
                ],

            "OVERALL_NET_MEDIAN_RETURN_%":
                overall[
                    "NET_MEDIAN_RETURN_%"
                ],

            "OVERALL_PROFIT_FACTOR":
                overall[
                    "PROFIT_FACTOR"
                ],

            "AVG_MFE_%":
                overall[
                    "AVG_MFE_%"
                ],

            "AVG_MAE_%":
                overall[
                    "AVG_MAE_%"
                ],

            "WILSON_LOW_95_%":
                overall[
                    "WILSON_LOW_95_%"
                ],

            "WILSON_HIGH_95_%":
                overall[
                    "WILSON_HIGH_95_%"
                ],

            "VALID_BLOCKS":
                valid_blocks,

            "POSITIVE_NET_BLOCKS":
                positive_blocks,

            "BLOCKS_WITH_MIN_SAMPLE":
                blocks_with_min_sample,

            "BLOCKS_55PLUS":
                blocks_55,

            "BLOCKS_60PLUS":
                blocks_60,

            "BLOCKS_65PLUS":
                blocks_65,

            "BLOCKS_70PLUS":
                blocks_70,

            "MIN_BLOCK_WIN_RATE_%":
                round(
                    min_rate,
                    2,
                )
                if pd.notna(min_rate)
                else np.nan,

            "MEDIAN_BLOCK_WIN_RATE_%":
                round(
                    median_rate,
                    2,
                )
                if pd.notna(median_rate)
                else np.nan,

            "MAX_BLOCK_WIN_RATE_%":
                round(
                    max_rate,
                    2,
                )
                if pd.notna(max_rate)
                else np.nan,

            "BLOCK_RATE_STD_%":
                round(
                    std_rate,
                    2,
                )
                if pd.notna(std_rate)
                else np.nan,
        }

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    if summary.empty:

        return summary

    summary["STATUS"] = (
        summary.apply(
            classify_candidate,
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
        .map(status_rank)
        .fillna(0)
    )

    summary = (
        summary
        .sort_values(
            by=[
                "_STATUS_RANK",
                "OVERALL_NET_AVG_RETURN_%",
                "OVERALL_PROFIT_FACTOR",
                "OVERALL_WIN_RATE_%",
                "TOTAL_TRADES",
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

    return summary


# ============================================================
# PRINT WIN VS LOSS PROFILE
# ============================================================

def print_profile(profile):

    print()
    print("=" * 150)
    print(
        "V14 WINNING VS LOSING ENTRY PROFILE"
    )
    print("=" * 150)

    if profile.empty:

        print(
            "No profile data."
        )

        return

    for side in [
        "LONG",
        "SHORT",
    ]:

        side_profile = (
            profile[
                profile["SIDE"]
                ==
                side
            ]
            .copy()
        )

        print()
        print("-" * 150)

        print(
            side,
            "WIN vs LOSS ENTRY FEATURES"
        )

        print("-" * 150)

        if side_profile.empty:

            print(
                "No profile rows."
            )

        else:

            print(
                side_profile.to_string(
                    index=False
                )
            )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(summary):

    print()
    print("=" * 180)

    print(
        "V14 ENTRY FILTER RESULTS"
    )

    print("=" * 180)

    if summary.empty:

        print(
            "No V14 results."
        )

        return

    display = [

        "CANDIDATE",

        "STATUS",

        "SIDE",

        "TOTAL_TRADES",

        "TOTAL_DECISIVE",

        "WINS",

        "LOSSES",

        "TIME_EXITS",

        "OVERALL_WIN_RATE_%",

        "OVERALL_NET_AVG_RETURN_%",

        "OVERALL_PROFIT_FACTOR",

        "WILSON_LOW_95_%",

        "WILSON_HIGH_95_%",

        "VALID_BLOCKS",

        "POSITIVE_NET_BLOCKS",

        "BLOCKS_60PLUS",

        "BLOCKS_65PLUS",

        "BLOCKS_70PLUS",

        "MIN_BLOCK_WIN_RATE_%",

        "MEDIAN_BLOCK_WIN_RATE_%",

        "BLOCK_RATE_STD_%",
    ]

    print(
        summary[
            display
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# PRINT FINAL
# ============================================================

def print_final_results(summary):

    if summary.empty:

        return

    display = [

        "CANDIDATE",

        "STATUS",

        "SIDE",

        "TOTAL_TRADES",

        "OVERALL_WIN_RATE_%",

        "OVERALL_NET_AVG_RETURN_%",

        "OVERALL_PROFIT_FACTOR",

        "WILSON_LOW_95_%",

        "POSITIVE_NET_BLOCKS",

        "BLOCKS_70PLUS",

        "MIN_BLOCK_WIN_RATE_%",

        "BLOCK_RATE_STD_%",
    ]

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
    print("=" * 180)

    print(
        "V14 FINAL PASS CANDIDATES:",
        len(passed),
    )

    print("=" * 180)

    if passed.empty:

        print(
            "No candidate passed V14 strict rules."
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
    print("=" * 180)

    print(
        "V14 WATCH CANDIDATES:",
        len(watch),
    )

    print("=" * 180)

    if watch.empty:

        print(
            "No WATCH candidates."
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
    print("=" * 180)

    print(
        "V14 REJECT CANDIDATES:",
        len(rejected),
    )

    print("=" * 180)

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

    # --------------------------------------------------------
    # BEST LONG
    # --------------------------------------------------------

    long_results = (
        summary[
            summary["SIDE"]
            ==
            "LONG"
        ]
        .copy()
    )

    short_results = (
        summary[
            summary["SIDE"]
            ==
            "SHORT"
        ]
        .copy()
    )

    print()
    print("=" * 180)
    print("V14 BEST LONG")
    print("=" * 180)

    if long_results.empty:

        print(
            "No LONG candidate."
        )

    else:

        print(
            long_results[
                display
            ]
            .head(1)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 180)
    print("V14 BEST SHORT")
    print("=" * 180)

    if short_results.empty:

        print(
            "No SHORT candidate."
        )

    else:

        print(
            short_results[
                display
            ]
            .head(1)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # BEST OVERALL
    # --------------------------------------------------------

    best = summary.iloc[0]

    print()
    print("=" * 180)
    print(
        "V14 BEST OVERALL CANDIDATE"
    )
    print("=" * 180)

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
        "TOTAL TRADES:",
        int(
            best["TOTAL_TRADES"]
        ),
    )

    print(
        "DECISIVE TRADES:",
        int(
            best["TOTAL_DECISIVE"]
        ),
    )

    print(
        "WIN RATE:",
        f'{best["OVERALL_WIN_RATE_%"]:.2f}%',
    )

    print(
        "NET AVG RETURN:",
        f'{best["OVERALL_NET_AVG_RETURN_%"]:.4f}%',
    )

    print(
        "PROFIT FACTOR:",
        f'{best["OVERALL_PROFIT_FACTOR"]:.4f}',
    )

    print(
        "95% WILSON RANGE:",
        f'{best["WILSON_LOW_95_%"]:.2f}%'
        " to "
        f'{best["WILSON_HIGH_95_%"]:.2f}%',
    )

    print(
        "POSITIVE NET BLOCKS:",
        int(
            best["POSITIVE_NET_BLOCKS"]
        ),
        "/",
        WALK_FORWARD_BLOCKS,
    )

    print(
        "BLOCKS >= 70%:",
        int(
            best["BLOCKS_70PLUS"]
        ),
    )

    print(
        "MIN BLOCK WIN RATE:",
        f'{best["MIN_BLOCK_WIN_RATE_%"]:.2f}%',
    )

    print(
        "BLOCK STD:",
        f'{best["BLOCK_RATE_STD_%"]:.2f}',
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 125)

    print(
        "COINDCX FUTURES HISTORICAL V14"
    )

    print(
        "WINNING VS LOSING SIGNAL ANALYSIS"
    )

    print(
        "+ ENTRY FILTER DISCOVERY"
    )

    print("=" * 125)

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
        "days",
    )

    print(
        "WALK-FORWARD BLOCKS:",
        WALK_FORWARD_BLOCKS,
    )

    print(
        "FIXED TP:",
        f"{TP_PCT:.2f}%",
    )

    print(
        "FIXED SL:",
        f"{SL_PCT:.2f}%",
    )

    print(
        "MAX HOLD:",
        MAX_HOLD_BARS,
        "x 5-minute candles",
    )

    print(
        "ROUND-TRIP COST:",
        f"{ROUND_TRIP_COST_PCT:.2f}%",
    )

    print(
        "SAME-BAR POLICY:",
        SAME_BAR_POLICY,
    )

    print()
    print(
        "V14 IS HISTORICAL RESEARCH ONLY."
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

    print_history_range(
        df_5m,
        df_1h,
        requested_days,
    )

    # ========================================================
    # PREPARE
    # ========================================================

    df = prepare_data(
        df_5m,
        df_1h,
    )

    print()
    print(
        "USABLE 5M CANDLES:",
        len(df),
    )

    print()
    print(
        "REGIME COUNTS:"
    )

    print(
        df[
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
        "Collecting V14 LONG / SHORT base events..."
    )

    events = collect_base_events(
        df
    )

    if events.empty:

        print(
            "No V14 base events found."
        )

        return

    print()
    print(
        "TOTAL V14 EVENTS:",
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
    # PROFILE
    # ========================================================

    print()
    print(
        "Analysing winning vs losing entry features..."
    )

    profile = build_win_loss_profile(
        events
    )

    print_profile(
        profile
    )

    # ========================================================
    # CANDIDATES
    # ========================================================

    print()
    print(
        "Building V14 entry-filter candidates..."
    )

    candidates = build_candidates(
        events
    )

    print(
        "TOTAL V14 CANDIDATES:",
        len(candidates),
    )

    print()

    for name, candidate_data in candidates.items():

        print(
            name,
            "->",
            len(candidate_data),
            "trades"
        )

    # ========================================================
    # WALK FORWARD
    # ========================================================

    print()
    print(
        "Running V14 8-block walk-forward validation..."
    )

    block_results = (
        walk_forward_test(
            events,
            candidates,
        )
    )

    summary = build_summary(
        candidates,
        block_results,
    )

    # ========================================================
    # SAVE
    # ========================================================

    events.to_csv(
        EVENT_FILE,
        index=False,
    )

    profile.to_csv(
        PROFILE_FILE,
        index=False,
    )

    block_results.to_csv(
        DETAIL_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        "V14 EVENTS SAVED:",
        EVENT_FILE,
    )

    print(
        "V14 WIN/LOSS PROFILE SAVED:",
        PROFILE_FILE,
    )

    print(
        "V14 BLOCK RESULTS SAVED:",
        DETAIL_FILE,
    )

    print(
        "V14 SUMMARY SAVED:",
        SUMMARY_FILE,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print_summary(
        summary
    )

    print_final_results(
        summary
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 125)

    print(
        "V14 INTERPRETATION"
    )

    print("=" * 125)

    print(
        "PASS = Entry filter produced positive net expectancy"
    )

    print(
        "and acceptable walk-forward consistency."
    )

    print()

    print(
        "WATCH = Promising filter, but not strong enough"
    )

    print(
        "for live use yet."
    )

    print()

    print(
        "REJECT = Entry filter did not produce robust edge."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "V14 does NOT optimize TP/SL."
    )

    print(
        "It uses one fixed TP/SL model so that we can"
    )

    print(
        "study ENTRY QUALITY instead of fitting exits."
    )

    print()

    print(
        "A high win rate alone is NOT sufficient."
    )

    print(
        "Net average return, profit factor, sample size,"
    )

    print(
        "Wilson confidence interval and block consistency"
    )

    print(
        "must also be considered."
    )

    print()

    print(
        "Any successful V14 filter must be validated again"
    )

    print(
        "on unseen data before Telegram/live deployment."
    )

    print()

    print(
        "Backtest results do not guarantee future performance."
    )

    print(
        "No real orders were placed."
    )

    print()
    print("=" * 125)

    print(
        "FUTURES HISTORICAL V14 COMPLETE"
    )

    print("=" * 125)


if __name__ == "__main__":
    main()
