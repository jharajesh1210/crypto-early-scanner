import math
import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v11 as v11


# ============================================================
# COINDCX FUTURES HISTORICAL V12
# ============================================================
#
# PURPOSE
# -------
# V11 showed that a LONG-only Weak Hammer approach was not
# robust enough across a full 365-day / 8-block test.
#
# V12 changes the STRUCTURE of the test:
#
#   1. LONG + SHORT setups
#   2. Market regime classification
#   3. TP / SL path-based outcome
#   4. MFE / MAE analysis
#   5. 365-day history attempt
#   6. 8 chronological walk-forward blocks
#   7. Estimated round-trip cost
#   8. PASS / WATCH / REJECT
#
# HISTORICAL RESEARCH ONLY
# NO REAL ORDERS
# ============================================================


DETAIL_FILE = "crypto_futures_v12_results.csv"
SUMMARY_FILE = "crypto_futures_v12_summary.csv"
EVENT_FILE = "crypto_futures_v12_events.csv"


# ============================================================
# SETTINGS
# ============================================================

HISTORY_ATTEMPTS = [
    365,
    270,
    180,
]

WALK_FORWARD_BLOCKS = 8

ROUND_TRIP_COST_PCT = 0.10

# ------------------------------------------------------------
# TP / SL TEST
# ------------------------------------------------------------
#
# These are research assumptions.
# They are NOT guaranteed live execution prices.
#
# TP and SL are measured from the signal candle close.
# ------------------------------------------------------------

TAKE_PROFIT_PCT = 0.80
STOP_LOSS_PCT = 0.50

# 4 hours on 5-minute candles = 48 bars
MAX_HOLD_BARS = 48

# If both TP and SL are touched inside the SAME 5m candle,
# OHLC data cannot tell us which happened first.
#
# To avoid optimistic backtesting, V12 counts this as SL.
SAME_BAR_POLICY = "SL_FIRST"


# ============================================================
# SAMPLE / ROBUSTNESS RULES
# ============================================================

MIN_TOTAL_TRADES_PASS = 50
MIN_TOTAL_TRADES_WATCH = 30

MIN_TRADES_PER_BLOCK = 4

PASS_MIN_WIN_RATE = 70.0
PASS_MIN_POSITIVE_BLOCKS = 6
PASS_MIN_BLOCKS_65PLUS = 5
PASS_MIN_VALID_BLOCKS = 6
PASS_MIN_BLOCK_WIN_RATE = 55.0
PASS_MAX_BLOCK_STD = 20.0
PASS_MIN_PROFIT_FACTOR = 1.20

WATCH_MIN_WIN_RATE = 62.0
WATCH_MIN_POSITIVE_BLOCKS = 5
WATCH_MIN_VALID_BLOCKS = 5
WATCH_MIN_PROFIT_FACTOR = 1.00

WILSON_Z = 1.96


# ============================================================
# SAFE HELPERS
# ============================================================

def ensure_dataframe(data, name="data"):

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        return pd.DataFrame(data)

    except Exception as exc:
        raise TypeError(
            f"{name} could not be converted to DataFrame. "
            f"Received type: {type(data)}"
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

        if b == 0:
            return default

        return a / b

    except Exception:
        return default


# ============================================================
# WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(
    wins,
    total,
    z=WILSON_Z,
):

    if total <= 0:
        return np.nan, np.nan

    p = wins / total

    denominator = (
        1.0
        +
        (z ** 2 / total)
    )

    centre = (
        p
        +
        (z ** 2 / (2.0 * total))
    )

    margin = (
        z
        *
        math.sqrt(
            (
                p * (1.0 - p)
                +
                (z ** 2 / (4.0 * total))
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
            print("=" * 100)
            print(
                "V12 TRYING HISTORY:",
                days,
                "DAYS",
            )
            print("=" * 100)

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
                    "No 5-minute data."
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
                    "No 1-hour data."
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
                "days",
            )

            print(
                "ERROR:",
                str(exc),
            )

            print(
                "Trying shorter history..."
            )

    raise RuntimeError(
        "All V12 history download attempts failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# DATA RANGE
# ============================================================

def print_history_range(
    df_5m,
    df_1h,
    requested_days,
):

    print()
    print("=" * 100)
    print("V12 DATA RANGE")
    print("=" * 100)

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
                round(
                    actual_days,
                    2,
                ),
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
# PREPARE EXISTING INDICATORS
# ============================================================

def prepare_data(
    df_5m,
    df_1h,
):

    print()
    print(
        "Calculating base 5M indicators..."
    )

    df_5m = v1.calculate_indicators(
        df_5m
    )

    df_5m = ensure_dataframe(
        df_5m,
        "base indicators result",
    )

    print(
        "Calculating Volume Profile..."
    )

    df_5m = v1.add_volume_profile(
        df_5m
    )

    df_5m = ensure_dataframe(
        df_5m,
        "volume profile result",
    )

    print(
        "Attaching original 1H trend..."
    )

    df_5m = v1.attach_1h_trend(
        df_5m,
        df_1h,
    )

    df_5m = ensure_dataframe(
        df_5m,
        "original 1H trend",
    )

    print(
        "Adding V6 indicators..."
    )

    df_5m = (
        v11.v10.v6.add_v6_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V6 result",
    )

    print(
        "Calculating V7 1H indicators..."
    )

    df_1h_v7 = (
        v11.v10.v7.prepare_1h_indicators(
            df_1h
        )
    )

    df_1h_v7 = ensure_dataframe(
        df_1h_v7,
        "V7 1H result",
    )

    print(
        "Attaching V7 1H indicators..."
    )

    df_5m = (
        v11.v10.v7.attach_v7_1h_data(
            df_5m,
            df_1h_v7,
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V7 attach result",
    )

    print(
        "Adding V8 indicators..."
    )

    df_5m = (
        v11.v10.v8.add_v8_indicators(
            df_5m
        )
    )

    print(
        "Adding V9 indicators..."
    )

    df_5m = (
        v11.v10.v9.add_v9_indicators(
            df_5m
        )
    )

    print(
        "Adding V10 indicators..."
    )

    df_5m = (
        v11.v10.add_v10_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V10 result",
    )

    return df_5m


# ============================================================
# V12 MARKET REGIME
# ============================================================

def add_v12_market_regime(df):

    data = ensure_dataframe(
        df,
        "V12 regime input",
    )

    # --------------------------------------------------------
    # 5M EMA slope
    # --------------------------------------------------------

    data["V12_EMA20_RISING"] = (
        data["ema20"]
        >
        data["ema20"].shift(3)
    ).astype(int)

    data["V12_EMA20_FALLING"] = (
        data["ema20"]
        <
        data["ema20"].shift(3)
    ).astype(int)

    # --------------------------------------------------------
    # 1H trend
    # --------------------------------------------------------

    data["V12_H1_BULL"] = (
        (
            data["ema20_1h"]
            >
            data["ema50_1h"]
        )
        &
        (
            data["macd_hist_1h"]
            >= 0
        )
    ).astype(int)

    data["V12_H1_BEAR"] = (
        (
            data["ema20_1h"]
            <
            data["ema50_1h"]
        )
        &
        (
            data["macd_hist_1h"]
            <= 0
        )
    ).astype(int)

    # --------------------------------------------------------
    # 5M directional trend
    # --------------------------------------------------------

    data["V12_5M_BULL"] = (
        (
            data["close"]
            >
            data["ema20"]
        )
        &
        (
            data["ema20"]
            >
            data["ema50"]
        )
        &
        (
            data["V12_EMA20_RISING"]
            == 1
        )
    ).astype(int)

    data["V12_5M_BEAR"] = (
        (
            data["close"]
            <
            data["ema20"]
        )
        &
        (
            data["ema20"]
            <
            data["ema50"]
        )
        &
        (
            data["V12_EMA20_FALLING"]
            == 1
        )
    ).astype(int)

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    bull = (
        (data["V12_H1_BULL"] == 1)
        &
        (data["V12_5M_BULL"] == 1)
    )

    bear = (
        (data["V12_H1_BEAR"] == 1)
        &
        (data["V12_5M_BEAR"] == 1)
    )

    data["V12_REGIME"] = "RANGE"

    data.loc[
        bull,
        "V12_REGIME",
    ] = "UPTREND"

    data.loc[
        bear,
        "V12_REGIME",
    ] = "DOWNTREND"

    return data


# ============================================================
# CANDLE STRUCTURE
# ============================================================

def add_v12_candle_features(df):

    data = ensure_dataframe(
        df,
        "V12 candle input",
    )

    candle_range = (
        data["high"]
        -
        data["low"]
    ).replace(
        0,
        np.nan,
    )

    body = (
        data["close"]
        -
        data["open"]
    ).abs()

    upper_wick = (
        data["high"]
        -
        data[
            [
                "open",
                "close",
            ]
        ].max(
            axis=1
        )
    )

    lower_wick = (
        data[
            [
                "open",
                "close",
            ]
        ].min(
            axis=1
        )
        -
        data["low"]
    )

    data["V12_BODY_RATIO"] = (
        body
        /
        candle_range
    )

    data["V12_UPPER_WICK_RATIO"] = (
        upper_wick
        /
        candle_range
    )

    data["V12_LOWER_WICK_RATIO"] = (
        lower_wick
        /
        candle_range
    )

    data["V12_CLOSE_POSITION"] = (
        (
            data["close"]
            -
            data["low"]
        )
        /
        candle_range
    )

    data["V12_GREEN"] = (
        data["close"]
        >
        data["open"]
    ).astype(int)

    data["V12_RED"] = (
        data["close"]
        <
        data["open"]
    ).astype(int)

    return data


# ============================================================
# LONG SETUP
# ============================================================

def long_setup(data, i):

    if i < 5:
        return False

    row = data.iloc[i]
    prev = data.iloc[i - 1]

    # Lower wick rejection / sweep
    wick_ok = (
        safe_float(
            row["V12_LOWER_WICK_RATIO"],
            0,
        )
        >=
        0.35
    )

    close_ok = (
        safe_float(
            row["V12_CLOSE_POSITION"],
            0,
        )
        >=
        0.55
    )

    sweep_ok = (
        row["low"]
        <=
        prev["low"]
    )

    momentum_ok = (
        safe_float(
            row["macd_hist"],
            -999,
        )
        >
        safe_float(
            prev["macd_hist"],
            -999,
        )
    )

    rsi = safe_float(
        row["rsi"]
    )

    rsi_ok = (
        pd.notna(rsi)
        and
        rsi >= 35
        and
        rsi <= 70
    )

    return bool(
        wick_ok
        and
        close_ok
        and
        sweep_ok
        and
        momentum_ok
        and
        rsi_ok
    )


# ============================================================
# SHORT SETUP
# ============================================================

def short_setup(data, i):

    if i < 5:
        return False

    row = data.iloc[i]
    prev = data.iloc[i - 1]

    # Upper wick rejection / sweep
    wick_ok = (
        safe_float(
            row["V12_UPPER_WICK_RATIO"],
            0,
        )
        >=
        0.35
    )

    close_position = safe_float(
        row["V12_CLOSE_POSITION"]
    )

    close_ok = (
        pd.notna(close_position)
        and
        close_position
        <=
        0.45
    )

    sweep_ok = (
        row["high"]
        >=
        prev["high"]
    )

    momentum_ok = (
        safe_float(
            row["macd_hist"],
            999,
        )
        <
        safe_float(
            prev["macd_hist"],
            999,
        )
    )

    rsi = safe_float(
        row["rsi"]
    )

    rsi_ok = (
        pd.notna(rsi)
        and
        rsi >= 30
        and
        rsi <= 65
    )

    return bool(
        wick_ok
        and
        close_ok
        and
        sweep_ok
        and
        momentum_ok
        and
        rsi_ok
    )


# ============================================================
# TP / SL + MFE / MAE SIMULATOR
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
                TAKE_PROFIT_PCT / 100.0
            )
        )

        sl_price = (
            entry
            *
            (
                1.0
                -
                STOP_LOSS_PCT / 100.0
            )
        )

    else:

        tp_price = (
            entry
            *
            (
                1.0
                -
                TAKE_PROFIT_PCT / 100.0
            )
        )

        sl_price = (
            entry
            *
            (
                1.0
                +
                STOP_LOSS_PCT / 100.0
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

    tp_hit_bar = None
    sl_hit_bar = None

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

        # Same candle touched TP and SL.
        # Conservative assumption = SL first.
        if hit_tp and hit_sl:

            if SAME_BAR_POLICY == "SL_FIRST":

                outcome = "LOSS"
                exit_price = sl_price
                exit_bars = offset
                sl_hit_bar = offset

            else:

                outcome = "WIN"
                exit_price = tp_price
                exit_bars = offset
                tp_hit_bar = offset

            break

        if hit_tp:

            outcome = "WIN"
            exit_price = tp_price
            exit_bars = offset
            tp_hit_bar = offset
            break

        if hit_sl:

            outcome = "LOSS"
            exit_price = sl_price
            exit_bars = offset
            sl_hit_bar = offset
            break

    # --------------------------------------------------------
    # MFE / MAE across the observed path until exit
    # --------------------------------------------------------

    observed = (
        future.iloc[
            :exit_bars
        ]
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

        "ENTRY_PRICE":
            round(
                entry,
                8,
            ),

        "EXIT_PRICE":
            round(
                exit_price,
                8,
            ),

        "TP_PRICE":
            round(
                tp_price,
                8,
            ),

        "SL_PRICE":
            round(
                sl_price,
                8,
            ),

        "EXIT_BARS":
            int(
                exit_bars
            ),

        "TP_HIT_BAR":
            tp_hit_bar,

        "SL_HIT_BAR":
            sl_hit_bar,

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
# COLLECT V12 EVENTS
# ============================================================

def collect_events(df):

    data = ensure_dataframe(
        df,
        "collect events input",
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

        # ====================================================
        # LONG
        # ====================================================

        if (
            regime == "UPTREND"
            and
            long_setup(
                data,
                i,
            )
        ):

            result = simulate_trade(
                data,
                i,
                "LONG",
            )

            if result is not None:

                event = {

                    "PAIR":
                        v1.PAIR,

                    "TIME":
                        row["datetime"].isoformat(),

                    "SIDE":
                        "LONG",

                    "REGIME":
                        regime,

                    "SETUP":
                        "LOWER_WICK_SWEEP",

                    "RSI_5M":
                        round(
                            safe_float(
                                row["rsi"]
                            ),
                            2,
                        ),

                    "MACD_HIST_5M":
                        round(
                            safe_float(
                                row["macd_hist"]
                            ),
                            8,
                        ),

                    "EMA20":
                        round(
                            safe_float(
                                row["ema20"]
                            ),
                            8,
                        ),

                    "EMA50":
                        round(
                            safe_float(
                                row["ema50"]
                            ),
                            8,
                        ),

                    "EMA20_1H":
                        round(
                            safe_float(
                                row["ema20_1h"]
                            ),
                            8,
                        ),

                    "EMA50_1H":
                        round(
                            safe_float(
                                row["ema50_1h"]
                            ),
                            8,
                        ),

                    "LOWER_WICK_RATIO":
                        round(
                            safe_float(
                                row[
                                    "V12_LOWER_WICK_RATIO"
                                ]
                            ),
                            4,
                        ),

                    "UPPER_WICK_RATIO":
                        round(
                            safe_float(
                                row[
                                    "V12_UPPER_WICK_RATIO"
                                ]
                            ),
                            4,
                        ),

                    "CLOSE_POSITION":
                        round(
                            safe_float(
                                row[
                                    "V12_CLOSE_POSITION"
                                ]
                            ),
                            4,
                        ),
                }

                event.update(
                    result
                )

                events.append(
                    event
                )

        # ====================================================
        # SHORT
        # ====================================================

        if (
            regime == "DOWNTREND"
            and
            short_setup(
                data,
                i,
            )
        ):

            result = simulate_trade(
                data,
                i,
                "SHORT",
            )

            if result is not None:

                event = {

                    "PAIR":
                        v1.PAIR,

                    "TIME":
                        row["datetime"].isoformat(),

                    "SIDE":
                        "SHORT",

                    "REGIME":
                        regime,

                    "SETUP":
                        "UPPER_WICK_SWEEP",

                    "RSI_5M":
                        round(
                            safe_float(
                                row["rsi"]
                            ),
                            2,
                        ),

                    "MACD_HIST_5M":
                        round(
                            safe_float(
                                row["macd_hist"]
                            ),
                            8,
                        ),

                    "EMA20":
                        round(
                            safe_float(
                                row["ema20"]
                            ),
                            8,
                        ),

                    "EMA50":
                        round(
                            safe_float(
                                row["ema50"]
                            ),
                            8,
                        ),

                    "EMA20_1H":
                        round(
                            safe_float(
                                row["ema20_1h"]
                            ),
                            8,
                        ),

                    "EMA50_1H":
                        round(
                            safe_float(
                                row["ema50_1h"]
                            ),
                            8,
                        ),

                    "LOWER_WICK_RATIO":
                        round(
                            safe_float(
                                row[
                                    "V12_LOWER_WICK_RATIO"
                                ]
                            ),
                            4,
                        ),

                    "UPPER_WICK_RATIO":
                        round(
                            safe_float(
                                row[
                                    "V12_UPPER_WICK_RATIO"
                                ]
                            ),
                            4,
                        ),

                    "CLOSE_POSITION":
                        round(
                            safe_float(
                                row[
                                    "V12_CLOSE_POSITION"
                                ]
                            ),
                            4,
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
# CREATE 8 CHRONOLOGICAL BLOCKS
# ============================================================

def create_blocks(events):

    data = ensure_dataframe(
        events,
        "create blocks input",
    )

    if data.empty:
        return []

    data = (
        data
        .sort_values(
            "TIME"
        )
        .reset_index(
            drop=True
        )
    )

    indexes = np.array_split(
        np.arange(
            len(data)
        ),
        WALK_FORWARD_BLOCKS,
    )

    blocks = []

    for number, idx in enumerate(
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
                number,
                block,
            )
        )

    return blocks


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

    trades = (
        wins
        +
        losses
        +
        time_exits
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

    positive_returns = (
        net_returns[
            net_returns > 0
        ].sum()
    )

    negative_returns = abs(
        net_returns[
            net_returns < 0
        ].sum()
    )

    if negative_returns > 0:

        profit_factor = (
            positive_returns
            /
            negative_returns
        )

    elif positive_returns > 0:

        profit_factor = np.inf

    else:

        profit_factor = 0.0

    return {

        "TRADES":
            int(
                trades
            ),

        "DECISIVE":
            int(
                decisive
            ),

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
            )
            if np.isfinite(
                profit_factor
            )
            else 999.0,

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
            if pd.notna(
                wilson_low
            )
            else np.nan,

        "WILSON_HIGH_95_%":
            round(
                float(
                    wilson_high
                ),
                2,
            )
            if pd.notna(
                wilson_high
            )
            else np.nan,
    }


# ============================================================
# TEST GROUPS
# ============================================================

def get_groups(events):

    groups = {

        "V12_ALL":
            events,

        "V12_LONG":
            events[
                events["SIDE"]
                ==
                "LONG"
            ].copy(),

        "V12_SHORT":
            events[
                events["SIDE"]
                ==
                "SHORT"
            ].copy(),

        "V12_UPTREND_LONG":
            events[
                (
                    events["SIDE"]
                    ==
                    "LONG"
                )
                &
                (
                    events["REGIME"]
                    ==
                    "UPTREND"
                )
            ].copy(),

        "V12_DOWNTREND_SHORT":
            events[
                (
                    events["SIDE"]
                    ==
                    "SHORT"
                )
                &
                (
                    events["REGIME"]
                    ==
                    "DOWNTREND"
                )
            ].copy(),
    }

    return groups


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward_test(events):

    rows = []

    blocks = create_blocks(
        events
    )

    for block_number, block in blocks:

        block_start = (
            block["TIME"].iloc[0]
            if not block.empty
            else ""
        )

        block_end = (
            block["TIME"].iloc[-1]
            if not block.empty
            else ""
        )

        groups = get_groups(
            block
        )

        for candidate, filtered in groups.items():

            stats = performance(
                filtered
            )

            if stats is None:

                rows.append({

                    "BLOCK":
                        block_number,

                    "BLOCK_START":
                        block_start,

                    "BLOCK_END":
                        block_end,

                    "CANDIDATE":
                        candidate,

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

                    "BLOCK_START":
                        block_start,

                    "BLOCK_END":
                        block_end,

                    "CANDIDATE":
                        candidate,
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

    blocks_65 = int(
        row["BLOCKS_65PLUS"]
    )

    blocks_min_sample = int(
        row["BLOCKS_WITH_MIN_SAMPLE"]
    )

    min_block = safe_float(
        row["MIN_BLOCK_WIN_RATE_%"]
    )

    std = safe_float(
        row["BLOCK_RATE_STD_%"]
    )

    pass_test = (

        trades
        >=
        MIN_TOTAL_TRADES_PASS

        and

        pd.notna(
            win_rate
        )

        and

        win_rate
        >=
        PASS_MIN_WIN_RATE

        and

        pd.notna(
            net_avg
        )

        and

        net_avg
        >
        0

        and

        pd.notna(
            profit_factor
        )

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

        blocks_65
        >=
        PASS_MIN_BLOCKS_65PLUS

        and

        blocks_min_sample
        >=
        PASS_MIN_VALID_BLOCKS

        and

        pd.notna(
            min_block
        )

        and

        min_block
        >=
        PASS_MIN_BLOCK_WIN_RATE

        and

        pd.notna(
            std
        )

        and

        std
        <=
        PASS_MAX_BLOCK_STD
    )

    if pass_test:
        return "PASS"

    watch_test = (

        trades
        >=
        MIN_TOTAL_TRADES_WATCH

        and

        pd.notna(
            win_rate
        )

        and

        win_rate
        >=
        WATCH_MIN_WIN_RATE

        and

        pd.notna(
            net_avg
        )

        and

        net_avg
        >
        0

        and

        pd.notna(
            profit_factor
        )

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
    events,
    block_results,
):

    rows = []

    groups = get_groups(
        events
    )

    for candidate, filtered in groups.items():

        overall = performance(
            filtered
        )

        if overall is None:
            continue

        blocks = (
            block_results[
                block_results["CANDIDATE"]
                ==
                candidate
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

        blocks_with_min_sample = int(
            (
                blocks["TRADES"]
                >=
                MIN_TRADES_PER_BLOCK
            ).sum()
        )

        positive_blocks = int(
            (
                blocks["NET_AVG_RETURN_%"]
                >
                0
            ).sum()
        )

        blocks_60 = int(
            (
                rates
                >=
                60
            ).sum()
        )

        blocks_65 = int(
            (
                rates
                >=
                65
            ).sum()
        )

        blocks_70 = int(
            (
                rates
                >=
                70
            ).sum()
        )

        blocks_75 = int(
            (
                rates
                >=
                75
            ).sum()
        )

        blocks_80 = int(
            (
                rates
                >=
                80
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

        row = {

            "CANDIDATE":
                candidate,

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

            "BLOCKS_60PLUS":
                blocks_60,

            "BLOCKS_65PLUS":
                blocks_65,

            "BLOCKS_70PLUS":
                blocks_70,

            "BLOCKS_75PLUS":
                blocks_75,

            "BLOCKS_80PLUS":
                blocks_80,

            "MIN_BLOCK_WIN_RATE_%":
                round(
                    min_rate,
                    2,
                )
                if pd.notna(
                    min_rate
                )
                else np.nan,

            "MEDIAN_BLOCK_WIN_RATE_%":
                round(
                    median_rate,
                    2,
                )
                if pd.notna(
                    median_rate
                )
                else np.nan,

            "MAX_BLOCK_WIN_RATE_%":
                round(
                    max_rate,
                    2,
                )
                if pd.notna(
                    max_rate
                )
                else np.nan,

            "BLOCK_RATE_STD_%":
                round(
                    std_rate,
                    2,
                )
                if pd.notna(
                    std_rate
                )
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

    rank = {
        "PASS": 3,
        "WATCH": 2,
        "REJECT": 1,
    }

    summary["_RANK"] = (
        summary["STATUS"]
        .map(
            rank
        )
        .fillna(0)
    )

    summary = (
        summary
        .sort_values(
            by=[
                "_RANK",
                "OVERALL_WIN_RATE_%",
                "TOTAL_TRADES",
                "OVERALL_PROFIT_FACTOR",
                "OVERALL_NET_AVG_RETURN_%",
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
                "_RANK"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ============================================================
# PRINT BLOCK RESULTS
# ============================================================

def print_block_results(
    block_results,
):

    print()
    print("=" * 145)
    print(
        "V12 WALK-FORWARD BLOCK RESULTS"
    )
    print("=" * 145)

    if block_results.empty:

        print(
            "No V12 block results."
        )

        return

    display = [

        "BLOCK",

        "CANDIDATE",

        "TRADES",

        "DECISIVE",

        "WINS",

        "LOSSES",

        "TIME_EXITS",

        "WIN_RATE_%",

        "NET_AVG_RETURN_%",

        "PROFIT_FACTOR",

        "AVG_MFE_%",

        "AVG_MAE_%",
    ]

    print(
        block_results[
            display
        ].to_string(
            index=False
        )
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary,
):

    print()
    print("=" * 165)
    print(
        "V12 OVERALL ROBUSTNESS RESULTS"
    )
    print("=" * 165)

    if summary.empty:

        print(
            "No V12 summary results."
        )

        return

    display = [

        "CANDIDATE",

        "STATUS",

        "TOTAL_TRADES",

        "TOTAL_DECISIVE",

        "WINS",

        "LOSSES",

        "TIME_EXITS",

        "OVERALL_WIN_RATE_%",

        "OVERALL_NET_AVG_RETURN_%",

        "OVERALL_PROFIT_FACTOR",

        "AVG_MFE_%",

        "AVG_MAE_%",

        "WILSON_LOW_95_%",

        "WILSON_HIGH_95_%",

        "VALID_BLOCKS",

        "POSITIVE_NET_BLOCKS",

        "BLOCKS_65PLUS",

        "BLOCKS_70PLUS",

        "BLOCKS_75PLUS",

        "BLOCKS_80PLUS",

        "MIN_BLOCK_WIN_RATE_%",

        "MEDIAN_BLOCK_WIN_RATE_%",

        "BLOCK_RATE_STD_%",
    ]

    print(
        summary[
            display
        ].to_string(
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
    print("=" * 165)

    print(
        "V12 FINAL PASS CANDIDATES:",
        len(
            passed
        ),
    )

    print("=" * 165)

    if passed.empty:

        print(
            "No candidate passed V12 strict robustness rules."
        )

    else:

        print(
            passed[
                display
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 165)

    print(
        "V12 WATCH CANDIDATES:",
        len(
            watch
        ),
    )

    print("=" * 165)

    if watch.empty:

        print(
            "No WATCH candidates."
        )

    else:

        print(
            watch[
                display
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 165)

    print(
        "V12 REJECT CANDIDATES:",
        len(
            rejected
        ),
    )

    print("=" * 165)

    if rejected.empty:

        print(
            "No rejected candidates."
        )

    else:

        print(
            rejected[
                display
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    print()
    print("=" * 165)
    print(
        "V12 BEST CANDIDATE"
    )
    print("=" * 165)

    if not passed.empty:

        best = passed.iloc[0]

    elif not watch.empty:

        best = watch.iloc[0]

    else:

        best = summary.iloc[0]

    print(
        "CANDIDATE:",
        best["CANDIDATE"],
    )

    print(
        "STATUS:",
        best["STATUS"],
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
        "WINS:",
        int(
            best["WINS"]
        ),
    )

    print(
        "LOSSES:",
        int(
            best["LOSSES"]
        ),
    )

    print(
        "TIME EXITS:",
        int(
            best["TIME_EXITS"]
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
        "AVG MFE:",
        f'{best["AVG_MFE_%"]:.4f}%',
    )

    print(
        "AVG MAE:",
        f'{best["AVG_MAE_%"]:.4f}%',
    )

    print(
        "95% WILSON RANGE:",
        f'{best["WILSON_LOW_95_%"]:.2f}%'
        " to "
        f'{best["WILSON_HIGH_95_%"]:.2f}%',
    )

    print(
        "VALID BLOCKS:",
        int(
            best["VALID_BLOCKS"]
        ),
        "/",
        WALK_FORWARD_BLOCKS,
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
        "BLOCKS >= 75%:",
        int(
            best["BLOCKS_75PLUS"]
        ),
    )

    print(
        "BLOCKS >= 80%:",
        int(
            best["BLOCKS_80PLUS"]
        ),
    )

    print(
        "MIN BLOCK WIN RATE:",
        f'{best["MIN_BLOCK_WIN_RATE_%"]:.2f}%',
    )

    print(
        "BLOCK WIN-RATE STD:",
        f'{best["BLOCK_RATE_STD_%"]:.2f}',
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 115)

    print(
        "COINDCX FUTURES HISTORICAL V12"
    )

    print(
        "LONG + SHORT / REGIME / TP-SL / MFE-MAE TEST"
    )

    print("=" * 115)

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
        "WALK-FORWARD BLOCKS:",
        WALK_FORWARD_BLOCKS,
    )

    print(
        "TAKE PROFIT:",
        f"{TAKE_PROFIT_PCT:.2f}%",
    )

    print(
        "STOP LOSS:",
        f"{STOP_LOSS_PCT:.2f}%",
    )

    print(
        "MAX HOLD:",
        MAX_HOLD_BARS,
        "x 5m candles",
    )

    print(
        "ROUND-TRIP COST:",
        f"{ROUND_TRIP_COST_PCT:.2f}%",
    )

    print(
        "SAME-BAR TP/SL POLICY:",
        SAME_BAR_POLICY,
    )

    print()
    print(
        "V12 IS HISTORICAL RESEARCH ONLY."
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

    df_5m = prepare_data(
        df_5m,
        df_1h,
    )

    print(
        "Adding V12 market regime..."
    )

    df_5m = add_v12_market_regime(
        df_5m
    )

    print(
        "Adding V12 candle structure..."
    )

    df_5m = add_v12_candle_features(
        df_5m
    )

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
    ]

    missing = [
        col
        for col in required
        if col not in df_5m.columns
    ]

    if missing:

        raise RuntimeError(
            "V12 missing required columns: "
            +
            ", ".join(
                missing
            )
        )

    df_5m = (
        df_5m
        .dropna(
            subset=[
                col
                for col in required
                if col != "V12_REGIME"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "USABLE 5M CANDLES:",
        len(
            df_5m
        ),
    )

    print()
    print(
        "REGIME COUNTS:"
    )

    print(
        df_5m[
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
        "Finding V12 LONG + SHORT events..."
    )

    events = collect_events(
        df_5m
    )

    if events.empty:

        print(
            "No V12 events found."
        )

        return

    print(
        "TOTAL V12 EVENTS:",
        len(
            events
        ),
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
    # WALK FORWARD
    # ========================================================

    print()
    print(
        "Running V12 8-block walk-forward validation..."
    )

    block_results = walk_forward_test(
        events
    )

    summary = build_summary(
        events,
        block_results,
    )

    # ========================================================
    # SAVE
    # ========================================================

    events.to_csv(
        EVENT_FILE,
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
        "V12 events saved:",
        EVENT_FILE,
    )

    print(
        "V12 detail saved:",
        DETAIL_FILE,
    )

    print(
        "V12 summary saved:",
        SUMMARY_FILE,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_block_results(
        block_results
    )

    print_summary(
        summary
    )

    print()
    print("=" * 115)

    print(
        "V12 INTERPRETATION"
    )

    print("=" * 115)

    print(
        "PASS   = Stronger multi-block evidence."
    )

    print(
        "WATCH  = Promising but needs more validation."
    )

    print(
        "REJECT = V12 robustness requirements not met."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "TP/SL results use 5-minute OHLC candles."
    )

    print(
        "If TP and SL touch in the same candle, "
        "V12 conservatively counts SL first."
    )

    print(
        "Backtest performance does not guarantee future results."
    )

    print(
        "No real orders were placed."
    )

    print()
    print("=" * 115)

    print(
        "FUTURES HISTORICAL V12 COMPLETE"
    )

    print("=" * 115)


if __name__ == "__main__":
    main()
