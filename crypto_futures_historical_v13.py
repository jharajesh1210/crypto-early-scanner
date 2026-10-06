import math
import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v12 as v12


# ============================================================
# COINDCX FUTURES HISTORICAL V13
# TP / SL OPTIMIZATION
# ============================================================
#
# PURPOSE
# -------
# V12 showed that fixed:
#
#     TP = 0.80%
#     SL = 0.50%
#
# was not robust.
#
# V13 keeps the V12 LONG / SHORT setup logic but tests
# multiple TP / SL combinations separately.
#
# IMPORTANT:
# - Historical research only
# - No Telegram alerts
# - No live orders
# - No real-money execution
# ============================================================


DETAIL_FILE = "crypto_futures_v13_results.csv"
SUMMARY_FILE = "crypto_futures_v13_summary.csv"
EVENT_FILE = "crypto_futures_v13_events.csv"


# ============================================================
# HISTORY / WALK FORWARD
# ============================================================

HISTORY_ATTEMPTS = [
    365,
    270,
    180,
]

WALK_FORWARD_BLOCKS = 8

MAX_HOLD_BARS = 48

ROUND_TRIP_COST_PCT = 0.10

SAME_BAR_POLICY = "SL_FIRST"


# ============================================================
# TP / SL GRID
# ============================================================
#
# V13 intentionally tests a moderate grid.
#
# We do NOT test hundreds of tiny combinations because
# that would increase overfitting risk.
# ============================================================

TP_LEVELS = [
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    1.00,
]

SL_LEVELS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


# ============================================================
# ROBUSTNESS RULES
# ============================================================

MIN_TOTAL_TRADES_PASS = 50
MIN_TOTAL_TRADES_WATCH = 30

MIN_TRADES_PER_BLOCK = 4

PASS_MIN_WIN_RATE = 60.0
PASS_MIN_POSITIVE_BLOCKS = 6
PASS_MIN_VALID_BLOCKS = 6
PASS_MIN_PROFIT_FACTOR = 1.20
PASS_MIN_BLOCKS_POSITIVE = 6
PASS_MAX_BLOCK_STD = 22.0

WATCH_MIN_WIN_RATE = 55.0
WATCH_MIN_POSITIVE_BLOCKS = 5
WATCH_MIN_VALID_BLOCKS = 5
WATCH_MIN_PROFIT_FACTOR = 1.00

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


# ============================================================
# WILSON CONFIDENCE INTERVAL
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
            print("=" * 110)
            print(
                "V13 TRYING HISTORY:",
                days,
                "DAYS",
            )
            print("=" * 110)

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
        "All V13 history attempts failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# PRINT HISTORY RANGE
# ============================================================

def print_history_range(
    df_5m,
    df_1h,
    requested_days,
):

    print()
    print("=" * 110)
    print("V13 DATA RANGE")
    print("=" * 110)

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
# PREPARE V12 DATA
# ============================================================

def prepare_data(df_5m, df_1h):

    print()
    print(
        "Preparing V12 indicator stack..."
    )

    df_5m = v12.prepare_data(
        df_5m,
        df_1h,
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V12 prepared data",
    )

    print(
        "Adding V12 market regime..."
    )

    df_5m = v12.add_v12_market_regime(
        df_5m
    )

    print(
        "Adding V12 candle features..."
    )

    df_5m = v12.add_v12_candle_features(
        df_5m
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V12 candle result",
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
            "V13 missing required columns: "
            +
            ", ".join(missing)
        )

    drop_columns = [
        col
        for col in required
        if col != "V12_REGIME"
    ]

    df_5m = (
        df_5m
        .dropna(
            subset=drop_columns
        )
        .reset_index(drop=True)
    )

    return df_5m


# ============================================================
# FIND BASE SIGNALS
# ============================================================

def collect_base_signals(df):

    data = ensure_dataframe(
        df,
        "base signal input",
    )

    signals = []

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

            signals.append({

                "DATA_INDEX":
                    i,

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

                "ENTRY_PRICE":
                    safe_float(
                        row["close"]
                    ),

                "RSI_5M":
                    safe_float(
                        row["rsi"]
                    ),

                "MACD_HIST_5M":
                    safe_float(
                        row["macd_hist"]
                    ),

                "EMA20":
                    safe_float(
                        row["ema20"]
                    ),

                "EMA50":
                    safe_float(
                        row["ema50"]
                    ),

                "EMA20_1H":
                    safe_float(
                        row["ema20_1h"]
                    ),

                "EMA50_1H":
                    safe_float(
                        row["ema50_1h"]
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
            })

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        if (
            regime == "DOWNTREND"
            and
            v12.short_setup(
                data,
                i,
            )
        ):

            signals.append({

                "DATA_INDEX":
                    i,

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

                "ENTRY_PRICE":
                    safe_float(
                        row["close"]
                    ),

                "RSI_5M":
                    safe_float(
                        row["rsi"]
                    ),

                "MACD_HIST_5M":
                    safe_float(
                        row["macd_hist"]
                    ),

                "EMA20":
                    safe_float(
                        row["ema20"]
                    ),

                "EMA50":
                    safe_float(
                        row["ema50"]
                    ),

                "EMA20_1H":
                    safe_float(
                        row["ema20_1h"]
                    ),

                "EMA50_1H":
                    safe_float(
                        row["ema50_1h"]
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
            })

    return pd.DataFrame(
        signals
    )


# ============================================================
# GENERIC TP / SL SIMULATOR
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

    # --------------------------------------------------------
    # PRICES
    # --------------------------------------------------------

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
    # CHECK TP / SL
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

        # Conservative same-candle handling
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

    # --------------------------------------------------------
    # MFE / MAE
    # --------------------------------------------------------

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
# RUN ALL TP / SL COMBINATIONS
# ============================================================

def generate_optimization_events(
    data,
    base_signals,
):

    results = []

    total_combinations = (
        len(TP_LEVELS)
        *
        len(SL_LEVELS)
    )

    print()
    print(
        "TP LEVELS:",
        TP_LEVELS,
    )

    print(
        "SL LEVELS:",
        SL_LEVELS,
    )

    print(
        "TP/SL COMBINATIONS PER SIDE:",
        total_combinations,
    )

    print(
        "TOTAL SIDE-COMBINATIONS:",
        total_combinations * 2,
    )

    print()

    for signal_number, (_, signal) in enumerate(
        base_signals.iterrows(),
        start=1,
    ):

        index = int(
            signal["DATA_INDEX"]
        )

        side = str(
            signal["SIDE"]
        )

        for tp_pct in TP_LEVELS:

            for sl_pct in SL_LEVELS:

                result = simulate_trade(
                    data,
                    index,
                    side,
                    tp_pct,
                    sl_pct,
                )

                if result is None:
                    continue

                candidate = (
                    f"V13_{side}"
                    f"_TP{int(tp_pct * 100):03d}"
                    f"_SL{int(sl_pct * 100):03d}"
                )

                row = {

                    "CANDIDATE":
                        candidate,

                    "PAIR":
                        signal["PAIR"],

                    "TIME":
                        signal["TIME"],

                    "SIDE":
                        side,

                    "REGIME":
                        signal["REGIME"],

                    "SETUP":
                        signal["SETUP"],

                    "TP_%":
                        tp_pct,

                    "SL_%":
                        sl_pct,

                    "ENTRY_PRICE":
                        signal["ENTRY_PRICE"],

                    "RSI_5M":
                        signal["RSI_5M"],

                    "MACD_HIST_5M":
                        signal["MACD_HIST_5M"],

                    "EMA20":
                        signal["EMA20"],

                    "EMA50":
                        signal["EMA50"],

                    "EMA20_1H":
                        signal["EMA20_1H"],

                    "EMA50_1H":
                        signal["EMA50_1H"],

                    "LOWER_WICK_RATIO":
                        signal["LOWER_WICK_RATIO"],

                    "UPPER_WICK_RATIO":
                        signal["UPPER_WICK_RATIO"],

                    "CLOSE_POSITION":
                        signal["CLOSE_POSITION"],
                }

                row.update(
                    result
                )

                results.append(
                    row
                )

        if (
            signal_number % 100
            ==
            0
        ):

            print(
                "Processed base signals:",
                signal_number,
                "/",
                len(base_signals),
            )

    return pd.DataFrame(
        results
    )


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

def create_blocks(base_signals):

    data = ensure_dataframe(
        base_signals,
        "block input",
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


# ============================================================
# ASSIGN BLOCK NUMBER BY SIGNAL TIME
# ============================================================

def create_time_block_map(base_signals):

    blocks = create_blocks(
        base_signals
    )

    mapping = {}

    for block_number, block in blocks:

        for time_value in block["TIME"]:

            mapping[
                str(time_value)
            ] = block_number

    return mapping


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward_test(
    optimization_events,
    base_signals,
):

    events = ensure_dataframe(
        optimization_events,
        "optimization events",
    )

    block_map = create_time_block_map(
        base_signals
    )

    events["BLOCK"] = (
        events["TIME"]
        .astype(str)
        .map(block_map)
    )

    candidates = sorted(
        events["CANDIDATE"]
        .dropna()
        .unique()
    )

    rows = []

    for block_number in range(
        1,
        WALK_FORWARD_BLOCKS + 1,
    ):

        block_events = (
            events[
                events["BLOCK"]
                ==
                block_number
            ]
            .copy()
        )

        for candidate in candidates:

            filtered = (
                block_events[
                    block_events["CANDIDATE"]
                    ==
                    candidate
                ]
                .copy()
            )

            stats = performance(
                filtered
            )

            if stats is None:

                rows.append({

                    "BLOCK":
                        block_number,

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
# CLASSIFY CANDIDATE
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

    min_sample_blocks = int(
        row["BLOCKS_WITH_MIN_SAMPLE"]
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
        0

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

        positive_blocks
        >=
        PASS_MIN_BLOCKS_POSITIVE

        and

        min_sample_blocks
        >=
        PASS_MIN_VALID_BLOCKS

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
        0

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
    optimization_events,
    block_results,
):

    events = ensure_dataframe(
        optimization_events,
        "summary events",
    )

    candidates = sorted(
        events["CANDIDATE"]
        .dropna()
        .unique()
    )

    rows = []

    for candidate in candidates:

        filtered = (
            events[
                events["CANDIDATE"]
                ==
                candidate
            ]
            .copy()
        )

        overall = performance(
            filtered
        )

        if overall is None:
            continue

        side = str(
            filtered["SIDE"].iloc[0]
        )

        tp_pct = safe_float(
            filtered["TP_%"].iloc[0]
        )

        sl_pct = safe_float(
            filtered["SL_%"].iloc[0]
        )

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

        positive_blocks = int(
            (
                blocks["NET_AVG_RETURN_%"]
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

        blocks_75 = int(
            (
                rates >= 75
            ).sum()
        )

        blocks_80 = int(
            (
                rates >= 80
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

            "SIDE":
                side,

            "TP_%":
                tp_pct,

            "SL_%":
                sl_pct,

            "RISK_REWARD":
                round(
                    tp_pct / sl_pct,
                    3,
                )
                if sl_pct > 0
                else np.nan,

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

            "BLOCKS_75PLUS":
                blocks_75,

            "BLOCKS_80PLUS":
                blocks_80,

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
# PRINT TOP RESULTS
# ============================================================

def print_top_results(summary):

    print()
    print("=" * 175)
    print(
        "V13 TOP TP / SL OPTIMIZATION RESULTS"
    )
    print("=" * 175)

    if summary.empty:

        print(
            "No V13 summary results."
        )

        return

    display = [

        "CANDIDATE",

        "STATUS",

        "SIDE",

        "TP_%",

        "SL_%",

        "RISK_REWARD",

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
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# PRINT FINAL CANDIDATES
# ============================================================

def print_final_results(summary):

    if summary.empty:
        return

    display = [

        "CANDIDATE",

        "STATUS",

        "SIDE",

        "TP_%",

        "SL_%",

        "RISK_REWARD",

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
    print("=" * 175)

    print(
        "V13 FINAL PASS CANDIDATES:",
        len(passed),
    )

    print("=" * 175)

    if passed.empty:

        print(
            "No candidate passed V13 strict rules."
        )

    else:

        print(
            passed[
                display
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 175)

    print(
        "V13 WATCH CANDIDATES:",
        len(watch),
    )

    print("=" * 175)

    if watch.empty:

        print(
            "No WATCH candidates."
        )

    else:

        print(
            watch[
                display
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 175)

    print(
        "V13 REJECT CANDIDATES:",
        len(rejected),
    )

    print("=" * 175)

    if rejected.empty:

        print(
            "No rejected candidates."
        )

    else:

        print(
            rejected[
                display
            ]
            .head(10)
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
    print("=" * 175)
    print(
        "V13 BEST LONG"
    )
    print("=" * 175)

    if long_results.empty:

        print(
            "No LONG results."
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
    print("=" * 175)
    print(
        "V13 BEST SHORT"
    )
    print("=" * 175)

    if short_results.empty:

        print(
            "No SHORT results."
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

    print()
    print("=" * 175)
    print(
        "V13 BEST OVERALL CANDIDATE"
    )
    print("=" * 175)

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
        "SIDE:",
        best["SIDE"],
    )

    print(
        "TP:",
        f'{best["TP_%"]:.2f}%',
    )

    print(
        "SL:",
        f'{best["SL_%"]:.2f}%',
    )

    print(
        "RISK/REWARD:",
        f'{best["RISK_REWARD"]:.3f}',
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
    print("=" * 120)

    print(
        "COINDCX FUTURES HISTORICAL V13"
    )

    print(
        "TP / SL OPTIMIZATION + LONG / SHORT SEPARATE TEST"
    )

    print("=" * 120)

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

    print(
        "TP LEVELS:",
        TP_LEVELS,
    )

    print(
        "SL LEVELS:",
        SL_LEVELS,
    )

    print()
    print(
        "V13 IS HISTORICAL RESEARCH ONLY."
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

    print()
    print(
        "USABLE 5M CANDLES:",
        len(df_5m),
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
    # BASE SIGNALS
    # ========================================================

    print()
    print(
        "Finding V13 base LONG / SHORT signals..."
    )

    base_signals = collect_base_signals(
        df_5m
    )

    if base_signals.empty:

        print(
            "No V13 base signals found."
        )

        return

    print(
        "TOTAL BASE SIGNALS:",
        len(base_signals),
    )

    print(
        "LONG BASE SIGNALS:",
        int(
            (
                base_signals["SIDE"]
                ==
                "LONG"
            ).sum()
        ),
    )

    print(
        "SHORT BASE SIGNALS:",
        int(
            (
                base_signals["SIDE"]
                ==
                "SHORT"
            ).sum()
        ),
    )

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    print()
    print(
        "Running TP / SL optimization..."
    )

    optimization_events = (
        generate_optimization_events(
            df_5m,
            base_signals,
        )
    )

    if optimization_events.empty:

        print(
            "No V13 optimization events."
        )

        return

    print()
    print(
        "TOTAL OPTIMIZATION ROWS:",
        len(
            optimization_events
        ),
    )

    print(
        "TOTAL CANDIDATES:",
        optimization_events[
            "CANDIDATE"
        ]
        .nunique(),
    )

    # ========================================================
    # WALK FORWARD
    # ========================================================

    print()
    print(
        "Running V13 8-block walk-forward test..."
    )

    block_results = (
        walk_forward_test(
            optimization_events,
            base_signals,
        )
    )

    summary = (
        build_summary(
            optimization_events,
            block_results,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    optimization_events.to_csv(
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
        "V13 EVENTS SAVED:",
        EVENT_FILE,
    )

    print(
        "V13 RESULTS SAVED:",
        DETAIL_FILE,
    )

    print(
        "V13 SUMMARY SAVED:",
        SUMMARY_FILE,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print_top_results(
        summary
    )

    print_final_results(
        summary
    )

    print()
    print("=" * 120)

    print(
        "V13 INTERPRETATION"
    )

    print("=" * 120)

    print(
        "PASS = TP/SL combination passed V13 robustness rules."
    )

    print(
        "WATCH = Promising, but needs further validation."
    )

    print(
        "REJECT = Not robust enough."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "A high win rate alone is NOT enough."
    )

    print(
        "Net return, profit factor, sample size and"
    )

    print(
        "walk-forward consistency must also be strong."
    )

    print()

    print(
        "V13 tests many TP/SL combinations."
    )

    print(
        "Therefore the best V13 result must be validated"
    )

    print(
        "again out-of-sample before live use."
    )

    print()

    print(
        "Backtest results do not guarantee future performance."
    )

    print(
        "No real orders were placed."
    )

    print()
    print("=" * 120)

    print(
        "FUTURES HISTORICAL V13 COMPLETE"
    )

    print("=" * 120)


if __name__ == "__main__":
    main()
