import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v6 as v6
import crypto_futures_historical_v7 as v7
import crypto_futures_historical_v8 as v8


# ============================================================
# COINDCX FUTURES HISTORICAL V9
# ============================================================
#
# PURPOSE
# -------
# V9 builds on the strongest V8 ideas:
#
#   V8_BB_POSITION55
#   V8_EMA20_RISING
#
# Main objective:
#
#   Improve 70-80% reliability
#   Keep enough decisive trades
#   Improve walk-forward stability
#   Avoid over-fitting tiny samples
#
# HISTORICAL TEST ONLY
# NO REAL ORDERS
# ============================================================


DETAIL_FILE = "crypto_futures_v9_results.csv"
SUMMARY_FILE = "crypto_futures_v9_summary.csv"

WIN_THRESHOLD_PCT = 0.50

WALK_FORWARD_BLOCKS = 4

MIN_TOTAL_DECISIVE = 20


# ============================================================
# SAFE DATAFRAME CHECK
# ============================================================

def ensure_dataframe(data, name="data"):

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        converted = pd.DataFrame(data)
    except Exception as exc:
        raise TypeError(
            f"{name} could not be converted to DataFrame. "
            f"Received type: {type(data)}"
        ) from exc

    return converted


# ============================================================
# V9 INDICATORS
# ============================================================

def add_v9_indicators(df):

    data = ensure_dataframe(
        df,
        "add_v9_indicators input",
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
        "rsi",
        "macd_hist",
        "poc",
    ]

    missing = [
        col
        for col in required
        if col not in data.columns
    ]

    if missing:
        raise RuntimeError(
            "V9 missing required columns: "
            + ", ".join(missing)
        )

    data = (
        data
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # EMA momentum
    # --------------------------------------------------------

    data["v9_ema20_rising"] = (
        data["ema20"]
        >
        data["ema20"].shift(1)
    )

    data["v9_ema20_rising2"] = (
        (
            data["ema20"]
            >
            data["ema20"].shift(1)
        )
        &
        (
            data["ema20"].shift(1)
            >=
            data["ema20"].shift(2)
        )
    )

    data["v9_above_ema20"] = (
        data["close"]
        >
        data["ema20"]
    )

    data["v9_ema_bull"] = (
        data["ema20"]
        >
        data["ema50"]
    )

    # --------------------------------------------------------
    # MACD momentum
    # --------------------------------------------------------

    data["v9_macd_rising"] = (
        data["macd_hist"]
        >
        data["macd_hist"].shift(1)
    )

    data["v9_macd_rising2"] = (
        (
            data["macd_hist"]
            >
            data["macd_hist"].shift(1)
        )
        &
        (
            data["macd_hist"].shift(1)
            >=
            data["macd_hist"].shift(2)
        )
    )

    data["v9_macd_positive"] = (
        data["macd_hist"]
        >
        0
    )

    # --------------------------------------------------------
    # RSI momentum
    # --------------------------------------------------------

    data["v9_rsi_rising"] = (
        data["rsi"]
        >
        data["rsi"].shift(1)
    )

    data["v9_rsi_45_70"] = (
        data["rsi"]
        .between(
            45,
            70,
        )
    )

    data["v9_rsi_50_70"] = (
        data["rsi"]
        .between(
            50,
            70,
        )
    )

    # --------------------------------------------------------
    # Candle quality
    # --------------------------------------------------------

    candle_range = (
        data["high"]
        -
        data["low"]
    )

    safe_range = candle_range.replace(
        0,
        np.nan,
    )

    data["v9_close_position"] = (
        (
            data["close"]
            -
            data["low"]
        )
        /
        safe_range
    )

    data["v9_close_top55"] = (
        data["v9_close_position"]
        >= 0.55
    )

    data["v9_close_top60"] = (
        data["v9_close_position"]
        >= 0.60
    )

    data["v9_close_top70"] = (
        data["v9_close_position"]
        >= 0.70
    )

    data["v9_green"] = (
        data["close"]
        >
        data["open"]
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    data["v9_bb_middle"] = (
        data["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        data["close"]
        .rolling(20)
        .std()
    )

    data["v9_bb_upper"] = (
        data["v9_bb_middle"]
        +
        2.0 * bb_std
    )

    data["v9_bb_lower"] = (
        data["v9_bb_middle"]
        -
        2.0 * bb_std
    )

    bb_range = (
        data["v9_bb_upper"]
        -
        data["v9_bb_lower"]
    )

    data["v9_bb_position"] = (
        (
            data["close"]
            -
            data["v9_bb_lower"]
        )
        /
        bb_range.replace(
            0,
            np.nan,
        )
    )

    data["v9_bb_position55"] = (
        data["v9_bb_position"]
        >= 0.55
    )

    data["v9_bb_position60"] = (
        data["v9_bb_position"]
        >= 0.60
    )

    data["v9_above_bb_middle"] = (
        data["close"]
        >=
        data["v9_bb_middle"]
    )

    data["v9_bb_width_pct"] = (
        (
            data["v9_bb_upper"]
            -
            data["v9_bb_lower"]
        )
        /
        data["v9_bb_middle"].replace(
            0,
            np.nan,
        )
        * 100
    )

    data["v9_bb_expanding"] = (
        data["v9_bb_width_pct"]
        >
        data["v9_bb_width_pct"].shift(1)
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    data["v9_volume_avg20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["v9_volume_ratio"] = (
        data["volume"]
        /
        data["v9_volume_avg20"].replace(
            0,
            np.nan,
        )
    )

    data["v9_volume100"] = (
        data["v9_volume_ratio"]
        >= 1.00
    )

    data["v9_volume110"] = (
        data["v9_volume_ratio"]
        >= 1.10
    )

    data["v9_volume120"] = (
        data["v9_volume_ratio"]
        >= 1.20
    )

    # --------------------------------------------------------
    # POC
    # --------------------------------------------------------

    data["v9_poc_distance_pct"] = (
        (
            data["close"]
            -
            data["poc"]
        )
        /
        data["poc"].replace(
            0,
            np.nan,
        )
        * 100
    )

    data["v9_poc_near050"] = (
        data["v9_poc_distance_pct"]
        .between(
            0,
            0.50,
        )
    )

    data["v9_poc_near075"] = (
        data["v9_poc_distance_pct"]
        .between(
            0,
            0.75,
        )
    )

    data["v9_poc_near100"] = (
        data["v9_poc_distance_pct"]
        .between(
            0,
            1.00,
        )
    )

    return data


# ============================================================
# CONFIRMATIONS
# ============================================================

def get_v9_confirmations(row):

    c = {}

    # Original base

    c["TREND"] = (
        row["trend_1h"]
        ==
        "BULLISH"
    )

    c["MACD_POSITIVE"] = (
        pd.notna(
            row["macd_hist"]
        )
        and
        row["macd_hist"] > 0
    )

    c["POC"] = (
        pd.notna(
            row["poc"]
        )
        and
        row["close"] >= row["poc"]
    )

    c["SUPPORT075"] = (
        pd.notna(
            row["distance_support_4h_pct"]
        )
        and
        0
        <=
        row["distance_support_4h_pct"]
        <=
        0.75
    )

    c["CLOSE_TOP_HALF"] = (
        pd.notna(
            row["close_position"]
        )
        and
        row["close_position"] >= 0.50
    )

    # V9 EMA

    c["EMA20_RISING"] = bool(
        row["v9_ema20_rising"]
    )

    c["EMA20_RISING2"] = bool(
        row["v9_ema20_rising2"]
    )

    c["ABOVE_EMA20"] = bool(
        row["v9_above_ema20"]
    )

    c["EMA_BULL"] = bool(
        row["v9_ema_bull"]
    )

    # V9 MACD

    c["MACD_RISING"] = bool(
        row["v9_macd_rising"]
    )

    c["MACD_RISING2"] = bool(
        row["v9_macd_rising2"]
    )

    # RSI

    c["RSI_RISING"] = bool(
        row["v9_rsi_rising"]
    )

    c["RSI45_70"] = bool(
        row["v9_rsi_45_70"]
    )

    c["RSI50_70"] = bool(
        row["v9_rsi_50_70"]
    )

    # Candle

    c["GREEN"] = bool(
        row["v9_green"]
    )

    c["CLOSE_TOP55"] = bool(
        row["v9_close_top55"]
    )

    c["CLOSE_TOP60"] = bool(
        row["v9_close_top60"]
    )

    c["CLOSE_TOP70"] = bool(
        row["v9_close_top70"]
    )

    # Bollinger

    c["BB_POSITION55"] = bool(
        row["v9_bb_position55"]
    )

    c["BB_POSITION60"] = bool(
        row["v9_bb_position60"]
    )

    c["BB_MIDDLE"] = bool(
        row["v9_above_bb_middle"]
    )

    c["BB_EXPANDING"] = bool(
        row["v9_bb_expanding"]
    )

    # Volume

    c["VOLUME100"] = bool(
        row["v9_volume100"]
    )

    c["VOLUME110"] = bool(
        row["v9_volume110"]
    )

    c["VOLUME120"] = bool(
        row["v9_volume120"]
    )

    # POC

    c["POC_NEAR050"] = bool(
        row["v9_poc_near050"]
    )

    c["POC_NEAR075"] = bool(
        row["v9_poc_near075"]
    )

    c["POC_NEAR100"] = bool(
        row["v9_poc_near100"]
    )

    # 1H

    c["H1_EMA20_RISING"] = bool(
        row["ema20_rising_1h"]
    )

    c["H1_MACD_RISING"] = bool(
        row["macd_rising_1h"]
    )

    c["H1_EMA_BULL"] = (
        pd.notna(row["ema20_1h"])
        and
        pd.notna(row["ema50_1h"])
        and
        row["ema20_1h"]
        >
        row["ema50_1h"]
    )

    return c


# ============================================================
# BASE
# ============================================================

BASE = [
    "TREND",
    "MACD_POSITIVE",
    "POC",
    "SUPPORT075",
    "CLOSE_TOP_HALF",
]


# ============================================================
# V9 CANDIDATES
# ============================================================

CANDIDATES = {

    "V9_BASE":
        BASE,

    "V9_BB55":
        BASE
        + [
            "BB_POSITION55",
        ],

    "V9_EMA20":
        BASE
        + [
            "EMA20_RISING",
        ],

    "V9_BB55_EMA20":
        BASE
        + [
            "BB_POSITION55",
            "EMA20_RISING",
        ],

    "V9_BB55_MACD":
        BASE
        + [
            "BB_POSITION55",
            "MACD_RISING",
        ],

    "V9_BB55_RSI":
        BASE
        + [
            "BB_POSITION55",
            "RSI_RISING",
        ],

    "V9_BB55_GREEN":
        BASE
        + [
            "BB_POSITION55",
            "GREEN",
        ],

    "V9_BB55_TOP60":
        BASE
        + [
            "BB_POSITION55",
            "CLOSE_TOP60",
        ],

    "V9_BB55_POC075":
        BASE
        + [
            "BB_POSITION55",
            "POC_NEAR075",
        ],

    "V9_BB55_VOLUME100":
        BASE
        + [
            "BB_POSITION55",
            "VOLUME100",
        ],

    "V9_EMA20_MACD":
        BASE
        + [
            "EMA20_RISING",
            "MACD_RISING",
        ],

    "V9_EMA20_RSI":
        BASE
        + [
            "EMA20_RISING",
            "RSI_RISING",
        ],

    "V9_EMA20_GREEN":
        BASE
        + [
            "EMA20_RISING",
            "GREEN",
        ],

    "V9_EMA20_TOP60":
        BASE
        + [
            "EMA20_RISING",
            "CLOSE_TOP60",
        ],

    "V9_EMA20_POC075":
        BASE
        + [
            "EMA20_RISING",
            "POC_NEAR075",
        ],

    "V9_BB55_EMA20_MACD":
        BASE
        + [
            "BB_POSITION55",
            "EMA20_RISING",
            "MACD_RISING",
        ],

    "V9_BB55_EMA20_RSI":
        BASE
        + [
            "BB_POSITION55",
            "EMA20_RISING",
            "RSI_RISING",
        ],

    "V9_BB55_EMA20_GREEN":
        BASE
        + [
            "BB_POSITION55",
            "EMA20_RISING",
            "GREEN",
        ],

    "V9_BB55_EMA20_TOP60":
        BASE
        + [
            "BB_POSITION55",
            "EMA20_RISING",
            "CLOSE_TOP60",
        ],

    "V9_BB55_EMA20_VOLUME":
        BASE
        + [
            "BB_POSITION55",
            "EMA20_RISING",
            "VOLUME100",
        ],

    "V9_BB55_EMA20_POC":
        BASE
        + [
            "BB_POSITION55",
            "EMA20_RISING",
            "POC_NEAR075",
        ],

    "V9_BB60_EMA20":
        BASE
        + [
            "BB_POSITION60",
            "EMA20_RISING",
        ],

    "V9_EMA20_RISING2":
        BASE
        + [
            "EMA20_RISING2",
        ],

    "V9_MACD_RISING2":
        BASE
        + [
            "MACD_RISING2",
        ],

    "V9_BB55_H1_EMA":
        BASE
        + [
            "BB_POSITION55",
            "H1_EMA20_RISING",
        ],

    "V9_EMA20_H1_EMA":
        BASE
        + [
            "EMA20_RISING",
            "H1_EMA20_RISING",
        ],

    "V9_BB55_EMA20_H1":
        BASE
        + [
            "BB_POSITION55",
            "EMA20_RISING",
            "H1_EMA20_RISING",
        ],

    "V9_BB55_EMA20_H1_MACD":
        BASE
        + [
            "BB_POSITION55",
            "EMA20_RISING",
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
        ],
}


# ============================================================
# COLLECT EVENTS
# ============================================================

def collect_events(df):

    data = ensure_dataframe(
        df,
        "collect_events input",
    )

    events = []

    max_forward = (
        v1.FORWARD_BARS["4H"]
    )

    for i in range(
        60,
        len(data) - max_forward - 1,
    ):

        try:

            signal = (
                v1.setup_weak_hammer_sweep(
                    data,
                    i,
                )
            )

        except Exception:
            continue

        if signal != "LONG":
            continue

        row = data.iloc[i]

        confirmations = (
            get_v9_confirmations(
                row
            )
        )

        forward = (
            v1.calculate_forward_results(
                data,
                i,
                "LONG",
            )
        )

        event = {

            "PAIR":
                v1.PAIR,

            "TIME":
                row["datetime"].isoformat(),

            "SETUP":
                "WEAK_HAMMER_SWEEP",

            "SIGNAL":
                "LONG",

            "ENTRY_PRICE":
                float(
                    row["close"]
                ),

            "RSI_5M":
                round(
                    float(
                        row["rsi"]
                    ),
                    2,
                ),

            "MACD_HIST_5M":
                round(
                    float(
                        row["macd_hist"]
                    ),
                    8,
                ),

            "BB_POSITION":
                round(
                    float(
                        row["v9_bb_position"]
                    ),
                    4,
                )
                if pd.notna(
                    row["v9_bb_position"]
                )
                else np.nan,

            "VOLUME_RATIO":
                round(
                    float(
                        row["v9_volume_ratio"]
                    ),
                    4,
                )
                if pd.notna(
                    row["v9_volume_ratio"]
                )
                else np.nan,

            "POC_DISTANCE_%":
                round(
                    float(
                        row["v9_poc_distance_pct"]
                    ),
                    4,
                )
                if pd.notna(
                    row["v9_poc_distance_pct"]
                )
                else np.nan,
        }

        for name, value in confirmations.items():

            event[name] = int(
                bool(value)
            )

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
# APPLY FILTER
# ============================================================

def apply_filter(
    df,
    conditions,
):

    data = ensure_dataframe(
        df,
        "apply_filter input",
    )

    mask = pd.Series(
        True,
        index=data.index,
    )

    for condition in conditions:

        if condition not in data.columns:

            mask = (
                mask
                &
                False
            )

            continue

        mask = (
            mask
            &
            (
                data[condition]
                == 1
            )
        )

    return (
        data[mask]
        .copy()
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

    if "RETURN_4H_%" not in data.columns:
        return None

    returns = (
        data["RETURN_4H_%"]
        .dropna()
    )

    if returns.empty:
        return None

    wins = int(
        (
            returns
            >=
            WIN_THRESHOLD_PCT
        ).sum()
    )

    losses = int(
        (
            returns
            <=
            -WIN_THRESHOLD_PCT
        ).sum()
    )

    neutral = int(
        len(returns)
        -
        wins
        -
        losses
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
            * 100
        )

    else:

        win_rate = np.nan

    return {

        "SIGNALS":
            len(returns),

        "WINS":
            wins,

        "LOSSES":
            losses,

        "NEUTRAL":
            neutral,

        "DECISIVE":
            decisive,

        "WIN_RATE_%":
            round(
                float(win_rate),
                2,
            )
            if pd.notna(win_rate)
            else np.nan,

        "AVG_RETURN_%":
            round(
                float(
                    returns.mean()
                ),
                4,
            ),

        "MEDIAN_RETURN_%":
            round(
                float(
                    returns.median()
                ),
                4,
            ),
    }


# ============================================================
# WALK FORWARD BLOCKS
# ============================================================

def create_blocks(events):

    data = ensure_dataframe(
        events,
        "create_blocks input",
    )

    data = (
        data
        .sort_values("TIME")
        .reset_index(drop=True)
    )

    indexes = (
        np.array_split(
            np.arange(
                len(data)
            ),
            WALK_FORWARD_BLOCKS,
        )
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
# WALK FORWARD TEST
# ============================================================

def walk_forward_test(events):

    results = []

    blocks = (
        create_blocks(
            events
        )
    )

    for block_number, block in blocks:

        for candidate, conditions in CANDIDATES.items():

            filtered = (
                apply_filter(
                    block,
                    conditions,
                )
            )

            stats = (
                performance(
                    filtered
                )
            )

            if stats is None:

                results.append({

                    "BLOCK":
                        block_number,

                    "CANDIDATE":
                        candidate,

                    "SIGNALS":
                        0,

                    "WINS":
                        0,

                    "LOSSES":
                        0,

                    "NEUTRAL":
                        0,

                    "DECISIVE":
                        0,

                    "WIN_RATE_%":
                        np.nan,

                    "AVG_RETURN_%":
                        np.nan,

                    "MEDIAN_RETURN_%":
                        np.nan,
                })

            else:

                results.append({

                    "BLOCK":
                        block_number,

                    "CANDIDATE":
                        candidate,

                    **stats,
                })

    return pd.DataFrame(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    events,
    block_results,
):

    rows = []

    for candidate, conditions in CANDIDATES.items():

        filtered = (
            apply_filter(
                events,
                conditions,
            )
        )

        overall = (
            performance(
                filtered
            )
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

        avg_returns = (
            blocks["AVG_RETURN_%"]
            .dropna()
        )

        decisive_blocks = (
            blocks[
                blocks["DECISIVE"] > 0
            ]
        )

        valid_blocks = int(
            len(
                decisive_blocks
            )
        )

        positive_blocks = int(
            (
                avg_returns > 0
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
            max_rate = np.nan
            std_rate = np.nan

        else:

            min_rate = float(
                rates.min()
            )

            max_rate = float(
                rates.max()
            )

            std_rate = float(
                rates.std(
                    ddof=0
                )
            )

        rows.append({

            "CANDIDATE":
                candidate,

            "TOTAL_SIGNALS":
                overall["SIGNALS"],

            "TOTAL_DECISIVE":
                overall["DECISIVE"],

            "WINS":
                overall["WINS"],

            "LOSSES":
                overall["LOSSES"],

            "NEUTRAL":
                overall["NEUTRAL"],

            "OVERALL_WIN_RATE_%":
                overall["WIN_RATE_%"],

            "OVERALL_AVG_RETURN_%":
                overall["AVG_RETURN_%"],

            "MEDIAN_RETURN_%":
                overall["MEDIAN_RETURN_%"],

            "VALID_BLOCKS":
                valid_blocks,

            "POSITIVE_BLOCKS":
                positive_blocks,

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

            "ENOUGH_SAMPLE":
                (
                    overall["DECISIVE"]
                    >=
                    MIN_TOTAL_DECISIVE
                ),
        })

    summary = pd.DataFrame(
        rows
    )

    if summary.empty:
        return summary

    return (
        summary
        .sort_values(
            by=[
                "ENOUGH_SAMPLE",
                "OVERALL_WIN_RATE_%",
                "TOTAL_DECISIVE",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    block_results,
    summary,
):

    print()
    print("=" * 120)
    print("V9 WALK-FORWARD BLOCK RESULTS")
    print("=" * 120)

    print(
        block_results.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("V9 OVERALL RESULTS")
    print("=" * 120)

    display = [

        "CANDIDATE",

        "TOTAL_DECISIVE",

        "OVERALL_WIN_RATE_%",

        "OVERALL_AVG_RETURN_%",

        "MEDIAN_RETURN_%",

        "VALID_BLOCKS",

        "POSITIVE_BLOCKS",

        "BLOCKS_70PLUS",

        "BLOCKS_75PLUS",

        "BLOCKS_80PLUS",

        "MIN_BLOCK_WIN_RATE_%",

        "MAX_BLOCK_WIN_RATE_%",

        "BLOCK_RATE_STD_%",
    ]

    if summary.empty:

        print(
            "No V9 summary results."
        )

        return

    print(
        summary[
            display
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # RELIABLE GROUPS
    # ========================================================

    for target in [
        70,
        75,
        80,
    ]:

        reliable = (
            summary[
                (
                    summary["ENOUGH_SAMPLE"]
                    ==
                    True
                )
                &
                (
                    summary["OVERALL_WIN_RATE_%"]
                    >=
                    target
                )
                &
                (
                    summary["OVERALL_AVG_RETURN_%"]
                    >
                    0
                )
                &
                (
                    summary["POSITIVE_BLOCKS"]
                    >=
                    3
                )
            ]
            .copy()
        )

        print()
        print("=" * 120)

        print(
            f"V9 RELIABLE {target}%+ CANDIDATES:",
            len(reliable),
        )

        print("=" * 120)

        if not reliable.empty:

            print(
                reliable[
                    display
                ].to_string(
                    index=False
                )
            )

    # ========================================================
    # STABLE 70+
    # ========================================================

    stable70 = (
        summary[
            (
                summary["ENOUGH_SAMPLE"]
                ==
                True
            )
            &
            (
                summary["OVERALL_WIN_RATE_%"]
                >=
                70
            )
            &
            (
                summary["OVERALL_AVG_RETURN_%"]
                >
                0
            )
            &
            (
                summary["VALID_BLOCKS"]
                ==
                WALK_FORWARD_BLOCKS
            )
            &
            (
                summary["POSITIVE_BLOCKS"]
                ==
                WALK_FORWARD_BLOCKS
            )
            &
            (
                summary["BLOCK_RATE_STD_%"]
                <=
                20
            )
        ]
        .copy()
    )

    print()
    print("=" * 120)

    print(
        "V9 STABLE 70%+ CANDIDATES:",
        len(stable70),
    )

    print("=" * 120)

    if not stable70.empty:

        print(
            stable70[
                display
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # STABLE 75+
    # ========================================================

    stable75 = (
        stable70[
            stable70["OVERALL_WIN_RATE_%"]
            >=
            75
        ]
        .copy()
    )

    print()
    print("=" * 120)

    print(
        "V9 STABLE 75%+ CANDIDATES:",
        len(stable75),
    )

    print("=" * 120)

    if not stable75.empty:

        print(
            stable75[
                display
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # STABLE 80+
    # ========================================================

    stable80 = (
        stable70[
            stable70["OVERALL_WIN_RATE_%"]
            >=
            80
        ]
        .copy()
    )

    print()
    print("=" * 120)

    print(
        "V9 STABLE 80%+ CANDIDATES:",
        len(stable80),
    )

    print("=" * 120)

    if not stable80.empty:

        print(
            stable80[
                display
            ].to_string(
                index=False
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)

    print(
        "COINDCX FUTURES HISTORICAL V9"
    )

    print(
        "V8 WINNER COMBINATION OPTIMIZATION"
    )

    print("=" * 100)

    print(
        "PAIR:",
        v1.PAIR,
    )

    print(
        "HISTORY:",
        v1.DAYS_TO_TEST,
        "days",
    )

    print(
        "TREND TIMEFRAME: 1 HOUR"
    )

    print(
        "ENTRY TIMEFRAME: 5 MINUTES"
    )

    # ========================================================
    # DOWNLOAD 5M
    # ========================================================

    print()
    print(
        "Downloading 5-minute data..."
    )

    df_5m = (
        v1.fetch_history(
            v1.PAIR,
            "5",
            v1.DAYS_TO_TEST,
            v1.CHUNK_DAYS_5M,
        )
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
        "5M CANDLES:",
        len(df_5m),
    )

    # ========================================================
    # DOWNLOAD 1H
    # ========================================================

    print()
    print(
        "Downloading 1-hour data..."
    )

    df_1h = (
        v1.fetch_history(
            v1.PAIR,
            "60",
            v1.DAYS_TO_TEST,
            v1.CHUNK_DAYS_1H,
        )
    )

    df_1h = ensure_dataframe(
        df_1h,
        "1-hour history",
    )

    if df_1h.empty:

        raise RuntimeError(
            "No 1-hour data."
        )

    print(
        "1H CANDLES:",
        len(df_1h),
    )

    # ========================================================
    # BASE INDICATORS
    # ========================================================

    print()
    print(
        "Calculating 5M indicators..."
    )

    df_5m = (
        v1.calculate_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "calculate_indicators result",
    )

    print(
        "Calculating Volume Profile..."
    )

    df_5m = (
        v1.add_volume_profile(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "volume profile result",
    )

    print(
        "Attaching original 1H trend..."
    )

    df_5m = (
        v1.attach_1h_trend(
            df_5m,
            df_1h,
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "1H trend result",
    )

    # ========================================================
    # V6
    # ========================================================

    print(
        "Adding V6 indicators..."
    )

    df_5m = (
        v6.add_v6_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V6 result",
    )

    # ========================================================
    # V7 1H
    # ========================================================

    print(
        "Calculating V7 1H indicators..."
    )

    df_1h_v7 = (
        v7.prepare_1h_indicators(
            df_1h
        )
    )

    df_1h_v7 = ensure_dataframe(
        df_1h_v7,
        "V7 1H result",
    )

    print(
        "Attaching completed 1H indicators..."
    )

    df_5m = (
        v7.attach_v7_1h_data(
            df_5m,
            df_1h_v7,
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V7 attach result",
    )

    # ========================================================
    # V8
    # ========================================================

    print(
        "Adding V8 indicators..."
    )

    df_5m = (
        v8.add_v8_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V8 result",
    )

    # ========================================================
    # V9
    # ========================================================

    print(
        "Adding V9 optimization indicators..."
    )

    df_5m = (
        add_v9_indicators(
            df_5m
        )
    )

    # ========================================================
    # CLEAN
    # ========================================================

    required_clean = [
        "ema20",
        "ema50",
        "rsi",
        "macd_hist",
        "poc",
        "support_4h",
        "close_position",
        "ema20_1h",
        "ema50_1h",
        "macd_hist_1h",
        "rsi_1h",
        "v9_bb_middle",
        "v9_bb_position",
        "v9_volume_ratio",
        "v9_poc_distance_pct",
    ]

    df_5m = (
        df_5m
        .dropna(
            subset=required_clean
        )
        .reset_index(drop=True)
    )

    print(
        "USABLE 5M CANDLES:",
        len(df_5m),
    )

    # ========================================================
    # EVENTS
    # ========================================================

    print()
    print(
        "Finding Weak Hammer Sweep events..."
    )

    events = (
        collect_events(
            df_5m
        )
    )

    if events.empty:

        print(
            "No Weak Hammer events found."
        )

        return

    print(
        "TOTAL EVENTS:",
        len(events),
    )

    # ========================================================
    # WALK FORWARD
    # ========================================================

    print()
    print(
        "Running V9 walk-forward test..."
    )

    block_results = (
        walk_forward_test(
            events
        )
    )

    summary = (
        build_summary(
            events,
            block_results,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

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
        "V9 detail saved:",
        DETAIL_FILE,
    )

    print(
        "V9 summary saved:",
        SUMMARY_FILE,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_results(
        block_results,
        summary,
    )

    print()
    print("=" * 100)

    print(
        "FUTURES HISTORICAL V9 COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
