import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v6 as v6
import crypto_futures_historical_v7 as v7


# ============================================================
# COINDCX FUTURES HISTORICAL V8
# ============================================================
#
# PURPOSE
# -------
# Improve V7 using:
#
# 1. Original Weak Hammer Sweep entry
# 2. V6 CLOSE_TOP_HALF base
# 3. 1H trend confirmation
# 4. Bollinger Band quality
# 5. Volume expansion
# 6. POC / Volume Profile quality
# 7. Walk-forward stability
#
# IMPORTANT:
# HISTORICAL TEST ONLY
# NO REAL ORDERS
# ============================================================


DETAIL_FILE = "crypto_futures_v8_results.csv"
SUMMARY_FILE = "crypto_futures_v8_summary.csv"

WIN_THRESHOLD_PCT = 0.50

WALK_FORWARD_BLOCKS = 4

MIN_TOTAL_DECISIVE = 20


# ============================================================
# V8 ADDITIONAL INDICATORS
# ============================================================

def add_v8_indicators(df):

    data = df.copy()

    data = (
        data
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    data["bb_middle_v8"] = (
        data["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        data["close"]
        .rolling(20)
        .std()
    )

    data["bb_upper_v8"] = (
        data["bb_middle_v8"]
        +
        2.0 * bb_std
    )

    data["bb_lower_v8"] = (
        data["bb_middle_v8"]
        -
        2.0 * bb_std
    )

    data["bb_width_v8_pct"] = (
        (
            data["bb_upper_v8"]
            -
            data["bb_lower_v8"]
        )
        /
        data["bb_middle_v8"]
        * 100
    )

    # BB width average

    data["bb_width_avg_v8"] = (
        data["bb_width_v8_pct"]
        .rolling(50)
        .mean()
    )

    # Current BB expanding

    data["bb_expanding_v8"] = (
        data["bb_width_v8_pct"]
        >
        data["bb_width_v8_pct"].shift(1)
    )

    # Previous squeeze

    data["bb_was_squeeze_v8"] = (
        data["bb_width_v8_pct"].shift(1)
        <
        data["bb_width_avg_v8"].shift(1)
    )

    # Price above BB middle

    data["above_bb_middle_v8"] = (
        data["close"]
        >=
        data["bb_middle_v8"]
    )

    # Price position inside BB

    bb_range = (
        data["bb_upper_v8"]
        -
        data["bb_lower_v8"]
    )

    data["bb_position_v8"] = (
        (
            data["close"]
            -
            data["bb_lower_v8"]
        )
        /
        bb_range.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    data["volume_avg20_v8"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["volume_ratio_v8"] = (
        data["volume"]
        /
        data["volume_avg20_v8"]
        .replace(
            0,
            np.nan,
        )
    )

    data["volume100_v8"] = (
        data["volume_ratio_v8"]
        >= 1.00
    )

    data["volume120_v8"] = (
        data["volume_ratio_v8"]
        >= 1.20
    )

    data["volume150_v8"] = (
        data["volume_ratio_v8"]
        >= 1.50
    )

    # --------------------------------------------------------
    # POC distance
    # --------------------------------------------------------

    data["poc_distance_v8_pct"] = (
        (
            data["close"]
            -
            data["poc"]
        )
        /
        data["poc"]
        .replace(
            0,
            np.nan,
        )
        * 100
    )

    data["poc_near_050_v8"] = (
        data["poc_distance_v8_pct"]
        .between(
            0,
            0.50,
        )
    )

    data["poc_near_100_v8"] = (
        data["poc_distance_v8_pct"]
        .between(
            0,
            1.00,
        )
    )

    # --------------------------------------------------------
    # 5M candle quality
    # --------------------------------------------------------

    candle_range = (
        data["high"]
        -
        data["low"]
    )

    data["close_position_v8"] = (
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

    data["close_top60_v8"] = (
        data["close_position_v8"]
        >= 0.60
    )

    data["close_top70_v8"] = (
        data["close_position_v8"]
        >= 0.70
    )

    data["green_v8"] = (
        data["close"]
        >
        data["open"]
    )

    # --------------------------------------------------------
    # EMA quality 5M
    # --------------------------------------------------------

    data["ema20_rising_v8"] = (
        data["ema20"]
        >
        data["ema20"].shift(1)
    )

    data["above_ema20_v8"] = (
        data["close"]
        >
        data["ema20"]
    )

    data["ema_bull_v8"] = (
        data["ema20"]
        >
        data["ema50"]
    )

    # --------------------------------------------------------
    # MACD quality 5M
    # --------------------------------------------------------

    data["macd_rising_v8"] = (
        data["macd_hist"]
        >
        data["macd_hist"].shift(1)
    )

    data["macd_positive_v8"] = (
        data["macd_hist"]
        >
        0
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    data["rsi_rising_v8"] = (
        data["rsi"]
        >
        data["rsi"].shift(1)
    )

    data["rsi45_70_v8"] = (
        data["rsi"]
        .between(
            45,
            70,
        )
    )

    data["rsi50_70_v8"] = (
        data["rsi"]
        .between(
            50,
            70,
        )
    )

    return data


# ============================================================
# V8 CONFIRMATIONS
# ============================================================

def get_v8_confirmations(row):

    c = {}

    # ========================================================
    # ORIGINAL V6/V7 BASE
    # ========================================================

    c["TREND"] = (
        row["trend_1h"]
        == "BULLISH"
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
        row["close_position"]
        >= 0.50
    )

    # ========================================================
    # 1H FILTERS
    # ========================================================

    c["H1_EMA_BULL"] = (
        pd.notna(
            row["ema20_1h"]
        )
        and
        pd.notna(
            row["ema50_1h"]
        )
        and
        row["ema20_1h"]
        >
        row["ema50_1h"]
    )

    c["H1_EMA20_RISING"] = (
        bool(
            row["ema20_rising_1h"]
        )
    )

    c["H1_MACD_POSITIVE"] = (
        pd.notna(
            row["macd_hist_1h"]
        )
        and
        row["macd_hist_1h"]
        > 0
    )

    c["H1_MACD_RISING"] = (
        bool(
            row["macd_rising_1h"]
        )
    )

    c["H1_RSI_RISING"] = (
        bool(
            row["rsi_rising_1h"]
        )
    )

    c["H1_RSI_50_70"] = (
        pd.notna(
            row["rsi_1h"]
        )
        and
        50
        <=
        row["rsi_1h"]
        <=
        70
    )

    # ========================================================
    # V8 BOLLINGER
    # ========================================================

    c["BB_ABOVE_MIDDLE"] = (
        bool(
            row["above_bb_middle_v8"]
        )
    )

    c["BB_EXPANDING"] = (
        bool(
            row["bb_expanding_v8"]
        )
    )

    c["BB_SQUEEZE_EXPAND"] = (
        bool(
            row["bb_was_squeeze_v8"]
        )
        and
        bool(
            row["bb_expanding_v8"]
        )
    )

    c["BB_POSITION_55"] = (
        pd.notna(
            row["bb_position_v8"]
        )
        and
        row["bb_position_v8"]
        >= 0.55
    )

    # ========================================================
    # V8 VOLUME
    # ========================================================

    c["VOLUME100"] = (
        bool(
            row["volume100_v8"]
        )
    )

    c["VOLUME120"] = (
        bool(
            row["volume120_v8"]
        )
    )

    c["VOLUME150"] = (
        bool(
            row["volume150_v8"]
        )
    )

    # ========================================================
    # V8 POC
    # ========================================================

    c["POC_NEAR050"] = (
        bool(
            row["poc_near_050_v8"]
        )
    )

    c["POC_NEAR100"] = (
        bool(
            row["poc_near_100_v8"]
        )
    )

    # ========================================================
    # V8 CANDLE QUALITY
    # ========================================================

    c["GREEN"] = (
        bool(
            row["green_v8"]
        )
    )

    c["CLOSE_TOP60"] = (
        bool(
            row["close_top60_v8"]
        )
    )

    c["CLOSE_TOP70"] = (
        bool(
            row["close_top70_v8"]
        )
    )

    # ========================================================
    # V8 5M MOMENTUM
    # ========================================================

    c["EMA20_RISING"] = (
        bool(
            row["ema20_rising_v8"]
        )
    )

    c["ABOVE_EMA20"] = (
        bool(
            row["above_ema20_v8"]
        )
    )

    c["EMA_BULL"] = (
        bool(
            row["ema_bull_v8"]
        )
    )

    c["MACD_RISING"] = (
        bool(
            row["macd_rising_v8"]
        )
    )

    c["RSI_RISING"] = (
        bool(
            row["rsi_rising_v8"]
        )
    )

    c["RSI45_70"] = (
        bool(
            row["rsi45_70_v8"]
        )
    )

    c["RSI50_70"] = (
        bool(
            row["rsi50_70_v8"]
        )
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
# V8 CANDIDATES
# ============================================================

CANDIDATES = {

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    "V8_BASE":
        BASE,

    # --------------------------------------------------------
    # V7 BEST IDEAS
    # --------------------------------------------------------

    "V8_H1_EMA20":
        BASE
        + [
            "H1_EMA20_RISING",
        ],

    "V8_H1_MACD":
        BASE
        + [
            "H1_MACD_RISING",
        ],

    "V8_H1_EMA_MACD":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
        ],

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    "V8_BB_MIDDLE":
        BASE
        + [
            "BB_ABOVE_MIDDLE",
        ],

    "V8_BB_EXPAND":
        BASE
        + [
            "BB_EXPANDING",
        ],

    "V8_BB_SQUEEZE_EXPAND":
        BASE
        + [
            "BB_SQUEEZE_EXPAND",
        ],

    "V8_BB_POSITION55":
        BASE
        + [
            "BB_POSITION_55",
        ],

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    "V8_VOLUME100":
        BASE
        + [
            "VOLUME100",
        ],

    "V8_VOLUME120":
        BASE
        + [
            "VOLUME120",
        ],

    "V8_VOLUME150":
        BASE
        + [
            "VOLUME150",
        ],

    # --------------------------------------------------------
    # POC QUALITY
    # --------------------------------------------------------

    "V8_POC_NEAR050":
        BASE
        + [
            "POC_NEAR050",
        ],

    "V8_POC_NEAR100":
        BASE
        + [
            "POC_NEAR100",
        ],

    # --------------------------------------------------------
    # CANDLE QUALITY
    # --------------------------------------------------------

    "V8_GREEN":
        BASE
        + [
            "GREEN",
        ],

    "V8_CLOSE_TOP60":
        BASE
        + [
            "CLOSE_TOP60",
        ],

    "V8_CLOSE_TOP70":
        BASE
        + [
            "CLOSE_TOP70",
        ],

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    "V8_EMA20_RISING":
        BASE
        + [
            "EMA20_RISING",
        ],

    "V8_MACD_RISING":
        BASE
        + [
            "MACD_RISING",
        ],

    "V8_RSI_RISING":
        BASE
        + [
            "RSI_RISING",
        ],

    # --------------------------------------------------------
    # COMBINATIONS
    # --------------------------------------------------------

    "V8_EMA_MACD":
        BASE
        + [
            "EMA20_RISING",
            "MACD_RISING",
        ],

    "V8_EMA_MACD_VOLUME":
        BASE
        + [
            "EMA20_RISING",
            "MACD_RISING",
            "VOLUME100",
        ],

    "V8_MACD_BB":
        BASE
        + [
            "MACD_RISING",
            "BB_ABOVE_MIDDLE",
        ],

    "V8_MACD_BB_VOLUME":
        BASE
        + [
            "MACD_RISING",
            "BB_ABOVE_MIDDLE",
            "VOLUME100",
        ],

    "V8_H1_EMA_BB":
        BASE
        + [
            "H1_EMA20_RISING",
            "BB_ABOVE_MIDDLE",
        ],

    "V8_H1_MACD_BB":
        BASE
        + [
            "H1_MACD_RISING",
            "BB_ABOVE_MIDDLE",
        ],

    "V8_H1_EMA_MACD_BB":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
            "BB_ABOVE_MIDDLE",
        ],

    "V8_H1_EMA_MACD_VOL":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
            "VOLUME100",
        ],

    "V8_H1_EMA_MACD_POC":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
            "POC_NEAR100",
        ],

    "V8_H1_EMA_MACD_TOP60":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
            "CLOSE_TOP60",
        ],

    # --------------------------------------------------------
    # HIGH QUALITY COMBINATIONS
    # --------------------------------------------------------

    "V8_QUALITY_1":
        BASE
        + [
            "H1_EMA20_RISING",
            "MACD_RISING",
            "BB_ABOVE_MIDDLE",
            "VOLUME100",
        ],

    "V8_QUALITY_2":
        BASE
        + [
            "H1_MACD_RISING",
            "MACD_RISING",
            "BB_ABOVE_MIDDLE",
            "VOLUME100",
        ],

    "V8_QUALITY_3":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
            "MACD_RISING",
            "BB_ABOVE_MIDDLE",
        ],

    "V8_QUALITY_4":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
            "BB_ABOVE_MIDDLE",
            "POC_NEAR100",
        ],

    "V8_QUALITY_5":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
            "BB_ABOVE_MIDDLE",
            "VOLUME100",
            "CLOSE_TOP60",
        ],

    "V8_QUALITY_6":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
            "MACD_RISING",
            "BB_ABOVE_MIDDLE",
            "VOLUME100",
            "POC_NEAR100",
        ],
}


# ============================================================
# COLLECT EVENTS
# ============================================================

def collect_events(df):

    events = []

    max_forward = (
        v1.FORWARD_BARS["4H"]
    )

    for i in range(
        60,
        len(df) - max_forward - 1,
    ):

        try:

            signal = (
                v1.setup_weak_hammer_sweep(
                    df,
                    i,
                )
            )

        except Exception:

            continue

        if signal != "LONG":
            continue

        row = df.iloc[i]

        confirmations = (
            get_v8_confirmations(
                row
            )
        )

        forward = (
            v1.calculate_forward_results(
                df,
                i,
                "LONG",
            )
        )

        event = {

            "PAIR":
                v1.PAIR,

            "TIME":
                row[
                    "datetime"
                ].isoformat(),

            "SETUP":
                "WEAK_HAMMER_SWEEP",

            "SIGNAL":
                "LONG",

            "ENTRY_PRICE":
                float(
                    row["close"]
                ),

            "TREND_1H":
                row["trend_1h"],

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

            "VOLUME_RATIO":
                round(
                    float(
                        row["volume_ratio_v8"]
                    ),
                    4,
                )
                if pd.notna(
                    row["volume_ratio_v8"]
                )
                else np.nan,

            "BB_WIDTH_%":
                round(
                    float(
                        row["bb_width_v8_pct"]
                    ),
                    4,
                )
                if pd.notna(
                    row["bb_width_v8_pct"]
                )
                else np.nan,

            "BB_POSITION":
                round(
                    float(
                        row["bb_position_v8"]
                    ),
                    4,
                )
                if pd.notna(
                    row["bb_position_v8"]
                )
                else np.nan,

            "POC_DISTANCE_%":
                round(
                    float(
                        row["poc_distance_v8_pct"]
                    ),
                    4,
                )
                if pd.notna(
                    row["poc_distance_v8_pct"]
                )
                else np.nan,

            "RSI_1H":
                round(
                    float(
                        row["rsi_1h"]
                    ),
                    2,
                )
                if pd.notna(
                    row["rsi_1h"]
                )
                else np.nan,

            "MACD_HIST_1H":
                round(
                    float(
                        row["macd_hist_1h"]
                    ),
                    8,
                )
                if pd.notna(
                    row["macd_hist_1h"]
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
# FILTER
# ============================================================

def apply_filter(
    df,
    conditions,
):

    mask = pd.Series(
        True,
        index=df.index,
    )

    for condition in conditions:

        mask = (
            mask
            &
            (
                df[condition]
                == 1
            )
        )

    return (
        df[mask]
        .copy()
    )


# ============================================================
# PERFORMANCE
# ============================================================

def performance(df):

    if df.empty:
        return None

    returns = (
        df["RETURN_4H_%"]
        .dropna()
    )

    if returns.empty:
        return None

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
                win_rate,
                2,
            )
            if pd.notna(
                win_rate
            )
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
# WALK FORWARD
# ============================================================

def create_blocks(events):

    events = (
        events
        .sort_values("TIME")
        .reset_index(drop=True)
    )

    indexes = (
        np.array_split(
            np.arange(
                len(events)
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

        blocks.append(
            (
                block_number,
                events
                .iloc[idx]
                .copy(),
            )
        )

    return blocks


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
                block_results[
                    "CANDIDATE"
                ]
                ==
                candidate
            ]
            .copy()
        )

        rates = (
            blocks[
                "WIN_RATE_%"
            ]
            .dropna()
        )

        avg_returns = (
            blocks[
                "AVG_RETURN_%"
            ]
            .dropna()
        )

        decisive_blocks = (
            blocks[
                blocks["DECISIVE"] > 0
            ]
        )

        if rates.empty:

            min_rate = np.nan
            max_rate = np.nan
            std_rate = np.nan

        else:

            min_rate = rates.min()
            max_rate = rates.max()

            std_rate = (
                rates.std(
                    ddof=0
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

        valid_blocks = int(
            len(
                decisive_blocks
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
                overall[
                    "MEDIAN_RETURN_%"
                ],

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
                    float(
                        min_rate
                    ),
                    2,
                )
                if pd.notna(
                    min_rate
                )
                else np.nan,

            "MAX_BLOCK_WIN_RATE_%":
                round(
                    float(
                        max_rate
                    ),
                    2,
                )
                if pd.notna(
                    max_rate
                )
                else np.nan,

            "BLOCK_RATE_STD_%":
                round(
                    float(
                        std_rate
                    ),
                    2,
                )
                if pd.notna(
                    std_rate
                )
                else np.nan,

            "ENOUGH_SAMPLE":
                (
                    overall[
                        "DECISIVE"
                    ]
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
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    block_results,
    summary,
):

    print()
    print(
        "=" * 120
    )

    print(
        "V8 WALK-FORWARD BLOCK RESULTS"
    )

    print(
        "=" * 120
    )

    print(
        block_results.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "V8 OVERALL RESULTS"
    )

    print(
        "=" * 120
    )

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

    print(
        summary[
            display
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # RELIABLE 70 / 75 / 80
    # --------------------------------------------------------

    for target in [
        70,
        75,
        80,
    ]:

        reliable = (
            summary[
                (
                    summary[
                        "ENOUGH_SAMPLE"
                    ]
                    == True
                )
                &
                (
                    summary[
                        "OVERALL_WIN_RATE_%"
                    ]
                    >= target
                )
                &
                (
                    summary[
                        "OVERALL_AVG_RETURN_%"
                    ]
                    > 0
                )
                &
                (
                    summary[
                        "POSITIVE_BLOCKS"
                    ]
                    >= 3
                )
            ]
            .copy()
        )

        print()
        print(
            "=" * 120
        )

        print(
            f"RELIABLE {target}%+ "
            f"CANDIDATES: "
            f"{len(reliable)}"
        )

        if not reliable.empty:

            print()

            print(
                reliable[
                    display
                ].to_string(
                    index=False
                )
            )

    # --------------------------------------------------------
    # SPECIAL STABLE CANDIDATES
    # --------------------------------------------------------

    stable = (
        summary[
            (
                summary[
                    "TOTAL_DECISIVE"
                ]
                >= MIN_TOTAL_DECISIVE
            )
            &
            (
                summary[
                    "OVERALL_WIN_RATE_%"
                ]
                >= 70
            )
            &
            (
                summary[
                    "OVERALL_AVG_RETURN_%"
                ]
                > 0
            )
            &
            (
                summary[
                    "POSITIVE_BLOCKS"
                ]
                >= 3
            )
            &
            (
                summary[
                    "BLOCK_RATE_STD_%"
                ]
                <= 20
            )
        ]
        .copy()
    )

    print()
    print(
        "=" * 120
    )

    print(
        "V8 STABLE 70%+ CANDIDATES:",
        len(stable),
    )

    print(
        "=" * 120
    )

    if not stable.empty:

        print(
            stable[
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
    print(
        "=" * 100
    )

    print(
        "COINDCX FUTURES HISTORICAL V8"
    )

    print(
        "BB + VOLUME + POC + 1H TREND QUALITY"
    )

    print(
        "=" * 100
    )

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

    if df_1h.empty:

        raise RuntimeError(
            "No 1-hour data."
        )

    print(
        "1H CANDLES:",
        len(df_1h),
    )

    # ========================================================
    # ORIGINAL INDICATORS
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

    print(
        "Calculating Volume Profile..."
    )

    df_5m = (
        v1.add_volume_profile(
            df_5m
        )
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

    print(
        "Adding V6 indicators..."
    )

    df_5m = (
        v6.add_v6_indicators(
            df_5m
        )
    )

    # ========================================================
    # V7 1H INDICATORS
    # ========================================================

    print(
        "Calculating V7 1H indicators..."
    )

    df_1h_v7 = (
        v7.prepare_1h_indicators(
            df_1h
        )
    )

    print(
        "Attaching completed 1H data..."
    )

    df_5m = (
        v7.attach_v7_1h_data(
            df_5m,
            df_1h_v7,
        )
    )

    # ========================================================
    # V8 INDICATORS
    # ========================================================

    print(
        "Calculating V8 BB / Volume / "
        "POC quality indicators..."
    )

    df_5m = (
        add_v8_indicators(
            df_5m
        )
    )

    df_5m = (
        df_5m
        .dropna(
            subset=[
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
                "bb_middle_v8",
                "bb_width_v8_pct",
                "volume_ratio_v8",
                "poc_distance_v8_pct",
            ]
        )
        .reset_index(
            drop=True
        )
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
        "Running V8 walk-forward test..."
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
        "V8 results saved:",
        DETAIL_FILE,
    )

    print(
        "V8 summary saved:",
        SUMMARY_FILE,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print_results(
        block_results,
        summary,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FUTURES HISTORICAL V8 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
