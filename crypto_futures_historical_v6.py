import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1


# ============================================================
# COINDCX FUTURES HISTORICAL V6
#
# PURPOSE
# -------
# V5 BEST BASE:
#   1H bullish trend
#   MACD positive
#   Price >= POC
#   Price within 0.75% of 4H support
#
# V6 adds confirmation-quality filters WITHOUT changing
# the original Weak Hammer Sweep setup.
#
# 1H = TREND
# 5M = ENTRY
#
# HISTORICAL TEST ONLY
# NO REAL ORDERS
# ============================================================


DETAIL_FILE = "crypto_futures_v6_results.csv"
SUMMARY_FILE = "crypto_futures_v6_summary.csv"

WIN_THRESHOLD_PCT = 0.50

WALK_FORWARD_BLOCKS = 4

# A candidate must have enough decisive trades before
# we consider its win rate meaningful.
MIN_TOTAL_DECISIVE = 30

# We will highlight stronger candidates separately.
TARGET_WIN_RATE = 70.0


# ============================================================
# INDICATORS
# ============================================================

def add_v6_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # 4-HOUR SUPPORT
    # 48 x 5-minute candles = 4 hours
    # --------------------------------------------------------

    df["support_4h"] = (
        df["low"]
        .shift(1)
        .rolling(48)
        .min()
    )

    df["distance_support_4h_pct"] = (
        (
            df["close"]
            - df["support_4h"]
        )
        / df["close"]
        * 100
    )

    # --------------------------------------------------------
    # CANDLE STRUCTURE
    # --------------------------------------------------------

    df["candle_range"] = (
        df["high"]
        - df["low"]
    )

    df["body"] = (
        df["close"]
        - df["open"]
    ).abs()

    df["body_ratio"] = (
        df["body"]
        / df["candle_range"].replace(
            0,
            np.nan,
        )
    )

    df["close_position"] = (
        (
            df["close"]
            - df["low"]
        )
        / df["candle_range"].replace(
            0,
            np.nan,
        )
    )

    df["green_candle"] = (
        df["close"]
        > df["open"]
    )

    # --------------------------------------------------------
    # EMA MOMENTUM
    # --------------------------------------------------------

    df["ema20_slope"] = (
        df["ema20"]
        - df["ema20"].shift(1)
    )

    df["above_ema20"] = (
        df["close"]
        > df["ema20"]
    )

    # --------------------------------------------------------
    # MACD MOMENTUM
    # --------------------------------------------------------

    df["macd_rising"] = (
        df["macd_hist"]
        > df["macd_hist"].shift(1)
    )

    df["macd_rising_2"] = (
        (
            df["macd_hist"]
            > df["macd_hist"].shift(1)
        )
        &
        (
            df["macd_hist"].shift(1)
            >= df["macd_hist"].shift(2)
        )
    )

    # --------------------------------------------------------
    # RSI MOMENTUM
    # --------------------------------------------------------

    df["rsi_rising"] = (
        df["rsi"]
        > df["rsi"].shift(1)
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    df["volume_ma20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["v6_volume_ratio"] = (
        df["volume"]
        / df["volume_ma20"].replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # POC
    # --------------------------------------------------------

    df["price_vs_poc_pct"] = (
        (
            df["close"]
            - df["poc"]
        )
        / df["poc"]
        * 100
    )

    # --------------------------------------------------------
    # PREVIOUS CANDLE
    # --------------------------------------------------------

    df["prev_high"] = (
        df["high"].shift(1)
    )

    df["prev_close"] = (
        df["close"].shift(1)
    )

    # Current candle closing above previous close
    df["close_above_prev_close"] = (
        df["close"]
        > df["prev_close"]
    )

    # Stronger momentum:
    # current close above previous high
    df["close_above_prev_high"] = (
        df["close"]
        > df["prev_high"]
    )

    return df


# ============================================================
# CONFIRMATIONS
# ============================================================

def get_confirmations(row):

    c = {}

    # ========================================================
    # V5 WINNING BASE
    # ========================================================

    c["TREND"] = (
        row["trend_1h"]
        == "BULLISH"
    )

    c["MACD_POSITIVE"] = (
        pd.notna(row["macd_hist"])
        and
        row["macd_hist"] > 0
    )

    c["POC"] = (
        pd.notna(row["poc"])
        and
        row["close"] >= row["poc"]
    )

    c["SUPPORT075"] = (
        pd.notna(
            row["distance_support_4h_pct"]
        )
        and
        0
        <= row["distance_support_4h_pct"]
        <= 0.75
    )

    # ========================================================
    # V6 EXTRA CONFIRMATIONS
    # ========================================================

    c["EMA20_RISING"] = (
        pd.notna(row["ema20_slope"])
        and
        row["ema20_slope"] > 0
    )

    c["ABOVE_EMA20"] = (
        bool(row["above_ema20"])
    )

    c["MACD_RISING"] = (
        bool(row["macd_rising"])
    )

    c["MACD_RISING_2"] = (
        bool(row["macd_rising_2"])
    )

    c["RSI_RISING"] = (
        bool(row["rsi_rising"])
    )

    c["RSI_45_65"] = (
        pd.notna(row["rsi"])
        and
        45 <= row["rsi"] <= 65
    )

    c["RSI_40_70"] = (
        pd.notna(row["rsi"])
        and
        40 <= row["rsi"] <= 70
    )

    c["GREEN"] = (
        bool(row["green_candle"])
    )

    c["CLOSE_TOP_HALF"] = (
        pd.notna(row["close_position"])
        and
        row["close_position"] >= 0.50
    )

    c["CLOSE_TOP30"] = (
        pd.notna(row["close_position"])
        and
        row["close_position"] >= 0.70
    )

    c["VOLUME100"] = (
        pd.notna(row["v6_volume_ratio"])
        and
        row["v6_volume_ratio"] >= 1.00
    )

    c["VOLUME120"] = (
        pd.notna(row["v6_volume_ratio"])
        and
        row["v6_volume_ratio"] >= 1.20
    )

    c["CLOSE_ABOVE_PREV_CLOSE"] = (
        bool(
            row[
                "close_above_prev_close"
            ]
        )
    )

    c["CLOSE_ABOVE_PREV_HIGH"] = (
        bool(
            row[
                "close_above_prev_high"
            ]
        )
    )

    return c


# ============================================================
# FROZEN V6 CANDIDATES
#
# Important:
# We are not testing hundreds of combinations.
# These are a small number of logically chosen confirmations.
# ============================================================

BASE = [
    "TREND",
    "MACD_POSITIVE",
    "POC",
    "SUPPORT075",
]


CANDIDATES = {

    "V5_BASE_SUPPORT075":
        BASE,

    "BASE_EMA20_RISING":
        BASE
        + [
            "EMA20_RISING",
        ],

    "BASE_MACD_RISING":
        BASE
        + [
            "MACD_RISING",
        ],

    "BASE_MACD_RISING2":
        BASE
        + [
            "MACD_RISING_2",
        ],

    "BASE_RSI_RISING":
        BASE
        + [
            "RSI_RISING",
        ],

    "BASE_RSI45_65":
        BASE
        + [
            "RSI_45_65",
        ],

    "BASE_GREEN":
        BASE
        + [
            "GREEN",
        ],

    "BASE_CLOSE_TOP_HALF":
        BASE
        + [
            "CLOSE_TOP_HALF",
        ],

    "BASE_CLOSE_TOP30":
        BASE
        + [
            "CLOSE_TOP30",
        ],

    "BASE_VOLUME100":
        BASE
        + [
            "VOLUME100",
        ],

    "BASE_VOLUME120":
        BASE
        + [
            "VOLUME120",
        ],

    "BASE_PREV_CLOSE":
        BASE
        + [
            "CLOSE_ABOVE_PREV_CLOSE",
        ],

    "BASE_PREV_HIGH":
        BASE
        + [
            "CLOSE_ABOVE_PREV_HIGH",
        ],

    # --------------------------------------------------------
    # Combined quality confirmations
    # --------------------------------------------------------

    "BASE_EMA_MACD":
        BASE
        + [
            "EMA20_RISING",
            "MACD_RISING",
        ],

    "BASE_MACD_RSI":
        BASE
        + [
            "MACD_RISING",
            "RSI_RISING",
        ],

    "BASE_EMA_MACD_RSI":
        BASE
        + [
            "EMA20_RISING",
            "MACD_RISING",
            "RSI_RISING",
        ],

    "BASE_EMA_MACD_GREEN":
        BASE
        + [
            "EMA20_RISING",
            "MACD_RISING",
            "GREEN",
        ],

    "BASE_MACD_VOLUME":
        BASE
        + [
            "MACD_RISING",
            "VOLUME100",
        ],

    "BASE_EMA_MACD_VOLUME":
        BASE
        + [
            "EMA20_RISING",
            "MACD_RISING",
            "VOLUME100",
        ],

    "BASE_MACD_PREVCLOSE":
        BASE
        + [
            "MACD_RISING",
            "CLOSE_ABOVE_PREV_CLOSE",
        ],

    "BASE_EMA_MACD_PREVCLOSE":
        BASE
        + [
            "EMA20_RISING",
            "MACD_RISING",
            "CLOSE_ABOVE_PREV_CLOSE",
        ],
}


# ============================================================
# FIND WEAK HAMMER SWEEP EVENTS
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
            get_confirmations(row)
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

            "RSI":
                round(
                    float(
                        row["rsi"]
                    ),
                    2,
                ),

            "MACD_HIST":
                round(
                    float(
                        row["macd_hist"]
                    ),
                    8,
                ),

            "EMA20":
                round(
                    float(
                        row["ema20"]
                    ),
                    8,
                ),

            "POC":
                round(
                    float(
                        row["poc"]
                    ),
                    8,
                ),

            "SUPPORT_DISTANCE_%":
                round(
                    float(
                        row[
                            "distance_support_4h_pct"
                        ]
                    ),
                    4,
                ),

            "VOLUME_RATIO":
                round(
                    float(
                        row[
                            "v6_volume_ratio"
                        ]
                    ),
                    3,
                )
                if pd.notna(
                    row[
                        "v6_volume_ratio"
                    ]
                )
                else np.nan,

            "CLOSE_POSITION":
                round(
                    float(
                        row[
                            "close_position"
                        ]
                    ),
                    4,
                )
                if pd.notna(
                    row[
                        "close_position"
                    ]
                )
                else np.nan,
        }

        for name, value in (
            confirmations.items()
        ):

            event[name] = (
                int(value)
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

    mask = pd.Series(
        True,
        index=df.index,
    )

    for condition in conditions:

        mask = (
            mask
            & (
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
        - wins
        - losses
    )

    decisive = (
        wins
        + losses
    )

    if decisive > 0:

        win_rate = (
            wins
            / decisive
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
# CREATE WALK-FORWARD BLOCKS
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

        block = (
            events
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
# WALK-FORWARD TEST
# ============================================================

def walk_forward_test(events):

    blocks = (
        create_blocks(
            events
        )
    )

    results = []

    for block_number, block in blocks:

        print()
        print(
            "=" * 80
        )

        print(
            "BLOCK:",
            block_number,
        )

        print(
            "EVENTS:",
            len(block),
        )

        print(
            "FROM:",
            block.iloc[0]["TIME"],
        )

        print(
            "TO:",
            block.iloc[-1]["TIME"],
        )

        print(
            "=" * 80
        )

        for (
            candidate_name,
            conditions,
        ) in CANDIDATES.items():

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
                        candidate_name,

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
                })

                continue

            results.append({

                "BLOCK":
                    block_number,

                "CANDIDATE":
                    candidate_name,

                **stats,
            })

    return pd.DataFrame(
        results
    )


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    events,
    block_results,
):

    rows = []

    for (
        candidate_name,
        conditions,
    ) in CANDIDATES.items():

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
                == candidate_name
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

        if rates.empty:

            min_rate = np.nan
            max_rate = np.nan
            avg_rate = np.nan
            std_rate = np.nan

        else:

            min_rate = (
                rates.min()
            )

            max_rate = (
                rates.max()
            )

            avg_rate = (
                rates.mean()
            )

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

        blocks_60 = int(
            (
                rates >= 60
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

        rows.append({

            "CANDIDATE":
                candidate_name,

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

            "POSITIVE_BLOCKS":
                positive_blocks,

            "BLOCKS_60PLUS":
                blocks_60,

            "BLOCKS_70PLUS":
                blocks_70,

            "BLOCKS_75PLUS":
                blocks_75,

            "BLOCKS_80PLUS":
                blocks_80,

            "MIN_BLOCK_WIN_RATE_%":
                round(
                    float(min_rate),
                    2,
                )
                if pd.notna(
                    min_rate
                )
                else np.nan,

            "MAX_BLOCK_WIN_RATE_%":
                round(
                    float(max_rate),
                    2,
                )
                if pd.notna(
                    max_rate
                )
                else np.nan,

            "AVG_BLOCK_WIN_RATE_%":
                round(
                    float(avg_rate),
                    2,
                )
                if pd.notna(
                    avg_rate
                )
                else np.nan,

            "BLOCK_RATE_STD_%":
                round(
                    float(std_rate),
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
                    >= MIN_TOTAL_DECISIVE
                ),
        })

    summary = (
        pd.DataFrame(
            rows
        )
    )

    if summary.empty:
        return summary

    summary = (
        summary
        .sort_values(
            by=[
                "ENOUGH_SAMPLE",
                "OVERALL_WIN_RATE_%",
                "TOTAL_DECISIVE",
                "BLOCK_RATE_STD_%",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    block_results,
    summary,
):

    print()
    print(
        "=" * 115
    )

    print(
        "CRYPTO FUTURES HISTORICAL V6"
    )

    print(
        "WALK-FORWARD BLOCK RESULTS"
    )

    print(
        "=" * 115
    )

    print(
        block_results
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 115
    )

    print(
        "V6 OVERALL RESULTS"
    )

    print(
        "=" * 115
    )

    display_columns = [

        "CANDIDATE",

        "TOTAL_DECISIVE",

        "OVERALL_WIN_RATE_%",

        "OVERALL_AVG_RETURN_%",

        "POSITIVE_BLOCKS",

        "BLOCKS_70PLUS",

        "MIN_BLOCK_WIN_RATE_%",

        "MAX_BLOCK_WIN_RATE_%",

        "BLOCK_RATE_STD_%",
    ]

    print(
        summary[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # RELIABLE 70%+
    # ========================================================

    reliable70 = (
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
                == WALK_FORWARD_BLOCKS
            )
        ]
        .copy()
    )

    print()
    print(
        "=" * 115
    )

    print(
        "RELIABLE 70%+ CANDIDATES:",
        len(reliable70),
    )

    if not reliable70.empty:

        print()

        print(
            reliable70[
                display_columns
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 75%+
    # ========================================================

    reliable75 = (
        reliable70[
            reliable70[
                "OVERALL_WIN_RATE_%"
            ]
            >= 75
        ]
        .copy()
    )

    print()
    print(
        "=" * 115
    )

    print(
        "RELIABLE 75%+ CANDIDATES:",
        len(reliable75),
    )

    if not reliable75.empty:

        print()

        print(
            reliable75[
                display_columns
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # 80%+
    # ========================================================

    reliable80 = (
        reliable70[
            reliable70[
                "OVERALL_WIN_RATE_%"
            ]
            >= 80
        ]
        .copy()
    )

    print()
    print(
        "=" * 115
    )

    print(
        "RELIABLE 80%+ CANDIDATES:",
        len(reliable80),
    )

    if not reliable80.empty:

        print()

        print(
            reliable80[
                display_columns
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 115
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
        "COINDCX FUTURES HISTORICAL V6"
    )

    print(
        "WEAK HAMMER QUALITY OPTIMIZATION"
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
        "Downloading native "
        "5-minute Futures candles..."
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
        "Downloading "
        "1-hour Futures candles..."
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
    # INDICATORS
    # ========================================================

    print()
    print(
        "Calculating indicators..."
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
        "Attaching 1-hour trend..."
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
        add_v6_indicators(
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
        "Finding Weak Hammer "
        "Sweep events..."
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
        "TOTAL WEAK HAMMER EVENTS:",
        len(events),
    )

    # ========================================================
    # WALK FORWARD
    # ========================================================

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
    # SAVE CSV
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
        "V6 results saved:",
        DETAIL_FILE,
    )

    print(
        "V6 summary saved:",
        SUMMARY_FILE,
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print_summary(
        block_results,
        summary,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FUTURES HISTORICAL V6 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
