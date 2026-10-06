import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1


# ============================================================
# COINDCX FUTURES HISTORICAL V5
# WALK-FORWARD VALIDATION
#
# 1H = TREND
# 5M = ENTRY
#
# Frozen candidates from V4:
#
# 1. V3_BASE
# 2. BASE + SUPPORT075
# 3. BASE + CLOSE_TOP30
# 4. BASE + CLOSE_TOP_HALF
#
# Goal:
# Check whether the V4 results remain stable across
# different historical periods.
#
# IMPORTANT:
# This script DOES NOT place real trades.
# ============================================================


DETAIL_FILE = "crypto_futures_v5_results.csv"
SUMMARY_FILE = "crypto_futures_v5_summary.csv"

WIN_THRESHOLD_PCT = 0.50

# Minimum number of decisive trades required
# across all walk-forward blocks.
MIN_TOTAL_DECISIVE = 30

# We divide all historical events into blocks.
WALK_FORWARD_BLOCKS = 4


# ============================================================
# ADD V5 INDICATORS
# ============================================================

def add_v5_indicators(df):

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
    # POC DISTANCE
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

    # Close position:
    #
    # 0.0 = candle low
    # 1.0 = candle high
    # --------------------------------------------------------

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

    return df


# ============================================================
# FROZEN V5 CONFIRMATIONS
# ============================================================

def get_confirmations(row):

    confirmations = {}

    # --------------------------------------------------------
    # BASE:
    # 1H bullish trend
    # Positive MACD
    # Price >= POC
    # Near 4H support <= 1.5%
    # --------------------------------------------------------

    confirmations["TREND"] = (
        row["trend_1h"]
        == "BULLISH"
    )

    confirmations["MACD_POSITIVE"] = (
        pd.notna(row["macd_hist"])
        and
        row["macd_hist"] > 0
    )

    confirmations["POC"] = (
        pd.notna(row["poc"])
        and
        row["close"] >= row["poc"]
    )

    confirmations["SUPPORT4H"] = (
        pd.notna(
            row[
                "distance_support_4h_pct"
            ]
        )
        and
        0
        <= row[
            "distance_support_4h_pct"
        ]
        <= 1.50
    )

    # --------------------------------------------------------
    # V4 81.82% candidate:
    # tighter 4H support <= 0.75%
    # --------------------------------------------------------

    confirmations["SUPPORT075"] = (
        pd.notna(
            row[
                "distance_support_4h_pct"
            ]
        )
        and
        0
        <= row[
            "distance_support_4h_pct"
        ]
        <= 0.75
    )

    # --------------------------------------------------------
    # V4 75% candidate:
    # close in upper 30% of candle
    # --------------------------------------------------------

    confirmations["CLOSE_TOP30"] = (
        pd.notna(
            row["close_position"]
        )
        and
        row["close_position"]
        >= 0.70
    )

    # --------------------------------------------------------
    # V4 73.68% candidate:
    # close in upper half
    # --------------------------------------------------------

    confirmations["CLOSE_TOP_HALF"] = (
        pd.notna(
            row["close_position"]
        )
        and
        row["close_position"]
        >= 0.50
    )

    return confirmations


# ============================================================
# FROZEN CANDIDATES
# ============================================================

CANDIDATES = {

    "V3_BASE": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "SUPPORT4H",
    ],

    "BASE_SUPPORT075": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "SUPPORT075",
    ],

    "BASE_CLOSE_TOP30": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "SUPPORT4H",
        "CLOSE_TOP30",
    ],

    "BASE_CLOSE_TOP_HALF": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "SUPPORT4H",
        "CLOSE_TOP_HALF",
    ],
}


# ============================================================
# COLLECT WEAK HAMMER EVENTS
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
                float(row["close"]),

            "TREND_1H":
                row["trend_1h"],

            "RSI":
                round(
                    float(row["rsi"]),
                    2,
                ),

            "MACD_HIST":
                round(
                    float(
                        row["macd_hist"]
                    ),
                    8,
                ),

            "VOLUME_RATIO":
                round(
                    float(
                        row["volume_ratio"]
                    ),
                    3,
                ),

            "SUPPORT_DISTANCE_%":
                round(
                    float(
                        row[
                            "distance_support_4h_pct"
                        ]
                    ),
                    4,
                )
                if pd.notna(
                    row[
                        "distance_support_4h_pct"
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

        for (
            name,
            value,
        ) in confirmations.items():

            event[name] = int(value)

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
    columns,
):

    mask = pd.Series(
        True,
        index=df.index,
    )

    for column in columns:

        mask = (
            mask
            & (
                df[column]
                == 1
            )
        )

    return (
        df[mask]
        .copy()
    )


# ============================================================
# PERFORMANCE CALCULATION
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

    avg_return = (
        returns.mean()
    )

    median_return = (
        returns.median()
    )

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

        "WIN_RATE":
            round(
                win_rate,
                2,
            )
            if pd.notna(
                win_rate
            )
            else np.nan,

        "AVG_RETURN":
            round(
                float(
                    avg_return
                ),
                4,
            ),

        "MEDIAN_RETURN":
            round(
                float(
                    median_return
                ),
                4,
            ),
    }


# ============================================================
# CREATE TIME BLOCKS
# ============================================================

def create_blocks(events):

    events = (
        events
        .sort_values("TIME")
        .reset_index(drop=True)
    )

    if len(events) < WALK_FORWARD_BLOCKS:

        raise RuntimeError(
            "Not enough events "
            "for walk-forward blocks."
        )

    index_blocks = (
        np.array_split(
            np.arange(
                len(events)
            ),
            WALK_FORWARD_BLOCKS,
        )
    )

    blocks = []

    for block_number, indexes in enumerate(
        index_blocks,
        start=1,
    ):

        if len(indexes) == 0:
            continue

        block = (
            events
            .iloc[indexes]
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

    rows = []

    print()
    print("=" * 100)
    print(
        "V5 WALK-FORWARD BLOCKS"
    )
    print("=" * 100)

    for (
        block_number,
        block,
    ) in blocks:

        print()
        print(
            "BLOCK",
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

        for (
            candidate_name,
            columns,
        ) in CANDIDATES.items():

            filtered = (
                apply_filter(
                    block,
                    columns,
                )
            )

            stats = (
                performance(
                    filtered
                )
            )

            if stats is None:

                rows.append({

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

            rows.append({

                "BLOCK":
                    block_number,

                "CANDIDATE":
                    candidate_name,

                "SIGNALS":
                    stats["SIGNALS"],

                "WINS":
                    stats["WINS"],

                "LOSSES":
                    stats["LOSSES"],

                "NEUTRAL":
                    stats["NEUTRAL"],

                "DECISIVE":
                    stats["DECISIVE"],

                "WIN_RATE_%":
                    stats["WIN_RATE"],

                "AVG_RETURN_%":
                    stats["AVG_RETURN"],
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

def build_summary(
    events,
    walk_results,
):

    summary_rows = []

    for (
        candidate_name,
        columns,
    ) in CANDIDATES.items():

        all_filtered = (
            apply_filter(
                events,
                columns,
            )
        )

        overall = (
            performance(
                all_filtered
            )
        )

        if overall is None:
            continue

        candidate_blocks = (
            walk_results[
                walk_results[
                    "CANDIDATE"
                ]
                == candidate_name
            ]
            .copy()
        )

        valid_rates = (
            candidate_blocks[
                "WIN_RATE_%"
            ]
            .dropna()
        )

        positive_blocks = int(
            (
                candidate_blocks[
                    "AVG_RETURN_%"
                ]
                > 0
            ).sum()
        )

        blocks_60 = int(
            (
                valid_rates
                >= 60
            ).sum()
        )

        blocks_70 = int(
            (
                valid_rates
                >= 70
            ).sum()
        )

        if valid_rates.empty:

            min_rate = np.nan
            max_rate = np.nan
            avg_block_rate = np.nan
            rate_std = np.nan

        else:

            min_rate = (
                valid_rates.min()
            )

            max_rate = (
                valid_rates.max()
            )

            avg_block_rate = (
                valid_rates.mean()
            )

            rate_std = (
                valid_rates.std(
                    ddof=0
                )
            )

        summary_rows.append({

            "CANDIDATE":
                candidate_name,

            "TOTAL_SIGNALS":
                overall["SIGNALS"],

            "TOTAL_DECISIVE":
                overall["DECISIVE"],

            "TOTAL_WINS":
                overall["WINS"],

            "TOTAL_LOSSES":
                overall["LOSSES"],

            "OVERALL_WIN_RATE_%":
                overall["WIN_RATE"],

            "OVERALL_AVG_RETURN_%":
                overall["AVG_RETURN"],

            "POSITIVE_BLOCKS":
                positive_blocks,

            "BLOCKS_60PLUS":
                blocks_60,

            "BLOCKS_70PLUS":
                blocks_70,

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
                    float(
                        avg_block_rate
                    ),
                    2,
                )
                if pd.notna(
                    avg_block_rate
                )
                else np.nan,

            "BLOCK_RATE_STD_%":
                round(
                    float(rate_std),
                    2,
                )
                if pd.notna(
                    rate_std
                )
                else np.nan,

            "ENOUGH_TOTAL_SAMPLE":
                (
                    overall[
                        "DECISIVE"
                    ]
                    >= MIN_TOTAL_DECISIVE
                ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    if summary.empty:
        return summary

    summary = (
        summary
        .sort_values(
            by=[
                "ENOUGH_TOTAL_SAMPLE",
                "OVERALL_WIN_RATE_%",
                "BLOCKS_70PLUS",
                "TOTAL_DECISIVE",
                "BLOCK_RATE_STD_%",
            ],
            ascending=[
                False,
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
# PRINT RESULTS
# ============================================================

def print_results(
    walk_results,
    summary,
):

    print()
    print("=" * 110)
    print(
        "V5 WALK-FORWARD BLOCK RESULTS"
    )
    print("=" * 110)

    if walk_results.empty:

        print(
            "No walk-forward results."
        )

    else:

        print(
            walk_results
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 110)
    print(
        "V5 OVERALL STABILITY SUMMARY"
    )
    print("=" * 110)

    if summary.empty:

        print(
            "No summary results."
        )

        return

    columns = [

        "CANDIDATE",

        "TOTAL_DECISIVE",

        "OVERALL_WIN_RATE_%",

        "OVERALL_AVG_RETURN_%",

        "POSITIVE_BLOCKS",

        "BLOCKS_60PLUS",

        "BLOCKS_70PLUS",

        "MIN_BLOCK_WIN_RATE_%",

        "MAX_BLOCK_WIN_RATE_%",

        "BLOCK_RATE_STD_%",
    ]

    print()

    print(
        summary[
            columns
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Stable 70% candidates
    # --------------------------------------------------------

    stable70 = (
        summary[
            (
                summary[
                    "ENOUGH_TOTAL_SAMPLE"
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
        ]
        .copy()
    )

    print()
    print("=" * 110)

    print(
        "STABLE 70%+ CANDIDATES:",
        len(stable70),
    )

    if not stable70.empty:

        print()

        print(
            stable70[
                columns
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Stable 75%
    # --------------------------------------------------------

    stable75 = (
        summary[
            (
                summary[
                    "ENOUGH_TOTAL_SAMPLE"
                ]
                == True
            )
            &
            (
                summary[
                    "OVERALL_WIN_RATE_%"
                ]
                >= 75
            )
            &
            (
                summary[
                    "OVERALL_AVG_RETURN_%"
                ]
                > 0
            )
        ]
        .copy()
    )

    print()
    print("=" * 110)

    print(
        "STABLE 75%+ CANDIDATES:",
        len(stable75),
    )

    if not stable75.empty:

        print()

        print(
            stable75[
                columns
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Stable 80%
    # --------------------------------------------------------

    stable80 = (
        summary[
            (
                summary[
                    "ENOUGH_TOTAL_SAMPLE"
                ]
                == True
            )
            &
            (
                summary[
                    "OVERALL_WIN_RATE_%"
                ]
                >= 80
            )
            &
            (
                summary[
                    "OVERALL_AVG_RETURN_%"
                ]
                > 0
            )
        ]
        .copy()
    )

    print()
    print("=" * 110)

    print(
        "STABLE 80%+ CANDIDATES:",
        len(stable80),
    )

    if not stable80.empty:

        print()

        print(
            stable80[
                columns
            ]
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)

    print(
        "COINDCX FUTURES HISTORICAL V5"
    )

    print(
        "WALK-FORWARD VALIDATION"
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
        "WALK-FORWARD BLOCKS:",
        WALK_FORWARD_BLOCKS,
    )

    # --------------------------------------------------------
    # 5-MINUTE DATA
    # --------------------------------------------------------

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
            "No 5-minute Futures data."
        )

    print(
        "5M CANDLES:",
        len(df_5m),
    )

    # --------------------------------------------------------
    # 1-HOUR DATA
    # --------------------------------------------------------

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
            "No 1-hour Futures data."
        )

    print(
        "1H CANDLES:",
        len(df_1h),
    )

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

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
        "Attaching 1H trend..."
    )

    df_5m = (
        v1.attach_1h_trend(
            df_5m,
            df_1h,
        )
    )

    print(
        "Adding V5 indicators..."
    )

    df_5m = (
        add_v5_indicators(
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

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    print()
    print(
        "Finding Weak Hammer "
        "Sweep LONG events..."
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

    print()
    print(
        "TOTAL EVENTS:",
        len(events),
    )

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    walk_results = (
        walk_forward_test(
            events
        )
    )

    summary = (
        build_summary(
            events,
            walk_results,
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    walk_results.to_csv(
        DETAIL_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        "Walk-forward results saved:",
        DETAIL_FILE,
    )

    print(
        "Summary saved:",
        SUMMARY_FILE,
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_results(
        walk_results,
        summary,
    )

    print()
    print("=" * 100)

    print(
        "FUTURES HISTORICAL V5 COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
