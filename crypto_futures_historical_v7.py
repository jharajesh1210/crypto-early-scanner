import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v6 as v6


# ============================================================
# COINDCX FUTURES HISTORICAL V7
#
# PURPOSE
# -------
# V6 best reliable idea:
#   BASE_CLOSE_TOP_HALF
#
# V7:
#   Keep original Weak Hammer Sweep entry.
#   1H = trend confirmation
#   5M = entry
#
# We test stronger 1H trend quality without changing
# the original Weak Hammer setup.
#
# HISTORICAL TEST ONLY
# NO REAL ORDERS
# ============================================================


DETAIL_FILE = "crypto_futures_v7_results.csv"
SUMMARY_FILE = "crypto_futures_v7_summary.csv"

WIN_THRESHOLD_PCT = 0.50
WALK_FORWARD_BLOCKS = 4

MIN_TOTAL_DECISIVE = 30


# ============================================================
# ADD 1-HOUR QUALITY INDICATORS
# ============================================================

def prepare_1h_indicators(df_1h):

    df = df_1h.copy()

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # EMA 20 / EMA 50
    # --------------------------------------------------------

    df["ema20_1h"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    df["ema50_1h"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    # EMA20 rising
    df["ema20_rising_1h"] = (
        df["ema20_1h"]
        >
        df["ema20_1h"].shift(1)
    )

    # EMA50 rising
    df["ema50_rising_1h"] = (
        df["ema50_1h"]
        >
        df["ema50_1h"].shift(1)
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    df["macd_1h"] = (
        ema12 - ema26
    )

    df["macd_signal_1h"] = (
        df["macd_1h"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["macd_hist_1h"] = (
        df["macd_1h"]
        -
        df["macd_signal_1h"]
    )

    df["macd_rising_1h"] = (
        df["macd_hist_1h"]
        >
        df["macd_hist_1h"].shift(1)
    )

    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    delta = (
        df["close"]
        .diff()
    )

    gain = (
        delta.clip(lower=0)
    )

    loss = (
        -delta.clip(upper=0)
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / 14,
            adjust=False,
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / 14,
            adjust=False,
        )
        .mean()
    )

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan,
        )
    )

    df["rsi_1h"] = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    df["rsi_rising_1h"] = (
        df["rsi_1h"]
        >
        df["rsi_1h"].shift(1)
    )

    # --------------------------------------------------------
    # 1H CANDLE
    # --------------------------------------------------------

    df["green_1h"] = (
        df["close"]
        >
        df["open"]
    )

    # --------------------------------------------------------
    # TREND STRENGTH
    # --------------------------------------------------------

    df["ema_gap_1h_pct"] = (
        (
            df["ema20_1h"]
            -
            df["ema50_1h"]
        )
        /
        df["ema50_1h"]
        * 100
    )

    return df


# ============================================================
# ATTACH COMPLETED 1H DATA TO 5M
# ============================================================

def attach_v7_1h_data(
    df_5m,
    df_1h,
):

    five = (
        df_5m
        .copy()
        .sort_values("datetime")
    )

    hour = (
        df_1h
        .copy()
        .sort_values("datetime")
    )

    # Shift 1H information by one candle.
    # This prevents using an unfinished/current 1H candle.
    columns_to_shift = [
        "ema20_1h",
        "ema50_1h",
        "ema20_rising_1h",
        "ema50_rising_1h",
        "macd_1h",
        "macd_signal_1h",
        "macd_hist_1h",
        "macd_rising_1h",
        "rsi_1h",
        "rsi_rising_1h",
        "green_1h",
        "ema_gap_1h_pct",
    ]

    hour[
        columns_to_shift
    ] = (
        hour[
            columns_to_shift
        ]
        .shift(1)
    )

    keep = [
        "datetime",
        *columns_to_shift,
    ]

    merged = pd.merge_asof(
        five,
        hour[keep],
        on="datetime",
        direction="backward",
    )

    return merged


# ============================================================
# V7 CONFIRMATIONS
# ============================================================

def get_v7_confirmations(row):

    c = {}

    # ========================================================
    # V6 BASE
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

    # V6 winner
    c["CLOSE_TOP_HALF"] = (
        pd.notna(
            row["close_position"]
        )
        and
        row["close_position"] >= 0.50
    )

    # ========================================================
    # V7 1-HOUR FILTERS
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

    c["H1_EMA50_RISING"] = (
        bool(
            row["ema50_rising_1h"]
        )
    )

    c["H1_MACD_POSITIVE"] = (
        pd.notna(
            row["macd_hist_1h"]
        )
        and
        row["macd_hist_1h"] > 0
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
        50 <= row["rsi_1h"] <= 70
    )

    c["H1_GREEN"] = (
        bool(
            row["green_1h"]
        )
    )

    c["H1_EMA_GAP_POSITIVE"] = (
        pd.notna(
            row["ema_gap_1h_pct"]
        )
        and
        row["ema_gap_1h_pct"] > 0
    )

    c["H1_EMA_GAP_005"] = (
        pd.notna(
            row["ema_gap_1h_pct"]
        )
        and
        row["ema_gap_1h_pct"] >= 0.05
    )

    return c


# ============================================================
# V7 CANDIDATES
# ============================================================

BASE = [
    "TREND",
    "MACD_POSITIVE",
    "POC",
    "SUPPORT075",
    "CLOSE_TOP_HALF",
]


CANDIDATES = {

    # V6 benchmark
    "V6_CLOSE_TOP_HALF":
        BASE,

    # --------------------------------------------------------
    # Single 1H filters
    # --------------------------------------------------------

    "V7_H1_EMA_BULL":
        BASE
        + [
            "H1_EMA_BULL",
        ],

    "V7_H1_EMA20_RISING":
        BASE
        + [
            "H1_EMA20_RISING",
        ],

    "V7_H1_MACD_POSITIVE":
        BASE
        + [
            "H1_MACD_POSITIVE",
        ],

    "V7_H1_MACD_RISING":
        BASE
        + [
            "H1_MACD_RISING",
        ],

    "V7_H1_RSI_RISING":
        BASE
        + [
            "H1_RSI_RISING",
        ],

    "V7_H1_RSI50_70":
        BASE
        + [
            "H1_RSI_50_70",
        ],

    # --------------------------------------------------------
    # Combined 1H trend filters
    # --------------------------------------------------------

    "V7_H1_EMA_MACD":
        BASE
        + [
            "H1_EMA_BULL",
            "H1_MACD_POSITIVE",
        ],

    "V7_H1_EMA20_MACD":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_POSITIVE",
        ],

    "V7_H1_EMA_MACD_RISING":
        BASE
        + [
            "H1_EMA_BULL",
            "H1_MACD_RISING",
        ],

    "V7_H1_EMA20_MACD_RISING":
        BASE
        + [
            "H1_EMA20_RISING",
            "H1_MACD_RISING",
        ],

    "V7_H1_EMA_MACD_RSI":
        BASE
        + [
            "H1_EMA_BULL",
            "H1_MACD_POSITIVE",
            "H1_RSI_50_70",
        ],

    "V7_H1_FULL_TREND":
        BASE
        + [
            "H1_EMA_BULL",
            "H1_EMA20_RISING",
            "H1_MACD_POSITIVE",
        ],

    "V7_H1_STRONG_TREND":
        BASE
        + [
            "H1_EMA_BULL",
            "H1_EMA20_RISING",
            "H1_MACD_POSITIVE",
            "H1_MACD_RISING",
        ],

    "V7_H1_STRONG_RSI":
        BASE
        + [
            "H1_EMA_BULL",
            "H1_EMA20_RISING",
            "H1_MACD_POSITIVE",
            "H1_RSI_50_70",
        ],

    "V7_H1_EMA_GAP":
        BASE
        + [
            "H1_EMA_GAP_005",
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
            get_v7_confirmations(
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

            "EMA20_1H":
                float(
                    row["ema20_1h"]
                )
                if pd.notna(
                    row["ema20_1h"]
                )
                else np.nan,

            "EMA50_1H":
                float(
                    row["ema50_1h"]
                )
                if pd.notna(
                    row["ema50_1h"]
                )
                else np.nan,

            "EMA_GAP_1H_%":
                round(
                    float(
                        row["ema_gap_1h_pct"]
                    ),
                    4,
                )
                if pd.notna(
                    row["ema_gap_1h_pct"]
                )
                else np.nan,
        }

        for (
            name,
            value,
        ) in confirmations.items():

            event[name] = int(
                value
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
# WALK-FORWARD BLOCKS
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

    for (
        block_number,
        idx,
    ) in enumerate(
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

    for (
        block_number,
        block,
    ) in blocks:

        for (
            candidate,
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

    for (
        candidate,
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
                == candidate
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
                    >= MIN_TOTAL_DECISIVE
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
        "=" * 110
    )

    print(
        "V7 WALK-FORWARD BLOCK RESULTS"
    )

    print(
        "=" * 110
    )

    print(
        block_results.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 110
    )

    print(
        "V7 OVERALL RESULTS"
    )

    print(
        "=" * 110
    )

    display = [

        "CANDIDATE",

        "TOTAL_DECISIVE",

        "OVERALL_WIN_RATE_%",

        "OVERALL_AVG_RETURN_%",

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
                    == WALK_FORWARD_BLOCKS
                )
            ]
            .copy()
        )

        print()
        print(
            "=" * 110
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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 100
    )

    print(
        "COINDCX FUTURES HISTORICAL V7"
    )

    print(
        "1H TREND QUALITY + 5M ENTRY"
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

    # --------------------------------------------------------
    # 5M
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 1H
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5M indicators
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NEW V7 1H indicators
    # --------------------------------------------------------

    print(
        "Calculating V7 1H indicators..."
    )

    df_1h_v7 = (
        prepare_1h_indicators(
            df_1h
        )
    )

    print(
        "Attaching completed 1H data "
        "to 5M candles..."
    )

    df_5m = (
        attach_v7_1h_data(
            df_5m,
            df_1h_v7,
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
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

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
        "TOTAL EVENTS:",
        len(events),
    )

    # --------------------------------------------------------
    # Walk forward
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    block_results.to_csv(
        DETAIL_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "V7 results saved:",
        DETAIL_FILE,
    )

    print(
        "V7 summary saved:",
        SUMMARY_FILE,
    )

    print_results(
        block_results,
        summary,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FUTURES HISTORICAL V7 COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
