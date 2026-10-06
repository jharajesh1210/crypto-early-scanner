import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1


# ============================================================
# COINDCX FUTURES HISTORICAL V3
# WEAK HAMMER LONG OPTIMIZATION
#
# 1H = TREND
# 5M = ENTRY
#
# Goal:
# Train on older data
# Test on unseen recent data
#
# IMPORTANT:
# This script does NOT place real trades.
# ============================================================


DETAIL_FILE = "crypto_futures_v3_results.csv"
SUMMARY_FILE = "crypto_futures_v3_summary.csv"

TRAIN_RATIO = 0.67

WIN_THRESHOLD_PCT = 0.50

MIN_TRAIN_DECISIVE = 25
MIN_TEST_DECISIVE = 10


# ============================================================
# EXTRA INDICATORS
# ============================================================

def add_v3_indicators(df):
    df = df.copy()

    # -------------------------
    # EMA slope
    # -------------------------

    df["ema20_slope_pct"] = (
        (
            df["ema20"]
            - df["ema20"].shift(3)
        )
        / df["ema20"].shift(3)
        * 100
    )

    df["ema50_slope_pct"] = (
        (
            df["ema50"]
            - df["ema50"].shift(3)
        )
        / df["ema50"].shift(3)
        * 100
    )

    # -------------------------
    # MACD momentum
    # -------------------------

    df["macd_rising"] = (
        df["macd_hist"]
        > df["macd_hist"].shift(1)
    )

    # -------------------------
    # RSI momentum
    # -------------------------

    df["rsi_rising"] = (
        df["rsi"]
        > df["rsi"].shift(1)
    )

    # -------------------------
    # Bollinger Band width
    # -------------------------

    df["bb_width_pct"] = (
        (
            df["bb_upper"]
            - df["bb_lower"]
        )
        / df["bb_middle"]
        * 100
    )

    df["bb_width_avg"] = (
        df["bb_width_pct"]
        .rolling(20)
        .mean()
    )

    # -------------------------
    # Price vs BB middle
    # -------------------------

    df["above_bb_middle"] = (
        df["close"]
        >= df["bb_middle"]
    )

    # -------------------------
    # Previous support
    # 2 hours = 24 x 5M
    # -------------------------

    df["support_2h"] = (
        df["low"]
        .shift(1)
        .rolling(24)
        .min()
    )

    df["distance_support_pct"] = (
        (
            df["close"]
            - df["support_2h"]
        )
        / df["close"]
        * 100
    )

    # -------------------------
    # 4-hour support
    # -------------------------

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

    # -------------------------
    # Volume acceleration
    # -------------------------

    df["volume_ratio_prev"] = (
        df["volume_ratio"]
        .shift(1)
    )

    df["volume_increasing"] = (
        df["volume_ratio"]
        > df["volume_ratio_prev"]
    )

    # -------------------------
    # POC distance
    # -------------------------

    df["price_vs_poc_pct"] = (
        (
            df["close"]
            - df["poc"]
        )
        / df["poc"]
        * 100
    )

    return df


# ============================================================
# V3 CONFIRMATIONS
# ============================================================

def get_v3_confirmations(row):

    c = {}

    # ----------------------------------
    # Core V2 winning confirmations
    # ----------------------------------

    c["TREND"] = (
        row["trend_1h"]
        == "BULLISH"
    )

    c["MACD_POSITIVE"] = (
        pd.notna(row["macd_hist"])
        and row["macd_hist"] > 0
    )

    c["POC"] = (
        pd.notna(row["poc"])
        and row["close"] >= row["poc"]
    )

    # ----------------------------------
    # EMA
    # ----------------------------------

    c["EMA_BULL"] = (
        row["close"] > row["ema20"]
        and row["ema20"] > row["ema50"]
    )

    c["EMA20_RISING"] = (
        pd.notna(row["ema20_slope_pct"])
        and row["ema20_slope_pct"] > 0
    )

    # ----------------------------------
    # MACD strengthening
    # ----------------------------------

    c["MACD_RISING"] = (
        bool(row["macd_rising"])
    )

    # ----------------------------------
    # RSI zones
    # ----------------------------------

    c["RSI_40_65"] = (
        pd.notna(row["rsi"])
        and 40 <= row["rsi"] <= 65
    )

    c["RSI_45_65"] = (
        pd.notna(row["rsi"])
        and 45 <= row["rsi"] <= 65
    )

    c["RSI_RISING"] = (
        bool(row["rsi_rising"])
    )

    # ----------------------------------
    # Volume
    # ----------------------------------

    c["VOL_1"] = (
        pd.notna(row["volume_ratio"])
        and row["volume_ratio"] >= 1.0
    )

    c["VOL_1_2"] = (
        pd.notna(row["volume_ratio"])
        and row["volume_ratio"] >= 1.2
    )

    c["VOL_RISING"] = (
        bool(row["volume_increasing"])
    )

    # ----------------------------------
    # BB
    # ----------------------------------

    c["ABOVE_BB_MIDDLE"] = (
        bool(row["above_bb_middle"])
    )

    c["BB_EXPANDING"] = (
        pd.notna(row["bb_width_pct"])
        and pd.notna(row["bb_width_avg"])
        and row["bb_width_pct"]
        >= row["bb_width_avg"]
    )

    # ----------------------------------
    # Support
    # ----------------------------------

    c["SUPPORT_2H_1PCT"] = (
        pd.notna(
            row["distance_support_pct"]
        )
        and 0 <=
        row["distance_support_pct"]
        <= 1.0
    )

    c["SUPPORT_4H_1_5PCT"] = (
        pd.notna(
            row[
                "distance_support_4h_pct"
            ]
        )
        and 0 <=
        row[
            "distance_support_4h_pct"
        ]
        <= 1.5
    )

    # ----------------------------------
    # POC proximity
    # ----------------------------------

    c["POC_NEAR"] = (
        pd.notna(
            row["price_vs_poc_pct"]
        )
        and 0 <=
        row["price_vs_poc_pct"]
        <= 1.5
    )

    return c


# ============================================================
# GET WEAK HAMMER SIGNALS
# ============================================================

def collect_events(df):

    events = []

    max_forward = v1.FORWARD_BARS["4H"]

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
        except Exception as exc:
            print(
                "Weak Hammer error:",
                i,
                exc,
            )
            continue

        if signal != "LONG":
            continue

        row = df.iloc[i]

        confirmations = (
            get_v3_confirmations(row)
        )

        forward = (
            v1.calculate_forward_results(
                df,
                i,
                "LONG",
            )
        )

        event = {
            "PAIR": v1.PAIR,

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

            "POC":
                round(
                    float(row["poc"]),
                    8,
                )
                if pd.notna(row["poc"])
                else np.nan,

            "BB_WIDTH_PCT":
                round(
                    float(
                        row["bb_width_pct"]
                    ),
                    4,
                ),
        }

        for name, value in (
            confirmations.items()
        ):
            event[name] = int(value)

        event.update(forward)

        events.append(event)

    return pd.DataFrame(events)


# ============================================================
# PERFORMANCE
# ============================================================

def performance(df):

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

    neutral = (
        len(returns)
        - wins
        - losses
    )

    decisive = (
        wins
        + losses
    )

    if decisive == 0:
        win_rate = np.nan
    else:
        win_rate = (
            wins
            / decisive
            * 100
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
            if pd.notna(win_rate)
            else np.nan,

        "AVG_RETURN":
            round(
                float(
                    returns.mean()
                ),
                4,
            ),
    }


# ============================================================
# FILTER DEFINITIONS
# ============================================================

FILTERS = {

    # ----------------------------------
    # V2 benchmark
    # ----------------------------------

    "BASE_TREND_MACD_POC": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
    ],

    # ----------------------------------
    # EMA tests
    # ----------------------------------

    "BASE+EMA_BULL": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "EMA_BULL",
    ],

    "BASE+EMA20_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "EMA20_RISING",
    ],

    # ----------------------------------
    # MACD strengthening
    # ----------------------------------

    "BASE+MACD_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "MACD_RISING",
    ],

    # ----------------------------------
    # RSI
    # ----------------------------------

    "BASE+RSI40_65": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "RSI_40_65",
    ],

    "BASE+RSI45_65": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "RSI_45_65",
    ],

    "BASE+RSI40_65+RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "RSI_40_65",
        "RSI_RISING",
    ],

    # ----------------------------------
    # Volume
    # ----------------------------------

    "BASE+VOL1": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "VOL_1",
    ],

    "BASE+VOL1.2": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "VOL_1_2",
    ],

    "BASE+VOL_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "VOL_RISING",
    ],

    # ----------------------------------
    # Bollinger
    # ----------------------------------

    "BASE+BB_MIDDLE": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "ABOVE_BB_MIDDLE",
    ],

    "BASE+BB_EXPANDING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "BB_EXPANDING",
    ],

    # ----------------------------------
    # Support
    # ----------------------------------

    "BASE+SUPPORT2H": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "SUPPORT_2H_1PCT",
    ],

    "BASE+SUPPORT4H": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "SUPPORT_4H_1_5PCT",
    ],

    # ----------------------------------
    # POC proximity
    # ----------------------------------

    "BASE+POC_NEAR": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "POC_NEAR",
    ],

    # ----------------------------------
    # Strong combinations
    # ----------------------------------

    "BASE+EMA+MACD_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "EMA_BULL",
        "MACD_RISING",
    ],

    "BASE+MACD_RISING+VOL": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "MACD_RISING",
        "VOL_1",
    ],

    "BASE+RSI+MACD_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "RSI_40_65",
        "MACD_RISING",
    ],

    "BASE+RSI+VOL": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "RSI_40_65",
        "VOL_1",
    ],

    "BASE+EMA+RSI": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "EMA_BULL",
        "RSI_40_65",
    ],

    "BASE+EMA+VOL": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "EMA_BULL",
        "VOL_1",
    ],

    "BASE+RSI+VOL+MACD_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "RSI_40_65",
        "VOL_1",
        "MACD_RISING",
    ],

    "BASE+EMA+RSI+MACD_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "EMA_BULL",
        "RSI_40_65",
        "MACD_RISING",
    ],

    "BASE+EMA+VOL+MACD_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "EMA_BULL",
        "VOL_1",
        "MACD_RISING",
    ],

    "BASE+EMA+RSI+VOL": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "EMA_BULL",
        "RSI_40_65",
        "VOL_1",
    ],

    "BASE+EMA+RSI+VOL+MACD": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "EMA_BULL",
        "RSI_40_65",
        "VOL_1",
        "MACD_RISING",
    ],

    "BASE+POC_NEAR+MACD_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "POC_NEAR",
        "MACD_RISING",
    ],

    "BASE+SUPPORT+MACD_RISING": [
        "TREND",
        "MACD_POSITIVE",
        "POC",
        "SUPPORT_4H_1_5PCT",
        "MACD_RISING",
    ],
}


# ============================================================
# APPLY FILTER
# ============================================================

def apply_filter(df, columns):

    mask = pd.Series(
        True,
        index=df.index,
    )

    for column in columns:
        mask = (
            mask
            & (df[column] == 1)
        )

    return df[mask].copy()


# ============================================================
# TRAIN / TEST
# ============================================================

def build_train_test_summary(events):

    events = events.sort_values(
        "TIME"
    ).reset_index(drop=True)

    split_index = int(
        len(events)
        * TRAIN_RATIO
    )

    train = events.iloc[
        :split_index
    ].copy()

    test = events.iloc[
        split_index:
    ].copy()

    print()
    print("=" * 80)
    print("TRAIN / TEST SPLIT")
    print("=" * 80)

    print(
        "TOTAL EVENTS:",
        len(events),
    )

    print(
        "TRAIN EVENTS:",
        len(train),
    )

    print(
        "TEST EVENTS:",
        len(test),
    )

    if not train.empty:
        print(
            "TRAIN PERIOD:",
            train.iloc[0]["TIME"],
            "to",
            train.iloc[-1]["TIME"],
        )

    if not test.empty:
        print(
            "TEST PERIOD:",
            test.iloc[0]["TIME"],
            "to",
            test.iloc[-1]["TIME"],
        )

    rows = []

    for filter_name, columns in (
        FILTERS.items()
    ):

        train_filtered = apply_filter(
            train,
            columns,
        )

        test_filtered = apply_filter(
            test,
            columns,
        )

        train_stats = performance(
            train_filtered
        )

        test_stats = performance(
            test_filtered
        )

        if train_stats is None:
            continue

        if test_stats is None:
            continue

        row = {
            "FILTER":
                filter_name,

            "TRAIN_SIGNALS":
                train_stats["SIGNALS"],

            "TRAIN_WINS":
                train_stats["WINS"],

            "TRAIN_LOSSES":
                train_stats["LOSSES"],

            "TRAIN_NEUTRAL":
                train_stats["NEUTRAL"],

            "TRAIN_DECISIVE":
                train_stats["DECISIVE"],

            "TRAIN_WIN_RATE_%":
                train_stats["WIN_RATE"],

            "TRAIN_AVG_RETURN_%":
                train_stats["AVG_RETURN"],

            "TEST_SIGNALS":
                test_stats["SIGNALS"],

            "TEST_WINS":
                test_stats["WINS"],

            "TEST_LOSSES":
                test_stats["LOSSES"],

            "TEST_NEUTRAL":
                test_stats["NEUTRAL"],

            "TEST_DECISIVE":
                test_stats["DECISIVE"],

            "TEST_WIN_RATE_%":
                test_stats["WIN_RATE"],

            "TEST_AVG_RETURN_%":
                test_stats["AVG_RETURN"],
        }

        row["ENOUGH_SAMPLE"] = (
            row["TRAIN_DECISIVE"]
            >= MIN_TRAIN_DECISIVE
            and
            row["TEST_DECISIVE"]
            >= MIN_TEST_DECISIVE
        )

        if (
            pd.notna(
                row["TRAIN_WIN_RATE_%"]
            )
            and
            pd.notna(
                row["TEST_WIN_RATE_%"]
            )
        ):
            row["STABILITY_GAP_%"] = round(
                abs(
                    row["TRAIN_WIN_RATE_%"]
                    - row["TEST_WIN_RATE_%"]
                ),
                2,
            )
        else:
            row["STABILITY_GAP_%"] = (
                np.nan
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary = summary.sort_values(
        by=[
            "ENOUGH_SAMPLE",
            "TEST_WIN_RATE_%",
            "TEST_DECISIVE",
            "STABILITY_GAP_%",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return summary


# ============================================================
# PRINT RESULTS
# ============================================================

def print_summary(summary):

    print()
    print("=" * 100)
    print("FUTURES V3 OUT-OF-SAMPLE RESULTS")
    print("=" * 100)

    if summary.empty:
        print("No V3 results found.")
        return

    reliable = summary[
        summary["ENOUGH_SAMPLE"]
        == True
    ].copy()

    columns = [
        "FILTER",
        "TRAIN_DECISIVE",
        "TRAIN_WIN_RATE_%",
        "TEST_DECISIVE",
        "TEST_WIN_RATE_%",
        "TEST_AVG_RETURN_%",
        "STABILITY_GAP_%",
    ]

    if reliable.empty:

        print(
            "No combination has enough "
            "train/test decisive trades."
        )

        print()
        print(
            summary[
                columns
            ]
            .head(20)
            .to_string(index=False)
        )

        return

    print()
    print(
        "RELIABLE RESULTS:"
    )
    print()

    print(
        reliable[
            columns
        ]
        .head(25)
        .to_string(index=False)
    )

    target = reliable[
        (
            reliable[
                "TEST_WIN_RATE_%"
            ] >= 70
        )
        &
        (
            reliable[
                "TEST_AVG_RETURN_%"
            ] > 0
        )
    ]

    print()
    print("=" * 100)

    if target.empty:

        print(
            "NO RELIABLE 70%+ OUT-OF-SAMPLE "
            "SETUP FOUND."
        )

        print(
            "Do not force 70-80%. "
            "Keep the best stable setup "
            "and improve it in the next test."
        )

    else:

        print(
            "70%+ OUT-OF-SAMPLE "
            "CANDIDATES FOUND:",
            len(target),
        )

        print()

        print(
            target[
                columns
            ]
            .to_string(index=False)
        )

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "COINDCX FUTURES HISTORICAL V3"
    )
    print(
        "WEAK HAMMER OUT-OF-SAMPLE TEST"
    )
    print("=" * 80)

    print(
        "PAIR:",
        v1.PAIR,
    )

    print(
        "HISTORY:",
        v1.DAYS_TO_TEST,
        "days",
    )

    print()
    print(
        "Downloading native 5-minute "
        "Futures candles..."
    )

    df_5m = v1.fetch_history(
        v1.PAIR,
        "5",
        v1.DAYS_TO_TEST,
        v1.CHUNK_DAYS_5M,
    )

    if df_5m.empty:
        raise RuntimeError(
            "No 5-minute Futures data."
        )

    print()
    print(
        "5M CANDLES:",
        len(df_5m),
    )

    print()
    print(
        "Downloading 1-hour "
        "Futures candles..."
    )

    df_1h = v1.fetch_history(
        v1.PAIR,
        "60",
        v1.DAYS_TO_TEST,
        v1.CHUNK_DAYS_1H,
    )

    if df_1h.empty:
        raise RuntimeError(
            "No 1-hour Futures data."
        )

    print()
    print(
        "1H CANDLES:",
        len(df_1h),
    )

    print()
    print(
        "Calculating indicators..."
    )

    df_5m = v1.calculate_indicators(
        df_5m
    )

    print(
        "Calculating Volume Profile..."
    )

    df_5m = v1.add_volume_profile(
        df_5m
    )

    print(
        "Attaching 1H trend..."
    )

    df_5m = v1.attach_1h_trend(
        df_5m,
        df_1h,
    )

    print(
        "Adding V3 indicators..."
    )

    df_5m = add_v3_indicators(
        df_5m
    )

    df_5m = df_5m.dropna(
        subset=[
            "ema20",
            "ema50",
            "rsi",
            "macd_hist",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "poc",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        "Finding Weak Hammer "
        "Sweep LONG events..."
    )

    events = collect_events(
        df_5m
    )

    if events.empty:
        print(
            "No Weak Hammer events found."
        )
        return

    events.to_csv(
        DETAIL_FILE,
        index=False,
    )

    print()
    print(
        "EVENTS FOUND:",
        len(events),
    )

    print(
        "Detailed results saved:",
        DETAIL_FILE,
    )

    summary = (
        build_train_test_summary(
            events
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "Summary saved:",
        SUMMARY_FILE,
    )

    print_summary(
        summary
    )

    print()
    print("=" * 80)
    print(
        "FUTURES HISTORICAL V3 COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
