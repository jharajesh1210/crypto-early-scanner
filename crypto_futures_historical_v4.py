import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1


# ============================================================
# COINDCX FUTURES HISTORICAL V4
#
# 1H = TREND FILTER
# 5M = ENTRY
#
# BASE FROM V3:
# Weak Hammer Sweep LONG
# + 1H Bullish Trend
# + Positive MACD
# + Price >= POC
# + Near 4H Support
#
# V4 tests additional filters:
# RSI / EMA / MACD momentum / Volume / BB / Candle strength
#
# IMPORTANT:
# Historical research only.
# Does NOT place real trades.
# ============================================================


DETAIL_FILE = "crypto_futures_v4_results.csv"
SUMMARY_FILE = "crypto_futures_v4_summary.csv"

TRAIN_RATIO = 0.67

WIN_THRESHOLD_PCT = 0.50

MIN_TRAIN_DECISIVE = 20
MIN_TEST_DECISIVE = 10

TARGET_WIN_RATE = 70.0


# ============================================================
# V4 INDICATORS
# ============================================================

def add_v4_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA direction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MACD momentum
    # --------------------------------------------------------

    df["macd_hist_prev"] = (
        df["macd_hist"].shift(1)
    )

    df["macd_rising"] = (
        df["macd_hist"]
        > df["macd_hist_prev"]
    )

    # --------------------------------------------------------
    # RSI momentum
    # --------------------------------------------------------

    df["rsi_prev"] = (
        df["rsi"].shift(1)
    )

    df["rsi_rising"] = (
        df["rsi"]
        > df["rsi_prev"]
    )

    # --------------------------------------------------------
    # Bollinger Band width
    # --------------------------------------------------------

    df["bb_width_pct"] = (
        (
            df["bb_upper"]
            - df["bb_lower"]
        )
        / df["bb_middle"]
        * 100
    )

    df["bb_width_avg20"] = (
        df["bb_width_pct"]
        .rolling(20)
        .mean()
    )

    df["bb_width_prev"] = (
        df["bb_width_pct"]
        .shift(1)
    )

    df["bb_expanding"] = (
        df["bb_width_pct"]
        > df["bb_width_prev"]
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["volume_ratio_prev"] = (
        df["volume_ratio"].shift(1)
    )

    df["volume_rising"] = (
        df["volume_ratio"]
        > df["volume_ratio_prev"]
    )

    # --------------------------------------------------------
    # 4H SUPPORT
    #
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
    # POC distance
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
    # Candle measurements
    # --------------------------------------------------------

    df["body"] = (
        df["close"]
        - df["open"]
    ).abs()

    df["range"] = (
        df["high"]
        - df["low"]
    )

    df["body_ratio"] = (
        df["body"]
        / df["range"].replace(0, np.nan)
    )

    df["green"] = (
        df["close"] > df["open"]
    )

    df["close_position"] = (
        (
            df["close"]
            - df["low"]
        )
        / df["range"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Previous candle
    # --------------------------------------------------------

    df["prev_green"] = (
        df["green"].shift(1)
    )

    # --------------------------------------------------------
    # ATR-like volatility
    # --------------------------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = (
        df["high"]
        - previous_close
    ).abs()

    tr3 = (
        df["low"]
        - previous_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    df["atr14"] = (
        tr.rolling(14).mean()
    )

    df["atr_pct"] = (
        df["atr14"]
        / df["close"]
        * 100
    )

    return df


# ============================================================
# V4 CONFIRMATIONS
# ============================================================

def get_confirmations(row):

    c = {}

    # ========================================================
    # V3 BASE
    # ========================================================

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

    c["SUPPORT4H"] = (
        pd.notna(
            row["distance_support_4h_pct"]
        )
        and
        0 <= row["distance_support_4h_pct"] <= 1.5
    )

    # ========================================================
    # TIGHTER SUPPORT TESTS
    # ========================================================

    c["SUPPORT4H_1PCT"] = (
        pd.notna(
            row["distance_support_4h_pct"]
        )
        and
        0 <= row["distance_support_4h_pct"] <= 1.0
    )

    c["SUPPORT4H_075"] = (
        pd.notna(
            row["distance_support_4h_pct"]
        )
        and
        0 <= row["distance_support_4h_pct"] <= 0.75
    )

    # ========================================================
    # RSI
    # ========================================================

    c["RSI_35_65"] = (
        pd.notna(row["rsi"])
        and 35 <= row["rsi"] <= 65
    )

    c["RSI_40_65"] = (
        pd.notna(row["rsi"])
        and 40 <= row["rsi"] <= 65
    )

    c["RSI_40_60"] = (
        pd.notna(row["rsi"])
        and 40 <= row["rsi"] <= 60
    )

    c["RSI_45_65"] = (
        pd.notna(row["rsi"])
        and 45 <= row["rsi"] <= 65
    )

    c["RSI_RISING"] = (
        bool(row["rsi_rising"])
    )

    # ========================================================
    # EMA
    # ========================================================

    c["PRICE_ABOVE_EMA20"] = (
        row["close"] > row["ema20"]
    )

    c["EMA_BULL"] = (
        row["close"] > row["ema20"]
        and
        row["ema20"] > row["ema50"]
    )

    c["EMA20_RISING"] = (
        pd.notna(row["ema20_slope_pct"])
        and
        row["ema20_slope_pct"] > 0
    )

    # ========================================================
    # MACD
    # ========================================================

    c["MACD_RISING"] = (
        bool(row["macd_rising"])
    )

    # ========================================================
    # VOLUME
    # ========================================================

    c["VOL_08"] = (
        pd.notna(row["volume_ratio"])
        and row["volume_ratio"] >= 0.8
    )

    c["VOL_10"] = (
        pd.notna(row["volume_ratio"])
        and row["volume_ratio"] >= 1.0
    )

    c["VOL_12"] = (
        pd.notna(row["volume_ratio"])
        and row["volume_ratio"] >= 1.2
    )

    c["VOL_RISING"] = (
        bool(row["volume_rising"])
    )

    # ========================================================
    # BOLLINGER BAND
    # ========================================================

    c["ABOVE_BB_MIDDLE"] = (
        row["close"]
        >= row["bb_middle"]
    )

    c["BELOW_BB_UPPER"] = (
        row["close"]
        <= row["bb_upper"]
    )

    c["BB_EXPANDING"] = (
        bool(row["bb_expanding"])
    )

    # ========================================================
    # POC PROXIMITY
    # ========================================================

    c["POC_NEAR_1PCT"] = (
        pd.notna(row["price_vs_poc_pct"])
        and
        0 <= row["price_vs_poc_pct"] <= 1.0
    )

    c["POC_NEAR_05"] = (
        pd.notna(row["price_vs_poc_pct"])
        and
        0 <= row["price_vs_poc_pct"] <= 0.5
    )

    # ========================================================
    # ENTRY CANDLE
    # ========================================================

    c["GREEN_CANDLE"] = (
        bool(row["green"])
    )

    c["BODY_25"] = (
        pd.notna(row["body_ratio"])
        and row["body_ratio"] >= 0.25
    )

    c["BODY_40"] = (
        pd.notna(row["body_ratio"])
        and row["body_ratio"] >= 0.40
    )

    c["CLOSE_TOP_HALF"] = (
        pd.notna(row["close_position"])
        and row["close_position"] >= 0.50
    )

    c["CLOSE_TOP_30"] = (
        pd.notna(row["close_position"])
        and row["close_position"] >= 0.70
    )

    return c


# ============================================================
# WEAK HAMMER EVENTS
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
                row["datetime"].isoformat(),

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
                    float(row["macd_hist"]),
                    8,
                ),

            "VOLUME_RATIO":
                round(
                    float(row["volume_ratio"]),
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

            "POC_DISTANCE_%":
                round(
                    float(
                        row[
                            "price_vs_poc_pct"
                        ]
                    ),
                    4,
                )
                if pd.notna(
                    row["price_vs_poc_pct"]
                )
                else np.nan,
        }

        for name, value in confirmations.items():
            event[name] = int(value)

        event.update(forward)

        events.append(event)

    return pd.DataFrame(events)


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
            returns >= WIN_THRESHOLD_PCT
        ).sum()
    )

    losses = int(
        (
            returns <= -WIN_THRESHOLD_PCT
        ).sum()
    )

    neutral = int(
        len(returns)
        - wins
        - losses
    )

    decisive = (
        wins + losses
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

        "WIN_RATE":
            round(win_rate, 2)
            if pd.notna(win_rate)
            else np.nan,

        "AVG_RETURN":
            round(
                float(returns.mean()),
                4,
            ),

        "MEDIAN_RETURN":
            round(
                float(returns.median()),
                4,
            ),
    }


# ============================================================
# FILTERS
# ============================================================

BASE = [
    "TREND",
    "MACD_POSITIVE",
    "POC",
    "SUPPORT4H",
]


FILTERS = {

    # --------------------------------------------------------
    # V3 WINNER
    # --------------------------------------------------------

    "V3_BASE":
        BASE,

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    "BASE+SUPPORT1PCT":
        [
            "TREND",
            "MACD_POSITIVE",
            "POC",
            "SUPPORT4H_1PCT",
        ],

    "BASE+SUPPORT075":
        [
            "TREND",
            "MACD_POSITIVE",
            "POC",
            "SUPPORT4H_075",
        ],

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    "BASE+RSI35_65":
        BASE + [
            "RSI_35_65"
        ],

    "BASE+RSI40_65":
        BASE + [
            "RSI_40_65"
        ],

    "BASE+RSI40_60":
        BASE + [
            "RSI_40_60"
        ],

    "BASE+RSI45_65":
        BASE + [
            "RSI_45_65"
        ],

    "BASE+RSI_RISING":
        BASE + [
            "RSI_RISING"
        ],

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    "BASE+ABOVE_EMA20":
        BASE + [
            "PRICE_ABOVE_EMA20"
        ],

    "BASE+EMA_BULL":
        BASE + [
            "EMA_BULL"
        ],

    "BASE+EMA20_RISING":
        BASE + [
            "EMA20_RISING"
        ],

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    "BASE+MACD_RISING":
        BASE + [
            "MACD_RISING"
        ],

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    "BASE+VOL08":
        BASE + [
            "VOL_08"
        ],

    "BASE+VOL10":
        BASE + [
            "VOL_10"
        ],

    "BASE+VOL12":
        BASE + [
            "VOL_12"
        ],

    "BASE+VOL_RISING":
        BASE + [
            "VOL_RISING"
        ],

    # --------------------------------------------------------
    # BB
    # --------------------------------------------------------

    "BASE+BB_MIDDLE":
        BASE + [
            "ABOVE_BB_MIDDLE"
        ],

    "BASE+BB_EXPANDING":
        BASE + [
            "BB_EXPANDING"
        ],

    # --------------------------------------------------------
    # POC
    # --------------------------------------------------------

    "BASE+POC_NEAR1":
        BASE + [
            "POC_NEAR_1PCT"
        ],

    "BASE+POC_NEAR05":
        BASE + [
            "POC_NEAR_05"
        ],

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    "BASE+GREEN":
        BASE + [
            "GREEN_CANDLE"
        ],

    "BASE+BODY25":
        BASE + [
            "BODY_25"
        ],

    "BASE+BODY40":
        BASE + [
            "BODY_40"
        ],

    "BASE+CLOSE_TOP_HALF":
        BASE + [
            "CLOSE_TOP_HALF"
        ],

    "BASE+CLOSE_TOP30":
        BASE + [
            "CLOSE_TOP_30"
        ],

    # --------------------------------------------------------
    # TWO FILTER COMBINATIONS
    # --------------------------------------------------------

    "BASE+RSI+MACD":
        BASE + [
            "RSI_40_65",
            "MACD_RISING",
        ],

    "BASE+RSI+EMA":
        BASE + [
            "RSI_40_65",
            "EMA20_RISING",
        ],

    "BASE+RSI+VOL":
        BASE + [
            "RSI_40_65",
            "VOL_10",
        ],

    "BASE+MACD+VOL":
        BASE + [
            "MACD_RISING",
            "VOL_10",
        ],

    "BASE+EMA+MACD":
        BASE + [
            "EMA20_RISING",
            "MACD_RISING",
        ],

    "BASE+EMA+VOL":
        BASE + [
            "EMA20_RISING",
            "VOL_10",
        ],

    "BASE+BB+MACD":
        BASE + [
            "ABOVE_BB_MIDDLE",
            "MACD_RISING",
        ],

    "BASE+GREEN+MACD":
        BASE + [
            "GREEN_CANDLE",
            "MACD_RISING",
        ],

    "BASE+GREEN+VOL":
        BASE + [
            "GREEN_CANDLE",
            "VOL_10",
        ],

    "BASE+POCNEAR+MACD":
        BASE + [
            "POC_NEAR_1PCT",
            "MACD_RISING",
        ],

    # --------------------------------------------------------
    # STRONG COMBINATIONS
    # --------------------------------------------------------

    "BASE+RSI+MACD+VOL":
        BASE + [
            "RSI_40_65",
            "MACD_RISING",
            "VOL_10",
        ],

    "BASE+RSI+EMA+MACD":
        BASE + [
            "RSI_40_65",
            "EMA20_RISING",
            "MACD_RISING",
        ],

    "BASE+EMA+MACD+VOL":
        BASE + [
            "EMA20_RISING",
            "MACD_RISING",
            "VOL_10",
        ],

    "BASE+RSI+BB+MACD":
        BASE + [
            "RSI_40_65",
            "ABOVE_BB_MIDDLE",
            "MACD_RISING",
        ],

    "BASE+RSI+GREEN+MACD":
        BASE + [
            "RSI_40_65",
            "GREEN_CANDLE",
            "MACD_RISING",
        ],

    "BASE+RSI+MACD+POCNEAR":
        BASE + [
            "RSI_40_65",
            "MACD_RISING",
            "POC_NEAR_1PCT",
        ],

    "BASE+EMA+MACD+POCNEAR":
        BASE + [
            "EMA20_RISING",
            "MACD_RISING",
            "POC_NEAR_1PCT",
        ],

    "BASE+RSI+EMA+MACD+VOL":
        BASE + [
            "RSI_40_65",
            "EMA20_RISING",
            "MACD_RISING",
            "VOL_10",
        ],

    "BASE+RSI+EMA+MACD+BB":
        BASE + [
            "RSI_40_65",
            "EMA20_RISING",
            "MACD_RISING",
            "ABOVE_BB_MIDDLE",
        ],

    "BASE+RSI+MACD+VOL+GREEN":
        BASE + [
            "RSI_40_65",
            "MACD_RISING",
            "VOL_10",
            "GREEN_CANDLE",
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
# TRAIN TEST SUMMARY
# ============================================================

def build_summary(events):

    events = (
        events
        .sort_values("TIME")
        .reset_index(drop=True)
    )

    split_index = int(
        len(events) * TRAIN_RATIO
    )

    train = (
        events
        .iloc[:split_index]
        .copy()
    )

    test = (
        events
        .iloc[split_index:]
        .copy()
    )

    print()
    print("=" * 100)
    print("V4 TRAIN / UNSEEN TEST")
    print("=" * 100)

    print(
        "TOTAL WEAK HAMMER EVENTS:",
        len(events),
    )

    print(
        "TRAIN EVENTS:",
        len(train),
    )

    print(
        "UNSEEN TEST EVENTS:",
        len(test),
    )

    rows = []

    for name, columns in FILTERS.items():

        train_filtered = (
            apply_filter(
                train,
                columns,
            )
        )

        test_filtered = (
            apply_filter(
                test,
                columns,
            )
        )

        train_stats = (
            performance(
                train_filtered
            )
        )

        test_stats = (
            performance(
                test_filtered
            )
        )

        if train_stats is None:
            continue

        if test_stats is None:
            continue

        enough_sample = (
            train_stats["DECISIVE"]
            >= MIN_TRAIN_DECISIVE
            and
            test_stats["DECISIVE"]
            >= MIN_TEST_DECISIVE
        )

        if (
            pd.notna(
                train_stats["WIN_RATE"]
            )
            and
            pd.notna(
                test_stats["WIN_RATE"]
            )
        ):

            stability_gap = abs(
                train_stats["WIN_RATE"]
                - test_stats["WIN_RATE"]
            )

        else:

            stability_gap = np.nan

        rows.append({

            "FILTER":
                name,

            "TRAIN_SIGNALS":
                train_stats["SIGNALS"],

            "TRAIN_DECISIVE":
                train_stats["DECISIVE"],

            "TRAIN_WINS":
                train_stats["WINS"],

            "TRAIN_LOSSES":
                train_stats["LOSSES"],

            "TRAIN_WIN_RATE_%":
                train_stats["WIN_RATE"],

            "TRAIN_AVG_RETURN_%":
                train_stats["AVG_RETURN"],

            "TEST_SIGNALS":
                test_stats["SIGNALS"],

            "TEST_DECISIVE":
                test_stats["DECISIVE"],

            "TEST_WINS":
                test_stats["WINS"],

            "TEST_LOSSES":
                test_stats["LOSSES"],

            "TEST_WIN_RATE_%":
                test_stats["WIN_RATE"],

            "TEST_AVG_RETURN_%":
                test_stats["AVG_RETURN"],

            "STABILITY_GAP_%":
                round(
                    stability_gap,
                    2,
                )
                if pd.notna(
                    stability_gap
                )
                else np.nan,

            "ENOUGH_SAMPLE":
                enough_sample,
        })

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary = (
        summary
        .sort_values(
            by=[
                "ENOUGH_SAMPLE",
                "TEST_WIN_RATE_%",
                "TEST_DECISIVE",
                "TEST_AVG_RETURN_%",
                "STABILITY_GAP_%",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return summary


# ============================================================
# PRINT V4 RESULTS
# ============================================================

def print_results(summary):

    print()
    print("=" * 110)
    print(
        "CRYPTO FUTURES HISTORICAL V4 RESULTS"
    )
    print("=" * 110)

    if summary.empty:

        print(
            "No V4 results."
        )

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

    print()
    print(
        "BEST RELIABLE RESULTS"
    )
    print()

    if reliable.empty:

        print(
            "No filter has enough "
            "decisive trades."
        )

    else:

        print(
            reliable[
                columns
            ]
            .head(30)
            .to_string(index=False)
        )

    # ========================================================
    # 70%+
    # ========================================================

    target70 = reliable[
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
    ].copy()

    print()
    print("=" * 110)

    print(
        "70%+ OUT-OF-SAMPLE "
        "CANDIDATES:",
        len(target70),
    )

    if not target70.empty:

        print()

        print(
            target70[
                columns
            ]
            .to_string(index=False)
        )

    # ========================================================
    # 75%+
    # ========================================================

    target75 = reliable[
        (
            reliable[
                "TEST_WIN_RATE_%"
            ] >= 75
        )
        &
        (
            reliable[
                "TEST_AVG_RETURN_%"
            ] > 0
        )
    ].copy()

    print()
    print("=" * 110)

    print(
        "75%+ OUT-OF-SAMPLE "
        "CANDIDATES:",
        len(target75),
    )

    if not target75.empty:

        print()

        print(
            target75[
                columns
            ]
            .to_string(index=False)
        )

    # ========================================================
    # 80%+
    # ========================================================

    target80 = reliable[
        (
            reliable[
                "TEST_WIN_RATE_%"
            ] >= 80
        )
        &
        (
            reliable[
                "TEST_AVG_RETURN_%"
            ] > 0
        )
    ].copy()

    print()
    print("=" * 110)

    print(
        "80%+ OUT-OF-SAMPLE "
        "CANDIDATES:",
        len(target80),
    )

    if not target80.empty:

        print()

        print(
            target80[
                columns
            ]
            .to_string(index=False)
        )

    print()
    print("=" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)

    print(
        "COINDCX FUTURES HISTORICAL V4"
    )

    print(
        "WEAK HAMMER + V3 BASE "
        "+ EXTRA CONFIRMATIONS"
    )

    print("=" * 90)

    print(
        "PAIR:",
        v1.PAIR,
    )

    print(
        "HISTORY:",
        v1.DAYS_TO_TEST,
        "days",
    )

    # ========================================================
    # DOWNLOAD 5M
    # ========================================================

    print()
    print(
        "Downloading native "
        "5-minute Futures candles..."
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

    print(
        "5M CANDLES:",
        len(df_5m),
    )

    # ========================================================
    # DOWNLOAD 1H
    # ========================================================

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
        "Attaching 1H trend..."
    )

    df_5m = (
        v1.attach_1h_trend(
            df_5m,
            df_1h,
        )
    )

    print(
        "Adding V4 indicators..."
    )

    df_5m = (
        add_v4_indicators(
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
                "bb_middle",
                "bb_upper",
                "bb_lower",
                "poc",
                "support_4h",
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

    events.to_csv(
        DETAIL_FILE,
        index=False,
    )

    print(
        "Detailed results saved:",
        DETAIL_FILE,
    )

    # ========================================================
    # TRAIN / TEST
    # ========================================================

    summary = (
        build_summary(
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

    print_results(
        summary
    )

    print()
    print("=" * 90)

    print(
        "FUTURES HISTORICAL V4 COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
