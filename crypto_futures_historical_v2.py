import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1


# ============================================================
# COINDCX FUTURES HISTORICAL OPTIMIZATION - VERSION 2
# ============================================================
#
# V1 remains unchanged.
#
# V2 goal:
# Test which extra confirmations improve each original setup.
#
# 1H = Direction
# 5M = Entry
#
# Extra confirmations:
#   1H trend
#   EMA20 / EMA50
#   RSI
#   MACD
#   Volume Ratio
#   POC
#   Support / Resistance
#
# This script DOES NOT place real trades.
# ============================================================


OUTPUT_FILE = "crypto_futures_historical_v2_results.csv"
SUMMARY_FILE = "crypto_futures_v2_summary.csv"

MIN_DECISIVE_TRADES = 30

WIN_THRESHOLD_PCT = 0.50


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def add_support_resistance(df):
    df = df.copy()

    # Previous 24 x 5M candles = 2 hours.
    # shift(1) prevents using current candle in the level.
    df["support_2h"] = (
        df["low"]
        .shift(1)
        .rolling(24)
        .min()
    )

    df["resistance_2h"] = (
        df["high"]
        .shift(1)
        .rolling(24)
        .max()
    )

    df["distance_support_pct"] = (
        (
            df["close"]
            - df["support_2h"]
        )
        / df["close"]
        * 100
    )

    df["distance_resistance_pct"] = (
        (
            df["resistance_2h"]
            - df["close"]
        )
        / df["close"]
        * 100
    )

    return df


# ============================================================
# EXTRA CONFIRMATIONS
# ============================================================

def get_confirmations(row, signal):
    confirmations = {}

    if signal == "LONG":

        confirmations["TREND"] = (
            row["trend_1h"] == "BULLISH"
        )

        confirmations["EMA"] = (
            row["close"] > row["ema20"]
            and row["ema20"] > row["ema50"]
        )

        confirmations["RSI"] = (
            pd.notna(row["rsi"])
            and 45 <= row["rsi"] <= 70
        )

        confirmations["MACD"] = (
            pd.notna(row["macd_hist"])
            and row["macd_hist"] > 0
        )

        confirmations["VOLUME"] = (
            pd.notna(row["volume_ratio"])
            and row["volume_ratio"] >= 1.0
        )

        confirmations["POC"] = (
            pd.notna(row["poc"])
            and row["close"] >= row["poc"]
        )

        confirmations["SR"] = (
            pd.notna(
                row["distance_support_pct"]
            )
            and row["distance_support_pct"] <= 1.0
        )

    else:

        confirmations["TREND"] = (
            row["trend_1h"] == "BEARISH"
        )

        confirmations["EMA"] = (
            row["close"] < row["ema20"]
            and row["ema20"] < row["ema50"]
        )

        confirmations["RSI"] = (
            pd.notna(row["rsi"])
            and 30 <= row["rsi"] <= 55
        )

        confirmations["MACD"] = (
            pd.notna(row["macd_hist"])
            and row["macd_hist"] < 0
        )

        confirmations["VOLUME"] = (
            pd.notna(row["volume_ratio"])
            and row["volume_ratio"] >= 1.0
        )

        confirmations["POC"] = (
            pd.notna(row["poc"])
            and row["close"] <= row["poc"]
        )

        confirmations["SR"] = (
            pd.notna(
                row["distance_resistance_pct"]
            )
            and row["distance_resistance_pct"] <= 1.0
        )

    return confirmations


# ============================================================
# ORIGINAL V1 SETUPS
# ============================================================

SETUPS = [
    (
        "VOLUME_PROFILE_PPD",
        v1.setup_volume_profile,
    ),
    (
        "BB_SQUEEZE",
        v1.setup_bb_squeeze,
    ),
    (
        "REJECTION_WICK",
        v1.setup_rejection,
    ),
    (
        "WEAK_HAMMER_SWEEP",
        v1.setup_weak_hammer_sweep,
    ),
]


# ============================================================
# V2 HISTORICAL SCAN
# ============================================================

def scan_v2(df):
    events = []

    max_forward = v1.FORWARD_BARS["4H"]

    for i in range(
        60,
        len(df) - max_forward - 1,
    ):

        row = df.iloc[i]

        for setup_name, setup_function in SETUPS:

            try:
                signal = setup_function(
                    df,
                    i,
                )
            except Exception as exc:
                print(
                    "SETUP ERROR:",
                    setup_name,
                    i,
                    exc,
                )
                continue

            if signal is None:
                continue

            confirmations = get_confirmations(
                row,
                signal,
            )

            score = int(
                sum(confirmations.values())
            )

            passed = [
                name
                for name, passed_value
                in confirmations.items()
                if passed_value
            ]

            forward = (
                v1.calculate_forward_results(
                    df,
                    i,
                    signal,
                )
            )

            event = {
                "PAIR": v1.PAIR,
                "TIME": row[
                    "datetime"
                ].isoformat(),
                "SETUP": setup_name,
                "SIGNAL": signal,
                "ENTRY_PRICE": round(
                    float(row["close"]),
                    8,
                ),
                "TREND_1H": row[
                    "trend_1h"
                ],
                "CONFIRMATION_SCORE": score,
                "CONFIRMATIONS": ",".join(
                    passed
                ),
                "TREND_OK": int(
                    confirmations["TREND"]
                ),
                "EMA_OK": int(
                    confirmations["EMA"]
                ),
                "RSI_OK": int(
                    confirmations["RSI"]
                ),
                "MACD_OK": int(
                    confirmations["MACD"]
                ),
                "VOLUME_OK": int(
                    confirmations["VOLUME"]
                ),
                "POC_OK": int(
                    confirmations["POC"]
                ),
                "SR_OK": int(
                    confirmations["SR"]
                ),
                "RSI": round(
                    float(row["rsi"]),
                    2,
                )
                if pd.notna(row["rsi"])
                else np.nan,
                "MACD_HIST": round(
                    float(row["macd_hist"]),
                    8,
                )
                if pd.notna(
                    row["macd_hist"]
                )
                else np.nan,
                "VOLUME_RATIO": round(
                    float(row["volume_ratio"]),
                    3,
                )
                if pd.notna(
                    row["volume_ratio"]
                )
                else np.nan,
                "VP_SHAPE": row[
                    "vp_shape"
                ],
                "POC": round(
                    float(row["poc"]),
                    8,
                )
                if pd.notna(row["poc"])
                else np.nan,
            }

            event.update(forward)

            events.append(event)

    return pd.DataFrame(events)


# ============================================================
# PERFORMANCE CALCULATION
# ============================================================

def performance(group):
    returns = (
        group["RETURN_4H_%"]
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

    decisive = wins + losses

    if decisive > 0:
        decisive_win_rate = (
            wins
            / decisive
            * 100
        )
    else:
        decisive_win_rate = np.nan

    all_signal_win_rate = (
        wins
        / len(returns)
        * 100
    )

    return {
        "TOTAL_SIGNALS": len(returns),
        "WINS": wins,
        "LOSSES": losses,
        "NEUTRAL": neutral,
        "DECISIVE_TRADES": decisive,
        "DECISIVE_WIN_RATE_%":
            round(
                decisive_win_rate,
                2,
            )
            if pd.notna(
                decisive_win_rate
            )
            else np.nan,
        "ALL_SIGNAL_WIN_RATE_%":
            round(
                all_signal_win_rate,
                2,
            ),
        "AVG_4H_RETURN_%":
            round(
                float(returns.mean()),
                4,
            ),
    }


# ============================================================
# TEST CONFIRMATION SCORE THRESHOLDS
# ============================================================

def build_summary(results):
    summary_rows = []

    for (
        setup,
        signal,
    ), base_group in results.groupby(
        [
            "SETUP",
            "SIGNAL",
        ]
    ):

        # Test minimum scores 0 through 7.
        for minimum_score in range(0, 8):

            group = base_group[
                base_group[
                    "CONFIRMATION_SCORE"
                ] >= minimum_score
            ]

            if group.empty:
                continue

            stats = performance(group)

            if stats is None:
                continue

            row = {
                "SETUP": setup,
                "SIGNAL": signal,
                "FILTER":
                    f"SCORE>={minimum_score}",
                "MIN_SCORE":
                    minimum_score,
            }

            row.update(stats)

            summary_rows.append(row)

        # ------------------------------------
        # Important specific combinations
        # ------------------------------------

        combinations = {
            "TREND+MACD":
                ["TREND_OK", "MACD_OK"],

            "TREND+EMA+MACD":
                [
                    "TREND_OK",
                    "EMA_OK",
                    "MACD_OK",
                ],

            "TREND+MACD+VOLUME":
                [
                    "TREND_OK",
                    "MACD_OK",
                    "VOLUME_OK",
                ],

            "TREND+EMA+MACD+VOLUME":
                [
                    "TREND_OK",
                    "EMA_OK",
                    "MACD_OK",
                    "VOLUME_OK",
                ],

            "TREND+MACD+POC":
                [
                    "TREND_OK",
                    "MACD_OK",
                    "POC_OK",
                ],

            "TREND+EMA+MACD+POC":
                [
                    "TREND_OK",
                    "EMA_OK",
                    "MACD_OK",
                    "POC_OK",
                ],

            "TREND+MACD+VOLUME+POC":
                [
                    "TREND_OK",
                    "MACD_OK",
                    "VOLUME_OK",
                    "POC_OK",
                ],

            "TREND+RSI+MACD+VOLUME+POC":
                [
                    "TREND_OK",
                    "RSI_OK",
                    "MACD_OK",
                    "VOLUME_OK",
                    "POC_OK",
                ],

            "ALL_7_CONFIRMATIONS":
                [
                    "TREND_OK",
                    "EMA_OK",
                    "RSI_OK",
                    "MACD_OK",
                    "VOLUME_OK",
                    "POC_OK",
                    "SR_OK",
                ],
        }

        for combo_name, columns in combinations.items():

            mask = pd.Series(
                True,
                index=base_group.index,
            )

            for column in columns:
                mask = (
                    mask
                    & (
                        base_group[column]
                        == 1
                    )
                )

            group = base_group[
                mask
            ]

            if group.empty:
                continue

            stats = performance(group)

            if stats is None:
                continue

            row = {
                "SETUP": setup,
                "SIGNAL": signal,
                "FILTER": combo_name,
                "MIN_SCORE": np.nan,
            }

            row.update(stats)

            summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    if summary.empty:
        return summary

    summary["ENOUGH_SAMPLE"] = (
        summary["DECISIVE_TRADES"]
        >= MIN_DECISIVE_TRADES
    )

    summary = summary.sort_values(
        by=[
            "ENOUGH_SAMPLE",
            "DECISIVE_WIN_RATE_%",
            "DECISIVE_TRADES",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return summary


# ============================================================
# PRINT BEST RESULTS
# ============================================================

def print_best(summary):
    print()
    print("=" * 80)
    print("FUTURES V2 OPTIMIZATION SUMMARY")
    print("=" * 80)

    if summary.empty:
        print("No results.")
        return

    reliable = summary[
        summary["ENOUGH_SAMPLE"]
        == True
    ].copy()

    if reliable.empty:
        print(
            "No filter currently has enough "
            "decisive trades."
        )
        return

    print()
    print(
        "BEST RESULTS WITH AT LEAST",
        MIN_DECISIVE_TRADES,
        "DECISIVE TRADES"
    )
    print()

    columns = [
        "SETUP",
        "SIGNAL",
        "FILTER",
        "TOTAL_SIGNALS",
        "WINS",
        "LOSSES",
        "NEUTRAL",
        "DECISIVE_TRADES",
        "DECISIVE_WIN_RATE_%",
        "AVG_4H_RETURN_%",
    ]

    print(
        reliable[
            columns
        ]
        .head(25)
        .to_string(index=False)
    )

    sixty_plus = reliable[
        reliable[
            "DECISIVE_WIN_RATE_%"
        ] >= 60
    ]

    print()
    print("=" * 80)

    if sixty_plus.empty:
        print(
            "NO RELIABLE 60%+ COMBINATION FOUND YET."
        )
        print(
            "Do not force the strategy. "
            "Further optimization is required."
        )
    else:
        print(
            "60%+ CANDIDATES FOUND:",
            len(sixty_plus),
        )

        print()

        print(
            sixty_plus[
                columns
            ]
            .head(20)
            .to_string(index=False)
        )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "COINDCX FUTURES HISTORICAL OPTIMIZATION V2"
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
        "Downloading 5-minute Futures history..."
    )

    df_5m = v1.fetch_history(
        v1.PAIR,
        "5",
        v1.DAYS_TO_TEST,
        v1.CHUNK_DAYS_5M,
    )

    if df_5m.empty:
        raise RuntimeError(
            "No 5-minute Futures data received."
        )

    print()
    print(
        "5M candles:",
        len(df_5m),
    )

    print()
    print(
        "Downloading 1-hour Futures history..."
    )

    df_1h = v1.fetch_history(
        v1.PAIR,
        "60",
        v1.DAYS_TO_TEST,
        v1.CHUNK_DAYS_1H,
    )

    if df_1h.empty:
        raise RuntimeError(
            "No 1-hour Futures data received."
        )

    print()
    print(
        "1H candles:",
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
        "Attaching 1H direction..."
    )

    df_5m = v1.attach_1h_trend(
        df_5m,
        df_1h,
    )

    print(
        "Calculating support/resistance..."
    )

    df_5m = add_support_resistance(
        df_5m
    )

    df_5m = df_5m.dropna(
        subset=[
            "ema20",
            "ema50",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "rsi",
            "macd_hist",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        "Testing original setups "
        "with V2 confirmations..."
    )

    results = scan_v2(
        df_5m
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Detailed CSV saved:",
        OUTPUT_FILE,
    )

    if results.empty:
        print(
            "No historical signals found."
        )
        return

    summary = build_summary(
        results
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "Summary CSV saved:",
        SUMMARY_FILE,
    )

    print_best(
        summary
    )

    print()
    print("=" * 80)
    print(
        "FUTURES HISTORICAL V2 COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
