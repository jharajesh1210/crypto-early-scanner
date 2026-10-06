import math
import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v10 as v10


# ============================================================
# COINDCX FUTURES HISTORICAL V11
# ============================================================
#
# PURPOSE
# -------
# V11 is a ROBUSTNESS / VALIDATION test.
#
# We are NOT adding more indicators.
# We are locking the strongest V10 families and testing them
# with stricter walk-forward validation.
#
# Main goals:
#
#   1. Try longer historical data
#   2. Use 8 chronological walk-forward blocks
#   3. Test only locked V10 winners
#   4. Add estimated round-trip trading cost
#   5. Measure raw + cost-adjusted performance
#   6. Add Wilson confidence interval
#   7. Require minimum sample size
#   8. Classify candidates as PASS / WATCH / REJECT
#
# HISTORICAL TEST ONLY
# NO REAL ORDERS
#
# IMPORTANT:
# Trading cost below is only a configurable test assumption.
# It is NOT a claim about exact CoinDCX fees.
# ============================================================


DETAIL_FILE = "crypto_futures_v11_results.csv"
SUMMARY_FILE = "crypto_futures_v11_summary.csv"


# ============================================================
# V11 SETTINGS
# ============================================================

# Try 365 days first.
# If CoinDCX/API does not provide enough data, V11 will
# automatically try 270 days and then 180 days.
HISTORY_ATTEMPTS = [
    365,
    270,
    180,
]

WALK_FORWARD_BLOCKS = 8

# Same decisive threshold concept used previously:
# raw 4H return >= +0.50% = WIN
# raw 4H return <= -0.50% = LOSS
WIN_THRESHOLD_PCT = 0.50

# Configurable assumed total round-trip cost.
# Example:
# 0.10 means 0.10% deducted from each trade return.
ROUND_TRIP_COST_PCT = 0.10

# Minimum sample targets.
MIN_TOTAL_DECISIVE_PASS = 40
MIN_TOTAL_DECISIVE_WATCH = 25

MIN_DECISIVE_PER_BLOCK = 3

# PASS robustness requirements.
PASS_MIN_OVERALL_WIN_RATE = 75.0
PASS_MIN_POSITIVE_BLOCKS = 6
PASS_MIN_BLOCKS_70PLUS = 5
PASS_MIN_VALID_BLOCKS = 6
PASS_MIN_BLOCK_WIN_RATE = 60.0
PASS_MAX_BLOCK_STD = 20.0

# WATCH requirements.
WATCH_MIN_OVERALL_WIN_RATE = 70.0
WATCH_MIN_POSITIVE_BLOCKS = 5
WATCH_MIN_VALID_BLOCKS = 5

# Wilson confidence level:
# z = 1.96 => approximately 95%
WILSON_Z = 1.96


# ============================================================
# LOCKED V10 CANDIDATES
# ============================================================
#
# IMPORTANT:
# These are selected BEFORE V11 testing.
# We do not search dozens of new combinations in V11.
#
# This reduces overfitting / selection bias.
# ============================================================

LOCKED_CANDIDATES = {

    # --------------------------------------------------------
    # PRIMARY V10 WINNER
    # --------------------------------------------------------

    "V11_EMA20_RISING3":
        v10.BASE
        + [
            "EMA20_RISING3",
        ],

    # --------------------------------------------------------
    # V10 / V9 CORE WINNER
    # --------------------------------------------------------

    "V11_EMA20_RISING2":
        v10.BASE
        + [
            "EMA20_RISING2",
        ],

    # --------------------------------------------------------
    # HIGHER WIN-RATE BUT SMALLER SAMPLE IN V10
    # --------------------------------------------------------

    "V11_EMA20_RISING2_MACD":
        v10.BASE
        + [
            "EMA20_RISING2",
            "MACD_RISING",
        ],

    # --------------------------------------------------------
    # BB55 + EMA20 RISING2
    # --------------------------------------------------------

    "V11_EMA20_RISING2_BB55":
        v10.BASE
        + [
            "EMA20_RISING2",
            "BB55",
        ],

    # --------------------------------------------------------
    # BB55 + H1 EMA
    # --------------------------------------------------------

    "V11_BB55_H1_EMA":
        v10.BASE
        + [
            "BB55",
            "H1_EMA_RISING",
        ],

    # --------------------------------------------------------
    # EMA20 + H1 EMA
    # --------------------------------------------------------

    "V11_EMA20_H1_EMA":
        v10.BASE
        + [
            "EMA20_RISING",
            "H1_EMA_RISING",
        ],

    # --------------------------------------------------------
    # BB55 + EMA20 + H1 EMA
    # --------------------------------------------------------

    "V11_BB55_EMA20_H1":
        v10.BASE
        + [
            "BB55",
            "EMA20_RISING",
            "H1_EMA_RISING",
        ],
}


# ============================================================
# SAFE DATAFRAME
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
# SAFE NUMBER
# ============================================================

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

def wilson_interval(
    wins,
    total,
    z=WILSON_Z,
):

    if total <= 0:
        return (
            np.nan,
            np.nan,
        )

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
# APPLY LOCKED FILTER
# ============================================================

def apply_filter(
    df,
    conditions,
):

    data = ensure_dataframe(
        df,
        "apply_filter input",
    )

    if data.empty:
        return data

    mask = pd.Series(
        True,
        index=data.index,
        dtype=bool,
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
# ADD V11 RETURN COLUMNS
# ============================================================

def add_cost_adjusted_returns(events):

    data = ensure_dataframe(
        events,
        "add_cost_adjusted_returns input",
    )

    if data.empty:
        return data

    if "RETURN_4H_%" not in data.columns:

        raise RuntimeError(
            "RETURN_4H_% column missing."
        )

    data["RAW_RETURN_4H_%"] = (
        pd.to_numeric(
            data["RETURN_4H_%"],
            errors="coerce",
        )
    )

    data["NET_RETURN_4H_%"] = (
        data["RAW_RETURN_4H_%"]
        -
        ROUND_TRIP_COST_PCT
    )

    return data


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

    if "RAW_RETURN_4H_%" not in data.columns:
        return None

    if "NET_RETURN_4H_%" not in data.columns:
        return None

    raw_returns = (
        data["RAW_RETURN_4H_%"]
        .dropna()
    )

    if raw_returns.empty:
        return None

    net_returns = (
        data.loc[
            raw_returns.index,
            "NET_RETURN_4H_%",
        ]
    )

    # --------------------------------------------------------
    # RAW DECISIVE RESULTS
    # --------------------------------------------------------

    raw_wins = int(
        (
            raw_returns
            >=
            WIN_THRESHOLD_PCT
        ).sum()
    )

    raw_losses = int(
        (
            raw_returns
            <=
            -WIN_THRESHOLD_PCT
        ).sum()
    )

    raw_neutral = int(
        len(raw_returns)
        -
        raw_wins
        -
        raw_losses
    )

    raw_decisive = (
        raw_wins
        +
        raw_losses
    )

    if raw_decisive > 0:

        raw_win_rate = (
            raw_wins
            /
            raw_decisive
            * 100.0
        )

        wilson_low, wilson_high = (
            wilson_interval(
                raw_wins,
                raw_decisive,
            )
        )

    else:

        raw_win_rate = np.nan
        wilson_low = np.nan
        wilson_high = np.nan

    # --------------------------------------------------------
    # COST-ADJUSTED DECISIVE RESULTS
    # --------------------------------------------------------
    #
    # We apply the same +/-0.50% decisive threshold AFTER
    # subtracting the assumed round-trip cost.
    # --------------------------------------------------------

    net_wins = int(
        (
            net_returns
            >=
            WIN_THRESHOLD_PCT
        ).sum()
    )

    net_losses = int(
        (
            net_returns
            <=
            -WIN_THRESHOLD_PCT
        ).sum()
    )

    net_neutral = int(
        len(net_returns)
        -
        net_wins
        -
        net_losses
    )

    net_decisive = (
        net_wins
        +
        net_losses
    )

    if net_decisive > 0:

        net_win_rate = (
            net_wins
            /
            net_decisive
            * 100.0
        )

    else:

        net_win_rate = np.nan

    return {

        "SIGNALS":
            int(
                len(raw_returns)
            ),

        "RAW_WINS":
            raw_wins,

        "RAW_LOSSES":
            raw_losses,

        "RAW_NEUTRAL":
            raw_neutral,

        "RAW_DECISIVE":
            raw_decisive,

        "RAW_WIN_RATE_%":
            round(
                float(raw_win_rate),
                2,
            )
            if pd.notna(raw_win_rate)
            else np.nan,

        "RAW_AVG_RETURN_%":
            round(
                float(
                    raw_returns.mean()
                ),
                4,
            ),

        "RAW_MEDIAN_RETURN_%":
            round(
                float(
                    raw_returns.median()
                ),
                4,
            ),

        "NET_WINS":
            net_wins,

        "NET_LOSSES":
            net_losses,

        "NET_NEUTRAL":
            net_neutral,

        "NET_DECISIVE":
            net_decisive,

        "NET_WIN_RATE_%":
            round(
                float(net_win_rate),
                2,
            )
            if pd.notna(net_win_rate)
            else np.nan,

        "NET_AVG_RETURN_%":
            round(
                float(
                    net_returns.mean()
                ),
                4,
            ),

        "NET_MEDIAN_RETURN_%":
            round(
                float(
                    net_returns.median()
                ),
                4,
            ),

        "WILSON_LOW_95_%":
            round(
                float(wilson_low),
                2,
            )
            if pd.notna(wilson_low)
            else np.nan,

        "WILSON_HIGH_95_%":
            round(
                float(wilson_high),
                2,
            )
            if pd.notna(wilson_high)
            else np.nan,
    }


# ============================================================
# CREATE CHRONOLOGICAL WALK-FORWARD BLOCKS
# ============================================================

def create_blocks(events):

    data = ensure_dataframe(
        events,
        "create_blocks input",
    )

    if data.empty:
        return []

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
# WALK-FORWARD TEST
# ============================================================

def walk_forward_test(events):

    results = []

    blocks = (
        create_blocks(
            events
        )
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

        for candidate, conditions in LOCKED_CANDIDATES.items():

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

                    "BLOCK_START":
                        block_start,

                    "BLOCK_END":
                        block_end,

                    "CANDIDATE":
                        candidate,

                    "SIGNALS":
                        0,

                    "RAW_WINS":
                        0,

                    "RAW_LOSSES":
                        0,

                    "RAW_NEUTRAL":
                        0,

                    "RAW_DECISIVE":
                        0,

                    "RAW_WIN_RATE_%":
                        np.nan,

                    "RAW_AVG_RETURN_%":
                        np.nan,

                    "RAW_MEDIAN_RETURN_%":
                        np.nan,

                    "NET_WINS":
                        0,

                    "NET_LOSSES":
                        0,

                    "NET_NEUTRAL":
                        0,

                    "NET_DECISIVE":
                        0,

                    "NET_WIN_RATE_%":
                        np.nan,

                    "NET_AVG_RETURN_%":
                        np.nan,

                    "NET_MEDIAN_RETURN_%":
                        np.nan,

                    "WILSON_LOW_95_%":
                        np.nan,

                    "WILSON_HIGH_95_%":
                        np.nan,
                })

            else:

                results.append({

                    "BLOCK":
                        block_number,

                    "BLOCK_START":
                        block_start,

                    "BLOCK_END":
                        block_end,

                    "CANDIDATE":
                        candidate,

                    **stats,
                })

    return pd.DataFrame(
        results
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_candidate(row):

    total_decisive = int(
        row["TOTAL_RAW_DECISIVE"]
    )

    overall_win_rate = safe_float(
        row["OVERALL_RAW_WIN_RATE_%"]
    )

    net_avg_return = safe_float(
        row["OVERALL_NET_AVG_RETURN_%"]
    )

    valid_blocks = int(
        row["VALID_BLOCKS"]
    )

    positive_blocks = int(
        row["POSITIVE_NET_BLOCKS"]
    )

    blocks_70 = int(
        row["BLOCKS_70PLUS"]
    )

    min_block_rate = safe_float(
        row["MIN_BLOCK_WIN_RATE_%"]
    )

    block_std = safe_float(
        row["BLOCK_RATE_STD_%"]
    )

    blocks_with_min_sample = int(
        row["BLOCKS_WITH_MIN_SAMPLE"]
    )

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    pass_test = (
        total_decisive
        >=
        MIN_TOTAL_DECISIVE_PASS

        and

        pd.notna(
            overall_win_rate
        )

        and

        overall_win_rate
        >=
        PASS_MIN_OVERALL_WIN_RATE

        and

        net_avg_return
        >
        0

        and

        valid_blocks
        >=
        PASS_MIN_VALID_BLOCKS

        and

        positive_blocks
        >=
        PASS_MIN_POSITIVE_BLOCKS

        and

        blocks_70
        >=
        PASS_MIN_BLOCKS_70PLUS

        and

        pd.notna(
            min_block_rate
        )

        and

        min_block_rate
        >=
        PASS_MIN_BLOCK_WIN_RATE

        and

        pd.notna(
            block_std
        )

        and

        block_std
        <=
        PASS_MAX_BLOCK_STD

        and

        blocks_with_min_sample
        >=
        PASS_MIN_VALID_BLOCKS
    )

    if pass_test:
        return "PASS"

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    watch_test = (
        total_decisive
        >=
        MIN_TOTAL_DECISIVE_WATCH

        and

        pd.notna(
            overall_win_rate
        )

        and

        overall_win_rate
        >=
        WATCH_MIN_OVERALL_WIN_RATE

        and

        net_avg_return
        >
        0

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

    for candidate, conditions in LOCKED_CANDIDATES.items():

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

        raw_rates = (
            blocks["RAW_WIN_RATE_%"]
            .dropna()
        )

        net_rates = (
            blocks["NET_WIN_RATE_%"]
            .dropna()
        )

        raw_avg_returns = (
            blocks["RAW_AVG_RETURN_%"]
            .dropna()
        )

        net_avg_returns = (
            blocks["NET_AVG_RETURN_%"]
            .dropna()
        )

        valid_blocks = int(
            (
                blocks["RAW_DECISIVE"]
                >
                0
            ).sum()
        )

        positive_raw_blocks = int(
            (
                raw_avg_returns
                >
                0
            ).sum()
        )

        positive_net_blocks = int(
            (
                net_avg_returns
                >
                0
            ).sum()
        )

        blocks_70 = int(
            (
                raw_rates
                >=
                70
            ).sum()
        )

        blocks_75 = int(
            (
                raw_rates
                >=
                75
            ).sum()
        )

        blocks_80 = int(
            (
                raw_rates
                >=
                80
            ).sum()
        )

        blocks_with_min_sample = int(
            (
                blocks["RAW_DECISIVE"]
                >=
                MIN_DECISIVE_PER_BLOCK
            ).sum()
        )

        if raw_rates.empty:

            min_block_rate = np.nan
            max_block_rate = np.nan
            block_rate_std = np.nan
            median_block_rate = np.nan

        else:

            min_block_rate = float(
                raw_rates.min()
            )

            max_block_rate = float(
                raw_rates.max()
            )

            block_rate_std = float(
                raw_rates.std(
                    ddof=0
                )
            )

            median_block_rate = float(
                raw_rates.median()
            )

        if net_rates.empty:

            min_net_block_rate = np.nan
            median_net_block_rate = np.nan

        else:

            min_net_block_rate = float(
                net_rates.min()
            )

            median_net_block_rate = float(
                net_rates.median()
            )

        row = {

            "CANDIDATE":
                candidate,

            "TOTAL_SIGNALS":
                overall["SIGNALS"],

            "TOTAL_RAW_DECISIVE":
                overall["RAW_DECISIVE"],

            "RAW_WINS":
                overall["RAW_WINS"],

            "RAW_LOSSES":
                overall["RAW_LOSSES"],

            "RAW_NEUTRAL":
                overall["RAW_NEUTRAL"],

            "OVERALL_RAW_WIN_RATE_%":
                overall["RAW_WIN_RATE_%"],

            "OVERALL_RAW_AVG_RETURN_%":
                overall["RAW_AVG_RETURN_%"],

            "OVERALL_RAW_MEDIAN_RETURN_%":
                overall["RAW_MEDIAN_RETURN_%"],

            "TOTAL_NET_DECISIVE":
                overall["NET_DECISIVE"],

            "NET_WINS":
                overall["NET_WINS"],

            "NET_LOSSES":
                overall["NET_LOSSES"],

            "NET_NEUTRAL":
                overall["NET_NEUTRAL"],

            "OVERALL_NET_WIN_RATE_%":
                overall["NET_WIN_RATE_%"],

            "OVERALL_NET_AVG_RETURN_%":
                overall["NET_AVG_RETURN_%"],

            "OVERALL_NET_MEDIAN_RETURN_%":
                overall["NET_MEDIAN_RETURN_%"],

            "WILSON_LOW_95_%":
                overall["WILSON_LOW_95_%"],

            "WILSON_HIGH_95_%":
                overall["WILSON_HIGH_95_%"],

            "VALID_BLOCKS":
                valid_blocks,

            "POSITIVE_RAW_BLOCKS":
                positive_raw_blocks,

            "POSITIVE_NET_BLOCKS":
                positive_net_blocks,

            "BLOCKS_70PLUS":
                blocks_70,

            "BLOCKS_75PLUS":
                blocks_75,

            "BLOCKS_80PLUS":
                blocks_80,

            "BLOCKS_WITH_MIN_SAMPLE":
                blocks_with_min_sample,

            "MIN_BLOCK_WIN_RATE_%":
                round(
                    min_block_rate,
                    2,
                )
                if pd.notna(
                    min_block_rate
                )
                else np.nan,

            "MEDIAN_BLOCK_WIN_RATE_%":
                round(
                    median_block_rate,
                    2,
                )
                if pd.notna(
                    median_block_rate
                )
                else np.nan,

            "MAX_BLOCK_WIN_RATE_%":
                round(
                    max_block_rate,
                    2,
                )
                if pd.notna(
                    max_block_rate
                )
                else np.nan,

            "BLOCK_RATE_STD_%":
                round(
                    block_rate_std,
                    2,
                )
                if pd.notna(
                    block_rate_std
                )
                else np.nan,

            "MIN_NET_BLOCK_WIN_RATE_%":
                round(
                    min_net_block_rate,
                    2,
                )
                if pd.notna(
                    min_net_block_rate
                )
                else np.nan,

            "MEDIAN_NET_BLOCK_WIN_RATE_%":
                round(
                    median_net_block_rate,
                    2,
                )
                if pd.notna(
                    median_net_block_rate
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

    status_rank = {
        "PASS": 3,
        "WATCH": 2,
        "REJECT": 1,
    }

    summary["_STATUS_RANK"] = (
        summary["STATUS"]
        .map(
            status_rank
        )
        .fillna(0)
    )

    summary = (
        summary
        .sort_values(
            by=[
                "_STATUS_RANK",
                "OVERALL_RAW_WIN_RATE_%",
                "TOTAL_RAW_DECISIVE",
                "OVERALL_NET_AVG_RETURN_%",
                "WILSON_LOW_95_%",
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
                "_STATUS_RANK",
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
    print("=" * 130)
    print("V11 WALK-FORWARD BLOCK RESULTS")
    print("=" * 130)

    if block_results.empty:

        print(
            "No block results."
        )

        return

    display = [

        "BLOCK",

        "CANDIDATE",

        "SIGNALS",

        "RAW_DECISIVE",

        "RAW_WIN_RATE_%",

        "RAW_AVG_RETURN_%",

        "NET_DECISIVE",

        "NET_WIN_RATE_%",

        "NET_AVG_RETURN_%",

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
    print("=" * 150)
    print("V11 OVERALL ROBUSTNESS RESULTS")
    print("=" * 150)

    if summary.empty:

        print(
            "No V11 summary results."
        )

        return

    display = [

        "CANDIDATE",

        "STATUS",

        "TOTAL_RAW_DECISIVE",

        "OVERALL_RAW_WIN_RATE_%",

        "OVERALL_RAW_AVG_RETURN_%",

        "OVERALL_NET_AVG_RETURN_%",

        "WILSON_LOW_95_%",

        "WILSON_HIGH_95_%",

        "VALID_BLOCKS",

        "POSITIVE_NET_BLOCKS",

        "BLOCKS_70PLUS",

        "BLOCKS_75PLUS",

        "BLOCKS_80PLUS",

        "BLOCKS_WITH_MIN_SAMPLE",

        "MIN_BLOCK_WIN_RATE_%",

        "MEDIAN_BLOCK_WIN_RATE_%",

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
    # PASS
    # --------------------------------------------------------

    passed = (
        summary[
            summary["STATUS"]
            ==
            "PASS"
        ]
        .copy()
    )

    print()
    print("=" * 150)

    print(
        "V11 FINAL PASS CANDIDATES:",
        len(passed),
    )

    print("=" * 150)

    if passed.empty:

        print(
            "No candidate passed the strict V11 robustness rules."
        )

    else:

        print(
            passed[
                display
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    watch = (
        summary[
            summary["STATUS"]
            ==
            "WATCH"
        ]
        .copy()
    )

    print()
    print("=" * 150)

    print(
        "V11 WATCH CANDIDATES:",
        len(watch),
    )

    print("=" * 150)

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

    # --------------------------------------------------------
    # REJECT
    # --------------------------------------------------------

    rejected = (
        summary[
            summary["STATUS"]
            ==
            "REJECT"
        ]
        .copy()
    )

    print()
    print("=" * 150)

    print(
        "V11 REJECT CANDIDATES:",
        len(rejected),
    )

    print("=" * 150)

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
    # BEST CANDIDATE
    # --------------------------------------------------------

    print()
    print("=" * 150)
    print("V11 BEST CANDIDATE")
    print("=" * 150)

    if not passed.empty:

        best = (
            passed.iloc[0]
        )

        decision = "PASS"

    elif not watch.empty:

        best = (
            watch.iloc[0]
        )

        decision = "WATCH"

    else:

        best = (
            summary.iloc[0]
        )

        decision = "REJECT"

    print(
        "CANDIDATE:",
        best["CANDIDATE"],
    )

    print(
        "STATUS:",
        decision,
    )

    print(
        "RAW DECISIVE TRADES:",
        int(
            best["TOTAL_RAW_DECISIVE"]
        ),
    )

    print(
        "RAW WIN RATE:",
        f'{best["OVERALL_RAW_WIN_RATE_%"]:.2f}%',
    )

    print(
        "RAW AVG RETURN:",
        f'{best["OVERALL_RAW_AVG_RETURN_%"]:.4f}%',
    )

    print(
        "NET AVG RETURN:",
        f'{best["OVERALL_NET_AVG_RETURN_%"]:.4f}%',
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
# DOWNLOAD HISTORY WITH FALLBACK
# ============================================================

def download_history():

    last_error = None

    for days in HISTORY_ATTEMPTS:

        try:

            print()
            print("=" * 100)

            print(
                "TRYING HISTORY:",
                days,
                "DAYS",
            )

            print("=" * 100)

            # ------------------------------------------------
            # 5M
            # ------------------------------------------------

            print()
            print(
                "Downloading 5-minute data..."
            )

            df_5m = (
                v1.fetch_history(
                    v1.PAIR,
                    "5",
                    days,
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

            # ------------------------------------------------
            # 1H
            # ------------------------------------------------

            print(
                "Downloading 1-hour data..."
            )

            df_1h = (
                v1.fetch_history(
                    v1.PAIR,
                    "60",
                    days,
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
        "All V11 history download attempts failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df_5m,
    df_1h,
):

    # --------------------------------------------------------
    # BASE 5M INDICATORS
    # --------------------------------------------------------

    print()
    print(
        "Calculating base 5M indicators..."
    )

    df_5m = (
        v1.calculate_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "base indicators result",
    )

    # --------------------------------------------------------
    # VOLUME PROFILE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ORIGINAL 1H TREND
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # V6 INDICATORS
    # --------------------------------------------------------

    print(
        "Adding V6 indicators..."
    )

    df_5m = (
        v10.v6.add_v6_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V6 result",
    )

    # --------------------------------------------------------
    # V7 1H INDICATORS
    # --------------------------------------------------------

    print(
        "Calculating V7 1H indicators..."
    )

    df_1h_v7 = (
        v10.v7.prepare_1h_indicators(
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
        v10.v7.attach_v7_1h_data(
            df_5m,
            df_1h_v7,
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V7 attach result",
    )

    # --------------------------------------------------------
    # V8
    # --------------------------------------------------------

    print(
        "Adding V8 indicators..."
    )

    df_5m = (
        v10.v8.add_v8_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V8 result",
    )

    # --------------------------------------------------------
    # V9
    # --------------------------------------------------------

    print(
        "Adding V9 indicators..."
    )

    df_5m = (
        v10.v9.add_v9_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V9 result",
    )

    # --------------------------------------------------------
    # V10
    # --------------------------------------------------------

    print(
        "Adding V10 indicators..."
    )

    df_5m = (
        v10.add_v10_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V10 result",
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

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

        "v10_bb_middle",

        "v10_bb_position",

        "v10_volume_ratio",

        "v10_poc_distance_pct",
    ]

    missing = [
        col
        for col in required_clean
        if col not in df_5m.columns
    ]

    if missing:

        raise RuntimeError(
            "V11 missing required columns: "
            +
            ", ".join(
                missing
            )
        )

    df_5m = (
        df_5m
        .dropna(
            subset=required_clean
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "USABLE 5M CANDLES:",
        len(df_5m),
    )

    return df_5m


# ============================================================
# COLLECT V11 EVENTS
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
            v10.get_v10_confirmations(
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

            "EMA20":
                round(
                    float(
                        row["ema20"]
                    ),
                    8,
                ),

            "EMA50":
                round(
                    float(
                        row["ema50"]
                    ),
                    8,
                ),

            "BB_POSITION":
                round(
                    float(
                        row["v10_bb_position"]
                    ),
                    4,
                )
                if pd.notna(
                    row["v10_bb_position"]
                )
                else np.nan,

            "VOLUME_RATIO":
                round(
                    float(
                        row["v10_volume_ratio"]
                    ),
                    4,
                )
                if pd.notna(
                    row["v10_volume_ratio"]
                )
                else np.nan,

            "POC_DISTANCE_%":
                round(
                    float(
                        row["v10_poc_distance_pct"]
                    ),
                    4,
                )
                if pd.notna(
                    row["v10_poc_distance_pct"]
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

    events_df = pd.DataFrame(
        events
    )

    if events_df.empty:
        return events_df

    events_df = (
        add_cost_adjusted_returns(
            events_df
        )
    )

    return events_df


# ============================================================
# HISTORY RANGE INFORMATION
# ============================================================

def print_history_range(
    df_5m,
    df_1h,
    requested_days,
):

    print()
    print("=" * 100)
    print("V11 DATA RANGE")
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

        start_5m = (
            pd.to_datetime(
                df_5m["datetime"],
                utc=True,
                errors="coerce",
            )
            .min()
        )

        end_5m = (
            pd.to_datetime(
                df_5m["datetime"],
                utc=True,
                errors="coerce",
            )
            .max()
        )

        print(
            "5M START:",
            start_5m,
        )

        print(
            "5M END:",
            end_5m,
        )

        if (
            pd.notna(start_5m)
            and
            pd.notna(end_5m)
        ):

            actual_days = (
                end_5m
                -
                start_5m
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

        start_1h = (
            pd.to_datetime(
                df_1h["datetime"],
                utc=True,
                errors="coerce",
            )
            .min()
        )

        end_1h = (
            pd.to_datetime(
                df_1h["datetime"],
                utc=True,
                errors="coerce",
            )
            .max()
        )

        print(
            "1H START:",
            start_1h,
        )

        print(
            "1H END:",
            end_1h,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)

    print(
        "COINDCX FUTURES HISTORICAL V11"
    )

    print(
        "FINAL ROBUSTNESS VALIDATION"
    )

    print("=" * 110)

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
        "ROUND-TRIP COST ASSUMPTION:",
        f"{ROUND_TRIP_COST_PCT:.2f}%",
    )

    print(
        "LOCKED CANDIDATES:",
        len(
            LOCKED_CANDIDATES
        ),
    )

    print()
    print(
        "V11 WILL NOT SEARCH NEW INDICATOR COMBINATIONS."
    )

    print(
        "V11 ONLY VALIDATES LOCKED V10 WINNERS."
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

    df_5m = (
        prepare_data(
            df_5m,
            df_1h,
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
            "No Weak Hammer Sweep events found."
        )

        return

    print(
        "TOTAL WEAK HAMMER EVENTS:",
        len(events),
    )

    # ========================================================
    # WALK FORWARD
    # ========================================================

    print()
    print(
        "Running V11 8-block walk-forward validation..."
    )

    block_results = (
        walk_forward_test(
            events
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

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
        "V11 detail saved:",
        DETAIL_FILE,
    )

    print(
        "V11 summary saved:",
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
    print("=" * 110)

    print(
        "V11 INTERPRETATION"
    )

    print("=" * 110)

    print(
        "PASS   = Stronger robustness evidence."
    )

    print(
        "WATCH  = Promising, but more validation is required."
    )

    print(
        "REJECT = Does not meet V11 robustness requirements."
    )

    print()
    print(
        "Backtest results do not guarantee future live performance."
    )

    print(
        "No real orders were placed."
    )

    print()
    print("=" * 110)

    print(
        "FUTURES HISTORICAL V11 COMPLETE"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
