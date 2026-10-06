import numpy as np
import pandas as pd

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v6 as v6
import crypto_futures_historical_v7 as v7
import crypto_futures_historical_v8 as v8
import crypto_futures_historical_v9 as v9


# ============================================================
# COINDCX FUTURES HISTORICAL V10
# ============================================================
#
# PURPOSE
# -------
# V10 builds on V9 walk-forward results.
#
# Main focus:
#
#   1. EMA20_RISING2
#   2. BB55 + H1 EMA
#   3. EMA20 + H1 EMA
#   4. BB55 + EMA20 + H1 EMA
#   5. Add selective MACD / RSI / POC / candle filters
#
# Goal:
#
#   Improve stability
#   Keep reasonable sample size
#   Avoid tiny-sample 100% results
#   Prefer candidates working across all 4 blocks
#
# HISTORICAL TEST ONLY
# NO REAL ORDERS
# ============================================================


DETAIL_FILE = "crypto_futures_v10_results.csv"
SUMMARY_FILE = "crypto_futures_v10_summary.csv"

WIN_THRESHOLD_PCT = 0.50

WALK_FORWARD_BLOCKS = 4

MIN_TOTAL_DECISIVE = 20

MIN_VALID_BLOCKS = 4


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
# V10 INDICATORS
# ============================================================

def add_v10_indicators(df):

    data = ensure_dataframe(
        df,
        "add_v10_indicators input",
    )

    data = (
        data
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    required = [
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
        "ema20_1h",
        "ema50_1h",
        "macd_hist_1h",
    ]

    missing = [
        col
        for col in required
        if col not in data.columns
    ]

    if missing:

        raise RuntimeError(
            "V10 missing required columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # EMA 5M
    # --------------------------------------------------------

    data["v10_ema20_rising1"] = (
        data["ema20"]
        >
        data["ema20"].shift(1)
    )

    data["v10_ema20_rising2"] = (
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

    data["v10_ema20_rising3"] = (
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
        &
        (
            data["ema20"].shift(2)
            >=
            data["ema20"].shift(3)
        )
    )

    data["v10_above_ema20"] = (
        data["close"]
        >
        data["ema20"]
    )

    data["v10_ema_bull"] = (
        data["ema20"]
        >
        data["ema50"]
    )

    data["v10_ema_gap_pct"] = (
        (
            data["ema20"]
            -
            data["ema50"]
        )
        /
        data["ema50"].replace(0, np.nan)
        * 100
    )

    data["v10_ema_gap_positive"] = (
        data["v10_ema_gap_pct"]
        >
        0
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    data["v10_macd_positive"] = (
        data["macd_hist"]
        >
        0
    )

    data["v10_macd_rising1"] = (
        data["macd_hist"]
        >
        data["macd_hist"].shift(1)
    )

    data["v10_macd_rising2"] = (
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

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    data["v10_rsi_rising"] = (
        data["rsi"]
        >
        data["rsi"].shift(1)
    )

    data["v10_rsi_45_70"] = (
        data["rsi"]
        .between(
            45,
            70,
        )
    )

    data["v10_rsi_50_70"] = (
        data["rsi"]
        .between(
            50,
            70,
        )
    )

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    candle_range = (
        data["high"]
        -
        data["low"]
    )

    safe_range = (
        candle_range
        .replace(
            0,
            np.nan,
        )
    )

    data["v10_close_position"] = (
        (
            data["close"]
            -
            data["low"]
        )
        /
        safe_range
    )

    data["v10_green"] = (
        data["close"]
        >
        data["open"]
    )

    data["v10_close_top55"] = (
        data["v10_close_position"]
        >= 0.55
    )

    data["v10_close_top60"] = (
        data["v10_close_position"]
        >= 0.60
    )

    data["v10_close_top65"] = (
        data["v10_close_position"]
        >= 0.65
    )

    # --------------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------------

    data["v10_bb_middle"] = (
        data["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        data["close"]
        .rolling(20)
        .std()
    )

    data["v10_bb_upper"] = (
        data["v10_bb_middle"]
        +
        2.0 * bb_std
    )

    data["v10_bb_lower"] = (
        data["v10_bb_middle"]
        -
        2.0 * bb_std
    )

    bb_range = (
        data["v10_bb_upper"]
        -
        data["v10_bb_lower"]
    )

    data["v10_bb_position"] = (
        (
            data["close"]
            -
            data["v10_bb_lower"]
        )
        /
        bb_range.replace(
            0,
            np.nan,
        )
    )

    data["v10_bb55"] = (
        data["v10_bb_position"]
        >= 0.55
    )

    data["v10_bb60"] = (
        data["v10_bb_position"]
        >= 0.60
    )

    data["v10_bb_middle_pass"] = (
        data["close"]
        >=
        data["v10_bb_middle"]
    )

    data["v10_bb_width_pct"] = (
        (
            data["v10_bb_upper"]
            -
            data["v10_bb_lower"]
        )
        /
        data["v10_bb_middle"].replace(
            0,
            np.nan,
        )
        * 100
    )

    data["v10_bb_expanding"] = (
        data["v10_bb_width_pct"]
        >
        data["v10_bb_width_pct"].shift(1)
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    data["v10_volume_avg20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["v10_volume_ratio"] = (
        data["volume"]
        /
        data["v10_volume_avg20"].replace(
            0,
            np.nan,
        )
    )

    data["v10_volume100"] = (
        data["v10_volume_ratio"]
        >= 1.00
    )

    data["v10_volume110"] = (
        data["v10_volume_ratio"]
        >= 1.10
    )

    # --------------------------------------------------------
    # POC
    # --------------------------------------------------------

    data["v10_poc_distance_pct"] = (
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

    data["v10_above_poc"] = (
        data["close"]
        >=
        data["poc"]
    )

    data["v10_poc_near050"] = (
        data["v10_poc_distance_pct"]
        .between(
            0,
            0.50,
        )
    )

    data["v10_poc_near075"] = (
        data["v10_poc_distance_pct"]
        .between(
            0,
            0.75,
        )
    )

    data["v10_poc_near100"] = (
        data["v10_poc_distance_pct"]
        .between(
            0,
            1.00,
        )
    )

    # --------------------------------------------------------
    # 1H TREND QUALITY
    # --------------------------------------------------------

    data["v10_h1_ema_bull"] = (
        data["ema20_1h"]
        >
        data["ema50_1h"]
    )

    if "ema20_rising_1h" in data.columns:

        data["v10_h1_ema_rising"] = (
            data["ema20_rising_1h"]
            .fillna(False)
            .astype(bool)
        )

    else:

        data["v10_h1_ema_rising"] = False

    if "macd_rising_1h" in data.columns:

        data["v10_h1_macd_rising"] = (
            data["macd_rising_1h"]
            .fillna(False)
            .astype(bool)
        )

    else:

        data["v10_h1_macd_rising"] = False

    data["v10_h1_macd_positive"] = (
        data["macd_hist_1h"]
        >
        0
    )

    # --------------------------------------------------------
    # QUALITY COMBINATIONS
    # --------------------------------------------------------

    data["v10_quality_ema2_bb55"] = (
        data["v10_ema20_rising2"]
        &
        data["v10_bb55"]
    )

    data["v10_quality_ema2_h1"] = (
        data["v10_ema20_rising2"]
        &
        data["v10_h1_ema_rising"]
    )

    data["v10_quality_bb55_h1"] = (
        data["v10_bb55"]
        &
        data["v10_h1_ema_rising"]
    )

    data["v10_quality_ema2_bb55_h1"] = (
        data["v10_ema20_rising2"]
        &
        data["v10_bb55"]
        &
        data["v10_h1_ema_rising"]
    )

    return data


# ============================================================
# CONFIRMATIONS
# ============================================================

def get_v10_confirmations(row):

    c = {}

    # --------------------------------------------------------
    # ORIGINAL BASE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    c["EMA20_RISING"] = bool(
        row["v10_ema20_rising1"]
    )

    c["EMA20_RISING2"] = bool(
        row["v10_ema20_rising2"]
    )

    c["EMA20_RISING3"] = bool(
        row["v10_ema20_rising3"]
    )

    c["ABOVE_EMA20"] = bool(
        row["v10_above_ema20"]
    )

    c["EMA_BULL"] = bool(
        row["v10_ema_bull"]
    )

    c["EMA_GAP_POSITIVE"] = bool(
        row["v10_ema_gap_positive"]
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    c["MACD_RISING"] = bool(
        row["v10_macd_rising1"]
    )

    c["MACD_RISING2"] = bool(
        row["v10_macd_rising2"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    c["RSI_RISING"] = bool(
        row["v10_rsi_rising"]
    )

    c["RSI45_70"] = bool(
        row["v10_rsi_45_70"]
    )

    c["RSI50_70"] = bool(
        row["v10_rsi_50_70"]
    )

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    c["GREEN"] = bool(
        row["v10_green"]
    )

    c["CLOSE_TOP55"] = bool(
        row["v10_close_top55"]
    )

    c["CLOSE_TOP60"] = bool(
        row["v10_close_top60"]
    )

    c["CLOSE_TOP65"] = bool(
        row["v10_close_top65"]
    )

    # --------------------------------------------------------
    # BB
    # --------------------------------------------------------

    c["BB55"] = bool(
        row["v10_bb55"]
    )

    c["BB60"] = bool(
        row["v10_bb60"]
    )

    c["BB_MIDDLE"] = bool(
        row["v10_bb_middle_pass"]
    )

    c["BB_EXPANDING"] = bool(
        row["v10_bb_expanding"]
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    c["VOLUME100"] = bool(
        row["v10_volume100"]
    )

    c["VOLUME110"] = bool(
        row["v10_volume110"]
    )

    # --------------------------------------------------------
    # POC
    # --------------------------------------------------------

    c["POC_NEAR050"] = bool(
        row["v10_poc_near050"]
    )

    c["POC_NEAR075"] = bool(
        row["v10_poc_near075"]
    )

    c["POC_NEAR100"] = bool(
        row["v10_poc_near100"]
    )

    # --------------------------------------------------------
    # 1H
    # --------------------------------------------------------

    c["H1_EMA_BULL"] = bool(
        row["v10_h1_ema_bull"]
    )

    c["H1_EMA_RISING"] = bool(
        row["v10_h1_ema_rising"]
    )

    c["H1_MACD_RISING"] = bool(
        row["v10_h1_macd_rising"]
    )

    c["H1_MACD_POSITIVE"] = bool(
        row["v10_h1_macd_positive"]
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    c["QUALITY_EMA2_BB55"] = bool(
        row["v10_quality_ema2_bb55"]
    )

    c["QUALITY_EMA2_H1"] = bool(
        row["v10_quality_ema2_h1"]
    )

    c["QUALITY_BB55_H1"] = bool(
        row["v10_quality_bb55_h1"]
    )

    c["QUALITY_EMA2_BB55_H1"] = bool(
        row["v10_quality_ema2_bb55_h1"]
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
# V10 CANDIDATES
# ============================================================

CANDIDATES = {

    "V10_BASE":
        BASE,

    # --------------------------------------------------------
    # V9 WINNER: EMA20 RISING 2
    # --------------------------------------------------------

    "V10_EMA20_RISING2":
        BASE
        + [
            "EMA20_RISING2",
        ],

    "V10_EMA20_RISING2_BB55":
        BASE
        + [
            "EMA20_RISING2",
            "BB55",
        ],

    "V10_EMA20_RISING2_RSI":
        BASE
        + [
            "EMA20_RISING2",
            "RSI_RISING",
        ],

    "V10_EMA20_RISING2_GREEN":
        BASE
        + [
            "EMA20_RISING2",
            "GREEN",
        ],

    "V10_EMA20_RISING2_TOP60":
        BASE
        + [
            "EMA20_RISING2",
            "CLOSE_TOP60",
        ],

    "V10_EMA20_RISING2_POC075":
        BASE
        + [
            "EMA20_RISING2",
            "POC_NEAR075",
        ],

    "V10_EMA20_RISING2_MACD":
        BASE
        + [
            "EMA20_RISING2",
            "MACD_RISING",
        ],

    # --------------------------------------------------------
    # BB55 + H1 EMA FAMILY
    # --------------------------------------------------------

    "V10_BB55_H1_EMA":
        BASE
        + [
            "BB55",
            "H1_EMA_RISING",
        ],

    "V10_BB55_H1_EMA_RSI":
        BASE
        + [
            "BB55",
            "H1_EMA_RISING",
            "RSI_RISING",
        ],

    "V10_BB55_H1_EMA_GREEN":
        BASE
        + [
            "BB55",
            "H1_EMA_RISING",
            "GREEN",
        ],

    "V10_BB55_H1_EMA_TOP60":
        BASE
        + [
            "BB55",
            "H1_EMA_RISING",
            "CLOSE_TOP60",
        ],

    "V10_BB55_H1_EMA_POC075":
        BASE
        + [
            "BB55",
            "H1_EMA_RISING",
            "POC_NEAR075",
        ],

    "V10_BB55_H1_EMA_MACD":
        BASE
        + [
            "BB55",
            "H1_EMA_RISING",
            "MACD_RISING",
        ],

    # --------------------------------------------------------
    # EMA20 + H1 EMA
    # --------------------------------------------------------

    "V10_EMA20_H1_EMA":
        BASE
        + [
            "EMA20_RISING",
            "H1_EMA_RISING",
        ],

    "V10_EMA20_H1_EMA_RSI":
        BASE
        + [
            "EMA20_RISING",
            "H1_EMA_RISING",
            "RSI_RISING",
        ],

    "V10_EMA20_H1_EMA_GREEN":
        BASE
        + [
            "EMA20_RISING",
            "H1_EMA_RISING",
            "GREEN",
        ],

    "V10_EMA20_H1_EMA_TOP60":
        BASE
        + [
            "EMA20_RISING",
            "H1_EMA_RISING",
            "CLOSE_TOP60",
        ],

    "V10_EMA20_H1_EMA_POC075":
        BASE
        + [
            "EMA20_RISING",
            "H1_EMA_RISING",
            "POC_NEAR075",
        ],

    # --------------------------------------------------------
    # BB55 + EMA20 + H1
    # --------------------------------------------------------

    "V10_BB55_EMA20_H1":
        BASE
        + [
            "BB55",
            "EMA20_RISING",
            "H1_EMA_RISING",
        ],

    "V10_BB55_EMA20_H1_RSI":
        BASE
        + [
            "BB55",
            "EMA20_RISING",
            "H1_EMA_RISING",
            "RSI_RISING",
        ],

    "V10_BB55_EMA20_H1_GREEN":
        BASE
        + [
            "BB55",
            "EMA20_RISING",
            "H1_EMA_RISING",
            "GREEN",
        ],

    "V10_BB55_EMA20_H1_TOP60":
        BASE
        + [
            "BB55",
            "EMA20_RISING",
            "H1_EMA_RISING",
            "CLOSE_TOP60",
        ],

    "V10_BB55_EMA20_H1_POC075":
        BASE
        + [
            "BB55",
            "EMA20_RISING",
            "H1_EMA_RISING",
            "POC_NEAR075",
        ],

    "V10_BB55_EMA20_H1_MACD":
        BASE
        + [
            "BB55",
            "EMA20_RISING",
            "H1_EMA_RISING",
            "MACD_RISING",
        ],

    # --------------------------------------------------------
    # STRICTER V10 QUALITY
    # --------------------------------------------------------

    "V10_QUALITY_EMA2_BB55":
        BASE
        + [
            "QUALITY_EMA2_BB55",
        ],

    "V10_QUALITY_EMA2_H1":
        BASE
        + [
            "QUALITY_EMA2_H1",
        ],

    "V10_QUALITY_BB55_H1":
        BASE
        + [
            "QUALITY_BB55_H1",
        ],

    "V10_QUALITY_EMA2_BB55_H1":
        BASE
        + [
            "QUALITY_EMA2_BB55_H1",
        ],

    "V10_QUALITY_EMA2_BB55_H1_RSI":
        BASE
        + [
            "QUALITY_EMA2_BB55_H1",
            "RSI_RISING",
        ],

    "V10_QUALITY_EMA2_BB55_H1_TOP60":
        BASE
        + [
            "QUALITY_EMA2_BB55_H1",
            "CLOSE_TOP60",
        ],

    "V10_QUALITY_EMA2_BB55_H1_POC":
        BASE
        + [
            "QUALITY_EMA2_BB55_H1",
            "POC_NEAR075",
        ],

    # --------------------------------------------------------
    # EMA3 STRICT TEST
    # --------------------------------------------------------

    "V10_EMA20_RISING3":
        BASE
        + [
            "EMA20_RISING3",
        ],

    "V10_EMA20_RISING3_BB55":
        BASE
        + [
            "EMA20_RISING3",
            "BB55",
        ],

    "V10_EMA20_RISING3_H1":
        BASE
        + [
            "EMA20_RISING3",
            "H1_EMA_RISING",
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
            get_v10_confirmations(
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
# CREATE WALK-FORWARD BLOCKS
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

        enough_sample = (
            overall["DECISIVE"]
            >=
            MIN_TOTAL_DECISIVE
        )

        stable_all_blocks = (
            valid_blocks
            ==
            MIN_VALID_BLOCKS
            and
            positive_blocks
            ==
            MIN_VALID_BLOCKS
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
                enough_sample,

            "STABLE_ALL_BLOCKS":
                stable_all_blocks,
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
                "STABLE_ALL_BLOCKS",
                "OVERALL_WIN_RATE_%",
                "TOTAL_DECISIVE",
                "OVERALL_AVG_RETURN_%",
            ],
            ascending=[
                False,
                False,
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
    print("V10 WALK-FORWARD BLOCK RESULTS")
    print("=" * 120)

    print(
        block_results.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("V10 OVERALL RESULTS")
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
            "No V10 summary results."
        )

        return

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
            f"V10 RELIABLE {target}%+ CANDIDATES:",
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

    # --------------------------------------------------------
    # STABLE 70+
    # --------------------------------------------------------

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
        "V10 STABLE 70%+ CANDIDATES:",
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

    # --------------------------------------------------------
    # STABLE 75+
    # --------------------------------------------------------

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
        "V10 STABLE 75%+ CANDIDATES:",
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

    # --------------------------------------------------------
    # STABLE 80+
    # --------------------------------------------------------

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
        "V10 STABLE 80%+ CANDIDATES:",
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

    # --------------------------------------------------------
    # BEST V10 CANDIDATES
    # --------------------------------------------------------

    best = (
        summary[
            (
                summary["ENOUGH_SAMPLE"]
                ==
                True
            )
            &
            (
                summary["STABLE_ALL_BLOCKS"]
                ==
                True
            )
            &
            (
                summary["OVERALL_WIN_RATE_%"]
                >=
                75
            )
            &
            (
                summary["OVERALL_AVG_RETURN_%"]
                >
                0
            )
        ]
        .sort_values(
            by=[
                "OVERALL_WIN_RATE_%",
                "TOTAL_DECISIVE",
                "OVERALL_AVG_RETURN_%",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .copy()
    )

    print()
    print("=" * 120)

    print(
        "V10 BEST STABLE CANDIDATES:",
        len(best),
    )

    print("=" * 120)

    if not best.empty:

        print(
            best[
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
        "COINDCX FUTURES HISTORICAL V10"
    )

    print(
        "V9 WINNER STABILITY OPTIMIZATION"
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
        "Adding V9 indicators..."
    )

    df_5m = (
        v9.add_v9_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V9 result",
    )

    # ========================================================
    # V10
    # ========================================================

    print(
        "Adding V10 optimization indicators..."
    )

    df_5m = (
        add_v10_indicators(
            df_5m
        )
    )

    df_5m = ensure_dataframe(
        df_5m,
        "V10 result",
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
        "v10_bb_middle",
        "v10_bb_position",
        "v10_volume_ratio",
        "v10_poc_distance_pct",
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
        "Running V10 walk-forward test..."
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
        "V10 detail saved:",
        DETAIL_FILE,
    )

    print(
        "V10 summary saved:",
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
        "FUTURES HISTORICAL V10 COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
