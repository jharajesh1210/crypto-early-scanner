# ============================================================
# CRYPTO FUTURES HISTORICAL V20
# TOP-50 CRYPTO FUTURES MASTER LOGIC
#
# PURPOSE:
# Top major crypto coins only
# CoinDCX USDT Futures only
# Historical research only
# LONG + SHORT
# RR 1:2 + RR 1:3
#
# TARGET:
# 70%+ historical win rate
# + positive net profit
# + Profit Factor >= 1.20
# + minimum 20 trades
#
# NO REAL ORDERS
# ============================================================

import math
import time
import requests
import numpy as np
import pandas as pd

import crypto_futures_historical_v19 as v19


# ============================================================
# SETTINGS
# ============================================================

HISTORY_DAYS = 180

STOP_LOSS_PCT = 0.50

ROUND_TRIP_COST_PCT = 0.10

MIN_TRADES = 20
TARGET_WIN_RATE = 70.0
MIN_PROFIT_FACTOR = 1.20

MIN_SCORE = 6

VOLUME_SURGE_MIN = 1.50

PROFILE_LOOKBACK = 96
PROFILE_BINS = 24
VALUE_AREA_PERCENT = 0.70

WICK_RATIO_MIN = 0.55


# ============================================================
# TOP-50 MAJOR CRYPTO UNIVERSE
#
# CoinDCX availability is checked automatically.
# Stablecoins are excluded because they are not useful
# directional futures candidates for this strategy research.
# ============================================================

TOP_50_SYMBOLS = [

    "BTC",
    "ETH",
    "BNB",
    "XRP",
    "SOL",
    "DOGE",
    "ADA",
    "TRX",
    "AVAX",
    "LINK",

    "SUI",
    "XLM",
    "HBAR",
    "TON",
    "SHIB",
    "DOT",
    "LTC",
    "BCH",
    "UNI",
    "NEAR",

    "APT",
    "ICP",
    "ETC",
    "POL",
    "ARB",
    "OP",
    "FIL",
    "ATOM",
    "AAVE",
    "ALGO",

    "VET",
    "RENDER",
    "INJ",
    "SEI",
    "TIA",
    "IMX",
    "GRT",
    "THETA",
    "MKR",
    "RUNE",

    "FET",
    "KAS",
    "STX",
    "LDO",
    "QNT",
    "JUP",
    "BONK",
    "WIF",
    "PEPE",
    "ENA",
]


# ============================================================
# FIND TOP-50 COINS AVAILABLE IN COINDCX FUTURES
# ============================================================

def get_top50_futures():

    active_pairs = (
        v19.get_active_usdt_futures()
    )

    active_set = set(active_pairs)

    selected = []

    missing = []

    for symbol in TOP_50_SYMBOLS:

        expected_pair = (
            f"B-{symbol}_USDT"
        )

        if expected_pair in active_set:

            selected.append(
                expected_pair
            )

        else:

            # Fallback in case naming differs
            matches = [
                pair
                for pair in active_pairs
                if pair.endswith(
                    f"-{symbol}_USDT"
                )
                or pair == (
                    f"{symbol}_USDT"
                )
            ]

            if matches:

                selected.append(
                    matches[0]
                )

            else:

                missing.append(
                    symbol
                )

    # Remove duplicates while keeping order
    selected = list(
        dict.fromkeys(
            selected
        )
    )

    print()
    print("=" * 75)
    print("V20 TOP-50 FUTURES UNIVERSE")
    print("=" * 75)

    print(
        f"Requested major coins : "
        f"{len(TOP_50_SYMBOLS)}"
    )

    print(
        f"CoinDCX Futures found : "
        f"{len(selected)}"
    )

    print(
        f"Unavailable           : "
        f"{len(missing)}"
    )

    if missing:

        print(
            "Unavailable symbols:",
            ", ".join(missing)
        )

    print()
    print("COINS TO TEST:")

    for number, pair in enumerate(
        selected,
        start=1
    ):

        print(
            f"{number:02d}. {pair}"
        )

    return selected


# ============================================================
# VOLUME PROFILE
# ============================================================

def volume_profile_at(
    df,
    index
):

    start = max(
        0,
        index - PROFILE_LOOKBACK + 1
    )

    window = df.iloc[
        start:index + 1
    ]

    if len(window) < 30:
        return None

    low_price = float(
        window["low"].min()
    )

    high_price = float(
        window["high"].max()
    )

    if not (
        math.isfinite(low_price)
        and
        math.isfinite(high_price)
    ):

        return None

    if high_price <= low_price:
        return None

    edges = np.linspace(
        low_price,
        high_price,
        PROFILE_BINS + 1
    )

    volume_bins = np.zeros(
        PROFILE_BINS
    )

    typical_price = (
        window["high"].to_numpy()
        +
        window["low"].to_numpy()
        +
        window["close"].to_numpy()
    ) / 3.0

    volumes = (
        window["volume"]
        .to_numpy()
    )

    bins = np.searchsorted(
        edges,
        typical_price,
        side="right"
    ) - 1

    bins = np.clip(
        bins,
        0,
        PROFILE_BINS - 1
    )

    for bin_number, volume in zip(
        bins,
        volumes
    ):

        if math.isfinite(volume):

            volume_bins[
                bin_number
            ] += volume

    total_volume = (
        volume_bins.sum()
    )

    if total_volume <= 0:
        return None

    centers = (
        edges[:-1]
        +
        edges[1:]
    ) / 2.0

    poc_index = int(
        np.argmax(
            volume_bins
        )
    )

    poc = float(
        centers[poc_index]
    )

    # --------------------------------------------------------
    # VALUE AREA
    # --------------------------------------------------------

    target_volume = (
        total_volume
        *
        VALUE_AREA_PERCENT
    )

    selected_bins = {
        poc_index
    }

    accumulated = (
        volume_bins[
            poc_index
        ]
    )

    left = (
        poc_index - 1
    )

    right = (
        poc_index + 1
    )

    while (
        accumulated
        <
        target_volume
    ):

        left_volume = (
            volume_bins[left]
            if left >= 0
            else -1
        )

        right_volume = (
            volume_bins[right]
            if right < PROFILE_BINS
            else -1
        )

        if (
            left_volume < 0
            and
            right_volume < 0
        ):

            break

        if (
            right_volume
            >
            left_volume
        ):

            selected_bins.add(
                right
            )

            accumulated += (
                volume_bins[
                    right
                ]
            )

            right += 1

        else:

            selected_bins.add(
                left
            )

            accumulated += (
                volume_bins[
                    left
                ]
            )

            left -= 1

    selected_bins = [
        x
        for x in selected_bins
        if 0 <= x < PROFILE_BINS
    ]

    val = float(
        edges[
            min(
                selected_bins
            )
        ]
    )

    vah = float(
        edges[
            max(
                selected_bins
            )
            +
            1
        ]
    )

    # --------------------------------------------------------
    # PROFILE SHAPE
    # P / b / D / B
    # --------------------------------------------------------

    weights = (
        volume_bins
        /
        total_volume
    )

    positions = np.linspace(
        0,
        1,
        PROFILE_BINS
    )

    center_of_volume = float(
        np.sum(
            positions
            *
            weights
        )
    )

    top_nodes = np.argsort(
        volume_bins
    )[-2:]

    node_distance = abs(
        int(top_nodes[1])
        -
        int(top_nodes[0])
    )

    strongest = (
        volume_bins[
            top_nodes[-1]
        ]
    )

    second = (
        volume_bins[
            top_nodes[-2]
        ]
    )

    double_distribution = (
        node_distance >= 7
        and
        second >= strongest * 0.65
    )

    if double_distribution:

        shape = "B"

    elif center_of_volume >= 0.58:

        shape = "P"

    elif center_of_volume <= 0.42:

        shape = "b"

    else:

        shape = "D"

    return {

        "POC":
            poc,

        "VAH":
            vah,

        "VAL":
            val,

        "SHAPE":
            shape,

        "CENTER":
            center_of_volume,
    }


# ============================================================
# PREPARE INDICATORS
# ============================================================

def prepare_indicators(
    raw
):

    df = (
        v19.calculate_indicators(
            raw
        )
    )

    df = (
        v19.add_market_regime(
            df
        )
    )

    # --------------------------------------------------------
    # Bollinger expansion
    # --------------------------------------------------------

    df["bb_expansion"] = (

        (
            df["bb_width_pct"]
            >
            df[
                "bb_width_pct"
            ].shift(1)
        )

        &

        (
            df[
                "bb_width_pct"
            ].shift(1)
            >=
            df[
                "bb_width_pct"
            ].shift(2)
        )
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["volume_surge"] = (
        df["volume_ratio"]
        >=
        VOLUME_SURGE_MIN
    )

    # --------------------------------------------------------
    # Wick
    # --------------------------------------------------------

    candle_range = (
        df["high"]
        -
        df["low"]
    ).replace(
        0,
        np.nan
    )

    df[
        "lower_wick_ratio"
    ] = (
        df["lower_wick"]
        /
        candle_range
    )

    df[
        "upper_wick_ratio"
    ] = (
        df["upper_wick"]
        /
        candle_range
    )

    df["bull_rejection"] = (

        (
            df[
                "lower_wick_ratio"
            ]
            >=
            WICK_RATIO_MIN
        )

        &

        (
            df["close"]
            >
            df["open"]
        )
    )

    df["bear_rejection"] = (

        (
            df[
                "upper_wick_ratio"
            ]
            >=
            WICK_RATIO_MIN
        )

        &

        (
            df["close"]
            <
            df["open"]
        )
    )

    # --------------------------------------------------------
    # Hammer
    # --------------------------------------------------------

    df["weak_hammer"] = (

        (
            df["lower_wick"]
            >=
            df["body"] * 1.5
        )

        &

        (
            df[
                "close_position"
            ]
            >=
            0.55
        )
    )

    # --------------------------------------------------------
    # Shooting star
    # --------------------------------------------------------

    df["shooting_star"] = (

        (
            df["upper_wick"]
            >=
            df["body"] * 1.5
        )

        &

        (
            df[
                "close_position"
            ]
            <=
            0.45
        )
    )

    # --------------------------------------------------------
    # RSI direction
    # --------------------------------------------------------

    df["rsi_rising"] = (
        df["rsi"]
        >
        df["rsi"].shift(1)
    )

    df["rsi_falling"] = (
        df["rsi"]
        <
        df["rsi"].shift(1)
    )

    # --------------------------------------------------------
    # RSI divergence approximation
    # --------------------------------------------------------

    previous_low = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    previous_high = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    previous_rsi_low = (
        df["rsi"]
        .rolling(20)
        .min()
        .shift(1)
    )

    previous_rsi_high = (
        df["rsi"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df[
        "bullish_divergence"
    ] = (

        (
            df["low"]
            <
            previous_low
        )

        &

        (
            df["rsi"]
            >
            previous_rsi_low
        )
    )

    df[
        "bearish_divergence"
    ] = (

        (
            df["high"]
            >
            previous_high
        )

        &

        (
            df["rsi"]
            <
            previous_rsi_high
        )
    )

    # --------------------------------------------------------
    # Accumulation / Distribution range
    # --------------------------------------------------------

    df["range_high"] = (
        df["high"]
        .rolling(48)
        .max()
        .shift(1)
    )

    df["range_low"] = (
        df["low"]
        .rolling(48)
        .min()
        .shift(1)
    )

    df["range_width_pct"] = (

        (
            df["range_high"]
            -
            df["range_low"]
        )

        /

        df["close"]

        *

        100
    )

    range_threshold = (
        df[
            "range_width_pct"
        ]
        .rolling(100)
        .quantile(0.35)
    )

    df["accumulation_zone"] = (
        df["range_width_pct"]
        <=
        range_threshold
    )

    # --------------------------------------------------------
    # Breakout / Breakdown
    # --------------------------------------------------------

    df["bull_breakout"] = (

        (
            df["close"]
            >
            df["range_high"]
        )

        &

        df["volume_surge"]
    )

    df["bear_breakdown"] = (

        (
            df["close"]
            <
            df["range_low"]
        )

        &

        df["volume_surge"]
    )

    return df


# ============================================================
# SIGNAL SCORING
# ============================================================

def get_signal(
    df,
    i
):

    row = df.iloc[i]

    profile = (
        volume_profile_at(
            df,
            i
        )
    )

    if profile is None:
        return None

    close = float(
        row["close"]
    )

    poc = profile["POC"]
    vah = profile["VAH"]
    val = profile["VAL"]

    shape = profile[
        "SHAPE"
    ]

    poc_distance = (
        abs(
            close - poc
        )
        /
        close
        *
        100
    )

    vah_distance = (
        abs(
            close - vah
        )
        /
        close
        *
        100
    )

    val_distance = (
        abs(
            close - val
        )
        /
        close
        *
        100
    )

    near_poc = (
        poc_distance
        <=
        0.50
    )

    near_vah = (
        vah_distance
        <=
        0.50
    )

    near_val = (
        val_distance
        <=
        0.50
    )

    # ========================================================
    # LONG SCORE
    # ========================================================

    long_score = 0

    long_reasons = []

    if bool(
        row["bull_trend"]
    ):

        long_score += 1

        long_reasons.append(
            "BULL_TREND"
        )

    if shape in (
        "b",
        "D",
        "P"
    ):

        long_score += 1

        long_reasons.append(
            f"PROFILE_{shape}"
        )

    if (
        near_val
        or
        near_poc
    ):

        long_score += 1

        long_reasons.append(
            "VAL_POC_SUPPORT"
        )

    if (
        bool(
            row["bb_squeeze"]
        )
        or
        bool(
            row["bb_expansion"]
        )
    ):

        long_score += 1

        long_reasons.append(
            "BB"
        )

    if bool(
        row["volume_surge"]
    ):

        long_score += 1

        long_reasons.append(
            "VOLUME_1_5X"
        )

    if (
        bool(
            row["bull_rejection"]
        )
        or
        bool(
            row["weak_hammer"]
        )
    ):

        long_score += 1

        long_reasons.append(
            "BULL_REJECTION"
        )

    if (
        (
            30
            <=
            row["rsi"]
            <=
            55
        )
        or
        bool(
            row[
                "bullish_divergence"
            ]
        )
    ):

        long_score += 1

        long_reasons.append(
            "RSI_LONG"
        )

    if (
        row["macd_hist"]
        >
        0
    ):

        long_score += 1

        long_reasons.append(
            "MACD_POSITIVE"
        )

    if bool(
        row["bull_breakout"]
    ):

        long_score += 2

        long_reasons.append(
            "ACCUMULATION_BREAKOUT"
        )

    # ========================================================
    # SHORT SCORE
    # ========================================================

    short_score = 0

    short_reasons = []

    if bool(
        row["bear_trend"]
    ):

        short_score += 1

        short_reasons.append(
            "BEAR_TREND"
        )

    if shape in (
        "P",
        "D",
        "b"
    ):

        short_score += 1

        short_reasons.append(
            f"PROFILE_{shape}"
        )

    if (
        near_vah
        or
        near_poc
    ):

        short_score += 1

        short_reasons.append(
            "VAH_POC_RESISTANCE"
        )

    if (
        bool(
            row["bb_squeeze"]
        )
        or
        bool(
            row["bb_expansion"]
        )
    ):

        short_score += 1

        short_reasons.append(
            "BB"
        )

    if bool(
        row["volume_surge"]
    ):

        short_score += 1

        short_reasons.append(
            "VOLUME_1_5X"
        )

    if (
        bool(
            row["bear_rejection"]
        )
        or
        bool(
            row["shooting_star"]
        )
    ):

        short_score += 1

        short_reasons.append(
            "BEAR_REJECTION"
        )

    if (
        (
            50
            <=
            row["rsi"]
            <=
            75
        )
        or
        bool(
            row[
                "bearish_divergence"
            ]
        )
    ):

        short_score += 1

        short_reasons.append(
            "RSI_SHORT"
        )

    if (
        row["macd_hist"]
        <
        0
    ):

        short_score += 1

        short_reasons.append(
            "MACD_NEGATIVE"
        )

    if bool(
        row["bear_breakdown"]
    ):

        short_score += 2

        short_reasons.append(
            "DISTRIBUTION_BREAKDOWN"
        )

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    if (
        long_score >= MIN_SCORE
        and
        long_score
        >
        short_score
    ):

        return {

            "side":
                "LONG",

            "score":
                long_score,

            "reasons":
                "|".join(
                    long_reasons
                ),

            "profile_shape":
                shape,

            "poc":
                poc,

            "vah":
                vah,

            "val":
                val,
        }

    if (
        short_score >= MIN_SCORE
        and
        short_score
        >
        long_score
    ):

        return {

            "side":
                "SHORT",

            "score":
                short_score,

            "reasons":
                "|".join(
                    short_reasons
                ),

            "profile_shape":
                shape,

            "poc":
                poc,

            "vah":
                vah,

            "val":
                val,
        }

    return None


# ============================================================
# BACKTEST ONE COIN
# ============================================================

def backtest_coin(
    pair,
    df
):

    trades = []

    last_signal_index = -999

    for i in range(
        250,
        len(df) - 2
    ):

        row = df.iloc[i]

        # ----------------------------------------------------
        # Fast pre-filter
        # ----------------------------------------------------

        interesting = (

            bool(
                row["volume_surge"]
            )

            or

            bool(
                row["bull_rejection"]
            )

            or

            bool(
                row["bear_rejection"]
            )

            or

            bool(
                row["bull_breakout"]
            )

            or

            bool(
                row["bear_breakdown"]
            )

            or

            bool(
                row["bb_squeeze"]
            )
        )

        if not interesting:
            continue

        # Avoid signals too close together
        if (
            i
            -
            last_signal_index
            <
            3
        ):

            continue

        signal = get_signal(
            df,
            i
        )

        if signal is None:
            continue

        last_signal_index = i

        for (
            rr_name,
            target_pct
        ) in (
            v19.RR_SETTINGS.items()
        ):

            trade = (
                v19.simulate_trade(
                    df,
                    i,
                    signal["side"],
                    target_pct
                )
            )

            if trade is None:
                continue

            trade["pair"] = pair

            trade["strategy"] = (
                "V20_MASTER"
            )

            trade["rr"] = (
                rr_name
            )

            trade["v20_score"] = (
                signal["score"]
            )

            trade[
                "v20_reasons"
            ] = (
                signal["reasons"]
            )

            trade[
                "profile_shape"
            ] = (
                signal[
                    "profile_shape"
                ]
            )

            trade["poc"] = (
                signal["poc"]
            )

            trade["vah"] = (
                signal["vah"]
            )

            trade["val"] = (
                signal["val"]
            )

            trades.append(
                trade
            )

    return trades


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_result(
    pair,
    trades,
    rr_name
):

    temp = pd.DataFrame(
        [
            x
            for x in trades
            if x["rr"]
            ==
            rr_name
        ]
    )

    if temp.empty:
        return None

    total = len(temp)

    wins = int(
        (
            temp["outcome"]
            ==
            "WIN"
        ).sum()
    )

    losses = (
        total - wins
    )

    win_rate = (
        wins
        /
        total
        *
        100
    )

    net_profit = float(
        temp[
            "net_return_pct"
        ].sum()
    )

    gross_profit = float(

        temp.loc[
            temp[
                "net_return_pct"
            ]
            >
            0,
            "net_return_pct"
        ].sum()
    )

    gross_loss = abs(
        float(
            temp.loc[
                temp[
                    "net_return_pct"
                ]
                <
                0,
                "net_return_pct"
            ].sum()
        )
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    elif gross_profit > 0:

        profit_factor = 999.0

    else:

        profit_factor = 0.0

    drawdown = (
        v19.calculate_max_drawdown(
            temp[
                "net_return_pct"
            ]
        )
    )

    average_score = float(
        temp[
            "v20_score"
        ].mean()
    )

    pass_result = (

        total
        >=
        MIN_TRADES

        and

        win_rate
        >=
        TARGET_WIN_RATE

        and

        net_profit
        >
        0

        and

        profit_factor
        >=
        MIN_PROFIT_FACTOR
    )

    return {

        "pair":
            pair,

        "strategy":
            "V20_MASTER",

        "rr":
            rr_name,

        "total_trades":
            total,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate_pct":
            round(
                win_rate,
                2
            ),

        "net_profit_pct":
            round(
                net_profit,
                2
            ),

        "profit_factor":
            round(
                profit_factor,
                3
            ),

        "max_drawdown_pct_points":
            round(
                drawdown,
                2
            ),

        "average_v20_score":
            round(
                average_score,
                2
            ),

        "V20_PASS":
            pass_result,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CRYPTO FUTURES HISTORICAL V20")
    print("TOP-50 MASTER LOGIC")
    print("=" * 80)

    print(
        "MARKET          : "
        "CoinDCX USDT Futures"
    )

    print(
        "UNIVERSE        : "
        "Top-50 major crypto symbols"
    )

    print(
        f"HISTORY         : "
        f"{HISTORY_DAYS} days"
    )

    print(
        "TIMEFRAME       : 5 minute"
    )

    print(
        f"STOP LOSS       : "
        f"{STOP_LOSS_PCT:.2f}%"
    )

    print(
        "RR              : "
        "1:2 and 1:3"
    )

    print(
        f"COST            : "
        f"{ROUND_TRIP_COST_PCT:.2f}%"
    )

    print(
        f"MIN SCORE       : "
        f"{MIN_SCORE}"
    )

    print(
        "TARGET          : "
        "70%+ WR + Positive Net Profit"
    )

    print(
        "REAL ORDERS     : NO"
    )

    # Make V19 simulator use same assumptions
    v19.HISTORY_DAYS = (
        HISTORY_DAYS
    )

    v19.STOP_LOSS_PCT = (
        STOP_LOSS_PCT
    )

    v19.ROUND_TRIP_COST_PCT = (
        ROUND_TRIP_COST_PCT
    )

    # --------------------------------------------------------
    # TOP-50 FUTURES
    # --------------------------------------------------------

    pairs = (
        get_top50_futures()
    )

    if not pairs:

        print(
            "No eligible Top-50 "
            "CoinDCX Futures found."
        )

        return

    all_trades = []

    all_results = []

    # --------------------------------------------------------
    # TEST EACH COIN
    # --------------------------------------------------------

    for number, pair in enumerate(
        pairs,
        start=1
    ):

        print()
        print("=" * 80)

        print(
            f"[{number}/{len(pairs)}] "
            f"TESTING {pair}"
        )

        print("=" * 80)

        try:

            raw = (
                v19.fetch_history(
                    pair,
                    HISTORY_DAYS
                )
            )

            if raw.empty:

                print("NO DATA")
                continue

            print(
                f"Candles: "
                f"{len(raw):,}"
            )

            if len(raw) < 3000:

                print(
                    "SKIPPED: "
                    "insufficient history"
                )

                continue

            df = (
                prepare_indicators(
                    raw
                )
            )

            trades = (
                backtest_coin(
                    pair,
                    df
                )
            )

            all_trades.extend(
                trades
            )

            coin_results = []

            for rr_name in (
                v19.RR_SETTINGS.keys()
            ):

                result = (
                    calculate_result(
                        pair,
                        trades,
                        rr_name
                    )
                )

                if result is not None:

                    all_results.append(
                        result
                    )

                    coin_results.append(
                        result
                    )

            if coin_results:

                best = sorted(
                    coin_results,
                    key=lambda x: (
                        x["V20_PASS"],
                        x["win_rate_pct"],
                        x["net_profit_pct"],
                        x["profit_factor"],
                    ),
                    reverse=True
                )[0]

                print()
                print(
                    "BEST RESULT:"
                )

                print(
                    f"RR={best['rr']} | "
                    f"Trades={best['total_trades']} | "
                    f"WR={best['win_rate_pct']:.2f}% | "
                    f"Net={best['net_profit_pct']:.2f}% | "
                    f"PF={best['profit_factor']:.2f} | "
                    f"PASS={best['V20_PASS']}"
                )

            else:

                print(
                    "NO V20 SIGNALS"
                )

        except Exception as exc:

            print(
                f"ERROR {pair}: {exc}"
            )

        time.sleep(
            v19.REQUEST_SLEEP
        )

    # ========================================================
    # SAVE ALL TRADES
    # ========================================================

    trade_df = pd.DataFrame(
        all_trades
    )

    trade_df.to_csv(
        "v20_all_trades.csv",
        index=False
    )

    # ========================================================
    # RESULTS
    # ========================================================

    result_df = pd.DataFrame(
        all_results
    )

    if result_df.empty:

        print()
        print(
            "NO V20 RESULTS GENERATED."
        )

        return

    # --------------------------------------------------------
    # Ranking score
    # --------------------------------------------------------

    result_df[
        "ranking_score"
    ] = (

        result_df[
            "win_rate_pct"
        ]

        +

        np.minimum(
            result_df[
                "profit_factor"
            ],
            5
        )
        *
        5

        +

        np.minimum(
            result_df[
                "total_trades"
            ],
            100
        )
        /
        10

        +

        result_df[
            "net_profit_pct"
        ]
        *
        0.10

        -

        result_df[
            "max_drawdown_pct_points"
        ]
        *
        0.10
    )

    result_df = (
        result_df
        .sort_values(
            by=[
                "V20_PASS",
                "ranking_score",
                "net_profit_pct",
                "total_trades",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    result_df.insert(
        0,
        "rank",
        range(
            1,
            len(result_df) + 1
        )
    )

    result_df.to_csv(
        "v20_coin_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # PASS only
    # --------------------------------------------------------

    passed = result_df[
        result_df[
            "V20_PASS"
        ]
        ==
        True
    ].copy()

    passed.to_csv(
        "v20_pass_strategies.csv",
        index=False
    )

    # --------------------------------------------------------
    # Best result for each coin
    # --------------------------------------------------------

    best_coins = (
        result_df
        .sort_values(
            by=[
                "V20_PASS",
                "ranking_score",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .groupby(
            "pair",
            as_index=False
        )
        .first()
    )

    best_coins = (
        best_coins
        .sort_values(
            by=[
                "V20_PASS",
                "ranking_score",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    best_coins.insert(
        0,
        "coin_rank",
        range(
            1,
            len(best_coins) + 1
        )
    )

    best_coins.to_csv(
        "v20_best_coins.csv",
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("V20 FINAL RESULTS")
    print("=" * 80)

    print(
        f"Eligible coins tested : "
        f"{result_df['pair'].nunique()}"
    )

    print(
        f"Total simulated trades: "
        f"{len(trade_df):,}"
    )

    print(
        f"V20 PASS combinations : "
        f"{len(passed)}"
    )

    print()
    print(
        "V20 PASS RULE:"
    )

    print(
        f"Win Rate >= "
        f"{TARGET_WIN_RATE:.0f}%"
    )

    print(
        f"Trades >= "
        f"{MIN_TRADES}"
    )

    print(
        "Net Profit > 0"
    )

    print(
        f"Profit Factor >= "
        f"{MIN_PROFIT_FACTOR:.2f}"
    )

    columns = [

        "rank",
        "pair",
        "rr",
        "total_trades",
        "wins",
        "losses",
        "win_rate_pct",
        "net_profit_pct",
        "profit_factor",
        "max_drawdown_pct_points",
        "average_v20_score",
        "V20_PASS",
    ]

    print()
    print("=" * 80)
    print("TOP RESULTS")
    print("=" * 80)

    print(
        result_df[
            columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    if not passed.empty:

        print()
        print("=" * 80)
        print("V20 PASS STRATEGIES")
        print("=" * 80)

        print(
            passed[
                columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 80)
    print("FILES CREATED")
    print("=" * 80)

    print(
        "v20_all_trades.csv"
    )

    print(
        "v20_coin_results.csv"
    )

    print(
        "v20_best_coins.csv"
    )

    print(
        "v20_pass_strategies.csv"
    )

    print()
    print(
        "NO REAL ORDERS WERE PLACED."
    )


if __name__ == "__main__":

    main()
