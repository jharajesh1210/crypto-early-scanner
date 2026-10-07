# ============================================================
# CRYPTO FUTURES HISTORICAL V19
# MULTI-COIN STRATEGY DISCOVERY ENGINE
#
# OBJECTIVE:
# Find robust crypto futures strategies with:
#   - Target historical Win Rate: 70% to 80%+
#   - Positive Net Profit after trading cost
#   - RR 1:2 and RR 1:3
#   - LONG + SHORT
#   - Maximum available CoinDCX USDT Futures coins
#   - Coin-wise ranking
#
# IMPORTANT:
# Historical research only.
# NO REAL ORDERS.
# ============================================================

import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# ============================================================
# SETTINGS
# ============================================================

ACTIVE_INSTRUMENTS_URL = (
    "https://api.coindcx.com/exchange/v1/derivatives/"
    "futures/data/active_instruments"
    "?margin_currency_short_name[]=USDT"
)

CANDLE_URL = "https://public.coindcx.com/market_data/candlesticks"

# ------------------------------------------------------------
# Start with 180 days.
# Later shortlisted coins can be tested on 365 days.
# ------------------------------------------------------------

HISTORY_DAYS = 180

RESOLUTION = "5"

# Download in chunks to reduce API problems
CHUNK_DAYS = 5

# ------------------------------------------------------------
# Risk management
# ------------------------------------------------------------

STOP_LOSS_PCT = 0.50

RR_SETTINGS = {
    "RR_1_2": 1.00,   # SL 0.50%, TP 1.00%
    "RR_1_3": 1.50,   # SL 0.50%, TP 1.50%
}

# Approx round-trip cost assumption
ROUND_TRIP_COST_PCT = 0.10

# Maximum holding period
# 5-minute bars:
# 288 bars = 24 hours
MAX_HOLD_BARS = 288

# Same candle hits TP and SL
# Conservative assumption
SAME_BAR_POLICY = "SL_FIRST"

# ------------------------------------------------------------
# Strategy requirements
# ------------------------------------------------------------

MIN_TRADES_FOR_RANKING = 20

TARGET_WIN_RATE_MIN = 70.0
TARGET_WIN_RATE_MAX = 100.0

MIN_PROFIT_FACTOR = 1.20

# ------------------------------------------------------------
# API / performance
# ------------------------------------------------------------

REQUEST_TIMEOUT = 20
REQUEST_SLEEP = 0.15

# Set None = scan all active USDT futures coins
MAX_COINS = None

# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

TRADE_FILE = "v19_all_trades.csv"
COIN_RESULT_FILE = "v19_coin_results.csv"
BEST_RESULT_FILE = "v19_best_coins.csv"


# ============================================================
# HELPER
# ============================================================

def safe_float(value, default=np.nan):
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass

    return default


# ============================================================
# GET ALL ACTIVE COINDCX USDT FUTURES
# ============================================================

def get_active_usdt_futures():

    print()
    print("=" * 70)
    print("GETTING ACTIVE COINDCX USDT FUTURES")
    print("=" * 70)

    response = requests.get(
        ACTIVE_INSTRUMENTS_URL,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise RuntimeError(
            "Unexpected active instruments response."
        )

    pairs = []

    for pair in data:

        pair = str(pair).strip()

        if pair.endswith("_USDT"):
            pairs.append(pair)

    pairs = sorted(set(pairs))

    if MAX_COINS is not None:
        pairs = pairs[:MAX_COINS]

    print(f"TOTAL ACTIVE USDT FUTURES: {len(pairs)}")

    return pairs


# ============================================================
# FETCH HISTORICAL CANDLES
# ============================================================

def fetch_chunk(pair, start_time, end_time):

    params = {
        "pair": pair,
        "from": int(start_time.timestamp()),
        "to": int(end_time.timestamp()),
        "resolution": RESOLUTION,
        "pcode": "f",
    }

    response = requests.get(
        CANDLE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    # CoinDCX may return list directly
    if isinstance(data, list):
        candles = data

    # Or dictionary containing candles/data
    elif isinstance(data, dict):

        candles = (
            data.get("data")
            or data.get("candles")
            or []
        )

    else:
        candles = []

    return candles


def fetch_history(pair, days):

    end_time = datetime.now(timezone.utc)

    start_time = end_time - timedelta(days=days)

    current = start_time

    all_rows = []

    while current < end_time:

        chunk_end = min(
            current + timedelta(days=CHUNK_DAYS),
            end_time
        )

        try:

            rows = fetch_chunk(
                pair,
                current,
                chunk_end
            )

            if rows:
                all_rows.extend(rows)

        except Exception as exc:

            print(
                f"    Chunk error "
                f"{current.date()} -> {chunk_end.date()}: {exc}"
            )

        current = chunk_end

        time.sleep(REQUEST_SLEEP)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "time",
    }

    df = df.rename(columns=rename_map)

    required = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # CoinDCX candle time is normally milliseconds
    numeric_time = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    median_time = numeric_time.dropna().median()

    if pd.isna(median_time):
        return pd.DataFrame()

    if median_time > 10_000_000_000:

        df["datetime"] = pd.to_datetime(
            numeric_time,
            unit="ms",
            utc=True,
            errors="coerce"
        )

    else:

        df["datetime"] = pd.to_datetime(
            numeric_time,
            unit="s",
            utc=True,
            errors="coerce"
        )

    df = (
        df
        .dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def ema(series, length):

    return series.ewm(
        span=length,
        adjust=False
    ).mean()


def calculate_rsi(close, length=14):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(df, length=14):

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / length,
        adjust=False
    ).mean()

    return atr


def calculate_indicators(df):

    data = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    data["ema20"] = ema(
        data["close"],
        20
    )

    data["ema50"] = ema(
        data["close"],
        50
    )

    data["ema200"] = ema(
        data["close"],
        200
    )

    # EMA slope
    data["ema20_rising3"] = (
        (data["ema20"] > data["ema20"].shift(1))
        &
        (data["ema20"].shift(1) >= data["ema20"].shift(2))
        &
        (data["ema20"].shift(2) >= data["ema20"].shift(3))
    )

    data["ema20_falling3"] = (
        (data["ema20"] < data["ema20"].shift(1))
        &
        (data["ema20"].shift(1) <= data["ema20"].shift(2))
        &
        (data["ema20"].shift(2) <= data["ema20"].shift(3))
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    data["rsi"] = calculate_rsi(
        data["close"],
        14
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    data["macd"] = (
        ema(data["close"], 12)
        -
        ema(data["close"], 26)
    )

    data["macd_signal"] = ema(
        data["macd"],
        9
    )

    data["macd_hist"] = (
        data["macd"]
        -
        data["macd_signal"]
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    data["bb_mid"] = (
        data["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        data["close"]
        .rolling(20)
        .std()
    )

    data["bb_upper"] = (
        data["bb_mid"]
        +
        2 * bb_std
    )

    data["bb_lower"] = (
        data["bb_mid"]
        -
        2 * bb_std
    )

    data["bb_width_pct"] = (
        (
            data["bb_upper"]
            -
            data["bb_lower"]
        )
        /
        data["bb_mid"]
        *
        100
    )

    data["bb_width_threshold"] = (
        data["bb_width_pct"]
        .rolling(100)
        .quantile(0.25)
    )

    data["bb_squeeze"] = (
        data["bb_width_pct"]
        <=
        data["bb_width_threshold"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    data["atr"] = calculate_atr(
        data,
        14
    )

    data["atr_pct"] = (
        data["atr"]
        /
        data["close"]
        *
        100
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    data["volume_ma20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"]
        /
        data["volume_ma20"]
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    data["body"] = (
        data["close"]
        -
        data["open"]
    ).abs()

    data["range"] = (
        data["high"]
        -
        data["low"]
    )

    data["lower_wick"] = (
        np.minimum(
            data["open"],
            data["close"]
        )
        -
        data["low"]
    )

    data["upper_wick"] = (
        data["high"]
        -
        np.maximum(
            data["open"],
            data["close"]
        )
    )

    data["close_position"] = (
        (
            data["close"]
            -
            data["low"]
        )
        /
        data["range"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Rolling support/resistance
    # Approximation for multi-coin research
    # --------------------------------------------------------

    data["support_48"] = (
        data["low"]
        .rolling(48)
        .min()
        .shift(1)
    )

    data["resistance_48"] = (
        data["high"]
        .rolling(48)
        .max()
        .shift(1)
    )

    data["distance_support_pct"] = (
        (
            data["close"]
            -
            data["support_48"]
        )
        /
        data["close"]
        *
        100
    )

    data["distance_resistance_pct"] = (
        (
            data["resistance_48"]
            -
            data["close"]
        )
        /
        data["close"]
        *
        100
    )

    # --------------------------------------------------------
    # Approx volume-weighted reference price
    # This is not full price-bin Volume Profile POC.
    # It is used as a lightweight multi-coin VWAP-style filter.
    # --------------------------------------------------------

    typical_price = (
        data["high"]
        +
        data["low"]
        +
        data["close"]
    ) / 3

    pv = (
        typical_price
        *
        data["volume"]
    )

    data["vwap_48"] = (
        pv.rolling(48).sum()
        /
        data["volume"]
        .rolling(48)
        .sum()
        .replace(0, np.nan)
    )

    return data


# ============================================================
# MARKET REGIME
# ============================================================

def add_market_regime(data):

    df = data.copy()

    df["bull_trend"] = (
        (df["close"] > df["ema50"])
        &
        (df["ema20"] > df["ema50"])
    )

    df["strong_bull_trend"] = (
        df["bull_trend"]
        &
        (df["close"] > df["ema200"])
    )

    df["bear_trend"] = (
        (df["close"] < df["ema50"])
        &
        (df["ema20"] < df["ema50"])
    )

    df["strong_bear_trend"] = (
        df["bear_trend"]
        &
        (df["close"] < df["ema200"])
    )

    return df


# ============================================================
# ENTRY CONDITION LIBRARY
# ============================================================

def build_signal_columns(data):

    df = data.copy()

    # --------------------------------------------------------
    # LONG SETUPS
    # --------------------------------------------------------

    df["LONG_TREND_MACD"] = (
        df["bull_trend"]
        &
        df["ema20_rising3"]
        &
        (df["macd_hist"] > 0)
        &
        (df["rsi"] >= 45)
        &
        (df["rsi"] <= 70)
    )

    df["LONG_STRONG_TREND"] = (
        df["strong_bull_trend"]
        &
        df["ema20_rising3"]
        &
        (df["macd_hist"] > 0)
        &
        (df["volume_ratio"] >= 1.0)
    )

    df["LONG_SQUEEZE_BREAK"] = (
        df["bull_trend"]
        &
        df["bb_squeeze"].shift(1).fillna(False)
        &
        (df["close"] > df["bb_upper"].shift(1))
        &
        (df["macd_hist"] > 0)
        &
        (df["volume_ratio"] >= 1.20)
    )

    df["LONG_SUPPORT_REVERSAL"] = (
        df["bull_trend"]
        &
        (df["distance_support_pct"] >= 0)
        &
        (df["distance_support_pct"] <= 0.75)
        &
        (df["close_position"] >= 0.60)
        &
        (df["lower_wick"] > df["body"])
        &
        (df["macd_hist"] > 0)
    )

    df["LONG_VWAP_MOMENTUM"] = (
        df["bull_trend"]
        &
        (df["close"] >= df["vwap_48"])
        &
        df["ema20_rising3"]
        &
        (df["rsi"] >= 50)
        &
        (df["rsi"] <= 68)
        &
        (df["macd_hist"] > 0)
    )

    df["LONG_V10_STYLE"] = (
        df["bull_trend"]
        &
        df["ema20_rising3"]
        &
        (df["macd_hist"] > 0)
        &
        (df["close"] >= df["vwap_48"])
        &
        (df["distance_support_pct"] >= 0)
        &
        (df["distance_support_pct"] <= 0.75)
        &
        (df["close_position"] >= 0.50)
    )

    # --------------------------------------------------------
    # SHORT SETUPS
    # --------------------------------------------------------

    df["SHORT_TREND_MACD"] = (
        df["bear_trend"]
        &
        df["ema20_falling3"]
        &
        (df["macd_hist"] < 0)
        &
        (df["rsi"] >= 30)
        &
        (df["rsi"] <= 55)
    )

    df["SHORT_STRONG_TREND"] = (
        df["strong_bear_trend"]
        &
        df["ema20_falling3"]
        &
        (df["macd_hist"] < 0)
        &
        (df["volume_ratio"] >= 1.0)
    )

    df["SHORT_SQUEEZE_BREAK"] = (
        df["bear_trend"]
        &
        df["bb_squeeze"].shift(1).fillna(False)
        &
        (df["close"] < df["bb_lower"].shift(1))
        &
        (df["macd_hist"] < 0)
        &
        (df["volume_ratio"] >= 1.20)
    )

    df["SHORT_RESISTANCE_REVERSAL"] = (
        df["bear_trend"]
        &
        (df["distance_resistance_pct"] >= 0)
        &
        (df["distance_resistance_pct"] <= 0.75)
        &
        (df["close_position"] <= 0.40)
        &
        (df["upper_wick"] > df["body"])
        &
        (df["macd_hist"] < 0)
    )

    df["SHORT_VWAP_MOMENTUM"] = (
        df["bear_trend"]
        &
        (df["close"] <= df["vwap_48"])
        &
        df["ema20_falling3"]
        &
        (df["rsi"] >= 32)
        &
        (df["rsi"] <= 50)
        &
        (df["macd_hist"] < 0)
    )

    return df


# ============================================================
# STRATEGY DEFINITIONS
# ============================================================

STRATEGIES = {

    "LONG_TREND_MACD":
        "LONG",

    "LONG_STRONG_TREND":
        "LONG",

    "LONG_SQUEEZE_BREAK":
        "LONG",

    "LONG_SUPPORT_REVERSAL":
        "LONG",

    "LONG_VWAP_MOMENTUM":
        "LONG",

    "LONG_V10_STYLE":
        "LONG",

    "SHORT_TREND_MACD":
        "SHORT",

    "SHORT_STRONG_TREND":
        "SHORT",

    "SHORT_SQUEEZE_BREAK":
        "SHORT",

    "SHORT_RESISTANCE_REVERSAL":
        "SHORT",

    "SHORT_VWAP_MOMENTUM":
        "SHORT",
}


# ============================================================
# TRADE SIMULATOR
# ============================================================

def simulate_trade(
    df,
    signal_index,
    side,
    target_pct
):

    # Entry at NEXT candle open.
    # This avoids using the already-closed signal candle price
    # as an unrealistic future fill.

    entry_index = signal_index + 1

    if entry_index >= len(df):
        return None

    entry_price = safe_float(
        df.iloc[entry_index]["open"]
    )

    if not math.isfinite(entry_price):
        return None

    if entry_price <= 0:
        return None

    if side == "LONG":

        stop_price = (
            entry_price
            *
            (1 - STOP_LOSS_PCT / 100)
        )

        target_price = (
            entry_price
            *
            (1 + target_pct / 100)
        )

    else:

        stop_price = (
            entry_price
            *
            (1 + STOP_LOSS_PCT / 100)
        )

        target_price = (
            entry_price
            *
            (1 - target_pct / 100)
        )

    last_index = min(
        entry_index + MAX_HOLD_BARS,
        len(df) - 1
    )

    exit_reason = None
    exit_price = None
    exit_index = None

    for i in range(
        entry_index,
        last_index + 1
    ):

        high = safe_float(
            df.iloc[i]["high"]
        )

        low = safe_float(
            df.iloc[i]["low"]
        )

        if not (
            math.isfinite(high)
            and
            math.isfinite(low)
        ):
            continue

        if side == "LONG":

            stop_hit = (
                low <= stop_price
            )

            target_hit = (
                high >= target_price
            )

        else:

            stop_hit = (
                high >= stop_price
            )

            target_hit = (
                low <= target_price
            )

        # Same bar ambiguity
        if stop_hit and target_hit:

            if SAME_BAR_POLICY == "SL_FIRST":

                exit_reason = "SL"
                exit_price = stop_price

            else:

                exit_reason = "TP"
                exit_price = target_price

            exit_index = i
            break

        if stop_hit:

            exit_reason = "SL"
            exit_price = stop_price
            exit_index = i
            break

        if target_hit:

            exit_reason = "TP"
            exit_price = target_price
            exit_index = i
            break

    # Time exit
    if exit_reason is None:

        exit_index = last_index

        exit_price = safe_float(
            df.iloc[exit_index]["close"]
        )

        exit_reason = "TIME"

    if side == "LONG":

        gross_return = (
            (
                exit_price
                -
                entry_price
            )
            /
            entry_price
            *
            100
        )

    else:

        gross_return = (
            (
                entry_price
                -
                exit_price
            )
            /
            entry_price
            *
            100
        )

    net_return = (
        gross_return
        -
        ROUND_TRIP_COST_PCT
    )

    if exit_reason == "TP":
        outcome = "WIN"

    elif exit_reason == "SL":
        outcome = "LOSS"

    else:

        if net_return > 0:
            outcome = "WIN"
        else:
            outcome = "LOSS"

    signal_row = df.iloc[signal_index]

    return {

        "signal_time":
            signal_row["datetime"],

        "entry_time":
            df.iloc[entry_index]["datetime"],

        "exit_time":
            df.iloc[exit_index]["datetime"],

        "side":
            side,

        "entry_price":
            entry_price,

        "stop_price":
            stop_price,

        "target_price":
            target_price,

        "exit_price":
            exit_price,

        "exit_reason":
            exit_reason,

        "outcome":
            outcome,

        "gross_return_pct":
            gross_return,

        "net_return_pct":
            net_return,

        "rsi":
            safe_float(
                signal_row["rsi"]
            ),

        "macd_hist":
            safe_float(
                signal_row["macd_hist"]
            ),

        "bb_squeeze":
            bool(
                signal_row["bb_squeeze"]
            ),

        "bb_width_pct":
            safe_float(
                signal_row["bb_width_pct"]
            ),

        "volume_ratio":
            safe_float(
                signal_row["volume_ratio"]
            ),

        "atr_pct":
            safe_float(
                signal_row["atr_pct"]
            ),
    }


# ============================================================
# BACKTEST ONE STRATEGY
# ============================================================

def backtest_strategy(
    pair,
    df,
    strategy_name,
    side,
    rr_name,
    target_pct
):

    trades = []

    signal_values = (
        df[strategy_name]
        .fillna(False)
        .to_numpy()
    )

    # Start after indicators have enough history
    start_index = 250

    # Need room for entry candle
    end_index = len(df) - 2

    last_trade_exit_time = None

    for i in range(
        start_index,
        end_index + 1
    ):

        if not signal_values[i]:
            continue

        # Avoid entering repeatedly on every candle
        # of the same continuous condition.
        if i > start_index:

            if signal_values[i - 1]:
                continue

        signal_time = df.iloc[i]["datetime"]

        # One position per strategy/RR at a time
        if last_trade_exit_time is not None:

            if signal_time <= last_trade_exit_time:
                continue

        trade = simulate_trade(
            df,
            i,
            side,
            target_pct
        )

        if trade is None:
            continue

        trade["pair"] = pair
        trade["strategy"] = strategy_name
        trade["rr"] = rr_name
        trade["target_pct"] = target_pct
        trade["stop_loss_pct"] = STOP_LOSS_PCT

        trades.append(trade)

        last_trade_exit_time = trade["exit_time"]

    return trades


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_max_drawdown(returns):

    if len(returns) == 0:
        return np.nan

    equity = (
        pd.Series(returns)
        .fillna(0)
        .cumsum()
    )

    peak = equity.cummax()

    drawdown = equity - peak

    return abs(
        safe_float(
            drawdown.min(),
            0
        )
    )


def calculate_performance(
    pair,
    strategy,
    rr,
    trades
):

    if not trades:
        return None

    df = pd.DataFrame(trades)

    total = len(df)

    wins = int(
        (df["outcome"] == "WIN").sum()
    )

    losses = int(
        (df["outcome"] == "LOSS").sum()
    )

    win_rate = (
        wins
        /
        total
        *
        100
    )

    net_profit = (
        df["net_return_pct"].sum()
    )

    average_net = (
        df["net_return_pct"].mean()
    )

    gross_profit = (
        df.loc[
            df["net_return_pct"] > 0,
            "net_return_pct"
        ]
        .sum()
    )

    gross_loss = abs(
        df.loc[
            df["net_return_pct"] < 0,
            "net_return_pct"
        ]
        .sum()
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

    max_drawdown = (
        calculate_max_drawdown(
            df["net_return_pct"]
        )
    )

    squeeze_trades = int(
        df["bb_squeeze"].sum()
    )

    return {

        "pair":
            pair,

        "strategy":
            strategy,

        "rr":
            rr,

        "total_trades":
            total,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate_pct":
            round(win_rate, 2),

        "net_profit_pct":
            round(net_profit, 2),

        "avg_net_return_pct":
            round(average_net, 4),

        "profit_factor":
            round(profit_factor, 3),

        "max_drawdown_pct_points":
            round(max_drawdown, 2),

        "bb_squeeze_trades":
            squeeze_trades,
    }


# ============================================================
# TEST ONE COIN
# ============================================================

def test_coin(
    pair,
    coin_number,
    total_coins
):

    print()
    print(
        f"[{coin_number}/{total_coins}] "
        f"Testing {pair}"
    )

    try:

        df = fetch_history(
            pair,
            HISTORY_DAYS
        )

    except Exception as exc:

        print(
            f"    DOWNLOAD FAILED: {exc}"
        )

        return [], []

    if df.empty:

        print("    NO DATA")

        return [], []

    print(
        f"    Candles: {len(df):,}"
    )

    # Require meaningful history
    if len(df) < 3000:

        print(
            "    SKIPPED - insufficient historical candles"
        )

        return [], []

    df = calculate_indicators(df)

    df = add_market_regime(df)

    df = build_signal_columns(df)

    all_coin_trades = []

    all_coin_results = []

    for strategy_name, side in STRATEGIES.items():

        for rr_name, target_pct in RR_SETTINGS.items():

            trades = backtest_strategy(
                pair=pair,
                df=df,
                strategy_name=strategy_name,
                side=side,
                rr_name=rr_name,
                target_pct=target_pct,
            )

            all_coin_trades.extend(
                trades
            )

            performance = (
                calculate_performance(
                    pair,
                    strategy_name,
                    rr_name,
                    trades,
                )
            )

            if performance is not None:

                all_coin_results.append(
                    performance
                )

    if all_coin_results:

        temp = pd.DataFrame(
            all_coin_results
        )

        best = temp.sort_values(
            by=[
                "win_rate_pct",
                "net_profit_pct",
                "profit_factor",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        ).iloc[0]

        print(
            f"    BEST: "
            f"{best['strategy']} | "
            f"{best['rr']} | "
            f"Trades {int(best['total_trades'])} | "
            f"WR {best['win_rate_pct']:.2f}% | "
            f"Net {best['net_profit_pct']:.2f}% | "
            f"PF {best['profit_factor']:.2f}"
        )

    else:

        print(
            "    NO QUALIFYING SIGNALS"
        )

    return (
        all_coin_trades,
        all_coin_results
    )


# ============================================================
# RANK RESULTS
# ============================================================

def rank_results(result_df):

    if result_df.empty:
        return result_df

    ranked = result_df.copy()

    ranked["enough_trades"] = (
        ranked["total_trades"]
        >=
        MIN_TRADES_FOR_RANKING
    )

    ranked["positive_net"] = (
        ranked["net_profit_pct"]
        >
        0
    )

    ranked["target_win_rate"] = (
        ranked["win_rate_pct"]
        >=
        TARGET_WIN_RATE_MIN
    )

    ranked["good_profit_factor"] = (
        ranked["profit_factor"]
        >=
        MIN_PROFIT_FACTOR
    )

    ranked["V19_PASS"] = (
        ranked["enough_trades"]
        &
        ranked["positive_net"]
        &
        ranked["target_win_rate"]
        &
        ranked["good_profit_factor"]
    )

    # --------------------------------------------------------
    # Ranking score
    #
    # Trade count is included so tiny sample strategies
    # do not automatically rank first.
    # --------------------------------------------------------

    ranked["trade_count_score"] = (
        np.minimum(
            ranked["total_trades"],
            100
        )
        /
        100
        *
        10
    )

    ranked["ranking_score"] = (
        ranked["win_rate_pct"]
        +
        np.minimum(
            ranked["profit_factor"],
            5
        )
        * 5
        +
        ranked["trade_count_score"]
        +
        np.maximum(
            ranked["net_profit_pct"],
            -50
        )
        * 0.10
        -
        ranked["max_drawdown_pct_points"]
        * 0.10
    )

    ranked = ranked.sort_values(
        by=[
            "V19_PASS",
            "ranking_score",
            "net_profit_pct",
            "total_trades",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "rank",
        range(
            1,
            len(ranked) + 1
        )
    )

    return ranked


# ============================================================
# BEST RESULT PER COIN
# ============================================================

def get_best_per_coin(ranked):

    if ranked.empty:
        return ranked

    eligible = ranked[
        ranked["total_trades"]
        >=
        MIN_TRADES_FOR_RANKING
    ].copy()

    if eligible.empty:
        return pd.DataFrame()

    best = (
        eligible
        .sort_values(
            by=[
                "V19_PASS",
                "ranking_score",
                "net_profit_pct",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .groupby(
            "pair",
            as_index=False
        )
        .first()
    )

    best = best.sort_values(
        by=[
            "V19_PASS",
            "ranking_score",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    best.insert(
        0,
        "coin_rank",
        range(
            1,
            len(best) + 1
        )
    )

    return best


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CRYPTO FUTURES HISTORICAL V19")
    print("MULTI-COIN STRATEGY DISCOVERY ENGINE")
    print("=" * 70)

    print(
        f"Historical Days : {HISTORY_DAYS}"
    )

    print(
        f"Timeframe       : {RESOLUTION} minute"
    )

    print(
        f"Stop Loss       : {STOP_LOSS_PCT:.2f}%"
    )

    print(
        "Risk Reward     : 1:2 and 1:3"
    )

    print(
        f"Trading Cost    : {ROUND_TRIP_COST_PCT:.2f}% round trip"
    )

    print(
        "Direction       : LONG + SHORT"
    )

    print(
        "Entry           : Next candle OPEN"
    )

    print(
        "Target          : 70-80%+ WR + Positive Net Profit"
    )

    print(
        "REAL ORDERS     : NO"
    )

    pairs = get_active_usdt_futures()

    if not pairs:

        print(
            "No active USDT Futures pairs found."
        )

        return

    all_trades = []

    all_results = []

    total_coins = len(pairs)

    for number, pair in enumerate(
        pairs,
        start=1
    ):

        try:

            trades, results = test_coin(
                pair,
                number,
                total_coins
            )

            all_trades.extend(
                trades
            )

            all_results.extend(
                results
            )

        except KeyboardInterrupt:
            raise

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

        time.sleep(REQUEST_SLEEP)

    # ========================================================
    # SAVE ALL TRADES
    # ========================================================

    if all_trades:

        trade_df = pd.DataFrame(
            all_trades
        )

        trade_df.to_csv(
            TRADE_FILE,
            index=False
        )

        print()
        print(
            f"Saved trades: {TRADE_FILE}"
        )

        print(
            f"Total simulated trades: "
            f"{len(trade_df):,}"
        )

    else:

        trade_df = pd.DataFrame()

        print(
            "No trades generated."
        )

    # ========================================================
    # RESULTS
    # ========================================================

    if not all_results:

        print(
            "No strategy results generated."
        )

        return

    result_df = pd.DataFrame(
        all_results
    )

    ranked = rank_results(
        result_df
    )

    ranked.to_csv(
        COIN_RESULT_FILE,
        index=False
    )

    best_coins = get_best_per_coin(
        ranked
    )

    best_coins.to_csv(
        BEST_RESULT_FILE,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("V19 FINAL RESULTS")
    print("=" * 70)

    print(
        f"Coins tested             : {total_coins}"
    )

    print(
        f"Strategy combinations    : {len(ranked)}"
    )

    pass_df = ranked[
        ranked["V19_PASS"] == True
    ]

    print(
        f"V19 PASS combinations    : {len(pass_df)}"
    )

    print()
    print(
        "PASS RULE:"
    )

    print(
        f"Win Rate >= {TARGET_WIN_RATE_MIN:.0f}%"
    )

    print(
        f"Trades >= {MIN_TRADES_FOR_RANKING}"
    )

    print(
        "Net Profit > 0"
    )

    print(
        f"Profit Factor >= {MIN_PROFIT_FACTOR:.2f}"
    )

    # ========================================================
    # TOP PASS RESULTS
    # ========================================================

    if not pass_df.empty:

        print()
        print("=" * 70)
        print("TOP V19 PASS STRATEGIES")
        print("=" * 70)

        columns = [
            "rank",
            "pair",
            "strategy",
            "rr",
            "total_trades",
            "win_rate_pct",
            "net_profit_pct",
            "profit_factor",
            "max_drawdown_pct_points",
        ]

        print(
            pass_df[
                columns
            ]
            .head(25)
            .to_string(
                index=False
            )
        )

    else:

        print()
        print("=" * 70)
        print(
            "NO STRATEGY MET ALL V19 PASS RULES"
        )
        print("=" * 70)

        print(
            "This is a valid research result."
        )

        print(
            "We will NOT force a fake 70-80% win rate."
        )

    # ========================================================
    # BEST COINS
    # ========================================================

    if not best_coins.empty:

        print()
        print("=" * 70)
        print("TOP COINS - BEST STRATEGY PER COIN")
        print("=" * 70)

        columns = [
            "coin_rank",
            "pair",
            "strategy",
            "rr",
            "total_trades",
            "win_rate_pct",
            "net_profit_pct",
            "profit_factor",
            "max_drawdown_pct_points",
            "V19_PASS",
        ]

        print(
            best_coins[
                columns
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # BEST RR COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("RR COMPARISON")
    print("=" * 70)

    rr_summary = (
        ranked[
            ranked["total_trades"]
            >=
            MIN_TRADES_FOR_RANKING
        ]
        .groupby("rr")
        .agg(
            combinations=(
                "pair",
                "count"
            ),
            avg_win_rate=(
                "win_rate_pct",
                "mean"
            ),
            total_net_profit=(
                "net_profit_pct",
                "sum"
            ),
            avg_profit_factor=(
                "profit_factor",
                "mean"
            ),
        )
        .reset_index()
    )

    if not rr_summary.empty:

        print(
            rr_summary.to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(TRADE_FILE)
    print(COIN_RESULT_FILE)
    print(BEST_RESULT_FILE)

    print()
    print(
        "V19 historical research complete."
    )

    print(
        "NO REAL ORDERS WERE PLACED."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
