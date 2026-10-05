import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# COINDCX 5-MIN BUY / SELL HISTORICAL SCANNER
# ============================================================

BASE_URL = "https://api.coindcx.com"

MIN_24H_VOLUME = 5_000_000
MAX_WORKERS = 8

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

TELEGRAM_ENABLED = True

MODEL_FILE = "crypto_6month_patterns.json"
ALERT_FILE = "telegram_early_alerted_signals.csv"
OUTPUT_FILE = "crypto_5m_signals.csv"

HISTORY_DAYS = 180

HIST_FORWARD_BARS = 8
HIST_SPIKE_PCT = 3.0

MIN_HISTORICAL_EVENTS = 5
MIN_MATCH_SCORE = 70.0

MODEL_REFRESH_HOURS = 24

MAX_CURRENT_CANDLE_MOVE = 2.5


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_ENABLED:
        return False

    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram BOT_TOKEN / CHAT_ID missing.")
        return False

    try:

        url = (
            f"https://api.telegram.org/"
            f"bot{BOT_TOKEN}/sendMessage"
        )

        r = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=20
        )

        if r.status_code == 200:
            print("Telegram alert sent.")
            return True

        print(
            "Telegram error:",
            r.text[:300]
        )

    except Exception as e:

        print(
            "Telegram exception:",
            e
        )

    return False


# ============================================================
# COINDCX USDT MARKETS
# MINIMUM $5 MILLION 24H VOLUME
# ============================================================

def get_symbols():

    print(
        "Downloading CoinDCX market "
        "details and 24H ticker..."
    )

    try:

        market_res = requests.get(
            f"{BASE_URL}/exchange/v1/markets_details",
            timeout=30
        )

        market_res.raise_for_status()

        markets = market_res.json()

        ticker_res = requests.get(
            f"{BASE_URL}/exchange/ticker",
            timeout=30
        )

        ticker_res.raise_for_status()

        tickers = ticker_res.json()

    except Exception as e:

        print(
            "Market/Ticker API error:",
            e
        )

        return []

    # --------------------------------------------------------
    # Ticker mapping
    #
    # ticker:
    # market = BTCUSDT
    #
    # markets_details:
    # coindcx_name = BTCUSDT
    # pair = B-BTC_USDT
    # --------------------------------------------------------

    ticker_map = {}

    for t in tickers:

        try:

            market_name = str(
                t.get("market") or ""
            ).upper().strip()

            volume = float(
                t.get("volume") or 0
            )

            last_price = float(
                t.get("last_price") or 0
            )

            if not market_name:
                continue

            if last_price <= 0:
                continue

            # approximate 24h turnover in USDT
            quote_volume_usdt = (
                volume * last_price
            )

            ticker_map[market_name] = {
                "volume": volume,
                "last_price": last_price,
                "quote_volume_usdt":
                    quote_volume_usdt
            }

        except Exception:
            continue

    print(
        "Ticker markets received:",
        len(ticker_map)
    )

    result = []
    seen = set()

    for m in markets:

        try:

            status = str(
                m.get("status") or ""
            ).lower().strip()

            base_currency = str(
                m.get(
                    "base_currency_short_name"
                ) or ""
            ).upper().strip()

            symbol = str(
                m.get("coindcx_name")
                or
                m.get("symbol")
                or ""
            ).upper().strip()

            pair = str(
                m.get("pair") or ""
            ).strip()

            if status != "active":
                continue

            # CoinDCX USDT markets
            if base_currency != "USDT":
                continue

            if not symbol:
                continue

            if not pair:
                continue

            ticker = ticker_map.get(
                symbol
            )

            if ticker is None:
                continue

            volume_24h_usdt = float(
                ticker[
                    "quote_volume_usdt"
                ]
            )

            if (
                volume_24h_usdt
                <
                MIN_24H_VOLUME
            ):
                continue

            if symbol in seen:
                continue

            seen.add(symbol)

            result.append({
                "symbol": symbol,
                "pair": pair,
                "volume_24h":
                    volume_24h_usdt
            })

        except Exception:
            continue

    result.sort(
        key=lambda x:
            x["volume_24h"],
        reverse=True
    )

    print(
        "USDT markets with >= "
        "$5M 24H volume:",
        len(result)
    )

    print(
        "\nTop qualifying markets:"
    )

    for x in result[:10]:

        print(
            x["symbol"],
            "| Pair:",
            x["pair"],
            "| Volume: $"
            +
            f"{x['volume_24h']/1_000_000:.2f}M"
        )

    return result


# ============================================================
# DOWNLOAD CANDLES
# ============================================================

def request_candles(
    pair,
    interval,
    start_ms=None,
    end_ms=None,
    limit=1000
):

    params = {
        "pair": pair,
        "interval": interval,
        "limit": limit
    }

    if start_ms is not None:

        params["startTime"] = int(
            start_ms
        )

    if end_ms is not None:

        params["endTime"] = int(
            end_ms
        )

    try:

        r = requests.get(
            f"{BASE_URL}/market_data/candles",
            params=params,
            timeout=30
        )

        if r.status_code != 200:
            return None

        data = r.json()

        if not isinstance(
            data,
            list
        ):
            return None

        if len(data) == 0:
            return None

        df = pd.DataFrame(data)

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "time"
        ]

        if not all(
            x in df.columns
            for x in required
        ):
            return None

        for c in required:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

        df = (
            df
            .dropna(
                subset=required
            )
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

        return df

    except Exception:
        return None


# ============================================================
# SIX MONTH 15-MINUTE HISTORY
# ============================================================

def get_six_month_history(pair):

    now = datetime.now(
        timezone.utc
    )

    end_ms = int(
        now.timestamp()
        * 1000
    )

    start_dt = (
        now
        -
        timedelta(
            days=HISTORY_DAYS
        )
    )

    start_ms = int(
        start_dt.timestamp()
        * 1000
    )

    all_frames = []

    # 1000 x 15-minute candles
    chunk_ms = (
        1000
        * 15
        * 60
        * 1000
    )

    cursor = start_ms

    while cursor < end_ms:

        chunk_end = min(
            cursor + chunk_ms,
            end_ms
        )

        df = request_candles(
            pair=pair,
            interval="15m",
            start_ms=cursor,
            end_ms=chunk_end,
            limit=1000
        )

        if (
            df is not None
            and
            not df.empty
        ):

            all_frames.append(
                df
            )

        cursor = (
            chunk_end + 1
        )

        time.sleep(0.05)

    if not all_frames:
        return None

    df = pd.concat(
        all_frames,
        ignore_index=True
    )

    df = (
        df
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# CREATE CLOSED 5-MINUTE CANDLES FROM 1-MINUTE DATA
# ============================================================

def get_live_5m(pair):

    df = request_candles(
        pair=pair,
        interval="1m",
        limit=300
    )

    if df is None:
        return None

    if len(df) < 100:
        return None

    df["datetime"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True
    )

    df = df.set_index(
        "datetime"
    )

    five = df.resample(
        "5min",
        label="left",
        closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    five = (
        five
        .dropna()
        .reset_index()
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    current_bucket = (
        now.floor("5min")
    )

    # only completed candles
    five = five[
        five["datetime"]
        <
        current_bucket
    ].copy()

    if len(five) < 55:
        return None

    five["time"] = (
        five["datetime"]
        .astype("int64")
        //
        10**6
    )

    return five.reset_index(
        drop=True
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    close,
    period=14
):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    return (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # RSI
    df["RSI14"] = calculate_rsi(
        df["close"],
        14
    )

    # MACD
    ema12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_HIST"] = (
        df["MACD"]
        -
        df["MACD_SIGNAL"]
    )

    # Bollinger Bands
    df["BB_MIDDLE"] = (
        df["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        df["close"]
        .rolling(20)
        .std()
    )

    df["BB_UPPER"] = (
        df["BB_MIDDLE"]
        +
        2 * bb_std
    )

    df["BB_LOWER"] = (
        df["BB_MIDDLE"]
        -
        2 * bb_std
    )

    df["BB_WIDTH"] = (
        (
            df["BB_UPPER"]
            -
            df["BB_LOWER"]
        )
        /
        df["BB_MIDDLE"]
    )

    # Volume
    df["VOLUME_AVG20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["VOLUME_RATIO"] = (
        df["volume"]
        /
        df["VOLUME_AVG20"]
    )

    return df


# ============================================================
# VOLUME PROFILE / POC
# ============================================================

def calculate_poc(
    df,
    bins=40
):

    if df is None:
        return np.nan

    if len(df) < 20:
        return np.nan

    low = float(
        df["low"].min()
    )

    high = float(
        df["high"].max()
    )

    if not np.isfinite(low):
        return np.nan

    if not np.isfinite(high):
        return np.nan

    if high <= low:
        return np.nan

    edges = np.linspace(
        low,
        high,
        bins + 1
    )

    profile = np.zeros(
        bins
    )

    for _, candle in df.iterrows():

        candle_low = float(
            candle["low"]
        )

        candle_high = float(
            candle["high"]
        )

        volume = float(
            candle["volume"]
        )

        if (
            candle_high
            <=
            candle_low
        ):
            continue

        for i in range(bins):

            overlap = max(
                0,
                min(
                    candle_high,
                    edges[i + 1]
                )
                -
                max(
                    candle_low,
                    edges[i]
                )
            )

            if overlap > 0:

                profile[i] += (
                    volume
                    *
                    overlap
                    /
                    (
                        candle_high
                        -
                        candle_low
                    )
                )

    if profile.sum() <= 0:
        return np.nan

    idx = int(
        np.argmax(
            profile
        )
    )

    poc = (
        edges[idx]
        +
        edges[idx + 1]
    ) / 2

    return float(poc)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def get_features(
    df,
    index
):

    if index < 50:
        return None

    row = df.iloc[index]

    prev = df.iloc[
        index - 1
    ]

    poc_start = max(
        0,
        index - 100
    )

    poc = calculate_poc(
        df.iloc[
            poc_start:
            index + 1
        ]
    )

    if not np.isfinite(poc):
        return None

    if not np.isfinite(
        row["RSI14"]
    ):
        return None

    if not np.isfinite(
        row["VOLUME_RATIO"]
    ):
        return None

    price = float(
        row["close"]
    )

    open_price = float(
        row["open"]
    )

    if price <= 0:
        return None

    if open_price <= 0:
        return None

    candle_move = (
        (
            price
            -
            open_price
        )
        /
        open_price
        *
        100
    )

    macd_rising = (
        1.0
        if
        row["MACD_HIST"]
        >
        prev["MACD_HIST"]
        else
        0.0
    )

    return {

        "rsi":
            float(
                row["RSI14"]
            ),

        "macd_hist_pct":
            (
                float(
                    row["MACD_HIST"]
                )
                /
                price
                *
                100
            ),

        "macd_rising":
            macd_rising,

        "ema_gap_pct":
            (
                (
                    float(
                        row["EMA20"]
                    )
                    -
                    float(
                        row["EMA50"]
                    )
                )
                /
                price
                *
                100
            ),

        "price_vs_bb_pct":
            (
                (
                    price
                    -
                    float(
                        row["BB_MIDDLE"]
                    )
                )
                /
                price
                *
                100
            ),

        "bb_width_pct":
            (
                float(
                    row["BB_WIDTH"]
                )
                *
                100
            ),

        "volume_ratio":
            float(
                row["VOLUME_RATIO"]
            ),

        "price_vs_poc_pct":
            (
                (
                    price
                    -
                    poc
                )
                /
                poc
                *
                100
            ),

        "candle_move_pct":
            float(
                candle_move
            ),

        "poc":
            float(poc)
    }


# ============================================================
# HISTORICAL SPIKE / DROP EVENTS
# ============================================================

def historical_events(df):

    df = add_indicators(
        df
    )

    buy_events = []
    sell_events = []

    last_buy_index = -999
    last_sell_index = -999

    end = (
        len(df)
        -
        HIST_FORWARD_BARS
        -
        1
    )

    for i in range(
        55,
        end
    ):

        current_price = float(
            df.iloc[i]["close"]
        )

        if current_price <= 0:
            continue

        future = df.iloc[
            i + 1:
            i + 1 + HIST_FORWARD_BARS
        ]

        if future.empty:
            continue

        future_high = float(
            future["high"].max()
        )

        future_low = float(
            future["low"].min()
        )

        future_up = (
            (
                future_high
                -
                current_price
            )
            /
            current_price
            *
            100
        )

        future_down = (
            (
                future_low
                -
                current_price
            )
            /
            current_price
            *
            100
        )

        feat = get_features(
            df,
            i
        )

        if feat is None:
            continue

        # BUY event
        if (
            future_up
            >=
            HIST_SPIKE_PCT
        ):

            if (
                i - last_buy_index
                >=
                HIST_FORWARD_BARS
            ):

                if (
                    feat[
                        "candle_move_pct"
                    ]
                    <=
                    MAX_CURRENT_CANDLE_MOVE
                ):

                    buy_events.append(
                        feat
                    )

                    last_buy_index = i

        # SELL event
        if (
            future_down
            <=
            -HIST_SPIKE_PCT
        ):

            if (
                i - last_sell_index
                >=
                HIST_FORWARD_BARS
            ):

                if (
                    feat[
                        "candle_move_pct"
                    ]
                    >=
                    -MAX_CURRENT_CANDLE_MOVE
                ):

                    sell_events.append(
                        feat
                    )

                    last_sell_index = i

    return (
        buy_events,
        sell_events
    )


# ============================================================
# HISTORICAL MODEL
# ============================================================

MODEL_FEATURES = [

    "rsi",
    "macd_hist_pct",
    "ema_gap_pct",
    "price_vs_bb_pct",
    "bb_width_pct",
    "volume_ratio",
    "price_vs_poc_pct",
    "candle_move_pct"

]


def summarize_events(
    events
):

    if (
        len(events)
        <
        MIN_HISTORICAL_EVENTS
    ):
        return None

    result = {
        "count":
            len(events)
    }

    for key in MODEL_FEATURES:

        values = []

        for x in events:

            value = x.get(key)

            if (
                value is not None
                and
                np.isfinite(value)
            ):
                values.append(
                    value
                )

        if (
            len(values)
            <
            MIN_HISTORICAL_EVENTS
        ):
            continue

        arr = np.array(
            values,
            dtype=float
        )

        median = float(
            np.median(arr)
        )

        mad = float(
            np.median(
                np.abs(
                    arr - median
                )
            )
        )

        minimum_scale = {

            "rsi": 5.0,

            "macd_hist_pct":
                0.02,

            "ema_gap_pct":
                0.15,

            "price_vs_bb_pct":
                0.20,

            "bb_width_pct":
                0.30,

            "volume_ratio":
                0.25,

            "price_vs_poc_pct":
                0.50,

            "candle_move_pct":
                0.20

        }.get(
            key,
            0.1
        )

        scale = max(
            mad * 1.4826,
            minimum_scale
        )

        result[key] = {
            "median":
                median,
            "scale":
                scale
        }

    return result


# ============================================================
# LEARN ONE SYMBOL
# ============================================================

def learn_symbol(
    symbol,
    pair
):

    print(
        f"\nLearning 6-month history: "
        f"{symbol}"
    )

    df = get_six_month_history(
        pair
    )

    if df is None:

        print(
            f"{symbol}: "
            f"historical data unavailable"
        )

        return None

    print(
        f"{symbol}: "
        f"{len(df)} historical candles"
    )

    if len(df) < 500:

        print(
            f"{symbol}: "
            f"insufficient historical candles"
        )

        return None

    buys, sells = historical_events(
        df
    )

    buy_model = summarize_events(
        buys
    )

    sell_model = summarize_events(
        sells
    )

    print(
        f"{symbol}: "
        f"BUY events={len(buys)} | "
        f"SELL events={len(sells)}"
    )

    return {

        "pair":
            pair,

        "buy":
            buy_model,

        "sell":
            sell_model
    }


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_FILE
    ):
        return None

    try:

        with open(
            MODEL_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            model = json.load(f)

        created_text = model.get(
            "created_utc"
        )

        if not created_text:
            return None

        created = pd.to_datetime(
            created_text,
            utc=True
        )

        age_hours = (
            pd.Timestamp.now(
                tz="UTC"
            )
            -
            created
        ).total_seconds() / 3600

        if (
            age_hours
            >
            MODEL_REFRESH_HOURS
        ):
            return None

        return model

    except Exception as e:

        print(
            "Model load error:",
            e
        )

        return None


# ============================================================
# BUILD 6-MONTH MODEL
# ============================================================

def build_model(
    markets
):

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "BUILDING 6-MONTH "
        "HISTORICAL MODEL"
    )

    print(
        "=" * 70
    )

    symbols_model = {}

    for n, market in enumerate(
        markets,
        1
    ):

        symbol = market[
            "symbol"
        ]

        pair = market[
            "pair"
        ]

        print(
            f"\n[{n}/{len(markets)}] "
            f"{symbol}"
        )

        try:

            learned = learn_symbol(
                symbol,
                pair
            )

            if learned is not None:

                symbols_model[
                    symbol
                ] = learned

        except Exception as e:

            print(
                f"{symbol} "
                f"learning error: {e}"
            )

    model = {

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "history_days":
            HISTORY_DAYS,

        "spike_pct":
            HIST_SPIKE_PCT,

        "symbols":
            symbols_model
    }

    with open(
        MODEL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            model,
            f,
            indent=2
        )

    print(
        "\nHistorical model saved:",
        MODEL_FILE
    )

    print(
        "Symbols learned:",
        len(symbols_model)
    )

    return model


# ============================================================
# HISTORICAL SIMILARITY
# ============================================================

def similarity_score(
    features,
    model
):

    if model is None:
        return 0.0

    scores = []

    for key in MODEL_FEATURES:

        if key not in model:
            continue

        value = features.get(
            key
        )

        if value is None:
            continue

        if not np.isfinite(
            value
        ):
            continue

        median = float(
            model[key][
                "median"
            ]
        )

        scale = float(
            model[key][
                "scale"
            ]
        )

        if scale <= 0:
            continue

        distance = (
            abs(
                value
                -
                median
            )
            /
            scale
        )

        score = max(
            0.0,
            100.0
            *
            (
                1.0
                -
                distance / 3.0
            )
        )

        scores.append(
            score
        )

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


# ============================================================
# LIVE ANALYSIS
# ============================================================

def analyze_live(
    market,
    historical_model
):

    symbol = market[
        "symbol"
    ]

    pair = market[
        "pair"
    ]

    df = get_live_5m(
        pair
    )

    if df is None:
        return None

    df = add_indicators(
        df
    )

    i = len(df) - 1

    features = get_features(
        df,
        i
    )

    if features is None:
        return None

    row = df.iloc[i]

    prev = df.iloc[
        i - 1
    ]

    candle_move = float(
        features[
            "candle_move_pct"
        ]
    )

    # Do not chase an already extended candle
    if (
        abs(candle_move)
        >
        MAX_CURRENT_CANDLE_MOVE
    ):
        return None

    buy_model = (
        historical_model.get(
            "buy"
        )
        if historical_model
        else None
    )

    sell_model = (
        historical_model.get(
            "sell"
        )
        if historical_model
        else None
    )

    buy_match = similarity_score(
        features,
        buy_model
    )

    sell_match = similarity_score(
        features,
        sell_model
    )

    price = float(
        row["close"]
    )

    # ========================================================
    # BUY CONDITIONS
    # ========================================================

    buy_rsi = (
        42
        <=
        row["RSI14"]
        <=
        70
    )

    buy_macd = (
        row["MACD_HIST"]
        >
        0
        and
        row["MACD_HIST"]
        >=
        prev["MACD_HIST"]
    )

    buy_ema = (

        row["EMA20"]
        >=
        row["EMA50"]

        or

        (
            row["EMA20"]
            >
            prev["EMA20"]

            and

            price
            >
            row["EMA20"]
        )
    )

    buy_bb = (
        price
        >=
        row["BB_MIDDLE"]

        and

        price
        <=
        row["BB_UPPER"]
        *
        1.01
    )

    buy_volume = (
        row["VOLUME_RATIO"]
        >=
        1.0
    )

    buy_poc = (
        features[
            "price_vs_poc_pct"
        ]
        >=
        0
    )

    buy_checks = [

        buy_rsi,
        buy_macd,
        buy_ema,
        buy_bb,
        buy_volume,
        buy_poc

    ]

    buy_confirmations = sum(
        bool(x)
        for x in buy_checks
    )

    # ========================================================
    # SELL CONDITIONS
    # ========================================================

    sell_rsi = (
        30
        <=
        row["RSI14"]
        <=
        58
    )

    sell_macd = (
        row["MACD_HIST"]
        <
        0

        and

        row["MACD_HIST"]
        <=
        prev["MACD_HIST"]
    )

    sell_ema = (

        row["EMA20"]
        <=
        row["EMA50"]

        or

        (
            row["EMA20"]
            <
            prev["EMA20"]

            and

            price
            <
            row["EMA20"]
        )
    )

    sell_bb = (
        price
        <=
        row["BB_MIDDLE"]

        and

        price
        >=
        row["BB_LOWER"]
        *
        0.99
    )

    sell_volume = (
        row["VOLUME_RATIO"]
        >=
        1.0
    )

    sell_poc = (
        features[
            "price_vs_poc_pct"
        ]
        <=
        0
    )

    sell_checks = [

        sell_rsi,
        sell_macd,
        sell_ema,
        sell_bb,
        sell_volume,
        sell_poc

    ]

    sell_confirmations = sum(
        bool(x)
        for x in sell_checks
    )

    signal = None
    match_score = 0.0
    confirmations = 0

    # ========================================================
    # DIRECT BUY CONFIRMATION
    # ========================================================

    if (

        buy_match
        >=
        MIN_MATCH_SCORE

        and

        buy_confirmations
        >=
        5

        and

        buy_match
        >
        sell_match

    ):

        signal = "BUY"

        match_score = (
            buy_match
        )

        confirmations = (
            buy_confirmations
        )

    # ========================================================
    # DIRECT SELL CONFIRMATION
    # ========================================================

    elif (

        sell_match
        >=
        MIN_MATCH_SCORE

        and

        sell_confirmations
        >=
        5

        and

        sell_match
        >
        buy_match

    ):

        signal = "SELL"

        match_score = (
            sell_match
        )

        confirmations = (
            sell_confirmations
        )

    if signal is None:
        return None

    candle_time = pd.to_datetime(
        row["time"],
        unit="ms",
        utc=True
    )

    close_time = (
        candle_time
        +
        pd.Timedelta(
            minutes=5
        )
    )

    return {

        "SYMBOL":
            symbol,

        "PAIR":
            pair,

        "SIGNAL":
            signal,

        "TIME":
            close_time.isoformat(),

        "PRICE":
            round(
                price,
                8
            ),

        "HISTORICAL_MATCH":
            round(
                match_score,
                1
            ),

        "CONFIRMATIONS":
            confirmations,

        "RSI14":
            round(
                float(
                    row["RSI14"]
                ),
                2
            ),

        "MACD_HIST":
            round(
                float(
                    row["MACD_HIST"]
                ),
                8
            ),

        "EMA20":
            round(
                float(
                    row["EMA20"]
                ),
                8
            ),

        "EMA50":
            round(
                float(
                    row["EMA50"]
                ),
                8
            ),

        "VOLUME_RATIO":
            round(
                float(
                    row["VOLUME_RATIO"]
                ),
                2
            ),

        "POC":
            round(
                float(
                    features["poc"]
                ),
                8
            ),

        "PRICE_VS_POC_PCT":
            round(
                float(
                    features[
                        "price_vs_poc_pct"
                    ]
                ),
                2
            ),

        "24H_VOLUME_USDT":
            round(
                float(
                    market[
                        "volume_24h"
                    ]
                ),
                2
            )
    }


# ============================================================
# DUPLICATE ALERT CHECK
# ============================================================

def already_alerted(
    symbol,
    signal,
    candle_time
):

    if not os.path.exists(
        ALERT_FILE
    ):
        return False

    try:

        old = pd.read_csv(
            ALERT_FILE
        )

        if old.empty:
            return False

        required = {
            "SYMBOL",
            "SIGNAL",
            "TIME"
        }

        if not required.issubset(
            old.columns
        ):
            return False

        found = (

            (
                old["SYMBOL"]
                .astype(str)
                ==
                str(symbol)
            )

            &

            (
                old["SIGNAL"]
                .astype(str)
                ==
                str(signal)
            )

            &

            (
                old["TIME"]
                .astype(str)
                ==
                str(candle_time)
            )

        )

        return bool(
            found.any()
        )

    except Exception:
        return False


# ============================================================
# SAVE ALERT
# ============================================================

def save_alert(row):

    new = pd.DataFrame(
        [{
            "SYMBOL":
                row["SYMBOL"],

            "SIGNAL":
                row["SIGNAL"],

            "TIME":
                row["TIME"]
        }]
    )

    if os.path.exists(
        ALERT_FILE
    ):

        try:

            old = pd.read_csv(
                ALERT_FILE
            )

            new = pd.concat(
                [
                    old,
                    new
                ],
                ignore_index=True
            )

        except Exception:
            pass

    new = new.drop_duplicates(
        subset=[
            "SYMBOL",
            "SIGNAL",
            "TIME"
        ],
        keep="last"
    )

    new.to_csv(
        ALERT_FILE,
        index=False
    )


# ============================================================
# TELEGRAM BUY / SELL MESSAGE
# ============================================================

def process_alert(row):

    if already_alerted(
        row["SYMBOL"],
        row["SIGNAL"],
        row["TIME"]
    ):

        print(
            "Duplicate alert skipped:",
            row["SYMBOL"],
            row["SIGNAL"]
        )

        return

    utc_time = pd.to_datetime(
        row["TIME"],
        utc=True
    )

    ist_time = (
        utc_time
        +
        pd.Timedelta(
            hours=5,
            minutes=30
        )
    )

    time_text = (
        ist_time.strftime(
            "%d-%m-%Y %I:%M %p"
        )
    )

    volume_m = (
        row[
            "24H_VOLUME_USDT"
        ]
        /
        1_000_000
    )

    if (
        row["SIGNAL"]
        ==
        "BUY"
    ):

        title = (
            "🟢 BUY CONFIRMATION"
        )

    else:

        title = (
            "🔴 SELL CONFIRMATION"
        )

    message = (

        f"{title} - CoinDCX\n\n"

        f"Symbol: "
        f"{row['SYMBOL']}\n"

        f"Timeframe: 5 Minute\n"

        f"Time (IST): "
        f"{time_text}\n"

        f"Price: "
        f"{row['PRICE']}\n\n"

        f"6-Month Historical Match: "
        f"{row['HISTORICAL_MATCH']}%\n"

        f"Live Confirmations: "
        f"{row['CONFIRMATIONS']}/6\n\n"

        f"RSI(14): "
        f"{row['RSI14']}\n"

        f"MACD Histogram: "
        f"{row['MACD_HIST']}\n"

        f"EMA20: "
        f"{row['EMA20']}\n"

        f"EMA50: "
        f"{row['EMA50']}\n"

        f"Volume Ratio: "
        f"{row['VOLUME_RATIO']}x\n"

        f"POC: "
        f"{row['POC']}\n"

        f"Price vs POC: "
        f"{row['PRICE_VS_POC_PCT']}%\n"

        f"24H Volume: "
        f"${volume_m:.2f}M\n\n"

        f"Historical pattern + "
        f"current technical conditions "
        f"confirmed."
    )

    if send_telegram(
        message
    ):

        save_alert(
            row
        )


# ============================================================
# MAIN
# ============================================================

def run():

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "COINDCX 5-MIN BUY / SELL "
        "HISTORICAL CONFIRMATION SCANNER"
    )

    print(
        "=" * 70
    )

    print(
        "UTC:",
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print(
        "Minimum 24H volume: "
        "$5,000,000"
    )

    print(
        "Historical period: "
        "6 months"
    )

    print(
        "Historical candles: "
        "15 minute"
    )

    print(
        "Live timeframe: "
        "5 minutes"
    )

    print(
        "Pre-alert: OFF"
    )

    print(
        "Signals: "
        "BUY + SELL confirmation"
    )

    print(
        "\nGetting CoinDCX markets..."
    )

    markets = get_symbols()

    print(
        "\nQualifying markets:",
        len(markets)
    )

    if not markets:

        print(
            "\nERROR:"
        )

        print(
            "No USDT markets passed "
            "$5M volume filter."
        )

        print(
            "Scanner stopped before "
            "historical analysis."
        )

        return

    # ========================================================
    # LOAD OR BUILD 6-MONTH MODEL
    # ========================================================

    model = load_model()

    if model is None:

        print(
            "\nNo fresh historical "
            "model found."
        )

        print(
            "Starting 6-month "
            "historical learning..."
        )

        model = build_model(
            markets
        )

    else:

        print(
            "\nUsing saved "
            "6-month historical model."
        )

        print(
            "Model created:",
            model.get(
                "created_utc"
            )
        )

    model_symbols = (
        model.get(
            "symbols",
            {}
        )
    )

    print(
        "Historical models available:",
        len(model_symbols)
    )

    if not model_symbols:

        print(
            "No usable historical "
            "models were created."
        )

        return

    # ========================================================
    # LIVE 5-MINUTE SCAN
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "STARTING LIVE CLOSED "
        "5-MINUTE SCAN"
    )

    print(
        "=" * 70
    )

    signals = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {}

        for market in markets:

            symbol = (
                market[
                    "symbol"
                ]
            )

            hist = (
                model_symbols.get(
                    symbol
                )
            )

            if hist is None:
                continue

            future = executor.submit(
                analyze_live,
                market,
                hist
            )

            futures[
                future
            ] = symbol

        total = len(
            futures
        )

        print(
            "Markets being scanned:",
            total
        )

        for n, future in enumerate(
            as_completed(
                futures
            ),
            1
        ):

            symbol = futures[
                future
            ]

            print(
                f"Scanning "
                f"{n}/{total}: "
                f"{symbol}"
            )

            try:

                result = (
                    future.result()
                )

                if result is not None:

                    signals.append(
                        result
                    )

                    print(
                        f">>> "
                        f"{result['SIGNAL']} "
                        f"CONFIRMATION: "
                        f"{symbol} | "
                        f"Historical Match "
                        f"{result['HISTORICAL_MATCH']}%"
                    )

                    process_alert(
                        result
                    )

            except Exception as e:

                print(
                    f"{symbol} "
                    f"live scan error: "
                    f"{e}"
                )

    # ========================================================
    # RESULTS
    # ========================================================

    if not signals:

        print(
            "\n"
            +
            "=" * 70
        )

        print(
            "NO BUY / SELL "
            "CONFIRMATION RIGHT NOW"
        )

        print(
            "=" * 70
        )

        print(
            "Scanner completed normally."
        )

        print(
            "Wait for the next "
            "5-minute scan."
        )

        return

    out = pd.DataFrame(
        signals
    )

    out = out.sort_values(

        [
            "HISTORICAL_MATCH",
            "CONFIRMATIONS",
            "VOLUME_RATIO"
        ],

        ascending=[
            False,
            False,
            False
        ]

    )

    out.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "CONFIRMATIONS FOUND:",
        len(out)
    )

    print(
        "=" * 70
    )

    print(
        out[
            [
                "SYMBOL",
                "SIGNAL",
                "HISTORICAL_MATCH",
                "CONFIRMATIONS",
                "RSI14",
                "VOLUME_RATIO",
                "24H_VOLUME_USDT"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        OUTPUT_FILE
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    try:

        run()

    except KeyboardInterrupt:

        print(
            "\nScanner stopped."
        )

    except Exception as e:

        print(
            "\nFATAL ERROR:",
            e
        )

        raise
