import os
import json
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# COINDCX FAST 5-MINUTE BUY / SELL SCANNER
# ============================================================

BASE_URL = "https://api.coindcx.com"

MODEL_FILE = "crypto_6month_patterns.json"
ALERT_FILE = "telegram_early_alerted_signals.csv"
OUTPUT_FILE = "crypto_5m_signals.csv"

MIN_24H_VOLUME = 5_000_000

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

MAX_WORKERS = 8

# Historical similarity required
MIN_MATCH_SCORE = 65.0

# At least 4 of 6 current conditions
MIN_CONFIRMATIONS = 4

# Avoid entering after an already huge 5m candle
MAX_CURRENT_CANDLE_MOVE = 2.5


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram BOT_TOKEN / CHAT_ID missing.")
        return False

    try:

        url = (
            f"https://api.telegram.org/"
            f"bot{BOT_TOKEN}/sendMessage"
        )

        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=20
        )

        if response.status_code == 200:
            print("Telegram alert sent.")
            return True

        print(
            "Telegram error:",
            response.text[:300]
        )

    except Exception as e:

        print(
            "Telegram exception:",
            e
        )

    return False


# ============================================================
# LOAD 6-MONTH HISTORICAL MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_FILE):

        print(
            "ERROR:",
            MODEL_FILE,
            "not found."
        )

        return None

    try:

        with open(
            MODEL_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            model = json.load(file)

        symbols = model.get(
            "symbols",
            {}
        )

        print(
            "Historical models loaded:",
            len(symbols)
        )

        print(
            "Historical model updated:",
            model.get(
                "updated_utc",
                "unknown"
            )
        )

        return model

    except Exception as e:

        print(
            "Model read error:",
            e
        )

        return None


# ============================================================
# GET CURRENT $5M+ COINDCX MARKETS
# ============================================================

def get_current_markets():

    try:

        markets_response = requests.get(
            f"{BASE_URL}/exchange/v1/markets_details",
            timeout=30
        )

        markets_response.raise_for_status()

        markets = markets_response.json()

        ticker_response = requests.get(
            f"{BASE_URL}/exchange/ticker",
            timeout=30
        )

        ticker_response.raise_for_status()

        tickers = ticker_response.json()

    except Exception as e:

        print(
            "CoinDCX market API error:",
            e
        )

        return {}

    ticker_map = {}

    for ticker in tickers:

        try:

            symbol = str(
                ticker.get("market") or ""
            ).upper().strip()

            volume = float(
                ticker.get("volume") or 0
            )

            last_price = float(
                ticker.get("last_price") or 0
            )

            if not symbol:
                continue

            if last_price <= 0:
                continue

            volume_usdt = (
                volume *
                last_price
            )

            ticker_map[symbol] = {
                "volume_usdt":
                    volume_usdt,

                "last_price":
                    last_price
            }

        except Exception:
            continue

    result = {}

    for market in markets:

        try:

            status = str(
                market.get("status") or ""
            ).lower().strip()

            base_currency = str(
                market.get(
                    "base_currency_short_name"
                ) or ""
            ).upper().strip()

            symbol = str(
                market.get("coindcx_name")
                or market.get("symbol")
                or ""
            ).upper().strip()

            pair = str(
                market.get("pair") or ""
            ).strip()

            if status != "active":
                continue

            if base_currency != "USDT":
                continue

            if not symbol or not pair:
                continue

            ticker = ticker_map.get(
                symbol
            )

            if ticker is None:
                continue

            volume_usdt = float(
                ticker["volume_usdt"]
            )

            if (
                volume_usdt
                <
                MIN_24H_VOLUME
            ):
                continue

            result[symbol] = {
                "symbol": symbol,
                "pair": pair,
                "volume_24h":
                    volume_usdt
            }

        except Exception:
            continue

    print(
        "Current $5M+ markets:",
        len(result)
    )

    return result


# ============================================================
# GET 1-MINUTE CANDLES
# ============================================================

def get_1m_candles(pair):

    try:

        response = requests.get(
            f"{BASE_URL}/market_data/candles",
            params={
                "pair": pair,
                "interval": "1m",
                "limit": 600
            },
            timeout=30
        )

        if response.status_code != 200:

            print(
                pair,
                "candle status:",
                response.status_code
            )

            return None

        data = response.json()

        if not isinstance(data, list):
            return None

        if len(data) < 100:
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
            column in df.columns
            for column in required
        ):
            return None

        for column in required:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = (
            df
            .dropna(subset=required)
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True)
        )

        return df

    except Exception as e:

        print(
            pair,
            "1m candle error:",
            e
        )

        return None


# ============================================================
# CONVERT 1M TO CLOSED 5M CANDLES
# ============================================================

def get_5m_candles(pair):

    df = get_1m_candles(
        pair
    )

    if df is None:
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

    # Remove incomplete current 5-minute candle
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
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    return (
        100 -
        (
            100 /
            (1 + rs)
        )
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA 20 / EMA 50
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

    # RSI 14
    df["RSI14"] = calculate_rsi(
        df["close"],
        14
    )

    # MACD 12,26,9
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

    # Bollinger Bands 20,2
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

    # Volume ratio
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

        if candle_high <= candle_low:
            continue

        candle_range = (
            candle_high -
            candle_low
        )

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
                    volume *
                    overlap /
                    candle_range
                )

    if profile.sum() <= 0:
        return np.nan

    index = int(
        np.argmax(profile)
    )

    poc = (
        edges[index]
        +
        edges[index + 1]
    ) / 2

    return float(poc)


# ============================================================
# CURRENT 5M FEATURES
# Must match historical model feature names
# ============================================================

def get_current_features(df):

    if len(df) < 55:
        return None

    index = len(df) - 1

    row = df.iloc[index]

    prev = df.iloc[
        index - 1
    ]

    price = float(
        row["close"]
    )

    open_price = float(
        row["open"]
    )

    if price <= 0 or open_price <= 0:
        return None

    if not np.isfinite(
        row["RSI14"]
    ):
        return None

    if not np.isfinite(
        row["VOLUME_RATIO"]
    ):
        return None

    # POC from latest available candles
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

    candle_move = (
        (
            price -
            open_price
        )
        /
        open_price
        *
        100
    )

    features = {

        "rsi":
            float(
                row["RSI14"]
            ),

        "macd_hist_pct":
            float(
                row["MACD_HIST"]
            )
            /
            price
            *
            100,

        "ema_gap_pct":
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
            100,

        "price_vs_bb_middle_pct":
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
            100,

        "bb_width_pct":
            float(
                row["BB_WIDTH"]
            )
            *
            100,

        "volume_ratio":
            float(
                row["VOLUME_RATIO"]
            ),

        "price_vs_poc_pct":
            (
                price - poc
            )
            /
            poc
            *
            100,

        "candle_move_pct":
            float(
                candle_move
            ),

        "poc":
            float(poc)
    }

    return (
        features,
        row,
        prev
    )


# ============================================================
# HISTORICAL SIMILARITY
# ============================================================

FEATURE_NAMES = [
    "rsi",
    "macd_hist_pct",
    "ema_gap_pct",
    "price_vs_bb_middle_pct",
    "bb_width_pct",
    "volume_ratio",
    "price_vs_poc_pct",
    "candle_move_pct"
]


def similarity_score(
    features,
    historical_pattern
):

    if not historical_pattern:
        return 0.0

    scores = []

    for feature in FEATURE_NAMES:

        if feature not in historical_pattern:
            continue

        current_value = features.get(
            feature
        )

        if current_value is None:
            continue

        if not np.isfinite(
            current_value
        ):
            continue

        data = historical_pattern[
            feature
        ]

        median = float(
            data["median"]
        )

        scale = float(
            data["scale"]
        )

        if scale <= 0:
            continue

        distance = (
            abs(
                current_value -
                median
            )
            /
            scale
        )

        # Median = 100 score
        # 3 robust deviations away = 0
        score = max(
            0.0,
            100.0 *
            (
                1.0 -
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
# ANALYZE ONE TRAINED COIN
# ============================================================

def analyze_symbol(
    market,
    historical
):

    symbol = market[
        "symbol"
    ]

    pair = market[
        "pair"
    ]

    df = get_5m_candles(
        pair
    )

    if df is None:
        return None

    df = add_indicators(
        df
    )

    result = get_current_features(
        df
    )

    if result is None:
        return None

    features, row, prev = result

    price = float(
        row["close"]
    )

    candle_move = float(
        features[
            "candle_move_pct"
        ]
    )

    # Avoid late entry after very large candle
    if (
        abs(candle_move)
        >
        MAX_CURRENT_CANDLE_MOVE
    ):
        return None

    buy_pattern = historical.get(
        "buy"
    )

    sell_pattern = historical.get(
        "sell"
    )

    buy_match = similarity_score(
        features,
        buy_pattern
    )

    sell_match = similarity_score(
        features,
        sell_pattern
    )

    # ========================================================
    # CURRENT BUY CONDITIONS
    # ========================================================

    buy_rsi = (
        42 <=
        row["RSI14"]
        <= 70
    )

    buy_macd = (
        row["MACD_HIST"] > 0
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
        row["BB_UPPER"] * 1.01
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

    buy_conditions = {
        "RSI": buy_rsi,
        "MACD": buy_macd,
        "EMA": buy_ema,
        "BB": buy_bb,
        "VOLUME": buy_volume,
        "POC": buy_poc
    }

    buy_count = sum(
        bool(value)
        for value
        in buy_conditions.values()
    )

    # ========================================================
    # CURRENT SELL CONDITIONS
    # ========================================================

    sell_rsi = (
        30 <=
        row["RSI14"]
        <= 58
    )

    sell_macd = (
        row["MACD_HIST"] < 0
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
        row["BB_LOWER"] * 0.99
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

    sell_conditions = {
        "RSI": sell_rsi,
        "MACD": sell_macd,
        "EMA": sell_ema,
        "BB": sell_bb,
        "VOLUME": sell_volume,
        "POC": sell_poc
    }

    sell_count = sum(
        bool(value)
        for value
        in sell_conditions.values()
    )

    signal = None
    historical_match = 0.0
    confirmations = 0
    conditions = None

    # ========================================================
    # DIRECT BUY CONFIRMATION
    # ========================================================

    if (
        buy_pattern
        and
        buy_match >= MIN_MATCH_SCORE
        and
        buy_count >= MIN_CONFIRMATIONS
        and
        buy_match > sell_match
    ):

        signal = "BUY"
        historical_match = buy_match
        confirmations = buy_count
        conditions = buy_conditions

    # ========================================================
    # DIRECT SELL CONFIRMATION
    # ========================================================

    elif (
        sell_pattern
        and
        sell_match >= MIN_MATCH_SCORE
        and
        sell_count >= MIN_CONFIRMATIONS
        and
        sell_match > buy_match
    ):

        signal = "SELL"
        historical_match = sell_match
        confirmations = sell_count
        conditions = sell_conditions

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

    passed = [
        name
        for name, value
        in conditions.items()
        if value
    ]

    return {

        "SYMBOL":
            symbol,

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
                historical_match,
                1
            ),

        "CONFIRMATIONS":
            confirmations,

        "PASSED":
            ", ".join(passed),

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

        "BB_MIDDLE":
            round(
                float(
                    row["BB_MIDDLE"]
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
# DUPLICATE ALERT PROTECTION
# ============================================================

def load_alert_history():

    if not os.path.exists(
        ALERT_FILE
    ):

        return pd.DataFrame(
            columns=[
                "SYMBOL",
                "SIGNAL",
                "TIME"
            ]
        )

    try:

        df = pd.read_csv(
            ALERT_FILE
        )

        # Support older alert CSV
        if "SIGNAL" not in df.columns:
            df["SIGNAL"] = ""

        if "SYMBOL" not in df.columns:
            df["SYMBOL"] = ""

        if "TIME" not in df.columns:
            df["TIME"] = ""

        return df

    except Exception:

        return pd.DataFrame(
            columns=[
                "SYMBOL",
                "SIGNAL",
                "TIME"
            ]
        )


def already_alerted(
    history,
    row
):

    if history.empty:
        return False

    match = (
        (
            history["SYMBOL"]
            .astype(str)
            ==
            str(
                row["SYMBOL"]
            )
        )
        &
        (
            history["SIGNAL"]
            .astype(str)
            ==
            str(
                row["SIGNAL"]
            )
        )
        &
        (
            history["TIME"]
            .astype(str)
            ==
            str(
                row["TIME"]
            )
        )
    )

    return bool(
        match.any()
    )


def save_alert(
    history,
    row
):

    new_row = pd.DataFrame(
        [{
            "SYMBOL":
                row["SYMBOL"],

            "SIGNAL":
                row["SIGNAL"],

            "TIME":
                row["TIME"]
        }]
    )

    history = pd.concat(
        [
            history,
            new_row
        ],
        ignore_index=True
    )

    history = history.drop_duplicates(
        subset=[
            "SYMBOL",
            "SIGNAL",
            "TIME"
        ],
        keep="last"
    )

    # Keep file manageable
    if len(history) > 5000:
        history = history.tail(
            5000
        )

    history.to_csv(
        ALERT_FILE,
        index=False
    )

    return history


# ============================================================
# TELEGRAM CONFIRMATION MESSAGE
# ============================================================

def alert_message(row):
    raw_time = row["TIME"]

    if isinstance(raw_time, (int, float, np.integer, np.floating)):
        if raw_time > 10_000_000_000:
            utc_time = pd.to_datetime(raw_time, unit="ms", utc=True)
        else:
            utc_time = pd.to_datetime(raw_time, unit="s", utc=True)
    else:
        utc_time = pd.to_datetime(raw_time, utc=True)

    ist_time = utc_time.tz_convert("Asia/Kolkata")

    time_text = ist_time.strftime(
        "%d-%m-%Y %I:%M %p"
    )

    volume_m = (
        row[
            "24H_VOLUME_USDT"
        ]
        /
        1_000_000
    )

    if row["SIGNAL"] == "BUY":

        title = (
            "🟢 BUY CONFIRMATION"
        )

    else:

        title = (
            "🔴 SELL CONFIRMATION"
        )

    return (

        f"{title}\n"
        f"CoinDCX Spot\n\n"

        f"Coin: "
        f"{row['SYMBOL']}\n"

        f"Timeframe: 5 Minute\n"

        f"Time: "
        f"{time_text} IST\n\n"

        f"Price: "
        f"{row['PRICE']}\n"

        f"Historical Match: "
        f"{row['HISTORICAL_MATCH']}%\n"

        f"Current Confirmation: "
        f"{row['CONFIRMATIONS']}/6\n"

        f"Passed: "
        f"{row['PASSED']}\n\n"

        f"RSI(14): "
        f"{row['RSI14']}\n"

        f"MACD Hist: "
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

        f"6-month historical pattern "
        f"+ current 5-minute setup confirmed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "COINDCX FAST 5-MINUTE "
        "BUY / SELL SCANNER"
    )

    print(
        "=" * 70
    )

    print(
        "Historical download: OFF"
    )

    print(
        "Saved 6-month model: ON"
    )

    print(
        "Minimum 24H volume: $5M"
    )

    print(
        "Timeframe: 5 minute"
    )

    print(
        "Pre-alert: OFF"
    )

    print(
        "Direct BUY / SELL: ON"
    )

    # --------------------------------------------------------
    # Historical model
    # --------------------------------------------------------

    model = load_model()

    if model is None:
        return

    trained_symbols = model.get(
        "symbols",
        {}
    )

    if not trained_symbols:

        print(
            "No trained symbols "
            "inside historical model."
        )

        return

    # --------------------------------------------------------
    # Current markets
    # --------------------------------------------------------

    current_markets = (
        get_current_markets()
    )

    if not current_markets:

        print(
            "No current $5M+ "
            "markets found."
        )

        return

    # Only coins that:
    # 1. have historical model
    # 2. still have >= $5M 24h volume

    scan_list = []

    for symbol, historical in (
        trained_symbols.items()
    ):

        market = (
            current_markets.get(
                symbol
            )
        )

        if market is None:
            continue

        scan_list.append(
            (
                market,
                historical
            )
        )

    print(
        "Trained coins:",
        len(trained_symbols)
    )

    print(
        "Eligible trained coins:",
        len(scan_list)
    )

    if not scan_list:

        print(
            "No trained coin currently "
            "passes $5M volume filter."
        )

        return

    # --------------------------------------------------------
    # Live scan
    # --------------------------------------------------------

    signals = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {}

        for market, historical in scan_list:

            future = executor.submit(
                analyze_symbol,
                market,
                historical
            )

            futures[future] = (
                market["symbol"]
            )

        total = len(futures)

        for number, future in enumerate(
            as_completed(futures),
            1
        ):

            symbol = futures[
                future
            ]

            try:

                result = future.result()

                print(
                    f"Scanned "
                    f"{number}/{total}: "
                    f"{symbol}"
                )

                if result is not None:

                    signals.append(
                        result
                    )

                    print(
                        f">>> "
                        f"{result['SIGNAL']} "
                        f"{symbol} | "
                        f"Match "
                        f"{result['HISTORICAL_MATCH']}% | "
                        f"{result['CONFIRMATIONS']}/6"
                    )

            except Exception as e:

                print(
                    symbol,
                    "scan error:",
                    e
                )

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    alert_history = (
        load_alert_history()
    )

    for row in signals:

        if already_alerted(
            alert_history,
            row
        ):

            print(
                "Duplicate skipped:",
                row["SYMBOL"],
                row["SIGNAL"]
            )

            continue

        message = alert_message(
            row
        )

        if send_telegram(
            message
        ):

            alert_history = save_alert(
                alert_history,
                row
            )

    # --------------------------------------------------------
    # CSV results
    # --------------------------------------------------------

    if signals:

        result_df = pd.DataFrame(
            signals
        )

        result_df = (
            result_df
            .sort_values(
                [
                    "HISTORICAL_MATCH",
                    "CONFIRMATIONS"
                ],
                ascending=[
                    False,
                    False
                ]
            )
        )

        result_df.to_csv(
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
            len(result_df)
        )

        print(
            "=" * 70
        )

        print(
            result_df[
                [
                    "SYMBOL",
                    "SIGNAL",
                    "HISTORICAL_MATCH",
                    "CONFIRMATIONS",
                    "RSI14",
                    "VOLUME_RATIO"
                ]
            ].to_string(
                index=False
            )
        )

    else:

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


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

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
