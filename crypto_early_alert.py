import requests
import pandas as pd
import os
import numpy as np
import time
from datetime import datetime
from signal_history_helper import save_signal_history
from concurrent.futures import ThreadPoolExecutor, as_completed
# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://api.binance.com/api/v3"

INTERVAL = "15m"
KLINE_LIMIT = 120

# Minimum score required to show a signal
MIN_SCORE = 4

# Telegram - keep False for first test
TELEGRAM_ENABLED = True

TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID", "")

OUTPUT_FILE = "crypto_early_signals.csv"


# ============================================================
# GET BINANCE USDT SPOT SYMBOLS
# ============================================================

def get_symbols():

    url = f"{BASE_URL}/exchangeInfo"

    response = requests.get(
        url,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    symbols = []

    for item in data["symbols"]:

        if (
            item["status"] == "TRADING"
            and item["quoteAsset"] == "USDT"
            and item.get("isSpotTradingAllowed", False)
        ):
            symbols.append(item["symbol"])

    return symbols


# ============================================================
# GET 15-MINUTE DATA
# ============================================================

def get_klines(symbol):

    url = f"{BASE_URL}/klines"

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": KLINE_LIMIT
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    if response.status_code != 200:
        return None

    data = response.json()

    if not isinstance(data, list):
        return None

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(
        data,
        columns=columns
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms"
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms"
    )

    return df


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = (
        average_gain /
        average_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    # EMA 20
    df["EMA20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # EMA 50
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

    # Bollinger Band Middle
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

    # Upper BB
    df["BB_UPPER"] = (
        df["BB_MIDDLE"]
        + (2 * bb_std)
    )

    # Lower BB
    df["BB_LOWER"] = (
        df["BB_MIDDLE"]
        - (2 * bb_std)
    )

    # Average Volume
    df["VOLUME_AVG20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    # Volume Ratio
    df["VOLUME_RATIO"] = (
        df["volume"]
        / df["VOLUME_AVG20"]
    )

    return df


# ============================================================
# SIMPLE VOLUME PROFILE / POC
# ============================================================

def calculate_volume_profile(
    df,
    bins=50
):

    if len(df) < 20:
        return None

    price_low = df["low"].min()

    price_high = df["high"].max()

    if price_high <= price_low:
        return None

    price_bins = np.linspace(
        price_low,
        price_high,
        bins + 1
    )

    volume_profile = np.zeros(
        bins
    )

    for _, candle in df.iterrows():

        low = candle["low"]

        high = candle["high"]

        volume = candle["volume"]

        if high <= low:
            continue

        candle_range = (
            high - low
        )

        for i in range(bins):

            bin_low = (
                price_bins[i]
            )

            bin_high = (
                price_bins[i + 1]
            )

            overlap = max(
                0,
                min(
                    high,
                    bin_high
                )
                -
                max(
                    low,
                    bin_low
                )
            )

            if overlap > 0:

                volume_profile[i] += (
                    volume
                    * overlap
                    / candle_range
                )

    poc_index = np.argmax(
        volume_profile
    )

    poc = (
        price_bins[poc_index]
        +
        price_bins[poc_index + 1]
    ) / 2

    return poc


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_ENABLED:
        return False

    if not TELEGRAM_BOT_TOKEN:
        return False

    if not TELEGRAM_CHAT_ID:
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:

        response = requests.post(
            url,
            data=data,
            timeout=15
        )

        if response.status_code == 200:
            print("Telegram alert sent successfully.")
            return True

        print(
            "Telegram send failed:",
            response.text
        )
        return False

    except Exception as error:

        print(
            "Telegram error:",
            error
        )
        return False
# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_symbol(symbol):

    df = get_klines(symbol)

    if df is None:
        return None

    if len(df) < 100:
        return None

    df = calculate_indicators(
        df
    )

    # --------------------------------------------------------
    # IMPORTANT
    # Last row may still be forming.
    # Use previous row = latest CLOSED 15m candle.
    # --------------------------------------------------------

    signal = df.iloc[-2]
    previous = df.iloc[-3]

    # Use recent candles for Volume Profile
    vp_df = df.iloc[-101:-1].copy()

    poc = calculate_volume_profile(
        vp_df
    )

    if poc is None:
        return None

    price = signal["close"]
    candle_gain_pct = (
        (signal["close"] - signal["open"])
        / signal["open"]
    ) * 100

    # ========================================================
    # SIX CONDITIONS
    # ========================================================

    rsi_bullish = (
    signal["RSI14"] >= 52
    and signal["RSI14"] <= 68
)

    macd_bullish = (
    signal["MACD"] > signal["MACD_SIGNAL"]
    and previous["MACD"] <= previous["MACD_SIGNAL"]
)

    price_above_poc = (
    price > poc
    and price <= poc * 1.03
)

    ema_bullish = (
        signal["EMA20"]
        >
        signal["EMA50"]
    )

    bb_breakout = (
    price >= signal["BB_MIDDLE"]
    and price <= signal["BB_UPPER"] * 1.01
)

    volume_confirm = (
    signal["VOLUME_RATIO"] >= 1.10
)

    # ========================================================
    # SCORE
    # ========================================================

    score = sum([
        rsi_bullish,
        macd_bullish,
        price_above_poc,
        ema_bullish,
        bb_breakout,
        volume_confirm
    ])

    # --------------------------------------------------------
    # CORE CONDITIONS MUST PASS
    # --------------------------------------------------------

    core_pass = (
        rsi_bullish
        and macd_bullish
        and price_above_poc
        and volume_confirm
        and candle_gain_pct <= 3.0
    )

    if not core_pass:
        return None

    if score < MIN_SCORE:
        return None

    return {

        "SYMBOL":
            symbol,

        "TIME":
            signal["close_time"],

        "PRICE":
            round(
                price,
                8
            ),

        "SCORE":
            score,

        "RSI14":
            round(
                signal["RSI14"],
                2
            ),

        "MACD_BULLISH":
            bool(macd_bullish),

        "PRICE_ABOVE_POC":
            bool(price_above_poc),

        "EMA20_ABOVE_EMA50":
            bool(ema_bullish),

        "BB_BREAKOUT":
            bool(bb_breakout),

        "VOLUME_RATIO":
            round(
                signal["VOLUME_RATIO"],
                2
            ),

        "VOLUME_CONFIRM":
            bool(volume_confirm),

        "POC":
            round(
                poc,
                8
            )
    }


# ============================================================
def send_instant_alert(row):

    if int(row["SCORE"]) < 4:
            return

    alert_file = "telegram_early_alerted_signals.csv"

    symbol = str(row["SYMBOL"])
    signal_time = str(row["TIME"])

    if os.path.exists(alert_file):
        alerted_df = pd.read_csv(alert_file)
    else:
        alerted_df = pd.DataFrame(
            columns=["SYMBOL", "TIME"]
        )

    already_sent = (
        (alerted_df["SYMBOL"].astype(str) == symbol)
        &
        (alerted_df["TIME"].astype(str) == signal_time)
    ).any()

    if already_sent:
        return

    message = (
        "EARLY CRYPTO ALERT\n\n"
        f"Symbol: {symbol}\n"
        f"Time (IST): "
        f"{(pd.to_datetime(row['TIME']) + pd.Timedelta(hours=5, minutes=30)).strftime('%d-%m-%Y %I:%M:%S %p')}\n"
        f"Price: {row['PRICE']}\n"
        f"Score: {row['SCORE']}/6\n"
        f"RSI: {row['RSI14']}\n"
        f"Volume Ratio: {row['VOLUME_RATIO']}x\n"
        f"POC: {row['POC']}\n\n"
        "Core confirmation:\n"
        "RSI Bullish = YES\n"
        "Fresh MACD Crossover = YES\n"
        "Price Above POC = YES"
    )

    telegram_sent = send_telegram(message)

    if telegram_sent:

        new_alert = pd.DataFrame([{
            "SYMBOL": symbol,
            "TIME": signal_time
        }])

        alerted_df = pd.concat(
            [alerted_df, new_alert],
            ignore_index=True
        )

        alerted_df.to_csv(
            alert_file,
            index=False
        )
# RUN LIVE SCAN
# ============================================================

def run_scan():

    print()
    print("=" * 70)

    print(
        "CRYPTO 15-MINUTE LIVE SIGNAL SCANNER"
    )

    print("=" * 70)

    print(
        "Time:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print()

    print(
        "Getting Binance USDT Spot symbols..."
    )

    symbols = get_symbols()

    print(
        f"Total symbols: {len(symbols)}"
    )

    print()

    signals = []

        # FAST PARALLEL SCANNING
    MAX_WORKERS = 20

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_symbol = {
            executor.submit(analyze_symbol, symbol): symbol
            for symbol in symbols
        }

        completed = 0

        for future in as_completed(future_to_symbol):

            symbol = future_to_symbol[future]
            completed += 1

            print(
                f"\rScanning {completed}/{len(symbols)} "
                f"{symbol}             ",
                end="",
                flush=True
            )

            try:

                result = future.result()

                if result is not None:
                  signals.append(result)
                  send_instant_alert(result)

            except Exception as error:
                print(
                    f"\nError scanning {symbol}: {error}"
                )
    print()
    print()

    # ========================================================
    # NO SIGNAL
    # ========================================================

    if len(signals) == 0:

        print(
            "No 5/6 or 6/6 strong setup found."
        )

        print(
            "Wait for next CLOSED 15-minute candle."
        )

        return

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        signals
    )

    result_df = result_df.sort_values(
        [
            "SCORE",
            "VOLUME_RATIO",
            "RSI14"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )

    # Save CSV
    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    result_df.to_csv(
        "crypto_early_signal_history.csv",
        index=False
    )
   



    print("=" * 70)

    print(
        f"STRONG SIGNALS FOUND: "
        f"{len(result_df)}"
    )

    print("=" * 70)

    print()

    print(
        result_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"CSV saved as: "
        f"{OUTPUT_FILE}"
    )

    
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        run_scan()

    except KeyboardInterrupt:

        print()
        print(
            "Scanner stopped by user."
        )

    except Exception as error:

        print()
        print(
            "ERROR:",
            error
        )
