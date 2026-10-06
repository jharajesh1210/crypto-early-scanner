import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta

# ============================================================
# COINDCX FUTURES HISTORICAL V9
# Goal:
#   1. Build on V8 winners:
#      - BB_POSITION55
#      - EMA20_RISING
#   2. Search combinations for:
#      - Overall win rate >= 80%
#      - Minimum block win rate >= 70%
#   3. Use 4-block walk-forward validation
#   4. Avoid tiny-sample "fake" 100% results
# ============================================================

BASE_URL = "https://public.coindcx.com"
PAIR = "B-BTC_USDT"

DAYS = 180
RESOLUTION = "5"

FORWARD_BARS = 48       # 48 x 5m = 4 hours
WIN_THRESHOLD = 0.50    # +0.50% in signal direction = WIN
LOSS_THRESHOLD = -0.50  # -0.50% = LOSS

MIN_TOTAL_DECISIVE = 20
MIN_BLOCK_DECISIVE = 4

OUTPUT_FILE = "crypto_futures_v9_results.csv"
SUMMARY_FILE = "crypto_futures_v9_summary.csv"


# ============================================================
# DATA DOWNLOAD
# ============================================================

def fetch_chunk(start_time, end_time):
    url = f"{BASE_URL}/market_data/candlesticks"

    params = {
        "pair": PAIR,
        "from": int(start_time.timestamp()),
        "to": int(end_time.timestamp()),
        "resolution": RESOLUTION,
        "pcode": "f"
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()

    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        elif "candles" in data:
            data = data["candles"]

    if not isinstance(data, list):
        return []

    return data


def download_history():
    print("=" * 80)
    print("COINDCX FUTURES HISTORICAL V9")
    print("=" * 80)
    print("PAIR:", PAIR)
    print("HISTORY:", DAYS, "days")
    print("TIMEFRAME: native 5-minute")
    print()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)

    all_data = []

    chunk_start = start
    chunk_no = 1

    while chunk_start < end:
        chunk_end = min(
            chunk_start + timedelta(days=5),
            end
        )

        print(
            f"Fetching chunk {chunk_no}: "
            f"{chunk_start.isoformat()} to "
            f"{chunk_end.isoformat()}"
        )

        try:
            data = fetch_chunk(
                chunk_start,
                chunk_end
            )

            print("Candles received:", len(data))

            all_data.extend(data)

        except Exception as e:
            print("Chunk error:", e)

        chunk_start = chunk_end
        chunk_no += 1

        time.sleep(0.20)

    if not all_data:
        raise RuntimeError("No Futures candles downloaded.")

    return all_data


# ============================================================
# NORMALIZE API DATA
# ============================================================

def normalize_data(data):
    rows = []

    for x in data:

        if not isinstance(x, dict):
            continue

        ts = (
            x.get("time")
            or x.get("timestamp")
            or x.get("t")
        )

        o = x.get("open")
        h = x.get("high")
        l = x.get("low")
        c = x.get("close")
        v = x.get("volume")

        if ts is None:
            continue

        try:
            ts = float(ts)

            if ts > 10_000_000_000:
                dt = pd.to_datetime(
                    ts,
                    unit="ms",
                    utc=True
                )
            else:
                dt = pd.to_datetime(
                    ts,
                    unit="s",
                    utc=True
                )

            rows.append({
                "datetime": dt,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v)
            })

        except Exception:
            continue

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "Downloaded data could not be converted."
        )

    df = (
        df
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print()
    print("TOTAL UNIQUE CANDLES:", len(df))
    print("FROM:", df["datetime"].min())
    print("TO:", df["datetime"].max())

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["ema20_rising"] = (
        df["ema20"] >
        df["ema20"].shift(1)
    )

    df["ema20_rising2"] = (
        (df["ema20"] > df["ema20"].shift(1))
        &
        (df["ema20"].shift(1) >
         df["ema20"].shift(2))
    )

    df["ema_bull"] = (
        df["ema20"] >
        df["ema50"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    df["rsi_rising"] = (
        df["rsi"] >
        df["rsi"].shift(1)
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        df["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    df["macd_positive"] = (
        df["macd_hist"] > 0
    )

    df["macd_rising"] = (
        df["macd_hist"] >
        df["macd_hist"].shift(1)
    )

    # --------------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------------

    df["bb_middle"] = (
        df["close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        df["close"]
        .rolling(20)
        .std()
    )

    df["bb_upper"] = (
        df["bb_middle"]
        + 2 * bb_std
    )

    df["bb_lower"] = (
        df["bb_middle"]
        - 2 * bb_std
    )

    bb_range = (
        df["bb_upper"]
        - df["bb_lower"]
    )

    df["bb_position"] = (
        (df["close"] - df["bb_lower"])
        /
        bb_range.replace(0, np.nan)
    )

    df["bb_width_pct"] = (
        bb_range
        /
        df["bb_middle"]
        * 100
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    df["volume_ma20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        /
        df["volume_ma20"]
        .replace(0, np.nan)
    )

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    df["green"] = (
        df["close"] >
        df["open"]
    )

    candle_range = (
        df["high"]
        - df["low"]
    )

    df["close_position"] = (
        (df["close"] - df["low"])
        /
        candle_range.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # TREND STRENGTH
    # --------------------------------------------------------

    df["ema_gap_pct"] = (
        (df["ema20"] - df["ema50"])
        /
        df["close"]
        * 100
    )

    return df


# ============================================================
# V8 BASE CONDITION
# ============================================================

def v8_base(df):
    """
    Approximate V8 base signal:
    Weak-hammer / bullish sweep style condition.

    We keep this deliberately broad, then V9 filters it.
    """

    body = abs(
        df["close"] -
        df["open"]
    )

    lower_wick = (
        np.minimum(
            df["open"],
            df["close"]
        )
        -
        df["low"]
    )

    upper_wick = (
        df["high"]
        -
        np.maximum(
            df["open"],
            df["close"]
        )
    )

    body_safe = body.replace(0, np.nan)

    weak_hammer = (
        (lower_wick >= body_safe * 0.75)
        &
        (lower_wick > upper_wick)
    )

    bullish_context = (
        (df["close"] > df["ema20"])
        |
        df["ema20_rising"]
    )

    return (
        weak_hammer
        &
        bullish_context
    )


# ============================================================
# V9 CANDIDATES
# ============================================================

def build_candidates(df):
    base = v8_base(df)

    bb55 = (
        df["bb_position"] >= 0.55
    )

    bb60 = (
        df["bb_position"] >= 0.60
    )

    bb65 = (
        df["bb_position"] >= 0.65
    )

    ema_rise = df["ema20_rising"]

    ema_rise2 = df["ema20_rising2"]

    ema_bull = df["ema_bull"]

    macd_pos = df["macd_positive"]

    macd_rise = df["macd_rising"]

    rsi_good = (
        (df["rsi"] >= 45)
        &
        (df["rsi"] <= 70)
    )

    rsi50 = (
        (df["rsi"] >= 50)
        &
        (df["rsi"] <= 70)
    )

    volume100 = (
        df["volume_ratio"] >= 1.00
    )

    close60 = (
        df["close_position"] >= 0.60
    )

    green = df["green"]

    candidates = {
        # V8 winners
        "V9_BB55":
            base & bb55,

        "V9_EMA20_RISING":
            base & ema_rise,

        # Core V9 combinations
        "V9_BB55_EMA20":
            base
            & bb55
            & ema_rise,

        "V9_BB60_EMA20":
            base
            & bb60
            & ema_rise,

        "V9_BB65_EMA20":
            base
            & bb65
            & ema_rise,

        "V9_BB55_EMA20_2BAR":
            base
            & bb55
            & ema_rise2,

        "V9_BB55_EMA_BULL":
            base
            & bb55
            & ema_bull,

        "V9_BB55_EMA20_MACD":
            base
            & bb55
            & ema_rise
            & macd_pos,

        "V9_BB55_EMA20_MACD_RISE":
            base
            & bb55
            & ema_rise
            & macd_rise,

        "V9_BB55_EMA20_RSI":
            base
            & bb55
            & ema_rise
            & rsi_good,

        "V9_BB55_EMA20_RSI50":
            base
            & bb55
            & ema_rise
            & rsi50,

        "V9_BB55_EMA20_GREEN":
            base
            & bb55
            & ema_rise
            & green,

        "V9_BB55_EMA20_CLOSE60":
            base
            & bb55
            & ema_rise
            & close60,

        "V9_BB55_EMA20_VOLUME":
            base
            & bb55
            & ema_rise
            & volume100,

        # Triple filters
        "V9_BB55_EMA_MACD_RSI":
            base
            & bb55
            & ema_rise
            & macd_pos
            & rsi_good,

        "V9_BB55_EMA_MACD_GREEN":
            base
            & bb55
            & ema_rise
            & macd_pos
            & green,

        "V9_BB55_EMA_MACD_CLOSE":
            base
            & bb55
            & ema_rise
            & macd_pos
            & close60,

        "V9_BB55_EMA_RSI_GREEN":
            base
            & bb55
            & ema_rise
            & rsi_good
            & green,

        "V9_BB55_EMA_RSI_CLOSE":
            base
            & bb55
            & ema_rise
            & rsi_good
            & close60,

        # High quality
        "V9_QUALITY_A":
            base
            & bb55
            & ema_rise
            & macd_pos
            & rsi_good
            & close60,

        "V9_QUALITY_B":
            base
            & bb55
            & ema_rise
            & macd_pos
            & rsi_good
            & green,

        "V9_QUALITY_C":
            base
            & bb55
            & ema_rise
            & macd_pos
            & rsi_good
            & volume100,

        "V9_QUALITY_D":
            base
            & bb55
            & ema_rise2
            & macd_pos
            & rsi_good,

        "V9_QUALITY_E":
            base
            & bb60
            & ema_rise
            & macd_pos
            & rsi_good,

        "V9_QUALITY_F":
            base
            & bb55
            & ema_bull
            & ema_rise
            & macd_pos
            & rsi_good,
    }

    return candidates


# ============================================================
# FUTURE 4H RETURN
# ============================================================

def calculate_forward_return(df):
    future_returns = []

    close_array = df["close"].to_numpy()

    n = len(df)

    for i in range(n):

        end = min(
            i + FORWARD_BARS + 1,
            n
        )

        if end <= i + 1:
            future_returns.append(np.nan)
            continue

        entry = close_array[i]

        future_prices = close_array[
            i + 1:end
        ]

        if len(future_prices) == 0:
            future_returns.append(np.nan)
            continue

        exit_price = future_prices[-1]

        ret = (
            (exit_price - entry)
            /
            entry
            * 100
        )

        future_returns.append(ret)

    df = df.copy()

    df["future_4h_return"] = future_returns

    return df


# ============================================================
# RESULT CLASSIFICATION
# ============================================================

def classify_return(ret):
    if pd.isna(ret):
        return "NEUTRAL"

    if ret >= WIN_THRESHOLD:
        return "WIN"

    if ret <= LOSS_THRESHOLD:
        return "LOSS"

    return "NEUTRAL"


# ============================================================
# TEST ONE CANDIDATE
# ============================================================

def evaluate_candidate(
    df,
    condition,
    candidate_name,
    block_no
):
    selected = df[
        condition.fillna(False)
    ].copy()

    selected = selected[
        selected["future_4h_return"].notna()
    ]

    if selected.empty:
        return {
            "BLOCK": block_no,
            "CANDIDATE": candidate_name,
            "SIGNALS": 0,
            "WINS": 0,
            "LOSSES": 0,
            "NEUTRAL": 0,
            "DECISIVE": 0,
            "WIN_RATE_%": np.nan,
            "AVG_RETURN_%": np.nan,
            "MEDIAN_RETURN_%": np.nan
        }

    selected["RESULT"] = (
        selected["future_4h_return"]
        .apply(classify_return)
    )

    wins = (
        selected["RESULT"] == "WIN"
    ).sum()

    losses = (
        selected["RESULT"] == "LOSS"
    ).sum()

    neutral = (
        selected["RESULT"] == "NEUTRAL"
    ).sum()

    decisive = wins + losses

    if decisive > 0:
        win_rate = (
            wins /
            decisive *
            100
        )
    else:
        win_rate = np.nan

    return {
        "BLOCK": block_no,
        "CANDIDATE": candidate_name,
        "SIGNALS": len(selected),
        "WINS": int(wins),
        "LOSSES": int(losses),
        "NEUTRAL": int(neutral),
        "DECISIVE": int(decisive),
        "WIN_RATE_%": win_rate,
        "AVG_RETURN_%":
            selected[
                "future_4h_return"
            ].mean(),
        "MEDIAN_RETURN_%":
            selected[
                "future_4h_return"
            ].median()
    }


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward_test(df):
    print()
    print("=" * 100)
    print("V9 WALK-FORWARD TEST")
    print("=" * 100)

    usable = df.dropna(
        subset=[
            "ema20",
            "ema50",
            "rsi",
            "macd_hist",
            "bb_position",
            "volume_ratio",
            "future_4h_return"
        ]
    ).copy()

    blocks = np.array_split(
        usable,
        4
    )

    results = []

    for block_no, block in enumerate(
        blocks,
        start=1
    ):

        block = block.copy()

        candidates = build_candidates(
            block
        )

        for name, condition in candidates.items():

            result = evaluate_candidate(
                block,
                condition,
                name,
                block_no
            )

            results.append(result)

    results_df = pd.DataFrame(results)

    print()
    print("V9 WALK-FORWARD BLOCK RESULTS")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


# ============================================================
# SUMMARY
# ============================================================

def create_summary(results_df):
    summary_rows = []

    for candidate in (
        results_df["CANDIDATE"]
        .unique()
    ):

        x = results_df[
            results_df["CANDIDATE"]
            == candidate
        ].copy()

        total_wins = x["WINS"].sum()
        total_losses = x["LOSSES"].sum()

        total_decisive = (
            total_wins +
            total_losses
        )

        if total_decisive > 0:
            overall_win = (
                total_wins /
                total_decisive *
                100
            )
        else:
            overall_win = np.nan

        valid_blocks = x[
            x["DECISIVE"]
            >= MIN_BLOCK_DECISIVE
        ]

        if len(valid_blocks) > 0:
            min_block = (
                valid_blocks[
                    "WIN_RATE_%"
                ].min()
            )

            max_block = (
                valid_blocks[
                    "WIN_RATE_%"
                ].max()
            )

            std_block = (
                valid_blocks[
                    "WIN_RATE_%"
                ].std(ddof=0)
            )

            positive_blocks = (
                valid_blocks[
                    "AVG_RETURN_%"
                ] > 0
            ).sum()

            blocks70 = (
                valid_blocks[
                    "WIN_RATE_%"
                ] >= 70
            ).sum()

            blocks75 = (
                valid_blocks[
                    "WIN_RATE_%"
                ] >= 75
            ).sum()

            blocks80 = (
                valid_blocks[
                    "WIN_RATE_%"
                ] >= 80
            ).sum()

        else:
            min_block = np.nan
            max_block = np.nan
            std_block = np.nan
            positive_blocks = 0
            blocks70 = 0
            blocks75 = 0
            blocks80 = 0

        weighted_return = np.average(
            x["AVG_RETURN_%"].fillna(0),
            weights=np.maximum(
                x["SIGNALS"],
                1
            )
        )

        median_return = (
            x["MEDIAN_RETURN_%"]
            .median()
        )

        summary_rows.append({
            "CANDIDATE":
                candidate,

            "TOTAL_DECISIVE":
                int(total_decisive),

            "OVERALL_WIN_RATE_%":
                overall_win,

            "OVERALL_AVG_RETURN_%":
                weighted_return,

            "MEDIAN_RETURN_%":
                median_return,

            "VALID_BLOCKS":
                len(valid_blocks),

            "POSITIVE_BLOCKS":
                int(positive_blocks),

            "BLOCKS_70PLUS":
                int(blocks70),

            "BLOCKS_75PLUS":
                int(blocks75),

            "BLOCKS_80PLUS":
                int(blocks80),

            "MIN_BLOCK_WIN_RATE_%":
                min_block,

            "MAX_BLOCK_WIN_RATE_%":
                max_block,

            "BLOCK_RATE_STD_%":
                std_block
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        by=[
            "OVERALL_WIN_RATE_%",
            "MIN_BLOCK_WIN_RATE_%",
            "TOTAL_DECISIVE"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).reset_index(drop=True)

    return summary


# ============================================================
# PRINT FILTERED RESULTS
# ============================================================

def print_candidates(
    summary,
    title,
    overall_threshold,
    min_block_threshold=None
):
    x = summary[
        summary["TOTAL_DECISIVE"]
        >= MIN_TOTAL_DECISIVE
    ].copy()

    x = x[
        x["VALID_BLOCKS"] >= 3
    ]

    x = x[
        x["OVERALL_WIN_RATE_%"]
        >= overall_threshold
    ]

    if min_block_threshold is not None:
        x = x[
            x["MIN_BLOCK_WIN_RATE_%"]
            >= min_block_threshold
        ]

    print()
    print("=" * 100)
    print(title + ":", len(x))
    print("=" * 100)

    if x.empty:
        print("NONE")
    else:
        print(
            x.to_string(
                index=False
            )
        )

    return x


# ============================================================
# MAIN
# ============================================================

def main():
    raw = download_history()

    df = normalize_data(raw)

    df = add_indicators(df)

    df = calculate_forward_return(df)

    results = walk_forward_test(df)

    summary = create_summary(results)

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print()
    print("=" * 100)
    print("V9 OVERALL RESULTS")
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )

    print_candidates(
        summary,
        "RELIABLE 70%+ CANDIDATES",
        70
    )

    print_candidates(
        summary,
        "RELIABLE 75%+ CANDIDATES",
        75
    )

    print_candidates(
        summary,
        "RELIABLE 80%+ CANDIDATES",
        80
    )

    print_candidates(
        summary,
        "TARGET: OVERALL 80% + MIN BLOCK 70%",
        80,
        70
    )

    print()
    print("=" * 100)
    print("V9 TARGET CHECK")
    print("=" * 100)

    target = summary[
        (summary["TOTAL_DECISIVE"]
         >= MIN_TOTAL_DECISIVE)
        &
        (summary["VALID_BLOCKS"]
         >= 3)
        &
        (summary["OVERALL_WIN_RATE_%"]
         >= 80)
        &
        (summary["MIN_BLOCK_WIN_RATE_%"]
         >= 70)
    ]

    if target.empty:

        print(
            "NO CANDIDATE YET MET BOTH:"
        )

        print(
            "OVERALL WIN RATE >= 80%"
        )

        print(
            "MIN BLOCK WIN RATE >= 70%"
        )

        print()
        print(
            "Do NOT force the strategy."
        )

    else:

        print(
            "TARGET ACHIEVED."
        )

        print()

        print(
            target.to_string(
                index=False
            )
        )

    print()
    print("Saved:", OUTPUT_FILE)
    print("Saved:", SUMMARY_FILE)

    print()
    print("=" * 100)
    print(
        "FUTURES HISTORICAL V9 COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
