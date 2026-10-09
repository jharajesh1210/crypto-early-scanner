# ============================================================
# COINDCX FUTURES V10 LIVE PAPER TRADING
# ============================================================
#
# STRATEGY:
#   Original V10_EMA20_RISING3 entry logic
#
# PAPER TEST:
#   RR 1:2
#   RR 1:3
#
# IMPORTANT:
#   PAPER TRADING ONLY
#   NO REAL ORDERS
#   NO COINDCX PRIVATE ORDER API
#
# BB SQUEEZE:
#   Recorded for later analysis
#   NOT used as an entry filter
# ============================================================

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import crypto_futures_historical_learning as v1
import crypto_futures_historical_v6 as v6
import crypto_futures_historical_v7 as v7
import crypto_futures_historical_v8 as v8
import crypto_futures_historical_v9 as v9
import crypto_futures_historical_v10 as v10


# ============================================================
# SETTINGS
# ============================================================

PAIR = "B-BTC_USDT"

# Enough recent candles for indicators / support / POC.
LOOKBACK_DAYS_5M = 7
LOOKBACK_DAYS_1H = 30

# Paper risk model.
# Risk = 0.50% from entry.
# RR 1:2 => TP = 1.00%
# RR 1:3 => TP = 1.50%
STOP_LOSS_PCT = 0.50

RR_2 = 2.0
RR_3 = 3.0

# Approximate round-trip cost used in our research.
ROUND_TRIP_COST_PCT = 0.10

# Maximum time we keep a paper trade open.
# 288 x 5m = 24 hours.
MAX_HOLD_BARS = 288

# Conservative rule:
# if both TP and SL are inside the same 5m candle,
# count SL first.
SAME_BAR_POLICY = "SL_FIRST"

# BB squeeze is ONLY recorded.
BB_SQUEEZE_LOOKBACK = 100
BB_SQUEEZE_QUANTILE = 0.25

# Files
PAPER_FILE = "crypto_futures_v10_paper.csv"
STATE_FILE = "crypto_futures_v10_paper_state.json"
SIGNAL_FILE = "crypto_futures_v10_paper_signals.csv"

# Telegram
TELEGRAM_ENABLED = True
TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID", "")


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dataframe(data, name="data"):

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        return pd.DataFrame(data)

    except Exception as exc:

        raise TypeError(
            f"{name} could not be converted to DataFrame. "
            f"Received type: {type(data)}"
        ) from exc


def utc_now():

    return datetime.now(timezone.utc)


def safe_float(value, default=np.nan):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def load_csv(path):

    if not Path(path).exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception as exc:

        print(
            f"Warning: could not read {path}:",
            exc,
        )

        return pd.DataFrame()


def save_csv(df, path):

    ensure_dataframe(
        df,
        f"save {path}",
    ).to_csv(
        path,
        index=False,
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_ENABLED:

        print("Telegram disabled.")
        return False

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:

        print(
            "Telegram BOT_TOKEN / CHAT_ID "
            "not configured."
        )

        return False

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
    }

    try:

        response = requests.post(
            url,
            data=payload,
            timeout=30,
        )

        if response.status_code == 200:

            print("Telegram sent.")
            return True

        print(
            "Telegram error:",
            response.status_code,
            response.text[:300],
        )

    except Exception as exc:

        print(
            "Telegram exception:",
            exc,
        )

    return False


# ============================================================
# STATE
# ============================================================

def load_state():

    default_state = {
        "last_signal_time": "",
    }

    if not Path(STATE_FILE).exists():
        return default_state

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            state = json.load(file)

        if not isinstance(state, dict):
            return default_state

        for key, value in default_state.items():

            if key not in state:
                state[key] = value

        return state

    except Exception as exc:

        print(
            "State read warning:",
            exc,
        )

        return default_state


def save_state(state):

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=2,
        )


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_market():

    print()
    print("=" * 100)
    print("DOWNLOADING COINDCX FUTURES DATA")
    print("=" * 100)

    print()
    print("Downloading native 5-minute data...")

    df_5m = v1.fetch_history(
        PAIR,
        "5",
        LOOKBACK_DAYS_5M,
        v1.CHUNK_DAYS_5M,
    )

    df_5m = ensure_dataframe(
        df_5m,
        "5-minute data",
    )

    if df_5m.empty:

        raise RuntimeError(
            "No 5-minute Futures data."
        )

    print(
        "5M CANDLES:",
        len(df_5m),
    )

    print()
    print("Downloading native 1-hour data...")

    df_1h = v1.fetch_history(
        PAIR,
        "60",
        LOOKBACK_DAYS_1H,
        v1.CHUNK_DAYS_1H,
    )

    df_1h = ensure_dataframe(
        df_1h,
        "1-hour data",
    )

    if df_1h.empty:

        raise RuntimeError(
            "No 1-hour Futures data."
        )

    print(
        "1H CANDLES:",
        len(df_1h),
    )

    return df_5m, df_1h


# ============================================================
# PREPARE EXACT V10 INDICATORS
# ============================================================

def prepare_v10_market(
    df_5m,
    df_1h,
):

    print()
    print("=" * 100)
    print("PREPARING ORIGINAL V10 INDICATORS")
    print("=" * 100)

    df_5m = ensure_dataframe(
        df_5m,
        "prepare 5m",
    )

    df_1h = ensure_dataframe(
        df_1h,
        "prepare 1h",
    )

    print("Calculating base 5M indicators...")

    df_5m = v1.calculate_indicators(
        df_5m
    )

    df_5m = ensure_dataframe(
        df_5m,
        "base indicators",
    )

    print("Calculating Volume Profile...")

    df_5m = v1.add_volume_profile(
        df_5m
    )

    df_5m = ensure_dataframe(
        df_5m,
        "volume profile",
    )

    print("Attaching original 1H trend...")

    df_5m = v1.attach_1h_trend(
        df_5m,
        df_1h,
    )

    df_5m = ensure_dataframe(
        df_5m,
        "1h trend",
    )

    print("Adding V6 indicators...")

    df_5m = v6.add_v6_indicators(
        df_5m
    )

    df_5m = ensure_dataframe(
        df_5m,
        "v6 indicators",
    )

    print("Preparing V7 1H indicators...")

    df_1h_v7 = v7.prepare_1h_indicators(
        df_1h
    )

    df_1h_v7 = ensure_dataframe(
        df_1h_v7,
        "v7 1h",
    )

    print("Attaching V7 completed 1H indicators...")

    df_5m = v7.attach_v7_1h_data(
        df_5m,
        df_1h_v7,
    )

    df_5m = ensure_dataframe(
        df_5m,
        "v7 attached",
    )

    print("Adding V8 indicators...")

    df_5m = v8.add_v8_indicators(
        df_5m
    )

    df_5m = ensure_dataframe(
        df_5m,
        "v8 indicators",
    )

    print("Adding V9 indicators...")

    df_5m = v9.add_v9_indicators(
        df_5m
    )

    df_5m = ensure_dataframe(
        df_5m,
        "v9 indicators",
    )

    print("Adding exact V10 indicators...")

    df_5m = v10.add_v10_indicators(
        df_5m
    )

    df_5m = ensure_dataframe(
        df_5m,
        "v10 indicators",
    )

    required = [
        "datetime",
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
        "support_4h",
        "distance_support_4h_pct",
        "close_position",
        "trend_1h",
        "ema20_1h",
        "ema50_1h",
        "macd_hist_1h",
        "rsi_1h",
        "v10_bb_middle",
        "v10_bb_position",
        "v10_bb_width_pct",
        "v10_volume_ratio",
        "v10_poc_distance_pct",
        "v10_ema20_rising3",
    ]

    missing = [
        col
        for col in required
        if col not in df_5m.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required V10 columns: "
            + ", ".join(missing)
        )

    df_5m = (
        df_5m
        .dropna(
            subset=required
        )
        .sort_values(
            "datetime"
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
# BB SQUEEZE RECORDING
# ============================================================

def add_bb_squeeze_information(df):

    data = ensure_dataframe(
        df,
        "bb squeeze input",
    )

    data["paper_bb_width_threshold"] = (
        data["v10_bb_width_pct"]
        .rolling(
            BB_SQUEEZE_LOOKBACK,
            min_periods=40,
        )
        .quantile(
            BB_SQUEEZE_QUANTILE
        )
    )

    data["paper_bb_squeeze"] = (
        data["v10_bb_width_pct"]
        <=
        data["paper_bb_width_threshold"]
    )

    data["paper_bb_squeeze"] = (
        data["paper_bb_squeeze"]
        .fillna(False)
        .astype(bool)
    )

    return data


# ============================================================
# EXACT V10 EMA20_RISING3 SIGNAL
# ============================================================

def is_v10_ema20_rising3_signal(
    data,
    index,
):

    if index < 60:
        return False, {}

    try:

        # Original V10 starts from
        # Weak Hammer Sweep LONG.
        setup = v1.setup_weak_hammer_sweep(
            data,
            index,
        )

    except Exception as exc:

        print(
            "Weak Hammer check error:",
            exc,
        )

        return False, {}

    if setup != "LONG":
        return False, {}

    row = data.iloc[index]

    try:

        confirmations = (
            v10.get_v10_confirmations(
                row
            )
        )

    except Exception as exc:

        print(
            "V10 confirmation error:",
            exc,
        )

        return False, {}

    required_conditions = (
        v10.CANDIDATES[
            "V10_EMA20_RISING3"
        ]
    )

    passed = True

    for condition in required_conditions:

        if not bool(
            confirmations.get(
                condition,
                False,
            )
        ):

            passed = False
            break

    return passed, confirmations


# ============================================================
# FIND NEW SIGNAL
# ============================================================

def find_latest_new_signal(
    market,
    state,
):

    data = ensure_dataframe(
        market,
        "signal market",
    )

    if len(data) < 65:
        return None

    last_signal_time = str(
        state.get(
            "last_signal_time",
            "",
        )
    )

    # We inspect only recent completed candles.
    # The newest candle may still be forming.
    end_index = len(data) - 2

    start_index = max(
        60,
        end_index - 12,
    )

    for index in range(
        end_index,
        start_index - 1,
        -1,
    ):

        row = data.iloc[index]

        candle_time = pd.Timestamp(
            row["datetime"]
        )

        if candle_time.tzinfo is None:

            candle_time = (
                candle_time
                .tz_localize("UTC")
            )

        else:

            candle_time = (
                candle_time
                .tz_convert("UTC")
            )

        signal_time = (
            candle_time.isoformat()
        )

        if (
            last_signal_time
            and
            signal_time <= last_signal_time
        ):

            continue

        passed, confirmations = (
            is_v10_ema20_rising3_signal(
                data,
                index,
            )
        )

        if not passed:
            continue

        return {
            "index": index,
            "time": signal_time,
            "row": row,
            "confirmations": confirmations,
        }

    return None


# ============================================================
# PAPER TRADE CREATION
# ============================================================

def create_trade_rows(signal):

    row = signal["row"]

    entry = float(
        row["close"]
    )

    risk_distance = (
        entry
        *
        STOP_LOSS_PCT
        /
        100.0
    )

    stop = (
        entry
        -
        risk_distance
    )

    target_2 = (
        entry
        +
        risk_distance
        * RR_2
    )

    target_3 = (
        entry
        +
        risk_distance
        * RR_3
    )

    squeeze = bool(
        row.get(
            "paper_bb_squeeze",
            False,
        )
    )

    bb_width = safe_float(
        row.get(
            "v10_bb_width_pct",
            np.nan,
        )
    )

    bb_threshold = safe_float(
        row.get(
            "paper_bb_width_threshold",
            np.nan,
        )
    )

    common = {
        "PAIR": PAIR,
        "STRATEGY": "V10_EMA20_RISING3",
        "SIDE": "LONG",
        "SIGNAL_TIME": signal["time"],
        "ENTRY_TIME": signal["time"],
        "ENTRY_PRICE": round(
            entry,
            8,
        ),
        "SL_PRICE": round(
            stop,
            8,
        ),
        "SL_PCT": STOP_LOSS_PCT,
        "STATUS": "OPEN",
        "EXIT_TIME": "",
        "EXIT_PRICE": np.nan,
        "EXIT_REASON": "",
        "GROSS_RETURN_%": np.nan,
        "NET_RETURN_%": np.nan,
        "BARS_HELD": 0,
        "BB_SQUEEZE": int(
            squeeze
        ),
        "BB_WIDTH_%": (
            round(
                bb_width,
                6,
            )
            if pd.notna(bb_width)
            else np.nan
        ),
        "BB_SQUEEZE_THRESHOLD_%": (
            round(
                bb_threshold,
                6,
            )
            if pd.notna(bb_threshold)
            else np.nan
        ),
        "RSI_5M": round(
            safe_float(
                row["rsi"]
            ),
            2,
        ),
        "MACD_HIST_5M": round(
            safe_float(
                row["macd_hist"]
            ),
            8,
        ),
        "EMA20": round(
            safe_float(
                row["ema20"]
            ),
            8,
        ),
        "EMA50": round(
            safe_float(
                row["ema50"]
            ),
            8,
        ),
        "POC": round(
            safe_float(
                row["poc"]
            ),
            8,
        ),
        "POC_DISTANCE_%": round(
            safe_float(
                row["v10_poc_distance_pct"]
            ),
            4,
        ),
        "SUPPORT_DISTANCE_%": round(
            safe_float(
                row["distance_support_4h_pct"]
            ),
            4,
        ),
        "BB_POSITION": round(
            safe_float(
                row["v10_bb_position"]
            ),
            4,
        ),
        "VOLUME_RATIO": round(
            safe_float(
                row["v10_volume_ratio"]
            ),
            4,
        ),
        "TREND_1H": str(
            row["trend_1h"]
        ),
        "CREATED_AT_UTC": (
            utc_now()
            .isoformat()
        ),
    }

    rr2 = dict(common)

    rr2.update({
        "TRADE_ID":
            signal["time"]
            + "_RR2",

        "RR":
            "1:2",

        "TP_PRICE":
            round(
                target_2,
                8,
            ),

        "TP_PCT":
            round(
                STOP_LOSS_PCT
                * RR_2,
                4,
            ),
    })

    rr3 = dict(common)

    rr3.update({
        "TRADE_ID":
            signal["time"]
            + "_RR3",

        "RR":
            "1:3",

        "TP_PRICE":
            round(
                target_3,
                8,
            ),

        "TP_PCT":
            round(
                STOP_LOSS_PCT
                * RR_3,
                4,
            ),
    })

    return [
        rr2,
        rr3,
    ]


# ============================================================
# SIGNAL HISTORY
# ============================================================

def save_signal_history(signal):

    row = signal["row"]

    history = load_csv(
        SIGNAL_FILE
    )

    signal_time = signal["time"]

    if (
        not history.empty
        and
        "SIGNAL_TIME" in history.columns
        and
        signal_time
        in history[
            "SIGNAL_TIME"
        ].astype(str).values
    ):

        return

    new_row = pd.DataFrame([
        {
            "PAIR":
                PAIR,

            "STRATEGY":
                "V10_EMA20_RISING3",

            "SIGNAL_TIME":
                signal_time,

            "ENTRY_PRICE":
                round(
                    float(
                        row["close"]
                    ),
                    8,
                ),

            "RSI_5M":
                round(
                    safe_float(
                        row["rsi"]
                    ),
                    2,
                ),

            "MACD_HIST_5M":
                round(
                    safe_float(
                        row["macd_hist"]
                    ),
                    8,
                ),

            "EMA20":
                round(
                    safe_float(
                        row["ema20"]
                    ),
                    8,
                ),

            "EMA50":
                round(
                    safe_float(
                        row["ema50"]
                    ),
                    8,
                ),

            "BB_SQUEEZE":
                int(
                    bool(
                        row.get(
                            "paper_bb_squeeze",
                            False,
                        )
                    )
                ),

            "BB_WIDTH_%":
                round(
                    safe_float(
                        row.get(
                            "v10_bb_width_pct",
                            np.nan,
                        )
                    ),
                    6,
                ),

            "BB_POSITION":
                round(
                    safe_float(
                        row["v10_bb_position"]
                    ),
                    4,
                ),

            "VOLUME_RATIO":
                round(
                    safe_float(
                        row["v10_volume_ratio"]
                    ),
                    4,
                ),

            "POC_DISTANCE_%":
                round(
                    safe_float(
                        row["v10_poc_distance_pct"]
                    ),
                    4,
                ),

            "SUPPORT_DISTANCE_%":
                round(
                    safe_float(
                        row[
                            "distance_support_4h_pct"
                        ]
                    ),
                    4,
                ),

            "TREND_1H":
                str(
                    row["trend_1h"]
                ),
        }
    ])

    history = pd.concat(
        [
            history,
            new_row,
        ],
        ignore_index=True,
    )

    save_csv(
        history,
        SIGNAL_FILE,
    )


# ============================================================
# ADD NEW PAPER TRADES
# ============================================================

def add_new_paper_trades(signal):

    paper = load_csv(
        PAPER_FILE
    )

    new_rows = (
        create_trade_rows(
            signal
        )
    )

    new_df = pd.DataFrame(
        new_rows
    )

    if not paper.empty:

        if "TRADE_ID" in paper.columns:

            existing = set(
                paper[
                    "TRADE_ID"
                ].astype(str)
            )

            new_df = (
                new_df[
                    ~new_df[
                        "TRADE_ID"
                    ]
                    .astype(str)
                    .isin(existing)
                ]
                .copy()
            )

    if new_df.empty:

        print(
            "Paper trades already exist "
            "for this signal."
        )

        return paper

    paper = pd.concat(
        [
            paper,
            new_df,
        ],
        ignore_index=True,
    )

    save_csv(
        paper,
        PAPER_FILE,
    )

    return paper


# ============================================================
# TRADE EXIT CALCULATION
# ============================================================

def close_long_trade(
    paper,
    row_index,
    exit_price,
    exit_time,
    reason,
    bars_held,
):

    entry = float(
        paper.at[
            row_index,
            "ENTRY_PRICE",
        ]
    )

    gross_return = (
        (
            exit_price
            -
            entry
        )
        /
        entry
        *
        100.0
    )

    net_return = (
        gross_return
        -
        ROUND_TRIP_COST_PCT
    )

    paper.at[
        row_index,
        "STATUS",
    ] = "CLOSED"

    paper.at[
        row_index,
        "EXIT_TIME",
    ] = exit_time

    paper.at[
        row_index,
        "EXIT_PRICE",
    ] = round(
        float(exit_price),
        8,
    )

    paper.at[
        row_index,
        "EXIT_REASON",
    ] = reason

    paper.at[
        row_index,
        "GROSS_RETURN_%",
    ] = round(
        gross_return,
        4,
    )

    paper.at[
        row_index,
        "NET_RETURN_%",
    ] = round(
        net_return,
        4,
    )

    paper.at[
        row_index,
        "BARS_HELD",
    ] = int(
        bars_held
    )

    return paper


# ============================================================
# UPDATE OPEN PAPER TRADES
# ============================================================

def update_open_trades(
    paper,
    market,
):

    if paper.empty:
        return paper, []

    if "STATUS" not in paper.columns:
        return paper, []

    data = ensure_dataframe(
        market,
        "trade tracking market",
    )

    closed_alerts = []

    open_indexes = (
        paper[
            paper["STATUS"]
            .astype(str)
            .str.upper()
            ==
            "OPEN"
        ]
        .index
        .tolist()
    )

    if not open_indexes:
        return paper, closed_alerts

    for row_index in open_indexes:

        entry_time = pd.to_datetime(
            paper.at[
                row_index,
                "ENTRY_TIME",
            ],
            utc=True,
            errors="coerce",
        )

        if pd.isna(entry_time):
            continue

        entry_price = float(
            paper.at[
                row_index,
                "ENTRY_PRICE",
            ]
        )

        stop = float(
            paper.at[
                row_index,
                "SL_PRICE",
            ]
        )

        target = float(
            paper.at[
                row_index,
                "TP_PRICE",
            ]
        )

        future = (
            data[
                pd.to_datetime(
                    data["datetime"],
                    utc=True,
                )
                >
                entry_time
            ]
            .copy()
        )

        if future.empty:
            continue

        future = (
            future
            .sort_values(
                "datetime"
            )
            .head(
                MAX_HOLD_BARS
            )
        )

        closed = False

        for bars_held, (_, candle) in enumerate(
            future.iterrows(),
            start=1,
        ):

            candle_high = float(
                candle["high"]
            )

            candle_low = float(
                candle["low"]
            )

            candle_close = float(
                candle["close"]
            )

            candle_time = pd.Timestamp(
                candle["datetime"]
            )

            if candle_time.tzinfo is None:

                candle_time = (
                    candle_time
                    .tz_localize("UTC")
                )

            else:

                candle_time = (
                    candle_time
                    .tz_convert("UTC")
                )

            exit_time = (
                candle_time
                .isoformat()
            )

            hit_sl = (
                candle_low
                <=
                stop
            )

            hit_tp = (
                candle_high
                >=
                target
            )

            # Conservative same-bar rule.
            if hit_sl and hit_tp:

                if SAME_BAR_POLICY == "SL_FIRST":

                    exit_price = stop
                    reason = "SL"

                else:

                    exit_price = target
                    reason = "TP"

                paper = close_long_trade(
                    paper,
                    row_index,
                    exit_price,
                    exit_time,
                    reason,
                    bars_held,
                )

                closed = True
                break

            if hit_sl:

                paper = close_long_trade(
                    paper,
                    row_index,
                    stop,
                    exit_time,
                    "SL",
                    bars_held,
                )

                closed = True
                break

            if hit_tp:

                paper = close_long_trade(
                    paper,
                    row_index,
                    target,
                    exit_time,
                    "TP",
                    bars_held,
                )

                closed = True
                break

        # Maximum 24h hold completed.
        if (
            not closed
            and
            len(future)
            >=
            MAX_HOLD_BARS
        ):

            last_candle = (
                future.iloc[
                    MAX_HOLD_BARS - 1
                ]
            )

            last_time = pd.Timestamp(
                last_candle[
                    "datetime"
                ]
            )

            if last_time.tzinfo is None:

                last_time = (
                    last_time
                    .tz_localize("UTC")
                )

            else:

                last_time = (
                    last_time
                    .tz_convert("UTC")
                )

            paper = close_long_trade(
                paper,
                row_index,
                float(
                    last_candle[
                        "close"
                    ]
                ),
                last_time.isoformat(),
                "TIME_EXIT",
                MAX_HOLD_BARS,
            )

            closed = True

        if closed:

            trade = (
                paper.loc[
                    row_index
                ]
                .to_dict()
            )

            closed_alerts.append(
                trade
            )

    return (
        paper,
        closed_alerts,
    )


# ============================================================
# TELEGRAM MESSAGES
# ============================================================

def new_signal_message(signal):

    row = signal["row"]

    entry = float(
        row["close"]
    )

    risk = (
        entry
        *
        STOP_LOSS_PCT
        /
        100.0
    )

    sl = entry - risk
    tp2 = entry + risk * 2.0
    tp3 = entry + risk * 3.0

    squeeze = bool(
        row.get(
            "paper_bb_squeeze",
            False,
        )
    )

    squeeze_text = (
        "YES"
        if squeeze
        else "NO"
    )

    return (
        "V10 FUTURES PAPER TRADE\n"
        "NO REAL ORDER\n\n"
        f"Pair: {PAIR}\n"
        "Strategy: V10_EMA20_RISING3\n"
        "Side: LONG\n"
        f"Signal UTC: {signal['time']}\n\n"

        f"Entry: {entry:.2f}\n"
        f"SL: {sl:.2f} (-{STOP_LOSS_PCT:.2f}%)\n\n"

        "Paper A - RR 1:2\n"
        f"TP: {tp2:.2f} "
        f"(+{STOP_LOSS_PCT * 2:.2f}%)\n\n"

        "Paper B - RR 1:3\n"
        f"TP: {tp3:.2f} "
        f"(+{STOP_LOSS_PCT * 3:.2f}%)\n\n"

        f"RSI: {safe_float(row['rsi']):.2f}\n"
        f"BB Squeeze: {squeeze_text}\n"
        f"BB Width: "
        f"{safe_float(row['v10_bb_width_pct']):.4f}%\n"
        f"Volume Ratio: "
        f"{safe_float(row['v10_volume_ratio']):.2f}\n"
        f"POC Distance: "
        f"{safe_float(row['v10_poc_distance_pct']):.3f}%\n\n"

        "PAPER TEST ONLY"
    )


def closed_trade_message(trade):

    rr = str(
        trade.get(
            "RR",
            "",
        )
    )

    reason = str(
        trade.get(
            "EXIT_REASON",
            "",
        )
    )

    net_return = safe_float(
        trade.get(
            "NET_RETURN_%",
            np.nan,
        )
    )

    result = reason

    if reason == "TP":
        result = "WIN - TP HIT"

    elif reason == "SL":
        result = "LOSS - SL HIT"

    elif reason == "TIME_EXIT":
        result = "TIME EXIT"

    return (
        "V10 PAPER TRADE CLOSED\n"
        "NO REAL ORDER\n\n"
        f"Pair: {PAIR}\n"
        "Strategy: V10_EMA20_RISING3\n"
        f"RR: {rr}\n"
        f"Result: {result}\n"
        f"Entry: "
        f"{safe_float(trade.get('ENTRY_PRICE')):.2f}\n"
        f"Exit: "
        f"{safe_float(trade.get('EXIT_PRICE')):.2f}\n"
        f"Net Return: {net_return:.4f}%\n"
        f"Bars Held: "
        f"{int(safe_float(trade.get('BARS_HELD'), 0))}\n\n"
        "PAPER TEST ONLY"
    )


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def print_performance_summary(
    paper,
):

    print()
    print("=" * 100)
    print("V10 PAPER PERFORMANCE")
    print("=" * 100)

    if paper.empty:

        print(
            "No paper trades yet."
        )

        return

    for rr in [
        "1:2",
        "1:3",
    ]:

        group = (
            paper[
                paper["RR"]
                .astype(str)
                ==
                rr
            ]
            .copy()
        )

        closed = (
            group[
                group["STATUS"]
                .astype(str)
                .str.upper()
                ==
                "CLOSED"
            ]
            .copy()
        )

        open_count = int(
            (
                group["STATUS"]
                .astype(str)
                .str.upper()
                ==
                "OPEN"
            )
            .sum()
        )

        print()
        print(
            "RR:",
            rr,
        )

        print(
            "TOTAL:",
            len(group),
        )

        print(
            "OPEN:",
            open_count,
        )

        print(
            "CLOSED:",
            len(closed),
        )

        if closed.empty:
            continue

        tp_count = int(
            (
                closed["EXIT_REASON"]
                ==
                "TP"
            )
            .sum()
        )

        sl_count = int(
            (
                closed["EXIT_REASON"]
                ==
                "SL"
            )
            .sum()
        )

        time_count = int(
            (
                closed["EXIT_REASON"]
                ==
                "TIME_EXIT"
            )
            .sum()
        )

        decisive = (
            tp_count
            +
            sl_count
        )

        if decisive > 0:

            win_rate = (
                tp_count
                /
                decisive
                *
                100.0
            )

        else:

            win_rate = np.nan

        net_returns = pd.to_numeric(
            closed[
                "NET_RETURN_%"
            ],
            errors="coerce",
        ).dropna()

        if net_returns.empty:

            avg_net = np.nan
            total_net = np.nan
            profit_factor = np.nan

        else:

            avg_net = float(
                net_returns.mean()
            )

            total_net = float(
                net_returns.sum()
            )

            gains = float(
                net_returns[
                    net_returns > 0
                ].sum()
            )

            losses = abs(
                float(
                    net_returns[
                        net_returns < 0
                    ].sum()
                )
            )

            if losses > 0:

                profit_factor = (
                    gains
                    /
                    losses
                )

            elif gains > 0:

                profit_factor = np.inf

            else:

                profit_factor = 0.0

        print(
            "TP WINS:",
            tp_count,
        )

        print(
            "SL LOSSES:",
            sl_count,
        )

        print(
            "TIME EXITS:",
            time_count,
        )

        print(
            "DECISIVE WIN RATE:",
            (
                f"{win_rate:.2f}%"
                if pd.notna(win_rate)
                else "N/A"
            ),
        )

        print(
            "AVG NET RETURN:",
            (
                f"{avg_net:.4f}%"
                if pd.notna(avg_net)
                else "N/A"
            ),
        )

        print(
            "TOTAL NET RETURN:",
            (
                f"{total_net:.4f}%"
                if pd.notna(total_net)
                else "N/A"
            ),
        )

        print(
            "PROFIT FACTOR:",
            (
                f"{profit_factor:.4f}"
                if pd.notna(
                    profit_factor
                )
                and
                np.isfinite(
                    profit_factor
                )
                else str(
                    profit_factor
                )
            ),
        )


# ============================================================
# MAIN
# ============================================================

# ============================================================
# INITIALIZE PERSISTENT OUTPUTS (NO SIGNAL REQUIRED)
# ============================================================

def initialize_output_files(state):
    """Create header-only CSVs and initial state on the first run.

    Existing files are never overwritten here.
    """
    paper_columns = [
        "PAIR", "STRATEGY", "SIDE", "SIGNAL_TIME", "ENTRY_TIME",
        "ENTRY_PRICE", "SL_PRICE", "SL_PCT", "STATUS", "EXIT_TIME",
        "EXIT_PRICE", "EXIT_REASON", "GROSS_RETURN_%", "NET_RETURN_%",
        "BARS_HELD", "BB_SQUEEZE", "BB_WIDTH_%",
        "BB_SQUEEZE_THRESHOLD_%", "RSI_5M", "MACD_HIST_5M",
        "EMA20", "EMA50", "POC", "POC_DISTANCE_%",
        "SUPPORT_DISTANCE_%", "BB_POSITION", "VOLUME_RATIO",
        "TREND_1H", "CREATED_AT_UTC", "TRADE_ID", "RR", "TP_PRICE",
        "TP_PCT",
    ]
    signal_columns = [
        "PAIR", "STRATEGY", "SIGNAL_TIME", "ENTRY_PRICE", "RSI_5M",
        "MACD_HIST_5M", "EMA20", "EMA50", "BB_SQUEEZE",
        "BB_WIDTH_%", "BB_POSITION", "VOLUME_RATIO",
        "POC_DISTANCE_%", "SUPPORT_DISTANCE_%", "TREND_1H",
    ]

    for path, columns in (
        (PAPER_FILE, paper_columns),
        (SIGNAL_FILE, signal_columns),
    ):
        if not Path(path).exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            print("Initialized empty output:", path)

    if not Path(STATE_FILE).exists():
        save_state(state)
        print("Initialized persistent state:", STATE_FILE)


def main():

    print()
    print("=" * 100)
    print("COINDCX FUTURES V10 PAPER TRADING")
    print("=" * 100)

    print()
    print(
        "PAIR:",
        PAIR,
    )

    print(
        "TIMEFRAME: 5 MINUTES"
    )

    print(
        "STRATEGY: V10_EMA20_RISING3"
    )

    print(
        "PAPER RR: 1:2 AND 1:3"
    )

    print(
        "STOP LOSS:",
        STOP_LOSS_PCT,
        "%",
    )

    print(
        "ROUND-TRIP COST:",
        ROUND_TRIP_COST_PCT,
        "%",
    )

    print(
        "BB SQUEEZE: RECORD ONLY"
    )

    print()
    print(
        "IMPORTANT: NO REAL ORDERS"
    )

    # --------------------------------------------------------
    # Load state
    # --------------------------------------------------------

    state = load_state()
    initialize_output_files(state)

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    df_5m, df_1h = (
        download_market()
    )

    # --------------------------------------------------------
    # Prepare exact V10
    # --------------------------------------------------------

    market = prepare_v10_market(
        df_5m,
        df_1h,
    )

    # --------------------------------------------------------
    # Add BB squeeze observation
    # --------------------------------------------------------

    market = (
        add_bb_squeeze_information(
            market
        )
    )

    # --------------------------------------------------------
    # First update any existing open trades
    # --------------------------------------------------------

    paper = load_csv(
        PAPER_FILE
    )

    paper, closed_alerts = (
        update_open_trades(
            paper,
            market,
        )
    )

    if not paper.empty:

        save_csv(
            paper,
            PAPER_FILE,
        )

    for trade in closed_alerts:

        print()
        print(
            "PAPER TRADE CLOSED:",
            trade.get(
                "TRADE_ID",
                "",
            ),
            trade.get(
                "EXIT_REASON",
                "",
            ),
        )

        send_telegram(
            closed_trade_message(
                trade
            )
        )

    # --------------------------------------------------------
    # Search latest completed candle for NEW V10 signal
    # --------------------------------------------------------

    signal = (
        find_latest_new_signal(
            market,
            state,
        )
    )

    if signal is None:

        print()
        print(
            "NO NEW V10_EMA20_RISING3 SIGNAL."
        )

    else:

        print()
        print("=" * 100)
        print("NEW V10 PAPER SIGNAL FOUND")
        print("=" * 100)

        print(
            "TIME:",
            signal["time"],
        )

        print(
            "ENTRY:",
            float(
                signal[
                    "row"
                ][
                    "close"
                ]
            ),
        )

        # Save signal observation.
        save_signal_history(
            signal
        )

        # Create both RR trades.
        paper = (
            add_new_paper_trades(
                signal
            )
        )

        # Telegram
        send_telegram(
            new_signal_message(
                signal
            )
        )

        # Mark signal as processed.
        state[
            "last_signal_time"
        ] = signal["time"]

        save_state(
            state
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    paper = load_csv(
        PAPER_FILE
    )

    print_performance_summary(
        paper
    )

    print()
    print("=" * 100)
    print("FILES")
    print("=" * 100)

    print(
        PAPER_FILE
    )

    print(
        SIGNAL_FILE
    )

    print(
        STATE_FILE
    )

    print()
    print("=" * 100)
    print(
        "V10 PAPER TRADING RUN COMPLETE"
    )
    print(
        "NO REAL ORDERS WERE PLACED"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
