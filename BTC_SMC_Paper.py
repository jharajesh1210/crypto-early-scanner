
"""BTC_SMC_Paper: CoinDCX BTC futures SMC signals, PAPER ONLY.
No order endpoints, no API keys. Uses completed candles only.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PAIR = "B-BTC_USDT"
API = "https://public.coindcx.com/market_data/candles"
STATE_FILE = Path("BTC_SMC_Paper_state.json")
TRADES_FILE = Path("BTC_SMC_Paper_trades.csv")
SWING = 3
SETUP_LIFE = 18                 # 18 x 5-minute candles
FVG_LIFE = 12
FEE_ROUND_TRIP = 0.001         # assumption: 0.10% of entry notional
SLIPPAGE_EACH_SIDE = 0.0002    # assumption: 0.02% each side
MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.025


def telegram(message):
    token = os.getenv("BOT_TOKEN", "").strip()
    chat_id = os.getenv("CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram secrets absent; message:", message)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message}, timeout=15)
        r.raise_for_status()
    except requests.RequestException as exc:
        print("Telegram delivery failed:", exc)


def get_candles(interval, days):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 86400 * 1000
    rows = []
    # CoinDCX API returns a bounded number of rows per call; use chunks.
    step_ms = (500 if interval == "5m" else 300) * (
        300000 if interval == "5m" else 3600000
    )
    for left in range(start_ms, now_ms, step_ms):
        right = min(left + step_ms, now_ms)
        response = requests.get(
            API,
            params={
                "pair": PAIR,
                "interval": interval,
                "startTime": left,
                "endTime": right,
            },
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("candles", []))
        if not isinstance(payload, list):
            raise RuntimeError(
                f"Unexpected CoinDCX response for {interval}"
            )
        rows.extend(payload)

    parsed = []
    for item in rows:
        try:
            if isinstance(item, dict):
                ts = item.get(
                    "time", item.get("timestamp", item.get("t"))
                )
                op = item.get("open", item.get("o"))
                hi = item.get("high", item.get("h"))
                lo = item.get("low", item.get("l"))
                cl = item.get("close", item.get("c"))
            else:
                continue

            parsed.append((
                float(ts),
                float(op),
                float(hi),
                float(lo),
                float(cl),
            ))
        except (ValueError, TypeError):
            continue

    if not parsed:
        raise RuntimeError(
            f"No usable CoinDCX candles for {interval}"
        )

    df = pd.DataFrame(
        parsed,
        columns=["time", "open", "high", "low", "close"],
    )

    # CoinDCX timestamps normally milliseconds; accept seconds too.
    df["time"] = df["time"].where(
        df["time"] > 1e11,
        df["time"] * 1000,
    )
    df = (
        df.drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    minutes = 5 if interval == "5m" else 60
    df = df[
        df["time"] + minutes * 60000 <= now_ms
    ].reset_index(drop=True)

    if len(df) < (100 if interval == "5m" else 30):
        raise RuntimeError(
            f"Too few completed {interval} candles: {len(df)}"
        )
    return df


def swings(df, n=SWING):
    highs, lows = [], []

    for i in range(n, len(df) - n):
        window = df.iloc[i - n:i + n + 1]

        if (
            df.iloc[i].high == window.high.max()
            and (window.high == df.iloc[i].high).sum() == 1
        ):
            highs.append((i, float(df.iloc[i].high)))

        if (
            df.iloc[i].low == window.low.min()
            and (window.low == df.iloc[i].low).sum() == 1
        ):
            lows.append((i, float(df.iloc[i].low)))

    return highs, lows


def h1_bias(df):
    highs, lows = swings(df)

    if len(highs) < 2 or len(lows) < 2:
        return "NEUTRAL"

    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]
    ll = lows[-1][1] < lows[-2][1]

    return (
        "LONG" if hh and hl
        else "SHORT" if lh and ll
        else "NEUTRAL"
    )


def fresh_state():
    return {
        "last_bar": 0,
        "setup": None,
        "positions": [],
        "closed": 0,
        "wins": 0,
        "losses": 0,
        "net_pct": 0.0,
    }


def load_state():
    if STATE_FILE.exists():
        with STATE_FILE.open(encoding="utf-8") as file:
            return json.load(file)
    return fresh_state()


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )
    tmp.replace(STATE_FILE)


def log_trade(trade):
    columns = [
        "time_utc", "side", "rr", "entry", "stop",
        "target", "exit", "reason", "gross_pct", "net_pct",
    ]
    row = {key: trade.get(key, "") for key in columns}

    pd.DataFrame([row]).to_csv(
        TRADES_FILE,
        mode="a",
        index=False,
        header=not TRADES_FILE.exists(),
    )


def close_positions(state, bar):
    survivors = []

    for pos in state["positions"]:
        # Do not exit on the same candle used to generate entry.
        if int(bar.time) <= pos["entry_time"]:
            survivors.append(pos)
            continue

        long = pos["side"] == "LONG"

        stop_hit = (
            bar.low <= pos["stop"]
            if long else bar.high >= pos["stop"]
        )
        target_hit = (
            bar.high >= pos["target"]
            if long else bar.low <= pos["target"]
        )

        if not stop_hit and not target_hit:
            survivors.append(pos)
            continue

        # If both touched in one candle, assume STOP first.
        reason = "SL" if stop_hit else "TP"
        exit_price = (
            pos["stop"] if stop_hit else pos["target"]
        )

        gross = (
            (exit_price / pos["entry"] - 1)
            * 100
            * (1 if long else -1)
        )
        net = gross - 100 * (
            FEE_ROUND_TRIP + 2 * SLIPPAGE_EACH_SIDE
        )

        state["closed"] += 1
        state["wins" if net > 0 else "losses"] += 1
        state["net_pct"] = round(
            state["net_pct"] + net, 6
        )

        stamp = datetime.fromtimestamp(
            bar.time / 1000, timezone.utc
        ).isoformat()

        log_trade({
            **pos,
            "time_utc": stamp,
            "exit": exit_price,
            "reason": reason,
            "gross_pct": round(gross, 4),
            "net_pct": round(net, 4),
        })

        count = state["closed"]

        telegram(
            f"BTC SMC PAPER EXIT | {pos['side']} "
            f"RR 1:{pos['rr']} | {reason}\n"
            f"Exit: {exit_price:.2f} | Net: {net:+.3f}%\n"
            f"Closed: {count} | Wins: {state['wins']} "
            f"| Losses: {state['losses']}\n"
            f"Win rate: {100 * state['wins'] / count:.1f}% "
            f"| Sum net returns: {state['net_pct']:+.3f}%"
        )

    state["positions"] = survivors


def find_sweep(df, i, side):
    # Only pivots confirmed BEFORE the current candle.
    previous = df.iloc[:i]
    highs, lows = swings(previous)
    bar = df.iloc[i]

    if side == "LONG" and lows:
        level = lows[-1][1]

        if bar.low < level and bar.close > level:
            return {
                "side": side,
                "stage": "SWEEP",
                "sweep_time": int(bar.time),
                "extreme": float(bar.low),
                "break_level": (
                    highs[-1][1] if highs else None
                ),
            }

    if side == "SHORT" and highs:
        level = highs[-1][1]

        if bar.high > level and bar.close < level:
            return {
                "side": side,
                "stage": "SWEEP",
                "sweep_time": int(bar.time),
                "extreme": float(bar.high),
                "break_level": (
                    lows[-1][1] if lows else None
                ),
            }

    return None


def process_setup(df, i, state, bias):
    bar = df.iloc[i]
    setup = state.get("setup")

    if setup and (
        (int(bar.time) - setup["sweep_time"])
        > SETUP_LIFE * 300000
        or setup["side"] != bias
    ):
        setup = None

    if setup:
        long = setup["side"] == "LONG"

        if (
            (long and bar.low < setup["extreme"])
            or (not long and bar.high > setup["extreme"])
        ):
            setup = None

        elif setup["stage"] == "SWEEP":
            level = setup.get("break_level")

            if level is not None and (
                (long and bar.close > level)
                or (not long and bar.close < level)
            ):
                setup["stage"] = "CHOCH"
                setup["choch_time"] = int(bar.time)

        elif setup["stage"] == "CHOCH" and i >= 2:
            a = df.iloc[i - 2]

            if long and bar.low > a.high:
                setup.update(
                    stage="FVG",
                    fvg_low=float(a.high),
                    fvg_high=float(bar.low),
                    fvg_time=int(bar.time),
                )

            elif not long and bar.high < a.low:
                setup.update(
                    stage="FVG",
                    fvg_low=float(bar.high),
                    fvg_high=float(a.low),
                    fvg_time=int(bar.time),
                )

        elif setup["stage"] == "FVG":
            if (
                int(bar.time) - setup["fvg_time"]
            ) > FVG_LIFE * 300000:
                setup = None

            else:
                touched = (
                    bar.low <= setup["fvg_high"]
                    and bar.high >= setup["fvg_low"]
                )

                if long:
                    rejection = (
                        bar.close > bar.open
                        and bar.close > setup["fvg_high"]
                    )
                else:
                    rejection = (
                        bar.close < bar.open
                        and bar.close < setup["fvg_low"]
                    )

                if touched and rejection:
                    entry = float(bar.close)
                    stop = float(setup["extreme"])
                    risk = abs(entry - stop)
                    risk_pct = risk / entry

                    valid_direction = (
                        (long and stop < entry)
                        or (not long and stop > entry)
                    )

                    if (
                        valid_direction
                        and MIN_RISK_PCT <= risk_pct <= MAX_RISK_PCT
                        and not state["positions"]
                    ):
                        stamp = datetime.fromtimestamp(
                            bar.time / 1000, timezone.utc
                        ).isoformat()

                        for rr in (2, 3):
                            target = (
                                entry + rr * risk
                                if long else entry - rr * risk
                            )

                            state["positions"].append({
                                "side": setup["side"],
                                "rr": rr,
                                "entry": entry,
                                "stop": stop,
                                "target": target,
                                "entry_time": int(bar.time),
                            })

                        target2 = (
                            entry + 2 * risk
                            if long else entry - 2 * risk
                        )
                        target3 = (
                            entry + 3 * risk
                            if long else entry - 3 * risk
                        )

                        telegram(
                            f"BTC SMC PAPER ENTRY | "
                            f"{setup['side']} | {stamp}\n"
                            f"5m Sweep + CHoCH + FVG retest "
                            f"| 1H {bias}\n"
                            f"Entry: {entry:.2f} "
                            f"| SL: {stop:.2f}\n"
                            f"RR2: {target2:.2f} "
                            f"| RR3: {target3:.2f}\n"
                            "SIMULATION ONLY - NO REAL ORDER"
                        )

                    setup = None

    if (
        setup is None
        and bias in ("LONG", "SHORT")
        and not state["positions"]
    ):
        setup = find_sweep(df, i, bias)

    state["setup"] = setup


def main():
    df5 = get_candles("5m", 6)
    df1 = get_candles("1h", 18)
    state = load_state()

    if not state["last_bar"]:
        state["last_bar"] = int(df5.iloc[-1].time)
        save_state(state)
        print(
            "Initialized at latest completed 5m candle; "
            "no historical trades fabricated."
        )
        return

    new_indices = df5.index[
        df5.time > state["last_bar"]
    ].tolist()

    if not new_indices:
        print("No new completed 5m candles.")
        return

    if len(new_indices) > 12:
        raise RuntimeError(
            "More than 12 candles missed; "
            "stop rather than replay stale alerts"
        )

    for i in new_indices:
        bar = df5.iloc[i]

        # Use only 1H candles completed by this 5m candle.
        eligible_h1 = df1[
            df1.time + 3600000 <= bar.time + 300000
        ]

        bias = h1_bias(eligible_h1)

        close_positions(state, bar)
        process_setup(df5, i, state, bias)

        state["last_bar"] = int(bar.time)
        save_state(state)

        print(
            f"5m candle {int(bar.time)} "
            f"| 1H bias {bias} "
            f"| open legs {len(state['positions'])}"
        )

    print(
        f"PAPER ONLY | closed {state['closed']} "
        f"| wins {state['wins']} "
        f"| losses {state['losses']} "
        f"| sum net returns "
        f"{state['net_pct']:+.3f}%"
    )


if __name__ == "__main__":
    main()
