"""NEW ASIA LONDO NY - CoinDCX futures PAPER scanner, 15m, five symbols including optional XAUUSDT."""
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

NAME = "NEW_ASIA_LONDO_NY"
STATE_FILE = Path(NAME + "_state.json")
TRADES_FILE = Path(NAME + "_trades.csv")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XAUUSDT")
API = "https://public.coindcx.com/market_data/candlesticks"
MARGIN, LEVERAGE = 100.0, 30.0
FEE_PER_SIDE, SLIPPAGE = 0.0005, 0.0002
BAR_MS = 15 * 60 * 1000


def timestamp(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def alert(message):
    print(message.replace("\n", " | "))
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if token and chat:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": message}, timeout=12,
            )
            r.raise_for_status()
        except requests.RequestException as exc:
            print("Telegram warning:", exc)


def load_state():
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        data = {}
    for key in ("last_bar", "setups", "positions", "traded_days"):
        data.setdefault(key, {})
    return data


def save_state(data):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def fetch_bars(symbol):
    now = int(time.time())
    pair = "B-" + symbol[:-4] + "_USDT"
    r = requests.get(API, params={
        "pair": pair, "pcode": "f", "resolution": "15",
        "from": now - 5 * 86400, "to": now,
    }, timeout=35)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Unexpected CoinDCX candle response")
    bars = {}
    for item in payload["data"]:
        try:
            t = int(float(item["time"]))
            if t < 100_000_000_000:
                t *= 1000
            if t + BAR_MS > int(time.time() * 1000):
                continue  # never trade an unfinished candle
            bar = {"t": t, "o": float(item["open"]),
                   "h": float(item["high"]), "l": float(item["low"]),
                   "c": float(item["close"]), "v": float(item.get("volume", 0))}
            if bar["c"] > 0 and bar["h"] >= bar["l"]:
                bars[t] = bar
        except (ValueError, TypeError, KeyError):
            continue
    return [bars[t] for t in sorted(bars)]


def atr_and_volume(history):
    if len(history) < 22:
        return None, None
    prev = history[-22:-1]  # exclude the signal candle
    trs = []
    for i in range(1, len(prev)):
        b, old = prev[i], prev[i - 1]
        trs.append(max(b["h"] - b["l"], abs(b["h"] - old["c"]),
                       abs(b["l"] - old["c"])))
    return sum(trs[-14:]) / 14, sum(b["v"] for b in prev[-20:]) / 20


def session_day(t):
    return datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d")


def asian_range(history, t):
    day = session_day(t)
    asia = [b for b in history if session_day(b["t"]) == day
            and 0 <= datetime.fromtimestamp(b["t"] / 1000, timezone.utc).hour < 8]
    if len(asia) != 32:
        return None
    return min(b["l"] for b in asia), max(b["h"] for b in asia)


def london(t):
    hour = datetime.fromtimestamp(t / 1000, timezone.utc).hour
    return 8 <= hour < 18


def record_trade(pos, exit_price, t, reason):
    sign = 1 if pos["side"] == "LONG" else -1
    entry = pos["entry"]
    actual_exit = exit_price * (1 - sign * SLIPPAGE)
    gross_pct = sign * (actual_exit / entry - 1) * 100
    fee_pct = FEE_PER_SIDE * (1 + actual_exit / entry) * 100
    net_pct = gross_pct - fee_pct
    pnl = MARGIN * LEVERAGE * net_pct / 100
    row = {
        "trade_id": pos["id"], "symbol": pos["symbol"],
        "strategy": "NEW_ASIA_LONDO_NY", "direction": pos["side"],
        "entry_time_utc": timestamp(pos["entry_t"]),
        "exit_time_utc": timestamp(t), "entry_price": entry,
        "exit_price": actual_exit, "initial_stop": pos["initial_stop"],
        "exit_reason": reason, "locked_rr": pos["locked_rr"],
        "margin_usdt": MARGIN, "leverage": LEVERAGE,
        "net_return_pct_per_leg": round(net_pct, 6),
        "paper_pnl_usdt": round(pnl, 2),
    }
    exists = TRADES_FILE.exists() and TRADES_FILE.stat().st_size > 0
    with TRADES_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    alert(f"NEW ASIA LONDO NY PAPER EXIT\n{pos['symbol']} {pos['side']}\n"
          f"{reason} | P/L {pnl:+.2f} USDT")


def manage_position(state, symbol, bar):
    pos = state["positions"].get(symbol)
    if not pos or bar["t"] <= pos["last_checked"]:
        return
    sign = 1 if pos["side"] == "LONG" else -1
    stop, target = pos["stop"], pos["target"]
    hit_stop = bar["l"] <= stop if sign == 1 else bar["h"] >= stop
    hit_target = bar["h"] >= target if sign == 1 else bar["l"] <= target
    if hit_stop or hit_target:
        # Conservative OHLC assumption: if both touched, stop occurs first.
        reason = "TRAIL_STOP" if hit_stop and pos["locked_rr"] else (
            "STOP" if hit_stop else "TARGET_4R")
        record_trade(pos, stop if hit_stop else target, bar["t"], reason)
        del state["positions"][symbol]
        return
    best = bar["h"] if sign == 1 else bar["l"]
    progress = sign * (best - pos["entry"]) / pos["risk"]
    lock = 3 if progress >= 3 else 2 if progress >= 2 else pos["locked_rr"]
    if lock > pos["locked_rr"]:
        pos["locked_rr"] = lock
        pos["stop"] = pos["entry"] + sign * lock * pos["risk"]
        alert(f"NEW ASIA LONDO NY PAPER TRAIL\n{symbol}\n"
              f"{lock}R reached; SL={pos['stop']:.8g} (next candle)")
    pos["last_checked"] = bar["t"]


def open_position(state, symbol, side, bar, stop):
    if symbol in state["positions"]:
        return
    sign = 1 if side == "LONG" else -1
    entry = bar["c"] * (1 + sign * SLIPPAGE)
    risk = sign * (entry - stop)
    if risk <= 0 or not 0.001 <= risk / entry <= 0.03:
        return
    pos = {
        "id": f"{symbol}-{bar['t']}", "symbol": symbol, "side": side,
        "entry": entry, "entry_t": bar["t"], "initial_stop": stop,
        "stop": stop, "target": entry + sign * 4 * risk,
        "risk": risk, "locked_rr": 0, "last_checked": bar["t"],
    }
    state["positions"][symbol] = pos
    state["traded_days"][symbol] = session_day(bar["t"])
    alert(f"NEW ASIA LONDO NY PAPER ENTRY\n{symbol} {side}\n"
          f"Entry {entry:.8g} | SL {stop:.8g} | 4R {pos['target']:.8g}\n"
          "$100 margin, 30x leverage, PAPER ONLY")


def evaluate(state, symbol, history):
    bar = history[-1]
    t, day = bar["t"], session_day(bar["t"])
    setup = state["setups"].get(symbol)
    if setup and (setup["day"] != day or t - setup["sweep_t"] > 16 * BAR_MS):
        setup = None
    if symbol in state["positions"] or state["traded_days"].get(symbol) == day:
        state["setups"].pop(symbol, None)
        return
    if not london(t):
        state["setups"].pop(symbol, None)
        return
    atr, avg_vol = atr_and_volume(history)
    asian = asian_range(history, t)
    if not asian or not atr or not avg_vol:
        return
    low, high = asian
    if setup:
        side = setup["side"]
        sign = 1 if side == "LONG" else -1
        if (bar["l"] < setup["extreme"] if sign == 1 else bar["h"] > setup["extreme"]):
            setup = None
        elif setup["stage"] == "SWEEP":
            choch = bar["c"] > setup["break_level"] if sign == 1 else bar["c"] < setup["break_level"]
            displacement = (sign * (bar["c"] - bar["o"]) >= 0.4 * atr
                            and bar["v"] >= 1.3 * avg_vol)
            if choch and displacement:
                setup["stage"] = "CHOCH"
                setup["choch_t"] = t
        elif setup["stage"] == "CHOCH" and len(history) >= 3:
            first = history[-3]
            fvg_low = first["h"] if sign == 1 else bar["h"]
            fvg_high = bar["l"] if sign == 1 else first["l"]
            if fvg_high > fvg_low and t > setup["choch_t"]:
                setup.update(stage="FVG", fvg_low=fvg_low,
                             fvg_high=fvg_high, fvg_t=t)
        elif setup["stage"] == "FVG" and t > setup["fvg_t"]:
            touched = bar["l"] <= setup["fvg_high"] and bar["h"] >= setup["fvg_low"]
            rejected = (bar["c"] > bar["o"] and bar["c"] > setup["fvg_high"]
                        if sign == 1 else
                        bar["c"] < bar["o"] and bar["c"] < setup["fvg_low"])
            if touched and rejected:
                stop = min(setup["extreme"], bar["l"]) - 0.3 * atr if sign == 1 else max(setup["extreme"], bar["h"]) + 0.3 * atr
                open_position(state, symbol, side, bar, stop)
                setup = None
    if setup is None and symbol not in state["positions"]:
        prior = history[-9:-1]
        if len(prior) == 8:
            if (bar["l"] <= low - 0.4 * atr and bar["c"] > low
                    and bar["v"] >= 1.3 * avg_vol):
                setup = {"stage": "SWEEP", "side": "LONG", "day": day,
                         "sweep_t": t, "extreme": bar["l"],
                         "break_level": max(b["h"] for b in prior)}
            elif (bar["h"] >= high + 0.4 * atr and bar["c"] < high
                  and bar["v"] >= 1.3 * avg_vol):
                setup = {"stage": "SWEEP", "side": "SHORT", "day": day,
                         "sweep_t": t, "extreme": bar["h"],
                         "break_level": min(b["l"] for b in prior)}
    if setup:
        state["setups"][symbol] = setup
    else:
        state["setups"].pop(symbol, None)


def main():
    state = load_state()
    errors = []
    for symbol in SYMBOLS:
        try:
            bars = fetch_bars(symbol)
            if len(bars) < 60:
                raise ValueError("Not enough completed candles")
            last = state["last_bar"].get(symbol)
            if last is None:
                # First run never opens historical trades.
                state["last_bar"][symbol] = bars[-1]["t"]
                print(symbol, "initialized; no retroactive entry")
                save_state(state)
                continue
            unseen = [i for i, b in enumerate(bars) if b["t"] > last]
            if len(unseen) > 8:
                # Continue checking existing positions over ALL missed candles,
                # but do not generate new entries from stale candles.
                state["setups"].pop(symbol, None)
            for i in unseen:
                manage_position(state, symbol, bars[i])
                if len(unseen) <= 8:
                    evaluate(state, symbol, bars[:i + 1])
                state["last_bar"][symbol] = bars[i]["t"]
                save_state(state)
            print(symbol, "OK", "new_bars=", len(unseen),
                  "open=", int(symbol in state["positions"]))
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            if symbol == "XAUUSDT":
                # CoinDCX may not list Gold futures as B-XAU_USDT.
                # Do not stop the four crypto scanners if Gold is unavailable.
                print("XAUUSDT SKIPPED (check CoinDCX futures listing):", exc)
            else:
                errors.append(f"{symbol}: {exc}")
                print("ERROR", symbol, exc)
    print(f"COMPLETE configured={len(SYMBOLS)} errors={len(errors)} open={len(state['positions'])}")
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
