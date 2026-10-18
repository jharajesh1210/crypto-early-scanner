
"""NEW ASIA LONDO NY V2 - CoinDCX Futures Paper Scanner."""

import csv
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

NAME = "NEW_ASIA_LONDO_NY"
STATE_FILE = Path(NAME + "_state.json")
TRADES_FILE = Path(NAME + "_trades.csv")

SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XAUUSDT",
)

API = "https://public.coindcx.com/market_data/candlesticks"

MARGIN = 100.0
LEVERAGE = 30.0
FEE_PER_SIDE = 0.0005
SLIPPAGE = 0.0002
BAR_MS = 15 * 60 * 1000

LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")


def timestamp(ms):
    return datetime.fromtimestamp(
        ms / 1000, timezone.utc
    ).isoformat()


def alert(message):
    print(message.replace("\n", " | "))

    token = os.getenv("BOT_TOKEN")
    chat = os.getenv("CHAT_ID")

    if token and chat:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": message},
                timeout=12,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print("Telegram warning:", exc)


def load_state():
    if STATE_FILE.exists():
        data = json.loads(
            STATE_FILE.read_text(encoding="utf-8")
        )
    else:
        data = {}

    for key in (
        "last_bar",
        "setups",
        "positions",
        "traded_days",
    ):
        data.setdefault(key, {})

    return data


def save_state(data):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    tmp.replace(STATE_FILE)


def fetch_bars(symbol):
    now = int(time.time())
    pair = "B-" + symbol[:-4] + "_USDT"

    response = requests.get(
        API,
        params={
            "pair": pair,
            "pcode": "f",
            "resolution": "15",
            "from": now - 5 * 86400,
            "to": now,
        },
        timeout=35,
    )
    response.raise_for_status()
    payload = response.json()

    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("data"), list)
    ):
        raise ValueError("Unexpected CoinDCX response")

    bars = {}

    for item in payload["data"]:
        try:
            t = int(float(item["time"]))

            if t < 100_000_000_000:
                t *= 1000

            if t + BAR_MS > int(time.time() * 1000):
                continue

            bar = {
                "t": t,
                "o": float(item["open"]),
                "h": float(item["high"]),
                "l": float(item["low"]),
                "c": float(item["close"]),
                "v": float(item.get("volume", 0)),
            }

            if bar["c"] > 0 and bar["h"] >= bar["l"]:
                bars[t] = bar

        except (ValueError, TypeError, KeyError):
            continue

    return [bars[t] for t in sorted(bars)]


def atr_and_volume(history):
    if len(history) < 22:
        return None, None

    previous = history[-22:-1]
    trs = []

    for i in range(1, len(previous)):
        bar = previous[i]
        old = previous[i - 1]

        trs.append(
            max(
                bar["h"] - bar["l"],
                abs(bar["h"] - old["c"]),
                abs(bar["l"] - old["c"]),
            )
        )

    atr = sum(trs[-14:]) / 14
    avg_vol = sum(
        bar["v"] for bar in previous[-20:]
    ) / 20

    return atr, avg_vol


def session_day(t):
    return datetime.fromtimestamp(
        t / 1000, timezone.utc
    ).strftime("%Y-%m-%d")


def local_time(t, tz):
    return datetime.fromtimestamp(
        t / 1000, timezone.utc
    ).astimezone(tz)


def asian_range(history, t):
    day = session_day(t)

    asia = [
        bar for bar in history
        if (
            session_day(bar["t"]) == day
            and 0 <= datetime.fromtimestamp(
                bar["t"] / 1000,
                timezone.utc,
            ).hour < 8
        )
    ]

    if len(asia) != 32:
        return None

    return (
        min(bar["l"] for bar in asia),
        max(bar["h"] for bar in asia),
    )


def session_windows(t):
    london = local_time(t, LONDON_TZ)
    new_york = local_time(t, NY_TZ)

    return {
        "LONDON": 8 <= london.hour < 16,
        "NEW_YORK": 8 <= new_york.hour < 16,
    }


def london_pre_ny_range(history, t):
    ny = local_time(t, NY_TZ)

    if ny.hour < 8:
        return None

    ny_open = ny.replace(
        hour=8,
        minute=0,
        second=0,
        microsecond=0,
    )

    ny_open_ms = int(
        ny_open.timestamp() * 1000
    )

    london_open = datetime.fromtimestamp(
        ny_open_ms / 1000,
        timezone.utc,
    ).astimezone(LONDON_TZ).replace(
        hour=8,
        minute=0,
        second=0,
        microsecond=0,
    )

    london_open_ms = int(
        london_open.timestamp() * 1000
    )

    selected = [
        bar for bar in history
        if (
            london_open_ms
            <= bar["t"]
            < ny_open_ms
        )
    ]

    if len(selected) < 4:
        return None

    return (
        min(bar["l"] for bar in selected),
        max(bar["h"] for bar in selected),
    )


def session_range(history, t, session):
    if session == "LONDON":
        return asian_range(history, t)

    return (
        london_pre_ny_range(history, t)
        or asian_range(history, t)
    )


def record_trade(pos, exit_price, t, reason):
    sign = 1 if pos["side"] == "LONG" else -1
    entry = pos["entry"]

    actual_exit = exit_price * (
        1 - sign * SLIPPAGE
    )

    gross_pct = (
        sign * (actual_exit / entry - 1) * 100
    )

    fee_pct = (
        FEE_PER_SIDE
        * (1 + actual_exit / entry)
        * 100
    )

    net_pct = gross_pct - fee_pct

    pnl = (
        MARGIN * LEVERAGE * net_pct / 100
    )

    row = {
        "trade_id": pos["id"],
        "symbol": pos["symbol"],
        "strategy": "NEW_ASIA_LONDO_NY",
        "session": pos.get("session", "LONDON"),
        "direction": pos["side"],
        "entry_time_utc": timestamp(pos["entry_t"]),
        "exit_time_utc": timestamp(t),
        "entry_price": entry,
        "exit_price": actual_exit,
        "initial_stop": pos["initial_stop"],
        "exit_reason": reason,
        "locked_rr": pos["locked_rr"],
        "margin_usdt": MARGIN,
        "leverage": LEVERAGE,
        "net_return_pct_per_leg": round(
            net_pct, 6
        ),
        "paper_pnl_usdt": round(pnl, 2),
    }

    if (
        TRADES_FILE.exists()
        and TRADES_FILE.stat().st_size
    ):
        with TRADES_FILE.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)

        fields = list(
            dict.fromkeys(old_fields + list(row))
        )

        if fields != old_fields:
            tmp = TRADES_FILE.with_suffix(
                ".csv.tmp"
            )

            with tmp.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=fields
                )
                writer.writeheader()
                writer.writerows(old_rows)

            tmp.replace(TRADES_FILE)
    else:
        fields = list(row)

    with TRADES_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )

        if f.tell() == 0:
            writer.writeheader()

        writer.writerow(row)

    alert(
        "NEW ASIA LONDO NY PAPER EXIT\n"
        f"{pos['symbol']} {pos['side']}\n"
        f"Session: {pos.get('session', 'LONDON')}\n"
        f"{reason}\n"
        f"P/L {pnl:+.2f} USDT"
    )


def manage_position(state, symbol, bar):
    pos = state["positions"].get(symbol)

    if not pos:
        return

    if bar["t"] <= pos["last_checked"]:
        return

    sign = 1 if pos["side"] == "LONG" else -1

    stop = pos["stop"]
    target = pos["target"]

    if sign == 1:
        hit_stop = bar["l"] <= stop
        hit_target = bar["h"] >= target
    else:
        hit_stop = bar["h"] >= stop
        hit_target = bar["l"] <= target

    if hit_stop or hit_target:
        if hit_stop:
            reason = (
                "TRAIL_STOP"
                if pos["locked_rr"]
                else "STOP"
            )
            exit_price = stop
        else:
            reason = "TARGET_4R"
            exit_price = target

        record_trade(
            pos,
            exit_price,
            bar["t"],
            reason,
        )

        del state["positions"][symbol]
        return

    best = (
        bar["h"] if sign == 1 else bar["l"]
    )

    progress = (
        sign * (best - pos["entry"])
        / pos["risk"]
    )

    if progress >= 3:
        lock = 3
    elif progress >= 2:
        lock = 2
    else:
        lock = pos["locked_rr"]

    if lock > pos["locked_rr"]:
        pos["locked_rr"] = lock

        pos["stop"] = (
            pos["entry"]
            + sign * lock * pos["risk"]
        )

        alert(
            "NEW ASIA LONDO NY PAPER TRAIL\n"
            f"{symbol}\n"
            f"Session: {pos.get('session', 'LONDON')}\n"
            f"{lock}R reached\n"
            f"SL: {pos['stop']:.8g}\n"
            "Effective next candle"
        )

    pos["last_checked"] = bar["t"]


def open_position(
    state,
    symbol,
    side,
    bar,
    stop,
    session,
):
    if symbol in state["positions"]:
        return

    sign = 1 if side == "LONG" else -1

    entry = bar["c"] * (
        1 + sign * SLIPPAGE
    )

    risk = sign * (entry - stop)

    if risk <= 0:
        return

    if not (
        0.001 <= risk / entry <= 0.03
    ):
        return

    pos = {
        "id": f"{symbol}-{session}-{bar['t']}",
        "symbol": symbol,
        "side": side,
        "session": session,
        "entry": entry,
        "entry_t": bar["t"],
        "initial_stop": stop,
        "stop": stop,
        "target": entry + sign * 4 * risk,
        "risk": risk,
        "locked_rr": 0,
        "last_checked": bar["t"],
    }

    state["positions"][symbol] = pos

    state["traded_days"][
        f"{symbol}:{session}"
    ] = session_day(bar["t"])

    alert(
        f"NEW ASIA LONDO NY [{session}]\n"
        f"{symbol} {side}\n"
        f"Entry: {entry:.8g}\n"
        f"SL: {stop:.8g}\n"
        f"4R Target: {pos['target']:.8g}\n"
        "$100 Margin | 30x | PAPER ONLY"
    )


def evaluate_session(
    state,
    symbol,
    history,
    session,
):
    bar = history[-1]
    t = bar["t"]
    day = session_day(t)
    key = f"{symbol}:{session}"

    if (
        symbol in state["positions"]
        or state["traded_days"].get(key) == day
        or state["traded_days"].get(symbol) == day
    ):
        state["setups"].pop(key, None)
        return

    setup = state["setups"].get(key)

    if setup and (
        setup["day"] != day
        or t - setup["sweep_t"] > 16 * BAR_MS
    ):
        setup = None

    atr, avg_vol = atr_and_volume(history)
    boundary = session_range(
        history, t, session
    )

    if (
        not boundary
        or not atr
        or not avg_vol
        or avg_vol <= 0
    ):
        return

    low, high = boundary

    if setup:
        side = setup["side"]
        sign = 1 if side == "LONG" else -1

        invalid = (
            bar["l"] < setup["extreme"]
            if sign == 1
            else bar["h"] > setup["extreme"]
        )

        if invalid:
            setup = None

        elif setup["stage"] == "SWEEP":
            choch = (
                bar["c"] > setup["break_level"]
                if sign == 1
                else bar["c"] < setup["break_level"]
            )

            displacement = (
                sign * (bar["c"] - bar["o"])
                >= 0.4 * atr
                and bar["v"] >= 1.3 * avg_vol
            )

            if choch and displacement:
                setup.update(
                    stage="CHOCH",
                    choch_t=t,
                )

        elif (
            setup["stage"] == "CHOCH"
            and len(history) >= 3
        ):
            first = history[-3]

            fvg_low = (
                first["h"]
                if sign == 1
                else bar["h"]
            )

            fvg_high = (
                bar["l"]
                if sign == 1
                else first["l"]
            )

            if (
                fvg_high > fvg_low
                and t > setup["choch_t"]
            ):
                setup.update(
                    stage="FVG",
                    fvg_low=fvg_low,
                    fvg_high=fvg_high,
                    fvg_t=t,
                )

        elif (
            setup["stage"] == "FVG"
            and t > setup["fvg_t"]
        ):
            touched = (
                bar["l"] <= setup["fvg_high"]
                and bar["h"] >= setup["fvg_low"]
            )

            if sign == 1:
                rejected = (
                    bar["c"] > bar["o"]
                    and bar["c"] > setup["fvg_high"]
                )
            else:
                rejected = (
                    bar["c"] < bar["o"]
                    and bar["c"] < setup["fvg_low"]
                )

            if touched and rejected:
                if sign == 1:
                    stop = (
                        min(
                            setup["extreme"],
                            bar["l"],
                        )
                        - 0.3 * atr
                    )
                else:
                    stop = (
                        max(
                            setup["extreme"],
                            bar["h"],
                        )
                        + 0.3 * atr
                    )

                open_position(
                    state,
                    symbol,
                    side,
                    bar,
                    stop,
                    session,
                )

                setup = None

    if (
        setup is None
        and symbol not in state["positions"]
    ):
        prior = history[-9:-1]

        if len(prior) == 8:
            if (
                bar["l"] <= low - 0.4 * atr
                and bar["c"] > low
                and bar["v"] >= 1.3 * avg_vol
            ):
                setup = {
                    "stage": "SWEEP",
                    "side": "LONG",
                    "day": day,
                    "sweep_t": t,
                    "extreme": bar["l"],
                    "break_level": max(
                        b["h"] for b in prior
                    ),
                }

            elif (
                bar["h"] >= high + 0.4 * atr
                and bar["c"] < high
                and bar["v"] >= 1.3 * avg_vol
            ):
                setup = {
                    "stage": "SWEEP",
                    "side": "SHORT",
                    "day": day,
                    "sweep_t": t,
                    "extreme": bar["h"],
                    "break_level": min(
                        b["l"] for b in prior
                    ),
                }

    if setup:
        state["setups"][key] = setup
    else:
        state["setups"].pop(key, None)


def evaluate(state, symbol, history):
    t = history[-1]["t"]
    windows = session_windows(t)

    for session, active in windows.items():
        key = f"{symbol}:{session}"

        if active:
            evaluate_session(
                state,
                symbol,
                history,
                session,
            )
        else:
            state["setups"].pop(key, None)


def main():
    state = load_state()
    errors = []

    for symbol in SYMBOLS:
        try:
            bars = fetch_bars(symbol)

            if len(bars) < 60:
                raise ValueError(
                    "Not enough completed candles"
                )

            last = state["last_bar"].get(symbol)

            if last is None:
                state["last_bar"][symbol] = (
                    bars[-1]["t"]
                )

                print(
                    symbol,
                    "initialized; no retroactive entry",
                )

                save_state(state)
                continue

            unseen = [
                i for i, bar in enumerate(bars)
                if bar["t"] > last
            ]

            if len(unseen) > 8:
                state["setups"].pop(symbol, None)

                for session in (
                    "LONDON",
                    "NEW_YORK",
                ):
                    state["setups"].pop(
                        f"{symbol}:{session}",
                        None,
                    )

            for i in unseen:
                manage_position(
                    state,
                    symbol,
                    bars[i],
                )

                if len(unseen) <= 8:
                    evaluate(
                        state,
                        symbol,
                        bars[:i + 1],
                    )

                state["last_bar"][symbol] = (
                    bars[i]["t"]
                )

                save_state(state)

            print(
                symbol,
                "OK",
                "new_bars=", len(unseen),
                "open=", int(
                    symbol in state["positions"]
                ),
            )

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            if symbol == "XAUUSDT":
                print(
                    "XAUUSDT SKIPPED:",
                    exc,
                )
            else:
                errors.append(
                    f"{symbol}: {exc}"
                )
                print("ERROR", symbol, exc)

    print(
        f"COMPLETE configured={len(SYMBOLS)} "
        f"errors={len(errors)} "
        f"open={len(state['positions'])}"
    )

    if errors:
        raise RuntimeError(
            "; ".join(errors)
        )


if __name__ == "__main__":
    main()
