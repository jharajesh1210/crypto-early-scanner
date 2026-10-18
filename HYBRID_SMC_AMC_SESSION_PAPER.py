
# HYBRID SMC + AMC POC + ASIAN LONDON NY
# 3 INDEPENDENT STRATEGIES - COINDCX FUTURES
# 20 USER SELECTED COINS - PAPER TRADING ONLY
# Single $100 position per strategy per coin
# 2R reached -> SL at 2R
# 3R reached -> SL at 3R
# 4R reached -> Full Exit. NO PARTIAL EXIT.

import os
import csv
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests

PREFIX = "HYBRID_SMC_AMC_SESSION_PAPER"
STATE = Path(PREFIX + "_state.json")
TRADES = Path(PREFIX + "_trades.csv")
SIGNALS = Path(PREFIX + "_signals.csv")

API = "https://public.coindcx.com/market_data/candlesticks"
ACTIVE_API = (
    "https://api.coindcx.com/exchange/v1/"
    "derivatives/futures/data/active_instruments"
)

MARGIN = 100.0
LEVERAGE = 30.0
FEE_RATE = 0.0005
SLIP_SIDE = 0.0002

MAX_ASIA_RANGE_PCT = 0.07
MIN_RISK_PCT = 0.001
MAX_RISK_PCT = 0.025

REQUESTED_COINS = (
    "BTC", "ETH", "USDT", "BNB", "XRP",
    "USDC", "SOL", "TRX", "HYPE", "ZEC",
    "DOGE", "XMR", "LINK", "LEO", "ADA",
    "NEAR", "XLM", "BCH", "USDE", "LTC"
)

STRATEGIES = (
    "BTC_SMC",
    "AMC_POC",
    "ASIAN_LONDON_NY"
)

LOG = logging.getLogger("hybrid_three_independent")


def utc(ts):
    return datetime.fromtimestamp(ts / 1000, timezone.utc)


def iso(ts):
    return utc(ts).isoformat()


def append_csv(path, row):
    # Preserve existing CSV columns and rows.
    if path.exists() and path.stat().st_size:
        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)

        fields = list(dict.fromkeys(
            old_fields + list(row)
        ))

        if fields != old_fields:
            tmp = path.with_suffix(".csv.tmp")
            with tmp.open(
                "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=fields
                )
                writer.writeheader()
                writer.writerows(old_rows)
            tmp.replace(path)
    else:
        fields = list(row)

    with path.open(
        "a", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def load_state():
    if STATE.exists():
        with STATE.open(encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {}

    defaults = {
        "positions": [],
        "last_closed_bar": {},
        "entered_days": {},
        "smc_setup": {},
        "amc_pending": {},
        "last_15m": {}
    }

    for key, value in defaults.items():
        state.setdefault(key, value)

    return state


def save_state(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )
    tmp.replace(STATE)


def notify(message):
    token = os.getenv("BOT_TOKEN")
    chat = os.getenv("CHAT_ID")

    if not token or not chat:
        LOG.info("Telegram not configured: %s", message)
        return

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat,
                "text": message
            },
            timeout=12
        )
        response.raise_for_status()
    except requests.RequestException as error:
        LOG.warning("Telegram failed: %s", error)


def top20_symbols():
    response = requests.get(
        ACTIVE_API,
        timeout=30
    )
    response.raise_for_status()
    active = response.json()

    if isinstance(active, dict):
        active = active.get(
            "data",
            active.get("instruments", [])
        )

    if not isinstance(active, list):
        raise ValueError(
            "CoinDCX active futures list unavailable"
        )

    pairs = set()

    for item in active:
        if isinstance(item, str):
            pair = item.upper()
        elif isinstance(item, dict):
            pair = str(
                item.get("pair")
                or item.get("instrument")
                or item.get("symbol")
                or item.get("coindcx_name")
                or ""
            ).upper()
        else:
            continue

        if pair.endswith("_USDT"):
            if not pair.startswith("B-"):
                pair = "B-" + pair
            pairs.add(pair)

    selected = []
    unavailable = []

    for coin in REQUESTED_COINS:
        pair = f"B-{coin}_USDT"

        if pair in pairs:
            selected.append(f"{coin}USDT")
        else:
            unavailable.append(coin)

    LOG.info(
        "USER SELECTED COINS: %s",
        ", ".join(REQUESTED_COINS)
    )
    LOG.info(
        "AVAILABLE COINDCX FUTURES (%d): %s",
        len(selected),
        ", ".join(selected) or "NONE"
    )

    if unavailable:
        LOG.warning(
            "SKIPPED: %s",
            ", ".join(unavailable)
        )

    if not selected:
        raise ValueError(
            "No requested active futures contracts"
        )

    return selected


def candles(symbol, resolution="5", days=4):
    now = int(time.time())
    start = now - days * 86400

    step = (
        2 * 86400
        if resolution == "5"
        else 10 * 86400
    )

    rows = []

    for left in range(start, now, step):
        response = requests.get(
            API,
            params={
                "pair": "B-" + symbol[:-4] + "_USDT",
                "pcode": "f",
                "resolution": resolution,
                "from": left,
                "to": min(left + step, now)
            },
            timeout=30
        )
        response.raise_for_status()

        payload = response.json()

        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("data"), list)
        ):
            raise ValueError(
                "Invalid CoinDCX candle response"
            )

        rows.extend(payload["data"])

    duration = int(resolution) * 60 * 1000
    result = {}

    for row in rows:
        try:
            timestamp = int(float(row["time"]))

            if timestamp < 100_000_000_000:
                timestamp *= 1000

            if (
                timestamp + duration
                > int(time.time() * 1000)
            ):
                continue

            bar = {
                "t": timestamp,
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
                "v": float(row.get("volume", 0))
            }

            if bar["h"] >= bar["l"] and bar["c"] > 0:
                result[timestamp] = bar

        except (ValueError, KeyError, TypeError):
            continue

    return [
        result[t]
        for t in sorted(result)
    ]


def approximate_poc(bars, bins=24):
    low = min(b["l"] for b in bars)
    high = max(b["h"] for b in bars)

    if high <= low:
        return low

    volume = [0.0] * bins

    for bar in bars:
        start = max(
            0,
            min(
                bins - 1,
                int(
                    (bar["l"] - low)
                    / (high - low) * bins
                )
            )
        )

        end = max(
            0,
            min(
                bins - 1,
                int(
                    (bar["h"] - low)
                    / (high - low) * bins
                )
            )
        )

        per_bin = bar["v"] / (end - start + 1)

        for index in range(start, end + 1):
            volume[index] += per_bin

    index = volume.index(max(volume))

    return (
        low
        + (index + 0.5)
        * (high - low) / bins
    )


def direction_sign(direction):
    return (
        1
        if direction in ("LONG", "BUY")
        else -1
    )


def open_trade(
    state, symbol, strategy,
    direction, bar, stop, poc=None
):
    # One open position per strategy per coin.
    for position in state["positions"]:
        if (
            position.get("symbol") == symbol
            and position.get("strategy") == strategy
        ):
            return False

    sign = direction_sign(direction)

    entry = bar["c"] * (1 + sign * SLIP_SIDE)
    risk = sign * (entry - stop)

    if (
        entry <= 0
        or risk <= 0
        or not 0.0005 <= risk / entry <= 0.10
    ):
        return False

    target = entry + sign * 4 * risk

    position = {
        "symbol": symbol,
        "strategy": strategy,
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "initial_stop": stop,
        "target": target,
        "risk": risk,
        "locked_rr": 0,
        "entry_time_utc": iso(bar["t"]),
        "entry_t": bar["t"],
        "last_checked_t": bar["t"],
        "margin_usdt": MARGIN,
        "rr": 4,
        "id": f"{symbol}-{strategy}-{bar['t']}",
        "poc": poc
    }

    state["positions"].append(position)

    append_csv(
        SIGNALS,
        {
            "trade_id": position["id"],
            "symbol": symbol,
            "strategy": strategy,
            "direction": direction,
            "entry_time_utc": position["entry_time_utc"],
            "entry": entry,
            "stop": stop,
            "target": target,
            "rr": 4,
            "margin_usdt": MARGIN
        }
    )

    notify(
        f"HYBRID PAPER ENTRY | {strategy}\n"
        f"{symbol} {direction}\n"
        f"Entry: {entry:.8g}\n"
        f"SL: {stop:.8g}\n"
        f"2R/3R Trailing SL, Final 4R: {target:.8g}\n"
        f"Margin: ${MARGIN:.0f}\n"
        f"Leverage: {LEVERAGE:.0f}x\n"
        "PAPER TRADING ONLY"
    )

    LOG.info(
        "ENTRY %s %s %s",
        strategy, symbol, direction
    )

    return True


def settle(state, symbol, bars):
    remaining = []

    for pos in state["positions"]:
        if pos.get("symbol") != symbol:
            remaining.append(pos)
            continue

        sign = direction_sign(pos["direction"])
        entry = float(pos["entry"])

        original = float(
            pos.get("initial_stop", pos["stop"])
        )
        risk = float(
            pos.get("risk", abs(entry - original))
        )

        outcome = None

        for bar in bars:
            last_checked = int(
                pos.get(
                    "last_checked_t",
                    pos.get("entry_t", 0)
                )
            )

            if bar["t"] <= last_checked:
                continue

            stop = float(pos["stop"])
            target = float(pos["target"])

            hit_stop = (
                bar["l"] <= stop
                if sign == 1
                else bar["h"] >= stop
            )

            hit_target = (
                bar["h"] >= target
                if sign == 1
                else bar["l"] <= target
            )

            # Conservative OHLC assumption:
            # existing SL is checked before TP.
            if hit_stop or hit_target:
                reason = (
                    "TRAIL_SL"
                    if hit_stop and pos.get("locked_rr", 0) > 0
                    else "STOP"
                    if hit_stop
                    else "TP_RR4"
                )

                exit_at = (
                    stop if hit_stop else target
                )

                outcome = (
                    reason,
                    exit_at,
                    bar["t"]
                )
                break

            # Old positions without a strategy retain
            # their previous per-leg management.
            if not pos.get("strategy"):
                leg = pos.get("leg", "")

                if leg in ("3R", "4R"):
                    threshold = (
                        2 if leg == "3R" else 3
                    )

                    best = (
                        bar["h"]
                        if sign == 1
                        else bar["l"]
                    )

                    progress = sign * (best - entry)

                    if progress >= threshold * risk:
                        proposed = (
                            entry
                            + sign * (threshold - 1) * risk
                        )

                        pos["stop"] = (
                            max(pos["stop"], proposed)
                            if sign == 1
                            else min(pos["stop"], proposed)
                        )

                pos["last_checked_t"] = bar["t"]
                continue

            best = (
                bar["h"]
                if sign == 1
                else bar["l"]
            )

            reached = (
                sign * (best - entry) / risk
                if risk > 0
                else 0
            )

            locked = int(
                pos.get("locked_rr", 0)
            )

            if reached >= 3:
                new_lock = 3
            elif reached >= 2:
                new_lock = 2
            else:
                new_lock = locked

            if new_lock > locked:
                pos["locked_rr"] = new_lock
                pos["stop"] = (
                    entry + sign * new_lock * risk
                )

                notify(
                    f"HYBRID PAPER TRAIL\n"
                    f"{pos.get('strategy')} {symbol}\n"
                    f"1:{new_lock} reached\n"
                    f"Full position SL: "
                    f"{pos['stop']:.8g}\n"
                    "Effective next candle"
                )

            pos["last_checked_t"] = bar["t"]

        if outcome is None:
            remaining.append(pos)
            continue

        reason, raw_exit, timestamp = outcome

        exit_price = (
            raw_exit * (1 - sign * SLIP_SIDE)
        )

        gross_pct = (
            sign * (exit_price / entry - 1) * 100
        )

        net_pct = (
            gross_pct
            - FEE_RATE
            * (1 + exit_price / entry) * 100
        )

        margin = float(
            pos.get("margin_usdt", MARGIN)
        )

        pnl = (
            net_pct / 100 * margin * LEVERAGE
        )

        append_csv(
            TRADES,
            {
                "trade_id": pos.get("id", ""),
                "symbol": symbol,
                "strategy": pos.get(
                    "strategy", "LEGACY_HYBRID"
                ),
                "direction": pos["direction"],
                "entry_time_utc": pos.get(
                    "entry_time_utc", ""
                ),
                "exit_time_utc": iso(timestamp),
                "entry_price": entry,
                "exit_price": exit_price,
                "stop_loss": original,
                "target": target,
                "exit_reason": reason,
                "rr": pos.get("rr", 4),
                "margin_usdt": margin,
                "net_return_pct_per_leg": round(
                    net_pct, 6
                ),
                "paper_pnl_usdt": round(pnl, 2)
            }
        )

        notify(
            f"HYBRID PAPER EXIT\n"
            f"{pos.get('strategy', 'LEGACY')} "
            f"{symbol}\n"
            f"{reason}\n"
            f"Net Return: {net_pct:+.3f}%\n"
            f"Paper P/L: {pnl:+.2f} USDT"
        )

        LOG.info(
            "EXIT %s %s %s %.2f USDT",
            pos.get("strategy"),
            symbol,
            reason,
            pnl
        )

    state["positions"] = remaining


def swing_points(bars, n=3):
    highs = []
    lows = []

    for i in range(n, len(bars) - n):
        window = bars[i - n:i + n + 1]
        bar = bars[i]

        if (
            bar["h"] == max(x["h"] for x in window)
            and sum(
                x["h"] == bar["h"]
                for x in window
            ) == 1
        ):
            highs.append(bar["h"])

        if (
            bar["l"] == min(x["l"] for x in window)
            and sum(
                x["l"] == bar["l"]
                for x in window
            ) == 1
        ):
            lows.append(bar["l"])

    return highs, lows


def h1_bias(hourly, bar):
    eligible = [
        b for b in hourly
        if b["t"] + 3600000
        <= bar["t"] + 300000
    ]

    highs, lows = swing_points(eligible)

    if len(highs) < 2 or len(lows) < 2:
        return "NEUTRAL"

    if (
        highs[-1] > highs[-2]
        and lows[-1] > lows[-2]
    ):
        return "LONG"

    if (
        highs[-1] < highs[-2]
        and lows[-1] < lows[-2]
    ):
        return "SHORT"

    return "NEUTRAL"


def btc_smc_step(
    state, symbol, history, hourly, bar
):
    # 1H trend + liquidity sweep
    # + CHOCH + FVG + rejection retest.
    bias = h1_bias(hourly, bar)

    pending = state["smc_setup"].get(symbol)

    if pending:
        expired = (
            bar["t"] - pending["sweep_t"]
            > 18 * 300000
        )

        if expired or pending["side"] != bias:
            pending = None

    if pending:
        is_long = pending["side"] == "LONG"

        invalidated = (
            bar["l"] < pending["extreme"]
            if is_long
            else bar["h"] > pending["extreme"]
        )

        if invalidated:
            pending = None

        elif pending["stage"] == "SWEEP":
            level = pending["break_level"]

            if level is not None:
                confirmed = (
                    bar["c"] > level
                    if is_long
                    else bar["c"] < level
                )

                if confirmed:
                    pending["stage"] = "CHOCH"

        elif (
            pending["stage"] == "CHOCH"
            and len(history) >= 3
        ):
            first = history[-3]

            fvg = (
                bar["l"] > first["h"]
                if is_long
                else bar["h"] < first["l"]
            )

            if fvg:
                pending.update({
                    "stage": "FVG",
                    "fvg_low": (
                        first["h"]
                        if is_long
                        else bar["h"]
                    ),
                    "fvg_high": (
                        bar["l"]
                        if is_long
                        else first["l"]
                    ),
                    "fvg_t": bar["t"]
                })

        elif pending["stage"] == "FVG":
            expired = (
                bar["t"] - pending["fvg_t"]
                > 12 * 300000
            )

            if expired:
                pending = None
            else:
                touched = (
                    bar["l"] <= pending["fvg_high"]
                    and bar["h"] >= pending["fvg_low"]
                )

                rejection = (
                    (
                        bar["c"] > bar["o"]
                        and bar["c"] > pending["fvg_high"]
                    )
                    if is_long
                    else (
                        bar["c"] < bar["o"]
                        and bar["c"] < pending["fvg_low"]
                    )
                )

                if touched and rejection:
                    entry = bar["c"]
                    stop = pending["extreme"]
                    risk_pct = (
                        abs(entry - stop) / entry * 100
                    )

                    valid_direction = (
                        stop < entry
                        if is_long
                        else stop > entry
                    )

                    if (
                        0.05 <= risk_pct <= 2.50
                        and valid_direction
                    ):
                        open_trade(
                            state,
                            symbol,
                            "BTC_SMC",
                            pending["side"],
                            bar,
                            stop
                        )

                    pending = None

    already_open = any(
        p.get("symbol") == symbol
        and p.get("strategy") == "BTC_SMC"
        for p in state["positions"]
    )

    if (
        pending is None
        and bias in ("LONG", "SHORT")
        and not already_open
    ):
        previous = history[:-1]
        highs, lows = swing_points(previous)

        if (
            bias == "LONG"
            and lows
            and bar["l"] < lows[-1]
            and bar["c"] > lows[-1]
        ):
            pending = {
                "side": "LONG",
                "stage": "SWEEP",
                "sweep_t": bar["t"],
                "extreme": bar["l"],
                "break_level": (
                    highs[-1] if highs else None
                )
            }

        elif (
            bias == "SHORT"
            and highs
            and bar["h"] > highs[-1]
            and bar["c"] < highs[-1]
        ):
            pending = {
                "side": "SHORT",
                "stage": "SWEEP",
                "sweep_t": bar["t"],
                "extreme": bar["h"],
                "break_level": (
                    lows[-1] if lows else None
                )
            }

    if pending is None:
        state["smc_setup"].pop(symbol, None)
    else:
        state["smc_setup"][symbol] = pending


def bars_15m(bars):
    groups = {}

    for bar in bars:
        start = bar["t"] // 900000 * 900000
        groups.setdefault(start, []).append(bar)

    result = []

    for timestamp, group in sorted(groups.items()):
        expected = [
            timestamp,
            timestamp + 300000,
            timestamp + 600000
        ]

        if (
            len(group) != 3
            or [b["t"] for b in group] != expected
        ):
            continue

        result.append({
            "t": timestamp,
            "o": group[0]["o"],
            "h": max(b["h"] for b in group),
            "l": min(b["l"] for b in group),
            "c": group[-1]["c"],
            "v": sum(b["v"] for b in group)
        })

    return result


def amc_poc_hourly(segment):
    typical = [
        (b["h"] + b["l"] + b["c"]) / 3
        for b in segment
    ]

    if not typical:
        return 0.0

    low = min(typical)
    high = max(typical)
    total = sum(b["v"] for b in segment)

    if high <= low or total <= 0:
        return typical[-1]

    bins = [0.0] * 30

    for price, bar in zip(typical, segment):
        index = min(
            29,
            max(
                0,
                int(
                    (price - low)
                    / (high - low) * 30
                )
            )
        )
        bins[index] += bar["v"]

    index = max(
        range(30),
        key=lambda i: bins[i]
    )

    return (
        low
        + (index + 0.5)
        * (high - low) / 30
    )


def amc_region(hourly, start):
    prior = [
        bar for bar in hourly
        if bar["t"] + 3600000 <= start
    ]

    if len(prior) < 4:
        return None

    segment = prior[-4:]

    high = max(b["h"] for b in segment)
    low = min(b["l"] for b in segment)
    avg = sum(b["c"] for b in segment) / 4

    if (
        avg <= 0
        or (high - low) / avg * 100 >= 2.0
    ):
        return None

    return {
        "high": high,
        "low": low,
        "poc": amc_poc_hourly(segment)
    }


def amc_step(state, symbol, bar, hourly):
    # 4H accumulation, 15M sweep,
    # next 15M confirmation.
    pending = state["amc_pending"].pop(
        symbol, None
    )

    if (
        pending
        and pending["t"] == bar["t"] - 900000
    ):
        is_long = pending["side"] == "LONG"
        buffer = 0.0008

        confirmed = (
            bar["c"]
            > pending["sweep_high"] * (1 + buffer)
            if is_long
            else bar["c"]
            < pending["sweep_low"] * (1 - buffer)
        )

        if confirmed:
            end = bar["t"] + 900000

            # Only recently completed candles.
            age_minutes = (
                time.time() * 1000 - end
            ) / 60000

            if 0 <= age_minutes <= 5:
                entry_bar = {
                    "t": end,
                    "c": bar["c"]
                }

                open_trade(
                    state,
                    symbol,
                    "AMC_POC",
                    pending["side"],
                    entry_bar,
                    pending["stop"],
                    pending["poc"]
                )

    region = amc_region(hourly, bar["t"])

    if region:
        high = region["high"]
        low = region["low"]

        if (
            bar["l"] < low
            and low < bar["c"] <= high
        ):
            side = "LONG"

        elif (
            bar["h"] > high
            and low <= bar["c"] < high
        ):
            side = "SHORT"

        else:
            side = None

        if side:
            state["amc_pending"][symbol] = {
                "t": bar["t"],
                "side": side,
                "sweep_high": bar["h"],
                "sweep_low": bar["l"],
                "stop": (
                    bar["l"]
                    if side == "LONG"
                    else bar["h"]
                ),
                "poc": region["poc"]
            }


def find_setup(bars, last_ts):
    # Asian 00:00-08:00 UTC
    # London sweep 08:00-10:00 UTC
    # CHOCH + FVG + fresh retest.

    if not bars:
        return None

    day = utc(bars[-1]["t"]).date()

    today = [
        b for b in bars
        if utc(b["t"]).date() == day
    ]

    asia = [
        b for b in today
        if 0 <= utc(b["t"]).hour < 8
    ]

    if len(asia) != 96:
        return None

    asian_high = max(b["h"] for b in asia)
    asian_low = min(b["l"] for b in asia)

    if (
        asian_low <= 0
        or (
            (asian_high - asian_low)
            / asian_low
        ) > MAX_ASIA_RANGE_PCT
    ):
        return None

    poc = approximate_poc(asia)

    london = [
        b for b in today
        if 8 <= utc(b["t"]).hour < 10
    ]

    if not london:
        return None

    for i, sweep in enumerate(london):
        if sweep["l"] < asian_low:
            direction = "BUY"
            extreme = sweep["l"]
            boundary = asian_low

            returned = next(
                (
                    j for j in range(
                        i, min(i + 3, len(london))
                    )
                    if london[j]["c"] > boundary
                ),
                None
            )

        elif sweep["h"] > asian_high:
            direction = "SELL"
            extreme = sweep["h"]
            boundary = asian_high

            returned = next(
                (
                    j for j in range(
                        i, min(i + 3, len(london))
                    )
                    if london[j]["c"] < boundary
                ),
                None
            )

        else:
            continue

        if returned is None:
            continue

        return_time = london[returned]["t"]

        prior = [
            b for b in today
            if b["t"] < sweep["t"]
        ][-3:]

        if len(prior) < 3:
            continue

        swing = (
            max(b["h"] for b in prior)
            if direction == "BUY"
            else min(b["l"] for b in prior)
        )

        post = [
            b for b in today
            if (
                b["t"] > return_time
                and 8 <= utc(b["t"]).hour < 18
            )
        ]

        choch = next(
            (
                b for b in post
                if (
                    b["c"] > swing
                    if direction == "BUY"
                    else b["c"] < swing
                )
            ),
            None
        )

        if choch is None:
            continue

        for k in range(2, len(today)):
            first = today[k - 2]
            middle = today[k - 1]
            third = today[k]

            if (
                third["t"] < choch["t"]
                or first["t"] < return_time
            ):
                continue

            if not (
                8 <= utc(third["t"]).hour < 18
            ):
                continue

            if direction == "BUY":
                low = first["h"]
                high = third["l"]

                displacement = (
                    middle["c"] > middle["o"]
                    and (
                        middle["c"] - middle["o"]
                    ) > 0.35 * (
                        middle["h"] - middle["l"]
                    )
                )

            else:
                low = third["h"]
                high = first["l"]

                displacement = (
                    middle["c"] < middle["o"]
                    and (
                        middle["o"] - middle["c"]
                    ) > 0.35 * (
                        middle["h"] - middle["l"]
                    )
                )

            if high <= low or not displacement:
                continue

            for retest in today[k + 1:]:
                if (
                    retest["t"] <= last_ts
                    or not (
                        8 <= utc(retest["t"]).hour < 18
                    )
                ):
                    continue

                touched = (
                    retest["l"] <= high
                    and retest["h"] >= low
                )

                if not touched:
                    continue

                entry = retest["c"]

                if (
                    direction == "BUY"
                    and (
                        entry <= low
                        or entry <= extreme
                    )
                ):
                    continue

                if (
                    direction == "SELL"
                    and (
                        entry >= high
                        or entry >= extreme
                    )
                ):
                    continue

                stop = (
                    min(extreme, retest["l"])
                    if direction == "BUY"
                    else max(extreme, retest["h"])
                )

                risk = (
                    entry - stop
                    if direction == "BUY"
                    else stop - entry
                )

                if (
                    entry <= 0
                    or not (
                        MIN_RISK_PCT
                        <= risk / entry
                        <= MAX_RISK_PCT
                    )
                ):
                    continue

                poc_confirms = (
                    entry > poc
                    if direction == "BUY"
                    else entry < poc
                )

                return {
                    "day": str(day),
                    "direction": direction,
                    "entry": entry,
                    "stop": stop,
                    "risk": risk,
                    "poc": poc,
                    "poc_confirms": poc_confirms,
                    "asian_high": asian_high,
                    "asian_low": asian_low,
                    "bar_t": retest["t"]
                }

    return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s"
    )

    state = load_state()
    selected = top20_symbols()

    # Include any symbols with open positions.
    held = {
        p["symbol"]
        for p in state["positions"]
    }

    symbols = list(
        dict.fromkeys(
            selected + sorted(held)
        )
    )

    failures = []
    scanned = 0

    for symbol in symbols:
        try:
            bars = candles(
                symbol, "5", days=6
            )
            hourly = candles(
                symbol, "60", days=18
            )

            if (
                len(bars) < 100
                or len(hourly) < 30
            ):
                LOG.warning(
                    "%s insufficient candles",
                    symbol
                )
                continue

            # Monitor existing positions first.
            settle(state, symbol, bars)

            last = int(
                state["last_closed_bar"].get(
                    symbol, 0
                )
            )

            # First run per coin:
            # initialize, no fake historical trades.
            if not last:
                state["last_closed_bar"][symbol] = (
                    bars[-1]["t"]
                )

                complete15 = bars_15m(bars)

                if complete15:
                    state["last_15m"][symbol] = (
                        complete15[-1]["t"]
                    )

                LOG.info(
                    "%s initialized",
                    symbol
                )

                save_state(state)
                scanned += 1
                continue

            new_indices = [
                i for i, bar in enumerate(bars)
                if bar["t"] > last
            ]

            if len(new_indices) > 12:
                LOG.warning(
                    "%s missed %d bars; "
                    "no backfill entries",
                    symbol,
                    len(new_indices)
                )

                state["smc_setup"].pop(
                    symbol, None
                )
                state["amc_pending"].pop(
                    symbol, None
                )

                new_indices = []

            # STRATEGY 1: BTC SMC
            for i in new_indices:
                btc_smc_step(
                    state,
                    symbol,
                    bars[:i + 1],
                    hourly,
                    bars[i]
                )

            # STRATEGY 2: AMC POC
            complete15 = bars_15m(bars)

            if (
                symbol not in state["last_15m"]
                and complete15
            ):
                state["last_15m"][symbol] = (
                    complete15[-1]["t"]
                )

            last15 = int(
                state["last_15m"].get(
                    symbol, 0
                )
            )

            for bar in complete15:
                if (
                    bar["t"] > last15
                    and bar["t"] + 900000
                    <= int(time.time() * 1000)
                ):
                    amc_step(
                        state,
                        symbol,
                        bar,
                        hourly
                    )

            if complete15:
                state["last_15m"][symbol] = (
                    complete15[-1]["t"]
                )

            # STRATEGY 3: ASIAN LONDON NY
            today = str(
                utc(bars[-1]["t"]).date()
            )

            if (
                state["entered_days"].get(
                    symbol
                ) != today
            ):
                setup = find_setup(
                    bars, last
                )

                if (
                    setup
                    and setup["bar_t"]
                    == bars[-1]["t"]
                ):
                    opened = open_trade(
                        state,
                        symbol,
                        "ASIAN_LONDON_NY",
                        setup["direction"],
                        {
                            "t": setup["bar_t"],
                            "c": setup["entry"]
                        },
                        setup["stop"],
                        setup["poc"]
                    )

                    if opened:
                        state["entered_days"][
                            symbol
                        ] = today

            state["last_closed_bar"][
                symbol
            ] = bars[-1]["t"]

            scanned += 1
            save_state(state)

            open_count = sum(
                p.get("symbol") == symbol
                for p in state["positions"]
            )

            LOG.info(
                "%s scan OK; open=%d",
                symbol,
                open_count
            )

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
            ZeroDivisionError
        ) as error:
            failures.append(
                f"{symbol}: {error}"
            )

            LOG.exception(
                "%s scan failed",
                symbol
            )

            save_state(state)

    LOG.info(
        "COMPLETE selected=%d "
        "scanned=%d failures=%d "
        "open_positions=%d",
        len(selected),
        scanned,
        len(failures),
        len(state["positions"])
    )

    if failures:
        LOG.error(
            "Failed coins: %s",
            "; ".join(failures)
        )

    if scanned == 0:
        raise RuntimeError(
            "No coins successfully scanned"
        )


if __name__ == "__main__":
    main()
