"""TOP100 CoinDCX futures AMD + approximate POC, FORWARD PAPER ONLY.

Requirements: pip install pandas numpy requests
Run once every 15 minutes; persist *_state.json and *_*.csv between runs.
No trading endpoint, credentials, leverage, or real orders are used.

Assumptions: CoinDCX candlestick `volume` is base-asset units, so the
approximate quote turnover is volume * close. Confirm this on your instrument.
POC is a candle-volume histogram estimate, NOT exchange tick-volume POC.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

NAME = "TOP100_CRIPTO_AMC_POC"
STATE_FILE = Path(f"{NAME}_state.json")
TRADES_FILE = Path(f"{NAME}_trades.csv")
SIGNALS_FILE = Path(f"{NAME}_signals.csv")
UNIVERSE_FILE = Path(f"{NAME}_universe.csv")
COINDCX_CANDLES = "https://public.coindcx.com/market_data/candlesticks"
COINDCX_INSTRUMENTS = "https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments"
COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
MIN_HOURLY_TURNOVER_USDT = 500_000.0
ACCUMULATION_HOURS = 20
MAX_ACCUMULATION_RANGE_PCT = 2.0
ATR_MULTIPLIER = 0.7
CONFIRM_BUFFER = 0.0008
RR_TARGETS = (2, 3)
FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002
MAX_WORKERS = 5
REQUEST_TIMEOUT = 20
LOG = logging.getLogger(NAME)


def utc_now():
    return pd.Timestamp.now(tz="UTC")


def iso(ts=None):
    return (ts if ts is not None else utc_now()).isoformat()


def get_json(url, params=None):
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                                    headers={"User-Agent": "Top100-AMC-POC-Paper/3.0"})
            if response.status_code == 429:
                LOG.warning("Rate limited by %s", url)
                time.sleep(3 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            LOG.warning("Request %d failed: %s: %s", attempt + 1, url, exc)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return None


def notify(message):
    token, chat_id = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat_id:
        LOG.warning("Telegram not configured: %s", message)
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": message[:4000]},
                          timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:
        LOG.error("Telegram notification failed: %s", exc)


def fresh_state():
    return {"version": 3, "last_bar": {}, "pending": {}, "open_trades": [],
            "closed_count": 0, "closed_wins": 0, "net_return_sum_pct": 0.0}


def load_state():
    if not STATE_FILE.exists():
        return fresh_state()
    try:
        with STATE_FILE.open(encoding="utf-8") as f:
            state = json.load(f)
        required = ("last_bar", "pending", "open_trades", "closed_count", "closed_wins")
        if not isinstance(state, dict) or any(k not in state for k in required):
            raise ValueError("Missing required state fields")
        return state
    except (OSError, ValueError, TypeError) as exc:
        # Never reset silently: resetting can duplicate live paper entries.
        raise RuntimeError(f"Invalid state file; fix/restore it before running: {exc}") from exc


def save_state(state):
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, allow_nan=False)
    tmp.replace(STATE_FILE)


def append_csv(path, row):
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def top100():
    data = get_json(COINGECKO_MARKETS, {"vs_currency": "usd", "order": "market_cap_desc",
                                      "per_page": 100, "page": 1, "sparkline": "false"})
    if not isinstance(data, list) or len(data) < 50:
        raise RuntimeError("CoinGecko Top 100 unavailable; no new entries")
    stable = {"usdt", "usdc", "dai", "fdusd", "tusd", "usde", "usdd",
              "pyusd", "usds", "rlusd", "eurc"}
    result = []
    for c in data:
        symbol = str(c.get("symbol", "")).upper()
        if symbol and symbol.lower() not in stable:
            result.append((symbol, str(c.get("name", "")), c.get("market_cap_rank")))
    return result


def futures_pairs():
    data = get_json(COINDCX_INSTRUMENTS)
    if isinstance(data, dict):
        data = data.get("data", data.get("instruments", []))
    if not isinstance(data, list):
        raise RuntimeError("CoinDCX active futures instruments unavailable")
    pairs = {}
    for item in data:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = (item.get("pair") or item.get("instrument") or
                    item.get("symbol") or item.get("coindcx_name") or "")
        else:
            continue
        name = str(name).upper()
        if name.endswith("_USDT"):
            if not name.startswith("B-"):
                name = "B-" + name
            symbol = name[2:-5]
            if symbol:
                pairs[symbol] = name
    if not pairs:
        raise RuntimeError("No CoinDCX USDT futures parsed; check endpoint/schema")
    return pairs


def candles(pair, minutes, days=4):
    end = int(time.time())
    raw = get_json(COINDCX_CANDLES, {"pair": pair, "pcode": "f",
                                     "resolution": str(minutes),
                                     "from": end - days * 86400, "to": end})
    if not isinstance(raw, dict) or raw.get("s") != "ok" or not raw.get("data"):
        return pd.DataFrame()
    df = pd.DataFrame(raw["data"])
    cols = ["time", "open", "high", "low", "close", "volume"]
    if any(col not in df.columns for col in cols):
        return pd.DataFrame()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=cols)
    if df.empty:
        return df
    unit = "ms" if df["time"].median() > 1e11 else "s"
    df["time"] = pd.to_datetime(df["time"], unit=unit, utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    df = df[(df.volume >= 0) & (df.high >= df.low) &
            (df.high >= df.open) & (df.high >= df.close) &
            (df.low <= df.open) & (df.low <= df.close) & (df.close > 0)]
    df = df[df.index + pd.Timedelta(minutes=minutes) <= utc_now()]
    return df


def bars15(df5):
    if df5.empty:
        return pd.DataFrame()
    agg = df5.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = df5.close.resample("15min", label="left", closed="left").count()
    # Exactly 3 completed, correctly aligned 5m candles, never 2/3.
    return agg[counts == 3].dropna()


def approximate_poc(segment):
    price = ((segment.high + segment.low + segment.close) / 3).to_numpy()
    vol = segment.volume.to_numpy()
    if not len(price) or vol.sum() <= 0:
        return None
    if np.ptp(price) == 0:
        return float(price[0])
    hist, edges = np.histogram(price, bins=30, weights=vol)
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def atr(segment, length=14):
    if len(segment) < length + 1:
        return None
    tr = pd.concat([segment.high - segment.low,
                    (segment.high - segment.close.shift()).abs(),
                    (segment.low - segment.close.shift()).abs()], axis=1).max(axis=1)
    return float(tr.tail(length).mean())


def accumulation(hourly, sweep_start):
    prior = hourly[hourly.index + pd.Timedelta(hours=1) <= sweep_start]
    if len(prior) < ACCUMULATION_HOURS:
        return None
    seg = prior.tail(ACCUMULATION_HOURS)
    # Reject gaps in hourly history; otherwise the 20h box is not 20h.
    if len(seg) != ACCUMULATION_HOURS or not (seg.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        return None
    hi, lo, mean = float(seg.high.max()), float(seg.low.min()), float(seg.close.mean())
    if mean <= 0 or (hi - lo) / mean * 100 >= MAX_ACCUMULATION_RANGE_PCT:
        return None
    # Low-volatility regime filter when enough comparison data exists.
    if len(prior) >= 60:
        short_atr, baseline_atr = atr(seg), atr(prior.tail(60))
        if short_atr is not None and baseline_atr and short_atr > baseline_atr * ATR_MULTIPLIER:
            return None
    poc = approximate_poc(seg)
    if poc is None:
        return None
    return {"high": hi, "low": lo, "poc": poc}


def eligible_hourly(item, pair):
    symbol, coin_name, rank = item
    hourly = candles(pair, 60, 4)
    if len(hourly) < ACCUMULATION_HOURS + 1:
        return None
    last = hourly.iloc[-1]
    if hourly.index[-1] + pd.Timedelta(hours=2) < utc_now():
        return None  # stale feed
    turnover = float(last.volume * last.close)  # approximate USDT, base volume assumption
    if not np.isfinite(turnover) or turnover <= MIN_HOURLY_TURNOVER_USDT:
        return None
    return {"symbol": symbol, "name": coin_name, "rank": rank, "pair": pair,
            "hourly": hourly, "turnover": turnover}


def open_paper_legs(state, pair, direction, signal_close_time, entry_raw, stop, poc, turnover):
    # No overlapping paper positions on a symbol (RR2 and RR3 count as one setup).
    if any(t["pair"] == pair for t in state["open_trades"]):
        return
    entry = entry_raw * (1 + SLIPPAGE_PER_SIDE if direction == "LONG" else 1 - SLIPPAGE_PER_SIDE)
    risk = (entry - stop) if direction == "LONG" else (stop - entry)
    if entry <= 0 or risk <= 0 or not 0.0005 <= risk / entry <= 0.10:
        return
    for rr in RR_TARGETS:
        target = entry + rr * risk if direction == "LONG" else entry - rr * risk
        trade = {"id": f"{pair}-{signal_close_time}-RR{rr}", "pair": pair,
                 "direction": direction, "rr": rr, "entry_time": signal_close_time,
                 "entry": float(entry), "stop": float(stop), "target": float(target),
                 "poc": float(poc), "turnover_1h_usdt": turnover,
                 "last_checked_5m": signal_close_time}
        state["open_trades"].append(trade)
        append_csv(SIGNALS_FILE, {"id": trade["id"], "time": signal_close_time,
                                  "pair": pair, "direction": direction, "rr": rr,
                                  "entry_indicative": entry, "stop": stop, "target": target,
                                  "poc_estimate": poc, "hourly_turnover_estimate": turnover})
    notify(f"PAPER ENTRY | {NAME}\n{pair} {direction}\nSignal close: {signal_close_time}\n"
           f"Indicative entry: {entry:.8g}\nSL: {stop:.8g}\nRR 1:2 + 1:3 legs\n"
           f"Approx POC: {poc:.8g}\nLast 1h approx turnover: {turnover:,.0f} USDT\n"
           "Simulation only; no real order. Entry is theoretical close + slippage.")
    LOG.info("Paper entry: %s %s", pair, direction)


def settle_trades(state):
    survivors = []
    cache = {}
    for trade in state["open_trades"]:
        pair = trade["pair"]
        if pair not in cache:
            cache[pair] = candles(pair, 5, 4)
        df = cache[pair]
        if df.empty:
            survivors.append(trade)
            continue
        start = pd.Timestamp(trade["last_checked_5m"])
        # Do not silently skip missing candles, which can conceal stop hits.
        newer = df[df.index >= start]
        if newer.empty or newer.index[0] > start:
            LOG.warning("Missing trade monitoring data %s from %s; not advancing", pair, start)
            survivors.append(trade)
            continue
        result = None
        last_checked = start
        for ts, bar in newer.iterrows():
            if ts != last_checked:
                LOG.warning("5m gap for %s at %s; monitoring paused", pair, last_checked)
                break
            if trade["direction"] == "LONG":
                sl_hit = bar.low <= trade["stop"]
                tp_hit = bar.high >= trade["target"]
            else:
                sl_hit = bar.high >= trade["stop"]
                tp_hit = bar.low <= trade["target"]
            # If both touch within same candle, conservatively assume SL first.
            if sl_hit or tp_hit:
                result = ("SL" if sl_hit else "TP", ts,
                          trade["stop"] if sl_hit else trade["target"])
                break
            last_checked = ts + pd.Timedelta(minutes=5)
        if result is None:
            trade["last_checked_5m"] = iso(last_checked)
            survivors.append(trade)
            continue
        outcome, exit_time, exit_raw = result
        exit_price = exit_raw * (1 - SLIPPAGE_PER_SIDE if trade["direction"] == "LONG" else 1 + SLIPPAGE_PER_SIDE)
        gross = (exit_price / trade["entry"] - 1) * 100
        if trade["direction"] == "SHORT":
            gross = -gross
        net_pct = gross - 2 * FEE_PER_SIDE * 100
        state["closed_count"] += 1
        state["closed_wins"] += int(net_pct > 0)
        state["net_return_sum_pct"] += net_pct
        append_csv(TRADES_FILE, {"id": trade["id"], "pair": pair,
                                 "direction": trade["direction"], "rr": trade["rr"],
                                 "entry_time": trade["entry_time"], "exit_time": iso(exit_time),
                                 "entry": trade["entry"], "stop": trade["stop"],
                                 "target": trade["target"], "exit": exit_price,
                                 "outcome": outcome, "net_return_pct_per_leg": round(net_pct, 4)})
        rate = 100 * state["closed_wins"] / state["closed_count"]
        notify(f"PAPER EXIT | {NAME}\n{trade['id']}\n{outcome} | {net_pct:+.2f}% per leg\n"
               f"Closed legs: {state['closed_count']} | Win rate (legs): {rate:.1f}%\n"
               "Simulation only; no real order.")
    state["open_trades"] = survivors


def process_coin(state, item):
    pair = item["pair"]
    five = candles(pair, 5, 4)
    bars = bars15(five)
    if len(bars) < 3:
        return
    latest = bars.index[-1]
    latest_s = iso(latest)
    prev_s = state["last_bar"].get(pair)
    if prev_s is None:
        state["last_bar"][pair] = latest_s
        LOG.info("Initialized %s at %s; no historical entries", pair, latest_s)
        return
    prev = pd.Timestamp(prev_s)
    if latest <= prev:
        return
    if latest - prev != pd.Timedelta(minutes=15):
        # Do not fabricate fills after GitHub delays / missed 15m bars.
        state["pending"].pop(pair, None)
        state["last_bar"][pair] = latest_s
        LOG.warning("Skipped missed 15m bars for %s; resynchronized", pair)
        return
    state["last_bar"][pair] = latest_s
    bar = bars.iloc[-1]
    pending = state["pending"].pop(pair, None)
    if pending and pd.Timestamp(pending["sweep_time"]) == bars.index[-2]:
        direction = pending["direction"]
        buffer = CONFIRM_BUFFER
        confirmed = (direction == "LONG" and
                     bar.close > pending["sweep_high"] * (1 + buffer) and
                     bar.close > pending["poc"])
        confirmed |= (direction == "SHORT" and
                      bar.close < pending["sweep_low"] * (1 - buffer) and
                      bar.close < pending["poc"])
        if confirmed:
            # 15m candle closes at latest + 15m; no earlier candle is traded.
            open_paper_legs(state, pair, direction, iso(latest + pd.Timedelta(minutes=15)),
                            float(bar.close), pending["stop"], pending["poc"], item["turnover"])
    box = accumulation(item["hourly"], latest)
    if box is None:
        return
    direction = None
    if bar.low < box["low"] and box["low"] < bar.close <= box["high"]:
        direction = "LONG"
    elif bar.high > box["high"] and box["low"] <= bar.close < box["high"]:
        direction = "SHORT"
    if direction:
        state["pending"][pair] = {"direction": direction, "sweep_time": latest_s,
                                  "sweep_high": float(bar.high), "sweep_low": float(bar.low),
                                  "stop": float(bar.low if direction == "LONG" else bar.high),
                                  "poc": box["poc"]}
        LOG.info("%s %s sweep; next completed 15m candle must confirm", pair, direction)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state = load_state()
    LOG.info("Starting %s | paper only | %s", NAME, iso())
    # Always monitor existing positions even if CoinGecko is down.
    settle_trades(state)
    save_state(state)
    try:
        market = top100()
        pairs = futures_pairs()
        candidates = [(coin, pairs[coin[0]]) for coin in market if coin[0] in pairs]
        LOG.info("CoinGecko top100 matched to CoinDCX futures: %d", len(candidates))
        if not candidates:
            raise RuntimeError("No matched futures; check CoinDCX instrument schema")
        eligible = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            jobs = [pool.submit(eligible_hourly, coin, pair) for coin, pair in candidates]
            for job in as_completed(jobs):
                try:
                    result = job.result()
                    if result:
                        eligible.append(result)
                except Exception as exc:
                    LOG.warning("Candidate skipped: %s", exc)
        eligible.sort(key=lambda c: c["rank"] if isinstance(c["rank"], int) else 99999)
        pd.DataFrame([{"symbol": x["symbol"], "rank": x["rank"],
                       "pair": x["pair"], "approx_1h_turnover_usdt": round(x["turnover"], 2)}
                      for x in eligible],
                     columns=["symbol", "rank", "pair", "approx_1h_turnover_usdt"]).to_csv(
                         UNIVERSE_FILE, index=False)
        LOG.info("Passed last completed 1h >500k approximate USDT turnover: %d", len(eligible))
        for item in eligible:
            try:
                process_coin(state, item)
                save_state(state)
            except Exception as exc:
                LOG.exception("Pair processing failed %s: %s", item["pair"], exc)
        LOG.info("Finished | eligible=%d open_legs=%d closed_legs=%d",
                 len(eligible), len(state["open_trades"]), state["closed_count"])
    finally:
        save_state(state)


if __name__ == "__main__":
    main()
