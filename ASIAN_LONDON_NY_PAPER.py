"""Asian range / London sweep / NY continuation, CoinDCX USDT perpetual PAPER scanner.
No live orders. 5m closed bars only. Four symbols. Persist JSON state + CSV legs.
"""
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

PREFIX = 'ASIAN_LONDON_NY_PAPER'
STATE = Path(PREFIX + '_state.json')
TRADES = Path(PREFIX + '_trades.csv')
SIGNALS = Path(PREFIX + '_signals.csv')
SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT')
API = 'https://public.coindcx.com/market_data/candlesticks'
MARGIN = 100.0
LEVERAGE = 30.0
FEE_RATE = 0.0005  # illustrative taker fee each side, adjust to account
MIN_RISK_PCT = 0.001
MAX_RISK_PCT = 0.025
MIN_RR = 1.2
MAX_ASIA_RANGE_PCT = 0.07
LOG = logging.getLogger('amd_paper')


def utc(ts):
    return datetime.fromtimestamp(ts / 1000, timezone.utc)


def iso(ts):
    return utc(ts).isoformat(timespec='minutes')


def append_csv(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def load_state():
    if STATE.exists():
        with STATE.open(encoding='utf-8') as f:
            s = json.load(f)
        s.setdefault('positions', [])
        s.setdefault('entered_days', {})
        s.setdefault('last_closed_bar', {})
        return s
    return {'positions': [], 'entered_days': {}, 'last_closed_bar': {}}


def save_state(state):
    tmp = STATE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2), encoding='utf-8')
    tmp.replace(STATE)


def notify(message):
    token, chat = os.getenv('BOT_TOKEN'), os.getenv('CHAT_ID')
    if not token or not chat:
        LOG.info('Telegram not configured: %s', message)
        return
    try:
        r = requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                          json={'chat_id': chat, 'text': message}, timeout=12)
        r.raise_for_status()
    except requests.RequestException as exc:
        LOG.warning('Telegram failed: %s', exc)


def candles(symbol):
    """CoinDCX USDT perpetual 5m candles, sorted ascending; closed bars only."""
    pair = f'B-{symbol[:-4]}_USDT'
    end_s = int(time.time())
    start_s = end_s - 4 * 24 * 60 * 60
    response = requests.get(API, params={
        'pair': pair, 'from': start_s, 'to': end_s,
        'resolution': '5', 'pcode': 'f'
    }, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get('s') != 'ok':
        raise ValueError(f'CoinDCX bad response for {pair}: {str(payload)[:250]}')
    raw = payload.get('data')
    if not isinstance(raw, list) or not raw:
        raise ValueError(f'CoinDCX returned no candles for {pair}')
    now_ms = int(time.time() * 1000)
    unique = {}
    for x in raw:
        t = int(x['time'])
        if t + 5 * 60 * 1000 <= now_ms - 1500:
