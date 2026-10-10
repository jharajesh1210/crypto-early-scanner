#!/usr/bin/env python3
"""V23 CoinDCX BTC perpetual: pure SMC, closed-candle paper simulator only."""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PAIR = 'B-BTC_USDT'
BASE = 'https://public.coindcx.com/market_data/candles'
STATE_FILE = Path('crypto_futures_v23_smc_state.json')
TRADES_FILE = Path('crypto_futures_v23_smc_trades.csv')
SIGNALS_FILE = Path('crypto_futures_v23_smc_signals.csv')
BOT_TOKEN = os.getenv('BOT_TOKEN', '')
CHAT_ID = os.getenv('CHAT_ID', '')
SWING = 3
SWEEP_WINDOW = 18  # five-minute candles
CHOCH_WINDOW = 12
FVG_WINDOW = 12
MAX_SETUP_AGE = 36
FEE_ROUNDTRIP_PCT = 0.10
SLIPPAGE_ROUNDTRIP_PCT = 0.04
MAX_OPEN_HOURS = 24
MIN_RISK_PCT = 0.08
MAX_RISK_PCT = 2.0
TRADE_COLUMNS = ['id','strategy','direction','rr','entry_time','entry','stop','target','exit_time','exit','reason','gross_pct','net_pct','status']
SIGNAL_COLUMNS = ['id','time','direction','sweep_time','choch_time','fvg_time','zone_low','zone_high','entry','stop','rr2_target','rr3_target']


def utcstr(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def notify(message):
    if not BOT_TOKEN or not CHAT_ID:
        print('TELEGRAM SKIPPED: BOT_TOKEN or CHAT_ID missing')
        return False
    try:
        r = requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',
                          json={'chat_id': CHAT_ID, 'text': message}, timeout=20)
        r.raise_for_status()
        if not r.json().get('ok'):
            raise ValueError('Telegram API returned ok=false')
        print('TELEGRAM SENT')
        return True
    except (requests.RequestException, ValueError) as e:
        print('TELEGRAM ERROR:', str(e)[:200])
        return False


def fetch_candles(minutes, days):
    """Try documented resolution/from/to and interval/startTime/endTime formats; reject malformed data."""
    now = int(time.time())
    end = now - now % (minutes * 60)
    start = end - days * 86400
    chunks = []
    chunk_seconds = min(5 * 86400, 1200 * minutes * 60)
    for left in range(start, end, chunk_seconds):
        right = min(end, left + chunk_seconds)
        attempts = [
            {'pair': PAIR, 'resolution': str(minutes), 'from': left, 'to': right},
            {'pair': PAIR, 'interval': f'{minutes}m' if minutes != 60 else '1h',
             'startTime': left * 1000, 'endTime': right * 1000},
        ]
        payload = None
        for params in attempts:
            try:
                resp = requests.get(BASE, params=params, timeout=25)
                resp.raise_for_status()
                result = resp.json()
                if isinstance(result, dict):
                    result = result.get('data', result.get('candles', []))
                if isinstance(result, list) and result:
                    payload = result
                    break
            except (requests.RequestException, ValueError) as e:
                print('API retry:', str(e)[:110])
        if payload is None:
            raise RuntimeError(f'No valid CoinDCX candles for {minutes}m {left}-{right}; failing safely')
        chunks.extend(payload)
        time.sleep(.12)
    records = []
    for x in chunks:
        try:
            if isinstance(x, dict):
                t = x.get('time', x.get('timestamp', x.get('t')))
                o = x.get('open', x.get('o'))
                h = x.get('high', x.get('h'))
                l = x.get('low', x.get('l'))
                c = x.get('close', x.get('c'))
