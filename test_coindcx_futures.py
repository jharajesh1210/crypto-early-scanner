import requests
import time
from datetime import datetime, timezone

URL = "https://public.coindcx.com/market_data/candlesticks"

PAIR = "B-BTC_USDT"

now = int(time.time())
from_time = now - (24 * 60 * 60)


def test_candles(resolution, name):
    params = {
        "pair": PAIR,
        "from": from_time,
        "to": now,
        "resolution": resolution,
        "pcode": "f"
    }

    print("=" * 60)
    print(f"TESTING {name}")
    print("=" * 60)

    response = requests.get(
        URL,
        params=params,
        timeout=20
    )

    print("HTTP STATUS:", response.status_code)

    response.raise_for_status()

    data = response.json()

    print("Candles received:", len(data))

    if not data:
        print("NO DATA RECEIVED")
        return

    print("First candle:")
    print(data[0])

    print()
    print("Last candle:")
    print(data[-1])

    print()
    print(f"{name} TEST = PASS")


print()
print("COINDCX FUTURES API TEST")
print("PAIR:", PAIR)
print()

test_candles("5", "5-MINUTE FUTURES")

print()

test_candles("60", "1-HOUR FUTURES")

print()
print("=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
