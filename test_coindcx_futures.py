import requests
import time
import json

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
    print("REQUEST URL:", response.url)

    response.raise_for_status()

    data = response.json()

    print("RESPONSE TYPE:", type(data).__name__)

    if isinstance(data, dict):
        print("RESPONSE KEYS:", list(data.keys()))
        print()
        print("FULL RESPONSE:")
        print(json.dumps(data, indent=2)[:5000])

    elif isinstance(data, list):
        print("CANDLES RECEIVED:", len(data))

        if len(data) > 0:
            print()
            print("FIRST CANDLE:")
            print(data[0])

            print()
            print("LAST CANDLE:")
            print(data[-1])

    else:
        print("UNKNOWN RESPONSE:")
        print(data)

    print()
    print(f"{name} API CONNECTION = PASS")


print()
print("COINDCX FUTURES API TEST")
print("PAIR:", PAIR)
print()

test_candles("5", "5-MINUTE FUTURES")

print()

test_candles("60", "1-HOUR FUTURES")

print()
print("=" * 60)
print("ALL API TESTS COMPLETED")
print("=" * 60)
