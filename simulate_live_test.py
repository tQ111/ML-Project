import pandas as pd
import requests
import time
import json

WEBHOOK_URL = "https://tradingml-u4lk.onrender.com/webhook"

df = pd.read_csv("spy_intraday.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# RTH only, grab one recent day for a clean test
df = df[(df['datetime'].dt.time >= pd.Timestamp('09:30').time()) &
        (df['datetime'].dt.time <= pd.Timestamp('16:00').time())]
df = df.sort_values('datetime').reset_index(drop=True)

# pick the most recent full day available
last_date = df['datetime'].dt.date.max()
day_df = df[df['datetime'].dt.date == last_date].reset_index(drop=True)

print(f"Simulating {len(day_df)} bars from {last_date}")

for i, row in day_df.iterrows():
    bar_payload = {
        "type": "bar",
        "time": str(int(row['datetime'].timestamp())),
        "open": str(row['Open']),
        "high": str(row['High']),
        "low": str(row['Low']),
        "close": str(row['Close']),
        "volume": str(row['Volume'])
    }
    r = requests.post(WEBHOOK_URL, json=bar_payload)
    print(f"Bar {i+1}/{len(day_df)} sent — status {r.status_code}")

    # simulate a LONG entry partway through, to test position tracking
    if i == 30:
        long_payload = {"type": "long", "price": str(row['Close'])}
        r2 = requests.post(WEBHOOK_URL, json=long_payload)
        print(f"  -> LONG signal sent — status {r2.status_code}")

    time.sleep(0.3)  # small delay so logs are readable, not required

print("Done. Check /positions and /buffer on your webhook server.")