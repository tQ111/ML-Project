import pandas as pd

data = pd.read_csv("spy_data.csv")

features = pd.DataFrame()
features['date'] = data['Date']

# return over last N days
features['return_5d'] = data['Close'].pct_change(5)
features['return_10d'] = data['Close'].pct_change(10)
features['return_20d'] = data['Close'].pct_change(20)

# moving averages, expressed as % distance from price
features['dist_from_ma20'] = (data['Close'] - data['Close'].rolling(20).mean()) / data['Close'].rolling(20).mean()
features['dist_from_ma50'] = (data['Close'] - data['Close'].rolling(50).mean()) / data['Close'].rolling(50).mean()

# volatility (rolling std of daily returns)
features['volatility_10d'] = data['Close'].pct_change().rolling(10).std()

# volume relative to its own recent average
features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

# consecutive up/down days
direction = (data['Close'].diff() > 0).astype(int)
features['consecutive_up'] = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
features['consecutive_up'] = features['consecutive_up'] * direction  # zero out if currently a down day

# RSI (14-day)
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
features['rsi_14'] = 100 - (100 / (1 + rs))

# MACD
ema12 = data['Close'].ewm(span=12).mean()
ema26 = data['Close'].ewm(span=26).mean()
features['macd'] = ema12 - ema26

# day of week (0=Monday ... 4=Friday)
features['day_of_week'] = pd.to_datetime(data['Date']).dt.dayofweek

# longer-term trend
features['dist_from_ma200'] = (data['Close'] - data['Close'].rolling(200).mean()) / data['Close'].rolling(200).mean()

# daily range relative to price
features['high_low_range'] = (data['High'] - data['Low']) / data['Close']

features.to_csv("features.csv", index=False)
print("Saved features. Rows:", len(features))
print(features.head(25))