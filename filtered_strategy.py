import pandas as pd
import xgboost as xgb

features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")

df = features.merge(labels, on="date")

feature_cols = ['return_5d', 'return_10d', 'return_20d', 'dist_from_ma20', 'dist_from_ma50',
                'volatility_10d', 'volume_ratio', 'consecutive_up', 'rsi_14', 'macd',
                'day_of_week', 'dist_from_ma200', 'high_low_range']
df = df.dropna(subset=feature_cols)

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:].copy()

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

long_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
long_model.fit(X_train, train_df['long_return_pct'])
test_df['long_pred_return'] = long_model.predict(X_test)

short_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
short_model.fit(X_train, train_df['short_return_pct'])
test_df['short_pred_return'] = short_model.predict(X_test)

# only act on the most confident predictions - top 10% for each direction
long_threshold = test_df['long_pred_return'].quantile(0.90)
short_threshold = test_df['short_pred_return'].quantile(0.90)

confident_longs = test_df[test_df['long_pred_return'] >= long_threshold]
confident_shorts = test_df[test_df['short_pred_return'] >= short_threshold]

print(f"Confident long trades: {len(confident_longs)}")
print(f"Their actual average return: {confident_longs['long_return_pct'].mean():.4f}")
print(f"Win rate (hit take profit): {(confident_longs['long_outcome'] == 'take_profit').mean():.2%}")

print(f"\nConfident short trades: {len(confident_shorts)}")
print(f"Their actual average return: {confident_shorts['short_return_pct'].mean():.4f}")
print(f"Win rate (hit take profit): {(confident_shorts['short_outcome'] == 'take_profit').mean():.2%}")
print(f"\n--- Flipped test: what if we did the OPPOSITE of confident short predictions? ---")
# if the model said "confident short" but shorts lose, check what going LONG instead would have done on those same days
print(f"If we went LONG instead on those same days:")
print(f"Their actual long return: {confident_shorts['long_return_pct'].mean():.4f}")
print(f"Long win rate on those days: {(confident_shorts['long_outcome'] == 'take_profit').mean():.2%}")

print(f"\nFor comparison, average return on ALL long predictions: {test_df['long_return_pct'].mean():.4f}")
print(f"For comparison, average return on ALL short predictions: {test_df['short_return_pct'].mean():.4f}")