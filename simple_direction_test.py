import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report

data = pd.read_csv("spy_data.csv")
features = pd.read_csv("features.csv")

df = features.copy()
df['date'] = features['date']

# simple label: did price go up the next day?
data['next_close'] = data['Close'].shift(-1)
data['up_next_day'] = (data['next_close'] > data['Close']).astype(int)

df = df.merge(data[['Date', 'up_next_day']], left_on='date', right_on='Date')

feature_cols = ['return_5d', 'return_10d', 'return_20d', 'dist_from_ma20', 'dist_from_ma50',
                'volatility_10d', 'volume_ratio', 'consecutive_up', 'rsi_14', 'macd',
                'day_of_week', 'dist_from_ma200', 'high_low_range']
df = df.dropna(subset=feature_cols + ['up_next_day'])

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train, X_test = train_df[feature_cols], test_df[feature_cols]
y_train, y_test = train_df['up_next_day'], test_df['up_next_day']

print(f"Baseline (always guess 'up'): {y_test.mean():.2%} of days actually went up")

model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds, zero_division=0))

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Top 5 features:", list(importances.head(5).index))