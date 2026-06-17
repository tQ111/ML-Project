import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from backtester import simulate_trade

data = pd.read_csv("spy_data.csv")
features = pd.read_csv("features.csv")

feature_cols = ['return_5d', 'return_10d', 'return_20d', 'dist_from_ma20', 'dist_from_ma50',
                'volatility_10d', 'volume_ratio', 'rsi_14', 'macd', 'day_of_week',
                'dist_from_ma200', 'high_low_range']

settings_to_try = [
    {'stop_loss_pct': 0.01, 'take_profit_pct': 0.02, 'max_hold_days': 10},
    {'stop_loss_pct': 0.03, 'take_profit_pct': 0.03, 'max_hold_days': 15},
    {'stop_loss_pct': 0.02, 'take_profit_pct': 0.06, 'max_hold_days': 30},
]

for settings in settings_to_try:
    print(f"\n=== Testing: {settings} ===")
    
    results = []
    for i in range(len(data) - 1):
        long_result = simulate_trade(data, entry_index=i, direction='long', **settings)
        results.append({'date': data.iloc[i]['Date'], 'long_outcome': long_result['outcome']})
    
    labels_df = pd.DataFrame(results)
    df = features.merge(labels_df, on="date")
    df['long_good'] = (df['long_outcome'] == 'take_profit').astype(int)
    df = df.dropna(subset=feature_cols)
    
    X = df[feature_cols]
    y = df['long_good']
    
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=scale)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"Positive label rate in data: {y.mean():.2%}")
    print(classification_report(y_test, preds, zero_division=0))
    
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 3 features:", list(importances.head(3).index))