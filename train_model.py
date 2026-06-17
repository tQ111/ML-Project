import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")

df = features.merge(labels, on="date")

# turn outcome into a simple good/bad label for long trades
df['long_good'] = (df['long_outcome'] == 'take_profit').astype(int)
df['short_good'] = (df['short_outcome'] == 'take_profit').astype(int)

# drop rows with missing features (the early days needing more history than exists)
feature_cols = ['return_5d', 'return_10d', 'return_20d', 'dist_from_ma20', 'dist_from_ma50', 'volatility_10d', 'volume_ratio', 'consecutive_up', 'rsi_14', 'macd', 'day_of_week', 'dist_from_ma200', 'high_low_range']
df = df.dropna(subset=feature_cols)

for label_col, name in [('long_good', 'LONG'), ('short_good', 'SHORT')]:
    print(f"\n=== {name} MODEL ===")
    X = df[feature_cols]
    y = df[label_col]

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=scale)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 3 features:", list(importances.head(3).index))