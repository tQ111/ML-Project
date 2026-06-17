import pandas as pd
from backtester import simulate_trade

data = pd.read_csv("spy_data.csv")

results = []

for i in range(len(data) - 1):  # -1 so we always have at least one future day
    long_result = simulate_trade(data, entry_index=i, direction='long', stop_loss_pct=0.03, take_profit_pct=0.03, max_hold_days=15)
    short_result = simulate_trade(data, entry_index=i, direction='short', stop_loss_pct=0.03, take_profit_pct=0.03, max_hold_days=15)
    
    results.append({
        'date': data.iloc[i]['Date'],
        'long_outcome': long_result['outcome'],
        'long_return_pct': long_result['return_pct'],
        'short_outcome': short_result['outcome'],
        'short_return_pct': short_result['return_pct'],
    })

labels_df = pd.DataFrame(results)
labels_df.to_csv("labels.csv", index=False)
print("Saved labels. Rows:", len(labels_df))
print(labels_df.head())