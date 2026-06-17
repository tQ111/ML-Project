import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go
from backtester import simulate_trade

data = pd.read_csv("spy_data.csv")
features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")

df = features.merge(labels, on="date")
df['long_good'] = (df['long_outcome'] == 'take_profit').astype(int)
df['short_good'] = (df['short_outcome'] == 'take_profit').astype(int)

feature_cols = ['return_5d', 'return_10d', 'return_20d', 'dist_from_ma20', 'dist_from_ma50',
                'volatility_10d', 'volume_ratio', 'consecutive_up', 'rsi_14', 'macd',
                'day_of_week', 'dist_from_ma200', 'high_low_range']
df = df.dropna(subset=feature_cols)

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:].copy().reset_index(drop=True)

X_train = train_df[feature_cols]

long_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
long_model.fit(X_train, train_df['long_return_pct'])
test_df['long_pred_return'] = long_model.predict(test_df[feature_cols])

# only trade the top 10% most confident long predictions
long_threshold = test_df['long_pred_return'].quantile(0.90)
test_df['long_pred'] = (test_df['long_pred_return'] >= long_threshold).astype(int)
test_df['short_pred'] = 0  # not trading shorts for now, per the findings above

# map test_df dates back to indices in the full price data, since the backtester needs real OHLC lookahead
data_dates = data['Date'].tolist()
date_to_index = {d: i for i, d in enumerate(data_dates)}

# trading assumptions
starting_capital = 10000
position_size_pct = 0.5
commission_per_trade = 1.0
slippage_pct = 0.0005

capital = starting_capital
equity_curve = []
open_positions = []  # list of {exit_index, pnl} for trades not yet resolved
trades_taken = 0
max_concurrent_positions = 5  # cap total simultaneous trades
position_size_pct = 0.15  # smaller per-trade size since multiple can be open at once

for _, row in test_df.iterrows():
    current_date = row['date']
    current_data_index = date_to_index[current_date]

    # resolve and apply any positions whose exit day has arrived
    still_open = []
    for pos in open_positions:
        if current_data_index >= pos['exit_index']:
            capital += pos['pnl']
        else:
            still_open.append(pos)
    open_positions = still_open

    direction = None
    if row['long_pred'] == 1:
        direction = 'long'
    elif row['short_pred'] == 1:
        direction = 'short'

    if direction is not None and len(open_positions) < max_concurrent_positions:
        result = simulate_trade(data, entry_index=current_data_index, direction=direction,
                                  stop_loss_pct=0.03, take_profit_pct=0.03, max_hold_days=15)
        net_return_pct = result['return_pct'] - (slippage_pct * 2)
        position_value = capital * position_size_pct
        pnl = (position_value * net_return_pct) - commission_per_trade

        open_positions.append({
            'exit_index': current_data_index + result['days_held'],
            'pnl': pnl
        })
        trades_taken += 1

    equity_curve.append({'date': current_date, 'capital': capital})

equity_df = pd.DataFrame(equity_curve)

print(f"Starting capital: ${starting_capital:,.2f}")
print(f"Ending capital: ${capital:,.2f}")
print(f"Total return: {(capital - starting_capital) / starting_capital:.2%}")
print(f"Trades actually taken: {trades_taken}")
print(f"Total test days: {len(test_df)}")

# buy and hold comparison
test_prices = data[data['Date'].isin(test_df['date'])].reset_index(drop=True)
start_price = test_prices.iloc[0]['Close']
test_prices['buy_hold_value'] = starting_capital * (test_prices['Close'] / start_price)

buy_hold_final = test_prices.iloc[-1]['buy_hold_value']
print(f"Buy-and-hold ending value: ${buy_hold_final:,.2f}")
print(f"Buy-and-hold return: {(buy_hold_final - starting_capital) / starting_capital:.2%}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_df['date'], y=equity_df['capital'], mode='lines', name='Model Strategy'))
fig.add_trace(go.Scatter(x=test_prices['Date'], y=test_prices['buy_hold_value'], mode='lines', name='Buy & Hold SPY'))
fig.update_layout(title='Equity Curve: Model vs Buy & Hold (Test Period)', template='plotly_dark', xaxis_title='Date', yaxis_title='Account Value ($)')
fig.write_html("equity_curve.html", config={'scrollZoom': True})
print("Saved equity_curve.html")