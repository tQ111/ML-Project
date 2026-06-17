import pandas as pd

def simulate_trade(data, entry_index, direction, stop_loss_pct=0.02, take_profit_pct=0.04, max_hold_days=20):
    entry_price = data.iloc[entry_index]['Close']
    
    for i in range(entry_index + 1, min(entry_index + 1 + max_hold_days, len(data))):
        current_price = data.iloc[i]['Close']
        
        if direction == 'long':
            change_pct = (current_price - entry_price) / entry_price
        else:  # short
            change_pct = (entry_price - current_price) / entry_price
        
        if change_pct <= -stop_loss_pct:
            return {'outcome': 'stop_loss', 'return_pct': change_pct, 'days_held': i - entry_index}
        if change_pct >= take_profit_pct:
            return {'outcome': 'take_profit', 'return_pct': change_pct, 'days_held': i - entry_index}
    
    # ran out of time without hitting stop or target
    final_price = data.iloc[min(entry_index + max_hold_days, len(data) - 1)]['Close']
    if direction == 'long':
        change_pct = (final_price - entry_price) / entry_price
    else:
        change_pct = (entry_price - final_price) / entry_price
    
    return {'outcome': 'time_limit', 'return_pct': change_pct, 'days_held': max_hold_days}


if __name__ == "__main__":
    data = pd.read_csv("spy_data.csv")
    
    # test it on one day, e.g. row 100, going long
    result = simulate_trade(data, entry_index=100, direction='long')
    print(result)