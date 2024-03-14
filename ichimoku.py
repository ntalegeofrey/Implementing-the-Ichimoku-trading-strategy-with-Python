import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Define the exchange and timeframe
exchange = ccxt.binance()
timeframe = '4h'

# Define the start date, Convert the start date to a UNIX timestamp
start_date = datetime(2021, 12, 1)
since = int(start_date.timestamp() * 1000)

# Fetch the historical data and Convert the data to a Pandas dataframe
ohlcv = exchange.fetch_ohlcv('ETH/USDT', timeframe, since=since)
df = pd.DataFrame(
    ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Calculate the Ichimoku indicators
df['tenkan_sen'] = (df['high'] + df['low']) / 2
df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
df['senkou_span_a'] = (df['tenkan_sen'] + df['kijun_sen']) / 2
df['senkou_span_b'] = (df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2
df['chikou_span'] = df['close'].shift(-26)

# Calculate the Cloud (Kumo)
df['cloud_top'] = (df['senkou_span_a'] + df['senkou_span_b']) / 2
df['cloud_bottom'] = (df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2

# Calculate the 14 EMA
df['ema'] = df['close'].ewm(span=14).mean()

# Create the signals, Create buy and sell signals column for the Ichimoku strategy
signal_conditions = [
    (df['close'] > df['cloud_top']) ,
    (df['close'] < df['cloud_top']),
]

signal_choices = [1, -1]

df['signal'] = np.select(signal_conditions, signal_choices, default=0)

# remove the look ahead bias by creating a signal lag of one period
df['signal'] = df['signal'].shift(1)

# Initialize the strategy performance variables
initial_capital = 100
capital = initial_capital
profit = 0

# Keep track of the previous signal
prev_signal = df.loc[0, 'signal']

# Previous close prise
prev_close = df.loc[0, 'close']

# Backtest the strategy
for i in range(1, len(df)):
    current_signal = df.loc[i, 'signal']
    if current_signal != prev_signal:
        if current_signal == 1:
            # Buy signal
            profit = capital * (df.loc[i, 'close'] - prev_close) / prev_close
            capital += profit

        elif current_signal == -1:
            # Sell signal
            profit = capital * (prev_close - df.loc[i, 'close']) / df.loc[i, 'close']
            capital += profit
        
        prev_close = df.loc[i, 'close']
            
    # Update the previous signal
    prev_signal = current_signal

    # print(f'{i} | current Signal: {current_signal} | Previous Signal: {prev_signal} | Capital: ${capital} | Profit: ${profit} | Prev_close: {prev_close}')


# Create the Positions
conditions = [
    (df['signal'] == 1) & (df['signal'] != df['signal'].shift(1)),
    (df['signal'] == -1) & (df['signal'] != df['signal'].shift(1))
]

choices = [1, -1]

df['positions'] = np.select(conditions, choices, default=np.nan)

# Print the strategy performance
print(df.tail(30))
overall_profit = capital - initial_capital
pl_percentage = (capital - initial_capital) / initial_capital * 100
print(f'Initial capital: {initial_capital} USDT')
print(f'Final capital: {capital} USDT')
print(f'Overall Profit: {overall_profit:.2f} USDT')
print(f'Percentage Profit Gain: {pl_percentage:.2f}%')


# Plot the Ichimoku strategy and the close price
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['senkou_span_a'], label='Senkou Span A')
plt.plot(df['timestamp'], df['senkou_span_b'], label='Senkou Span B')
plt.plot(df['timestamp'], df['chikou_span'], label='Chikou Span')
plt.plot(df['timestamp'], df['cloud_top'], label='Cloud Top')
plt.plot(df['timestamp'], df['cloud_bottom'], label='Cloud Bottom')
plt.plot(df['timestamp'], df['tenkan_sen'], label='tenkan_sen')
plt.plot(df['timestamp'], df['kijun_sen'], label='kijun_sen')
plt.plot(df['timestamp'], df['close'], label='Close Price', color='black')


# Plot the 14 EMA
plt.plot(df['timestamp'], df['ema'], label='14 EMA')

# Plot the positions
plt.scatter(df['timestamp'][df['positions'] == 1], df['close']
            [df['positions'] == 1], color='green', label='Buy position', marker="^", s=100)
plt.scatter(df['timestamp'][df['positions'] == -1], df['close']
            [df['positions'] == -1], color='red', label='Sell position', marker="v", s=100)

# Fill area under close line for cloud top
plt.fill_between(df['timestamp'], df['close'], df['cloud_top'] , where=(df['close'] > df['cloud_top']),
                 color='lightgreen', alpha=0.3, label='Buy')

# Fill area under close line for cloud top
plt.fill_between(df['timestamp'], df['close'], df['cloud_top'], where=(df['close'] < df['cloud_top']),
                 color='lightcoral', alpha=0.3, label='Sell')


plt.legend()

# Add axis labels and a title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Ichimoku Trading Strategy')

# Show the plot
plt.show()
