from trading_bot.bot.indicator_bot import IndicatorBot
import datetime
import pandas as pd
import matplotlib.pyplot as plt

bot = IndicatorBot(symbol="ADAUSDT")

data = bot.get_data(
    symbol="ADAUSDT",
    interval="1m",
    target_date=datetime.datetime.strptime("2025-12-13 00:00:00", "%Y-%m-%d %H:%M:%S"),
    limit=1000,
)

signal = bot.transform(data)
df = bot.generate_signal(signal)

df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms', utc=True)

# Convert to Bangkok Time (UTC+7)
df['Close Time Bangkok'] = df['Close Time'].dt.tz_convert('Asia/Bangkok')


# Check the data
print(df[['Close Time Bangkok', 'Close', 'KAMA']].head())
print(df.info())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['Close Time Bangkok'], df['Close'], label='Close')
plt.plot(df['Close Time Bangkok'], df['KAMA'], label='KAMA', linestyle='--')

buys = df[df['signal'] == 1]
plt.scatter(buys['Close Time Bangkok'], buys['Close'], color='green', marker='^', s=100, zorder=5)
sells = df[df['signal'] == -1]
plt.scatter(sells['Close Time Bangkok'], sells['Close'], color='red', marker='v', s=100, zorder=5)

plt.title('Close Price and KAMA over Time (Bangkok Time)')
plt.xlabel('Close Time (Bangkok)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()