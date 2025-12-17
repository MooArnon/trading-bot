from trading_bot.bot.indicator_bot import IndicatorBot
import datetime
import pandas as pd
import matplotlib.pyplot as plt

THRESHOLD = 0.007
symbol = "ETHUSDT"

bot = IndicatorBot(symbol=symbol, check_volatility=True, bandwidth_threshold=THRESHOLD)

data = bot.get_data(
    symbol=symbol,
    interval="1m",
    target_date=datetime.datetime.strptime("2025-12-17 17:00:00", "%Y-%m-%d %H:%M:%S"),
    limit=1000,
)

signal = bot.transform(data)
df = bot.generate_signal(signal)

df['close_time'] = pd.to_datetime(df['Close Time'], unit='ms', utc=True)

# Convert to Bangkok Time (UTC+7)
df['close_time_local'] = df['close_time'].dt.tz_convert('Asia/Bangkok')

# Plotting
# --- Design: Dual Subplots (Price on Top, Volatility on Bottom) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})

# === TOP PANEL: Price, KAMA, Signals ===
ax1.plot(df['close_time_local'], df['Close'], label='Close Price', color='black', alpha=0.6)
ax1.plot(df['close_time_local'], df['KAMA'], label='KAMA', color='orange', linestyle='--', linewidth=1.5)

# Plot Signals (Green Up / Red Down)
buys = df[df['signal'] == 1]
sells = df[df['signal'] == -1]

if not buys.empty:
    ax1.scatter(buys['close_time_local'], buys['Close'], 
                color='green', marker='^', s=150, zorder=5, label='Buy Signal')
if not sells.empty:
    ax1.scatter(sells['close_time_local'], sells['Close'], 
                color='red', marker='v', s=150, zorder=5, label='Sell Signal')

ax1.set_ylabel('Price')
ax1.set_title('Price Action vs KAMA (with Volatility Context)')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.3)

# === BOTTOM PANEL: Volatility (Bandwidth) ===
# Set your filter threshold

ax2.plot(df['close_time_local'], df['bandwidth'], color='tab:blue', label='Bandwidth')
ax2.axhline(y=THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD})')

# Visual Aid: Shade the background of the TOP chart based on BOTTOM chart condition
# Green Background = Safe Volatility, Red Background = Squeeze
# (We use fill_between on the top axes using the data from the bottom)
y_min, y_max = ax1.get_ylim()
safe_mask = df['bandwidth'] > THRESHOLD
unsafe_mask = df['bandwidth'] <= THRESHOLD

# Note: This shading method works best with continuous data. 
# For sparse data, it highlights specific points.
ax1.fill_between(df['close_time_local'], y_min, y_max, where=safe_mask, 
                 color='green', alpha=0.05, label='Safe Zone')
ax1.fill_between(df['close_time_local'], y_min, y_max, where=unsafe_mask, 
                 color='red', alpha=0.05, label='Squeeze Zone')

ax2.set_ylabel('Bandwidth')
ax2.set_xlabel('Close Time (Bangkok)')
ax2.fill_between(df['close_time_local'], df['bandwidth'], 0, color='tab:blue', alpha=0.1)
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()