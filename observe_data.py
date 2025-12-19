from trading_bot.bot.indicator_bot import IndicatorBot
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


BANDWIDTH_THRESHOLD = 0.003
symbol = "ADAUSDT"

bot = IndicatorBot(
    symbol=symbol, 
    check_volatility=True, 
    bandwidth_threshold=BANDWIDTH_THRESHOLD,
    signal_type = [
        'KAMA', 
        # 'RSI',
    ],
)

data = bot.get_data(
    symbol=symbol,
    interval="1m",
    target_date=datetime.datetime.strptime("2025-12-19 17:00:00", "%Y-%m-%d %H:%M:%S"),
    limit=1000,
)

signal = bot.transform(data)
df = bot.generate_signal(signal)

df.to_csv("indicator_output.csv", index=False)

# 1. Load Data
file_path = 'indicator_output.csv'
df = pd.read_csv(file_path)

# 2. Preprocessing
# Convert timestamp (ms) to datetime
df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')

# Calculate bar width for volume chart (80% of the interval)
if len(df) > 1:
    time_diff = df['Open Time'].iloc[1] - df['Open Time'].iloc[0]
    width = time_diff.total_seconds() / (24 * 3600) * 0.8
else:
    width = 0.01

# Check for indicators
has_rsi = 'rsi' in df.columns
has_bandwidth = 'bandwidth' in df.columns

# 3. Setup Plot
# Define layout dynamically
plots = ['price']
if has_rsi: plots.append('rsi')
if has_bandwidth: plots.append('bandwidth')
plots.append('volume')

n_plots = len(plots)
height_ratios = [3] + [1] * (n_plots - 1)  # Price gets 3x height, others get 1x

fig = plt.figure(figsize=(14, 3 * n_plots + 2)) 
gs = fig.add_gridspec(n_plots, 1, height_ratios=height_ratios)

axes = {}

# --- Plot Loop ---
for i, plot_type in enumerate(plots):
    if i == 0:
        ax = fig.add_subplot(gs[i])
        axes['price'] = ax
    else:
        # Share x-axis with price plot
        ax = fig.add_subplot(gs[i], sharex=axes['price'])
        axes[plot_type] = ax

    # Plot Content
    if plot_type == 'price':
        ax.plot(df['Open Time'], df['Close'], label='Price', color='gray', linewidth=1, alpha=0.6)
        if 'KAMA' in df.columns:
            ax.plot(df['Open Time'], df['KAMA'], label='KAMA', color='orange', linewidth=2)
        
        # Signals
        buy_signals = df[df['signal'] == 1]
        if not buy_signals.empty:
            ax.scatter(buy_signals['Open Time'], buy_signals['Close'], 
                       marker='^', color="#088B08", s=150, edgecolors='black', label='Buy', zorder=5)
        
        sell_signals = df[df['signal'] == -1]
        if not sell_signals.empty:
            ax.scatter(sell_signals['Open Time'], sell_signals['Close'], 
                       marker='v', color='red', s=150, edgecolors='black', label='Sell', zorder=5)
        
        ax.set_title('Strategy Analysis: Price vs Indicators')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)

    elif plot_type == 'rsi':
        ax.plot(df['Open Time'], df['rsi'], color='purple', linewidth=1.5, label='RSI')
        ax.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel('RSI')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2)

    elif plot_type == 'bandwidth':
        ax.plot(df['Open Time'], df['bandwidth'], color='teal', linewidth=1.5, label='Bandwidth')
        ax.axhline(BANDWIDTH_THRESHOLD, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({BANDWIDTH_THRESHOLD})')
        ax.set_ylabel('Bandwidth')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')

    elif plot_type == 'volume':
        vol_colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
        ax.bar(df['Open Time'], df['Volume'], color=vol_colors, alpha=0.5, width=width)
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.2)
        
        # Format X-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()

plt.tight_layout()
plt.savefig('kama_rsi_bandwidth_analysis.png')
plt.show()