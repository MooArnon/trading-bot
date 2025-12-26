##########
# Import #
##############################################################################

import os
from logging import Logger, INFO
import io
import base64
import json
import traceback
import re

import requests
from binance.client import Client
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from openai import OpenAI

from .binance import BinanceBot
from config.config import config
from trading_bot.util.logger import get_utc_logger

###########
# Classes #
##############################################################################

class LLMBot(BinanceBot):
    bot_type="LLMBot"
    kline_columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close Time', 'Quote Asset Volume', 'Number of Trades', 
        'Taker Buy Base', 'Taker Buy Quote', 'Ignore'
    ]
    def __init__(
            self, 
            symbol: str,
            api_key: str = os.environ["BINANCE_API_KEY"],
            secret_key: str = os.environ["BINANCE_SECRET_KEY"],
            logger: Logger=None,
            notify_object: object = None,
            # model: str = "anthropic/claude-3.5-sonnet",
            # model: str = "openai/gpt-oss-120b",
            model: str = "google/gemini-2.5-flash",
    ):
        if logger is None:
            logger = get_utc_logger(
                name=__name__,
                level=INFO,
            )
        
        self.notify_object = notify_object
        self.symbol = symbol
        self.logger = logger
        
        self.client = Client(api_key, secret_key)
        
        self.base_url_usdm_market = "https://fapi.binance.com/fapi/v1/klines"
        
        self.model = model

    ##########################################################################
    
    def transform(
            self, 
            data: pd.DataFrame,
    ) -> pd.DataFrame:
        pass
    
    ##########################################################################
    
    def get_text_promt(
            self, 
            df: pd.DataFrame,
    ) -> str:
        self.logger.info("Generating text promt for LLM...")
        """
        Builds the enhanced prompt for the VERY LAST row in the dataframe.
        """
        df = self.prepare_market_data(df)
        # Get the latest row index
        idx = df.index[-1]
        curr = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        # 1. Pre-calculate subjective logic to save LLM brainpower
        trend_short = "UP" if curr['EMA_9'] > curr['EMA_21'] else "DOWN"
        trend_long = "BULLISH" if curr['Close'] > curr['EMA_200'] else "BEARISH"
        
        macd_momentum = "Building" if abs(curr['MACD_Hist']) > abs(prev['MACD_Hist']) else "Fading"
        macd_signal = "Bullish" if curr['MACD_Hist'] > 0 else "Bearish"
        
        # Distance from EMA 200 (Mean Reversion Check)
        dist_ema200 = ((curr['Close'] - curr['EMA_200']) / curr['EMA_200']) * 100
        
        # Get History Context
        history_context = self.get_recent_price_action(df, idx, lookback=3)
        
        # 2. Build the Prompt
        prompt = f"""
You are an expert Crypto Scalper specializing in ADA/USDT 15m timeframe.
Your goal is to identify high-probability setups using Trend Following or Mean Reversion.

### 1. MARKET CONTEXT (Sequence of Events)
The last 3 completed candles before the current moment:
{history_context}

### 2. LIVE SNAPSHOT (Current Candle)
- **Price:** {curr['Close']:.4f}
- **Volume:** {curr['Volume']:.0f} (Previous: {prev['Volume']:.0f})
- **ATR (Volatility):** {curr['ATR']:.4f}

### 3. TECHNICAL INDICATORS
- **Trend Status:**
- Short-term (9/21 EMA): {trend_short}
- Long-term (200 EMA): {trend_long} (Price is {dist_ema200:.2f}% away from EMA200)

- **Momentum:**
- RSI (14): {curr['RSI']:.2f} (Overbought > 70, Oversold < 30)
- MACD: {macd_signal} signal, Momentum is {macd_momentum} (Hist: {curr['MACD_Hist']:.5f})

- **Structure (Bollinger Bands):**
- Bandwidth: {curr['BB_Width']:.4f} (Low = Squeeze/Breakout pending, High = Volatile)
- Position: Price is {"Near Upper Band" if curr['Close'] > curr['BB_Upper']*0.995 else ("Near Lower Band" if curr['Close'] < curr['BB_Lower']*1.005 else "Middle")}

### 4. TRADING RULES (Strict)
1. **LONG Criteria:** Trend is Bullish OR Price is at Lower BB (Reversion). RSI is rising.
2. **SHORT Criteria:** Trend is Bearish OR Price is at Upper BB (Reversion). RSI is falling.
3. **NO TRADE:** If market is flat (RSI 45-55, Low BB Width) or signals conflict (e.g., Price > EMA200 but MACD is bearish).

### TASK
Perform a "Chain of Thought" analysis. First, analyze the **sequence** of the last 3 candles combined with current indicators. Then output the JSON.

**OUTPUT FORMAT (JSON ONLY):**
{{
    "analysis_thought_process": "Analyze price action, volume, and indicators here...",
    "signal": "1" (LONG), "-1" (SHORT), or "0" (HOLD),
    "confidence": 0-100,
    "reasoning": "Under 20 words summary."
}}
"""
        return prompt

    ##########################################################################
    
    def prepare_market_data(self, df):
        """
        Takes raw kline list from Binance and returns a DataFrame with indicators.
        """
        # 1. Convert Types (Binance API returns strings)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        
        # 2. Calculate Indicators (Pandas TA)
        # Trend
        df['EMA_9'] = ta.ema(df['Close'], length=9)
        df['EMA_21'] = ta.ema(df['Close'], length=21)
        df['EMA_200'] = ta.ema(df['Close'], length=200)
        
        # Momentum
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9'] # Histogram
        
        # Volatility
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Calculate Bollinger Bands
        bb = ta.bbands(df['Close'], length=20, std=2)
        
        # METHOD 1: Join everything automatically (Safe & Easy)
        df = pd.concat([df, bb], axis=1)
        
        # METHOD 2: Rename columns to standard names so your Prompt logic always works
        # This finds the column ending in 'BBU...' regardless of the numbers
        bbu_col = [c for c in bb.columns if c.startswith('BBU')][0]
        bbl_col = [c for c in bb.columns if c.startswith('BBL')][0]
        bbw_col = [c for c in bb.columns if c.startswith('BBB')][0] # Bandwidth is already calculated!
        
        # Map them to simple names for your logic
        df['BB_Upper'] = bb[bbu_col]
        df['BB_Lower'] = bb[bbl_col]
        df['BB_Width'] = bb[bbw_col] # Use the library's calculation
            
        return df
    
    ##########################################################################
    
    def get_recent_price_action(self, df, current_index, lookback=3):
        """
        Creates a text summary of the last N candles to give the LLM context of flow.
        """
        summary = []
        # Loop backwards from current_index - 1
        start_idx = max(0, current_index - lookback)
        subset = df.iloc[start_idx:current_index]
        
        for i, row in subset.iterrows():
            # Determine candle color
            color = "GREEN" if row['Close'] > row['Open'] else "RED"
            
            # Calculate body size vs wicks (simple pattern detection context)
            body_size = abs(row['Close'] - row['Open'])
            total_range = row['High'] - row['Low']
            wick_ratio = (total_range - body_size) / total_range if total_range > 0 else 0
            
            shape = "Normal"
            if wick_ratio > 0.6: 
                shape = "Indecision/Doji"
            if body_size > row['ATR']:
                shape = "Big Momentum"
            
            summary.append(
                f"- T-{current_index - i}: {color} Candle. "
                f"Close: {row['Close']:.4f}, Vol: {row['Volume']:.0f}, Shape: {shape}"
            )
        
        return "\n".join(summary)
    
    ##########################################################################
    
    def get_image(self, df: pd.DataFrame) -> bytes:
        self.logger.info("Generating Sniper Chart for LLM...")
        
        # 1. SLICE DATA: Critical for LLM to see candle shapes. 
        # taking last 60 candles (approx 15 hours of data on 15m)
        plot_df = df.iloc[-60:].copy() 
        
        # Ensure numerics
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
            
        # 2. Setup Figure with 3 Subplots (Price, MACD, RSI)
        # Height Ratios: 3 parts Price, 1 part MACD, 1 part RSI
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

        # -------------------------------------------------------
        # TOP CHART: Price + Bands + EMAs + Candles
        # -------------------------------------------------------
        
        # A. Bollinger Bands (Background Context)
        # Check if they exist, otherwise calc (assuming standard names or calc on fly)
        if 'BB_Upper' in plot_df.columns and 'BB_Lower' in plot_df.columns:
            ax1.fill_between(plot_df.index, plot_df['BB_Upper'], plot_df['BB_Lower'], color='gray', alpha=0.1)
            ax1.plot(plot_df.index, plot_df['BB_Upper'], color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax1.plot(plot_df.index, plot_df['BB_Lower'], color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # B. EMAs (Aligning with your Prompt: 9/21/200)
        # We plot 21 (Short Trend) and 200 (Long Trend) to avoid clutter
        if 'EMA_21' not in plot_df.columns: plot_df['EMA_21'] = ta.ema(plot_df['Close'], length=21)
        if 'EMA_200' not in plot_df.columns: plot_df['EMA_200'] = ta.ema(plot_df['Close'], length=200)
        
        ax1.plot(plot_df.index, plot_df['EMA_21'], color='orange', linewidth=1.5, label='EMA 21 (Short Trend)')
        ax1.plot(plot_df.index, plot_df['EMA_200'], color='blue', linewidth=2, label='EMA 200 (Long Trend)')

        # C. DRAW CANDLES (Manual drawing to avoid mplfinance dependency)
        # This loop is fast for 60 items
        width = .6
        width2 = .1
        up = plot_df[plot_df.Close >= plot_df.Open]
        down = plot_df[plot_df.Close < plot_df.Open]
        
        # Draw Up Candles (Green)
        if not up.empty:
            ax1.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='green', alpha=0.8)
            ax1.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='green', alpha=0.8)
            ax1.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='green', alpha=0.8)

        # Draw Down Candles (Red)
        if not down.empty:
            ax1.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='red', alpha=0.8)
            ax1.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color='red', alpha=0.8)
            ax1.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color='red', alpha=0.8)

        ax1.set_title(f"ADA/USDT Sniper View (Last {len(plot_df)} candles)")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.2)
        ax1.legend(loc='upper left')

        # -------------------------------------------------------
        # MIDDLE CHART: MACD (New!)
        # -------------------------------------------------------
        # Ensure MACD exists
        if 'MACD' not in plot_df.columns or 'MACD_Hist' not in plot_df.columns:
            macd = ta.macd(plot_df['Close'])
            plot_df['MACD'] = macd['MACD_12_26_9']
            plot_df['MACD_Hist'] = macd['MACDh_12_26_9']

        # Colorize Histogram
        colors = ['green' if v >= 0 else 'red' for v in plot_df['MACD_Hist']]
        ax2.bar(plot_df.index, plot_df['MACD_Hist'], color=colors, alpha=0.5)
        ax2.plot(plot_df.index, plot_df['MACD'], color='blue', linewidth=1, label='MACD Line')
        ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax2.set_ylabel("MACD")
        ax2.grid(True, alpha=0.2)
        ax2.legend(loc='upper left', fontsize='small')

        # -------------------------------------------------------
        # BOTTOM CHART: RSI
        # -------------------------------------------------------
        if 'RSI' not in plot_df.columns:
            plot_df['RSI'] = ta.rsi(plot_df['Close'], length=14)

        ax3.plot(plot_df.index, plot_df['RSI'], label='RSI (14)', color='purple')
        ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax3.fill_between(plot_df.index, 70, 30, color='purple', alpha=0.05) # "Safe Zone" shading
        ax3.set_ylabel("RSI")
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.2)
        
        plt.tight_layout()

        # 4. Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100) # dpi=100 is good for LLM vision
        plt.close(fig) 

        # 5. Convert to Base64
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64
    
    ##########################################################################
    
    def generate_signal(
            self, 
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        self.logger.info("Generating signal using LLM...")
        
        text_prompt = self.get_text_promt(df=df)
        image_b64 = self.get_image(df=df)

        # B. Define Headers
        headers = {
            "Authorization": f"Bearer {config['OPENAI_API_KEY']}",
            "Content-Type": "application/json"
        }

        # C. Define Payload (Multimodal)
        payload = {
            # Use the correct slug for Claude 3.5 Sonnet
            "model": self.model, 
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional crypto scalper using a multimodel approach. "
                        "1. Cross-reference the textual indicators with the visual chart data. "
                        "2. If the text says 'Downtrend' but the chart shows 'Uptrend', lower your confidence. "
                        "3. Focus heavily on the candlestick shapes in the image for entry timing. "
                        "Return a JSON with signal [1,0,-1] (LONG/HOLD/SHORT) and reasoning."
                    )    
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                # OpenRouter uses the standard data URI format
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
        }

        # D. Send POST Request
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            
            # Check for errors
            response.raise_for_status()
            
            # E. Parse Response
            result = response.json()
            content = result['choices'][0]['message']['content']
            content = re.sub(r"```json\n|\n```", "", content).strip()

            self.logger.info(f"LLM total_tokens: {result['usage']['total_tokens']}"),
            self.logger.info(f"LLM cost: {result['usage']['cost']}")
            
            content_json = json.loads(content)
            self.logger.info(f"LLM Signal: {content_json['signal']}, Confidence: {content_json['confidence']}")
            self.logger.info(f"LLM Reasoning: {content_json['reasoning']}")
            if content_json["confidence"] < config['LLM_CONFIDENCE_PERCENTAGE_THRESHOLD']:
                self.logger.warning("Low confidence in signal. Defaulting to HOLD.")
                return 0
            return content_json['signal']

        except Exception as e:
            self.logger.info(f"Error calling OpenRouter: {e}")
            traceback.print_exc()
        return None
    
    ##########################################################################
        
##############################################################################
