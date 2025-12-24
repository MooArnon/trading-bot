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
        Takes a DataFrame with OHLCV data, calculates indicators, 
        and returns a structured prompt for the LLM.
        """

        # ---------------------------------------------------------
        # 1. CALCULATE INDICATORS
        # ---------------------------------------------------------
        # Ensure we are working with a copy to avoid SettingWithCopy warnings
        df = df.copy()
        # 1. Convert columns to numeric types immediately after creating the DataFrame
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        df['timestamp'] = pd.to_datetime(df['Open Time'], unit='ms') 
        df.set_index('timestamp', inplace=True)

        # This converts the columns to float numbers. 
        # errors='coerce' turns un-parseable text into NaN (prevents crashing)
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Trend: EMAs
        df['EMA_9'] = ta.ema(df['Close'], length=9)
        df['EMA_21'] = ta.ema(df['Close'], length=21)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df['EMA_200'] = ta.ema(df['Close'], length=200)

        # Trend Strength: ADX
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = adx['ADX_14']

        # Momentum: RSI & MACD
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD_Line'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']

        # Volatility: Bollinger Bands & ATR
        bb = ta.bbands(df['Close'], length=20, std=2)

        df['BB_Lower'] = bb.iloc[:, 0] 
        df['BB_Upper'] = bb.iloc[:, 2]
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        # Volume: VWAP & Volume SMA
        # Note: VWAP usually requires a datetime index, here we use a simple approximation or standard calculation
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['Vol_SMA'] = ta.sma(df['Volume'], length=20)

        # ---------------------------------------------------------
        # 2. GET LATEST DATA POINT
        # ---------------------------------------------------------
        # We take the last completed candle (iloc[-1])
        # If you are running this on a live "forming" candle, be careful as values change.
        current = df.iloc[-1]

        # ---------------------------------------------------------
        # 3. INTERPRETATION LOGIC (The "Reasoning" Layer)
        # ---------------------------------------------------------

        # Trend Analysis
        trend_long = "BULLISH" if current['Close'] > current['EMA_200'] else "BEARISH"
        trend_short = "BULLISH" if current['EMA_9'] > current['EMA_21'] else "BEARISH"

        adx_strength = "WEAK/RANGING"
        if current['ADX'] > 25: 
            adx_strength = "STRONG"
        if current['ADX'] > 50: 
            adx_strength = "VERY STRONG"

        # Momentum Analysis
        rsi_state = "NEUTRAL"
        if current['RSI'] > 70: 
            rsi_state = "OVERBOUGHT"
        elif current['RSI'] < 30: 
            rsi_state = "OVERSOLD"

        macd_state = "BULLISH" if current['MACD_Hist'] > 0 else "BEARISH"

        # Volatility Analysis
        bb_status = "INSIDE BANDS"
        if current['Close'] >= current['BB_Upper']: 
            bb_status = "TOUCHING UPPER BAND (Possible Reversal/Breakout)"
        elif current['Close'] <= current['BB_Lower']:
            bb_status = "TOUCHING LowER BAND (Possible Reversal/Breakout)"

        # Volume Analysis
        vol_status = "NORMAL"
        if current['Volume'] > (current['Vol_SMA'] * 1.5): 
            vol_status = "High (1.5x Avg)"
        elif current['Volume'] < (current['Vol_SMA'] * 0.5): 
            vol_status = "Low"

        # ---------------------------------------------------------
        # 4. CONSTRUCT THE PROMPT
        # ---------------------------------------------------------
        prompt = f"""
You are an expert crypto trading bot specializing in 15-minute timeframe scalping for ADA/USDT.
Analyze the folLowing technical data and provide a trading signal. You will use this signal to make real trades with leverage = {config['LEVERAGE']}.
The goal is to maximize profits.

### 1. MARKET DATA
- **Current Price:** {current['Close']:.4f}
- **Volume:** {current['Volume']:.0f} ({vol_status})
- **ATR (Volatility):** {current['ATR']:.4f}

### 2. TREND ANALYSIS
- **Long-Term Trend (EMA 200):** {trend_long} (Price is {"above" if current['Close'] > current['EMA_200'] else "beLow"} EMA 200)
- **Short-Term Trend (EMA 9/21):** {trend_short}
- **Trend Strength (ADX):** {current['ADX']:.2f} ({adx_strength})

### 3. MOMENTUM & OSCILLATORS
- **RSI (14):** {current['RSI']:.2f} ({rsi_state})
- **MACD:** {macd_state} (Histogram: {current['MACD_Hist']:.5f})

### 4. DYNAMIC SUPPORT/RESISTANCE
- **Bollinger Bands:** {bb_status}
    - Upper: {current['BB_Upper']:.4f}
    - Lower: {current['BB_Lower']:.4f}
- **VWAP:** {current['VWAP']:.4f} (Price is {"ABOVE" if current['Close'] > current['VWAP'] else "BELow"} VWAP)

### INSTRUCTIONS
Based on the data above, generate a JSON response in the folLowing format:
- signal: "1" for LONG, "-1" for SHORT, "0" for HOLD
- reasoning: must be short like not more thatn 200 words
{{
    "signal": "1" or "0" or "-1",
    "confidence": 0-100,
    "reasoning": "Concise explanation referencing specific indicators.",
}}
    """
        return prompt

    ##########################################################################
    
    def get_image(self, df: pd.DataFrame) -> bytes:
        self.logger.info("Generating image for LLM...")
        
        # Working with a copy ensures we don't mess up the main dataframe
        plot_df = df.copy() 
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
            
        # 1. Setup the Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
 
        # -------------------------------------------------------
        # TOP CHART: Price + Trends (EMA/BB)
        # -------------------------------------------------------
        ax1.plot(plot_df.index, plot_df['Close'], label='Price', color='black', linewidth=1.5)
 
        # A. Add EMAs (Visual Trend Context) - CRITICAL FOR LLM
        # Check if they exist, or calculate them on the fly for the plot
        if 'EMA_50' not in plot_df.columns:
            plot_df['EMA_50'] = ta.ema(plot_df['Close'], length=50)
        if 'EMA_200' not in plot_df.columns:
            plot_df['EMA_200'] = ta.ema(plot_df['Close'], length=200)

        # Plot them with distinct colors
        ax1.plot(plot_df.index, plot_df['EMA_50'], color='orange', linewidth=1, label='EMA 50')
        ax1.plot(plot_df.index, plot_df['EMA_200'], color='blue', linewidth=1, label='EMA 200')

        # B. Bollinger Bands (Visual Volatility)
        if 'BB_Upper' in plot_df.columns:
            ax1.plot(plot_df.index, plot_df['BB_Upper'], color='green', linestyle='--', alpha=0.3)
            ax1.plot(plot_df.index, plot_df['BB_Lower'], color='red', linestyle='--', alpha=0.3)
            ax1.fill_between(plot_df.index, plot_df['BB_Upper'], plot_df['BB_Lower'], color='gray', alpha=0.1)
 
        ax1.set_title("ADA/USDT Price Action (15m)")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.2)
        ax1.legend(loc='upper left') # Keep legend out of the way
 
        # -------------------------------------------------------
        # BOTTOM CHART: Momentum (RSI Only)
        # -------------------------------------------------------
        if 'RSI' not in plot_df.columns:
            plot_df['RSI'] = ta.rsi(plot_df['Close'], length=14)
 
        ax2.plot(plot_df.index, plot_df['RSI'], label='RSI (14)', color='purple')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.set_title("RSI Momentum")
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.2)
        
        # REMOVED plt.show() - This blocks code execution!
 
        # 4. Save plot to buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
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
                    "content": "You are a professional crypto scalper. Analyze the chart pattern and technical data. Return a JSON with signal [1,0,-1] (LONG/HOLD/SHORT) and reasoning."
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
