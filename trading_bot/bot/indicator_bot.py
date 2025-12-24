##########
# Import #
##############################################################################

import datetime
import os
from logging import Logger, INFO
import requests

from binance.client import Client
import numpy as np
import pandas as pd

from .binance import BinanceBot
from trading_bot.util.logger import get_utc_logger

###########
# Classes #
##############################################################################

class IndicatorBot(BinanceBot):
    bot_type="IndicatorBot"
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
            signal_type: str = ['KAMA', 'RSI'],
            kama_prce_buffer: float = 0.001, # 0.2% buffer,
            check_volatility: bool = True,
            check_rsi: bool = True,
            bandwidth_threshold: float = 0.020,
            notify_object: object = None,
            bb_lengthL: int = 20,
            bb_std: int = 2,
            rsi_length: int = 14,
            rsi_max: int = 70,
            rsi_min: int = 30,
    ):
        if logger is None:
            logger = get_utc_logger(
                name=__name__,
                level=INFO,
            )
        
        self.notify_object = notify_object
        self.symbol = symbol
        self.logger = logger
        self.signal_type = signal_type
        
        self.check_rsi = check_rsi
        self.rsi_max = rsi_max
        self.rsi_min = rsi_min
        self.rsi_length = rsi_length
        
        self.kama_prce_buffer = kama_prce_buffer
        
        self.check_volatility = check_volatility
        self.bandwidth_threshold = bandwidth_threshold
        
        self.client = Client(api_key, secret_key)
        
        self.base_url_usdm_market = "https://fapi.binance.com/fapi/v1/klines"
        
        if check_volatility:
            self.set_bb_params(bb_lengthL, bb_std)

    ##########################################################################
    
    @property
    def available_balance(self, asset: str = "USDT") -> float:
        """Return the amount balance for asset `asset`

        Parameters
        ----------
        asset : str, optional
            Target asset, by default "USDT"

        Returns
        -------
        float
            Float of available balance
        """
        future_account_asset = self.client.futures_account()['assets']
        return float(
            next(
                item['availableBalance'] \
                    for item in future_account_asset \
                        if item['asset'] == asset
            )
        )
    
    ##########################################################################
    
    @property
    def get_bb_params(self) -> dict:
        return {
            "bb_length": self.bb_length,
            "bb_std": self.bb_std
        }
        
    ##########################################################################
        
    def set_bb_params(self, bb_length: int, bb_std: float):
        self.bb_length = bb_length
        self.bb_std = bb_std

    ##########################################################################
    
    def transform(
            self, 
            data: pd.DataFrame,
            target_column: str = 'Close',
            n: int = 10,
            fast: int = 4, 
            slow: int = 50,
    ) -> pd.DataFrame:
        
        series = data[target_column].astype(float)
        
        # 1. Efficiency Ratio (ER)
        # Change = |Price - Price(n)|
        change = series.diff(n).abs()
        
        # Volatility = Sum(|Price(i) - Price(i-1)|) over n periods
        volatility = series.diff().abs().rolling(window=n).sum()
        
        er = change / volatility
        
        # Handle division by zero if volatility is 0 (price flatline)
        er = er.replace([np.inf, -np.inf], 0).fillna(0)

        # 2. Smoothing Constant (SC)
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # 3. KAMA Calculation
        kama = np.zeros_like(series)
        kama[:] = np.nan
        
        # Initialization: The first valid KAMA is often the price itself
        # We need 'n' data points before we can start.
        start_idx = n
        
        if start_idx < len(series):
            kama[start_idx-1] = series.iloc[start_idx-1]  # Initialize with previous Close
            
            # Iterative calculation
            for i in range(start_idx, len(series)):
                
                # KAMA = KAMA_prev + SC * (Price - KAMA_prev)
                kama[i] = kama[i-1] + sc.iloc[i] * (series.iloc[i] - kama[i-1])
                
        data['KAMA'] = kama

        return data
        
    ##########################################################################
    
    def generate_signal(
            self, 
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        # 1. Initialize a "combined_signal" column with 0s
        df = df.copy()
        df = self.transform(df)
        
        df['combined_signal'] = 0
        
        # 2. Check for KAMA
        if 'KAMA' in self.signal_type:
            self.logger.info("Activate KAMA signals...")
            
            # Generate temporary DF for KAMA
            kama_df = self.generate_kama_signal(df)
            # Add KAMA signals to the running total
            df['combined_signal'] = df['combined_signal'] + kama_df['signal']

        # 3. Check for RSI
        if 'RSI' in self.signal_type:
            self.logger.info("Activate RSI signals...")
            
            # Generate temporary DF for RSI
            rsi_df = self.generate_rsi_signal(df)
            # Add RSI signals to the running total
            df['combined_signal'] = df['combined_signal'] + rsi_df['signal']

        # 4. Normalize the Final Signal
        # If sum is positive (e.g., 1+0 or 1+1), result is 1
        # If sum is negative (e.g., -1+0 or -1-1), result is -1
        # If sum is 0 (e.g., 0+0 or 1-1), result is 0
        
        # Apply normalization:
        df['signal'] = 0
        df.loc[df['combined_signal'] >= 1, 'signal'] = 1
        df.loc[df['combined_signal'] <= -1, 'signal'] = -1
        
        # Clean up temporary column
        df.drop(columns=['combined_signal'], inplace=True)
        
        return df
    
    ##########################################################################
    
    def generate_rsi_signal(
            self,
            df: pd.DataFrame,
            target_column: str = 'Close',
    ) -> pd.DataFrame:
        """
        Generates Buy (1) and Sell (-1) signals based on RSI Mean Reversion.
        
        Logic:
        1. Buy: RSI crosses ABOVE the Oversold threshold (e.g., 30).
        2. Sell: RSI crosses BELOW the Overbought threshold (e.g., 70).
        3. Filter: Volatility (Bollinger Bandwidth) can be optionally applied.
        """
        # Ensure correct types
        df[target_column] = df[target_column].astype(float)
        
        # ---------------------------------------------------------
        # 1. Calculate RSI (Wilder's Smoothing)
        # ---------------------------------------------------------
        rsi_len = getattr(self, 'rsi_length', 14)
        
        # Calculate differences
        delta = df[target_column].diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use EWM for Wilder's Smoothing (standard RSI)
        avg_gain = gain.ewm(com=rsi_len - 1, adjust=False, min_periods=rsi_len).mean()
        avg_loss = loss.ewm(com=rsi_len - 1, adjust=False, min_periods=rsi_len).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        # Handle edge case where loss is 0 (pure uptrend)
        if avg_loss.sum() == 0:
            df.loc[avg_loss == 0, 'rsi'] = 100

        # ---------------------------------------------------------
        # 3. Generate Signals (Crossover Logic)
        # ---------------------------------------------------------
        rsi_overbought = self.rsi_max
        rsi_oversold = self.rsi_min

        curr_rsi = df['rsi']
        prev_rsi = df['rsi'].shift(1)

        # BUY: Previous RSI < 30 AND Current RSI > 30 (Crossing Up)
        crossover_up = (prev_rsi < rsi_oversold) & (curr_rsi > rsi_oversold)
        
        # SELL: Previous RSI > 70 AND Current RSI < 70 (Crossing Down)
        crossover_down = (prev_rsi > rsi_overbought) & (curr_rsi < rsi_overbought)
        
        # ---------------------------------------------------------
        # 4. Apply Combined Logic
        # ---------------------------------------------------------
        df['signal'] = 0
        
        # LONG Signal
        df.loc[crossover_up, 'signal'] = 1
        
        # SHORT Signal
        df.loc[crossover_down, 'signal'] = -1

        return df
    
    ##########################################################################
    
    def generate_kama_signal(
            self,
            df: pd.DataFrame,
            target_column: str = 'Close',
    ) -> pd.DataFrame:
        """
        Generates Buy (1) and Sell (-1) signals based on KAMA crossover.
        Filters:
        1. Volatility (Bollinger Bandwidth)
        2. RSI (Overbought/Oversold checks)
        """
        # df = df.copy()
        
        # Ensure correct types
        df['KAMA'] = df['KAMA'].astype(float)
        df[target_column] = df[target_column].astype(float)
        
        # ---------------------------------------------------------
        # 1. Volatility Filter
        # ---------------------------------------------------------
        is_safe_volatility = True 
        
        if getattr(self, 'check_volatility', False):
            self.logger.info("Calculating Bollinger Bandwidth for volatility filtering...")
            sma = df[target_column].rolling(window=self.get_bb_params['bb_length']).mean()
            std = df[target_column].rolling(window=self.get_bb_params['bb_std']).std()
            
            # CORRECTED: Use (2 * bb_std * std) instead of hardcoded (4 * std)
            # This ensures if you set bb_std=3, the bandwidth expands correctly.
            df['bandwidth'] = ((2 * self.get_bb_params['bb_std'] * std) / sma).fillna(0)
            
            is_safe_volatility = df['bandwidth'] > self.bandwidth_threshold

        # ---------------------------------------------------------
        # 2. RSI Filter
        # ---------------------------------------------------------
        # Default to True (Safe to trade)
        is_rsi_bullish = pd.Series(True, index=df.index)
        is_rsi_bearish = pd.Series(True, index=df.index)

        if getattr(self, 'check_rsi', False):
            # Calculate RSI (Wilder's)
            delta = df[target_column].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)

            # NEW LOGIC:
            # Long: RSI must be > 50 (Momentum is Up)
            # Short: RSI must be < 50 (Momentum is Down)
            is_rsi_bullish = df['rsi'] > 50
            is_rsi_bearish = df['rsi'] < 50

        # ---------------------------------------------------------
        # 3. KAMA Crossover Signals
        # ---------------------------------------------------------

        # Check if Price was above KAMA for the LAST 2 candles
        is_above = df[target_column] > df['KAMA']
        is_below = df[target_column] < df['KAMA']

        # BUY: Current is above, Previous was above, 2-bars ago was below
        crossover_up = (
            is_above &              # Currently Above
            is_above.shift(1) &     # Previous Candle Also Above (Confirmation)
            is_below.shift(2)       # 2 Candles ago was Below (The crossover start)
        )

        # SELL: Current is below, Previous was below, 2-bars ago was above
        crossover_down = (
            is_below & 
            is_below.shift(1) & 
            is_above.shift(2)
        )
        
        # ---------------------------------------------------------
        # 4. Signals
        # ---------------------------------------------------------
        df['signal'] = 0
        
        # Long: Price Crosses Up + Volatility exists + Momentum is Bullish
        df.loc[crossover_up & is_safe_volatility & is_rsi_bullish, 'signal'] = 1
        
        # Short: Price Crosses Down + Volatility exists + Momentum is Bearish
        df.loc[crossover_down & is_safe_volatility & is_rsi_bearish, 'signal'] = -1

        return df
    
    ##########################################################################
        
##############################################################################
