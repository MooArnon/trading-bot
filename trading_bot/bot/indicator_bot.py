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
            signal_type: str = 'KAMA',
    ):
        if logger is None:
            logger = get_utc_logger(
                name=__name__,
                level=INFO,
            )
            
        self.symbol = symbol
        self.logger = logger
        self.signal_type = signal_type
        
        self.client = Client(api_key, secret_key)
        
        self.base_url_usdm_market = "https://fapi.binance.com/fapi/v1/klines"

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
    
    def trasform(
            self, 
            data: pd.DataFrame,
            target_column: str = 'Close',
            n: int = 10,
            fast: int = 2, 
            slow: int = 30,
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
    
    def generate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.signal_type == 'KAMA':
            return self.generate_kama_signal(df)
    
    ##########################################################################
    
    def generate_kama_signal(
            self,
            df: pd.DataFrame,
            target_column: str = 'Close',
    ) -> pd.DataFrame:
        """
        Generates Buy (1) and Sell (-1) signals based on KAMA crossover.
        """
        df = df.copy()
        # 1. Create a 'Signal' column initialized to 0
        df['signal'] = 0
        
        df['KAMA'] = df['KAMA'].astype(float)
        df[target_column] = df[target_column].astype(float)

        # 2. Define the Buy Condition (Crossover Up)
        # current Close > current KAMA  AND  previous Close < previous KAMA
        buy_condition = (
            df[target_column] > df['KAMA']) & (df[target_column].shift(1) < df['KAMA'].shift(1)
        )
        
        # 3. Define the Sell Condition (Crossover Down)
        # current Close < current KAMA  AND  previous Close > previous KAMA
        sell_condition = (
            df[target_column] < df['KAMA']) & (df[target_column].shift(1) > df['KAMA'].shift(1)
        )
        
        # 4. Apply Signals
        df.loc[buy_condition, 'signal'] = 1   # BUY
        df.loc[sell_condition, 'signal'] = -1 # SELL

        return df
    
    ##########################################################################
        
##############################################################################
