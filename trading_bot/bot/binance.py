##########
# Import #
##############################################################################

import datetime
import os
from logging import Logger, INFO
import requests

from binance.client import Client
import pandas as pd

from . import BaseBot
from trading_bot.util.logger import get_utc_logger

###########
# Classes #
##############################################################################

class BinanceBot(BaseBot):
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
            logger: Logger=None
    ):
        if logger is None:
            logger = get_utc_logger(
                name=__name__,
                level=INFO,
            )
            
        self.symbol = symbol
        self.logger = logger
        
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
    
    def search(self, *args, **kawrgs) -> None:
        raise NotImplementedError("Child class must implement seach method")
    
    ##########################################################################
    
    def get_data(
            self, 
            symbol: str, 
            target_date: str, 
            limit: int = 30,
            interval: str = Client.KLINE_INTERVAL_1MINUTE,
    ) -> None:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        
        minutes_to_subtract = 1 * limit
        duration = datetime.timedelta(minutes=minutes_to_subtract)
        params["endTime"] = int(target_date.timestamp() * 1000)
        params["startTime"] = int(
            (target_date - duration).timestamp() * 1000
        )
        response = requests.get(self.base_url_usdm_market, params=params)
        self.logger.debug(f"{params}")
        return  pd.DataFrame(response.json(), columns=self.kline_columns)
    
    ##########################################################################
    
##############################################################################
