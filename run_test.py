##########
# Import #
##############################################################################

import logging
import time
import os
import schedule

from binance.client import Client

from config.config import config
from trading_bot.util.binance import check_weight_usage
from trading_bot.market.binance import BinanceMarket
from flow.live import check_open_algo_order
from trading_bot.util.logger import get_utc_logger

##########
# Static #
##############################################################################

leverage = config['LEVERAGE']

logger = get_utc_logger(
    name=__name__,
    level=logging.DEBUG,
)

binance = BinanceMarket(
    logger=logger,
    symbol="ADAUSDT",
    leverage = leverage,
)

client = Client(os.environ["BINANCE_API_KEY"], os.environ["BINANCE_SECRET_KEY"])

#############
# Functions #
##############################################################################

def main():
    result = check_open_algo_order(
        client=client,
        symbol="ADAUSDT",
        logger=logger,
        type_to_cancel="STOP_MARKET",
        return_price=True,
    )
    print(result)

##########
# Flows #
##############################################################################

if __name__ == "__main__":
    main()

##############################################################################
