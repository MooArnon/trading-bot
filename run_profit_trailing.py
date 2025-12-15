##########
# Import #
##############################################################################

import logging
import time
import requests
import schedule

from config.config import config
from trading_bot.util.binance import check_weight_usage
from trading_bot.market.binance import BinanceMarket
from flow.live import profit_trailing
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

#############
# Functions #
##############################################################################

def main():
    profit_trailing(
        symbol="ADAUSDT",
        logger=logger,
        leverage=leverage,
        trailing_levels = [
            (2, 1.5), (3, 2), (5, 4), (7, 5), (9, 7), (11, 9),
            (15, 12), (20, 18), (25, 20), (30, 25), (40, 35),
            (50, 45), (60, 55), (70, 65), (80, 75), (90, 85),
        ]
    )
    check_weight_usage(logger)
    logger.info("###################### END ONE LOOP ######################")

##########
# Flows #
##############################################################################

schedule.every(3).seconds.do(main)

if __name__ == "__main__":
    logger.info("Running profit trailing...")
    while True:
        schedule.run_pending()
        time.sleep(1)

##############################################################################
