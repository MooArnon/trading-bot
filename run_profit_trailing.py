##########
# Import #
##############################################################################

import logging
import time
import requests

from trading_bot.util.binance import check_weight_usage
from trading_bot.bot.binance_naive_bot import BinanceNaiveBot   
from trading_bot.util.logger import get_utc_logger

##########
# Static #
##############################################################################

logger = get_utc_logger(
    name=__name__,
    level=logging.DEBUG,
)

##########
# Flows #
##############################################################################

if __name__ == "__main__":
    bot = BinanceNaiveBot(logger=logger, symbol="ADAUSDT")
    logger.info("Running profit trailing...")
    bot.profit_trailing(
        "ADAUSDT",
        trailing_levels = [
            (3, 2), 
            (5, 3),
            (7, 5),
            (9, 7),
            (11, 9),
            (15, 12),
            (20, 18),
            (25, 20),
            (30, 25),
            (40, 35),
            (50, 45),
            (60, 55),
            (70, 65),
            (80, 75),
            (90, 85),
        ]
    )
    check_weight_usage()
    print("#"*72)

##############################################################################
