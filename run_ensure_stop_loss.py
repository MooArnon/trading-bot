##########
# Import #
##############################################################################

import logging
import time

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

bot = BinanceNaiveBot(logger=logger, symbol="ADAUSDT")
logger.info("Running profit trailing...")
bot.check_and_create_stop_loss(
    "ADAUSDT",
    stop_loss_percent=3.0,
)
check_weight_usage()
logger.info("Sleeping for 5 seconds.")
time.sleep(5)
print("#"*72)

##############################################################################
