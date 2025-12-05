##########
# Import #
##############################################################################

import logging
import time

from flow.run_bot import run_naive_bot
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
bot.profit_trailing(
    "ADAUSDT"
)
logger.info("Profit trailing run complete. Sleeping for 5 seconds.")
time.sleep(5)
print("#"*72)

##############################################################################
