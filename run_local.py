##########
# Import #
##############################################################################

import logging

from flow.simulation import running_simulation
from trading_bot.bot.binance import BinanceBot   
from trading_bot.util.logger import get_utc_logger

##########
# Static #
##############################################################################

START_TIME = '2025-10-25 00:00:00'

logger = get_utc_logger(
    name=__name__,
    level=logging.DEBUG,
)

##########
# Flows #
##############################################################################

bot = BinanceBot(logger=logger, symbol="ADAUSDT")
running_simulation(
    start_time=START_TIME,
    bot=bot,
)

##############################################################################
