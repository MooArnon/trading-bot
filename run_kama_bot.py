##########
# Import #
##############################################################################

from argparse import ArgumentParser
import time
import schedule
import os

from config.config import config
from flow.simulation import running_simulation
from flow.live import running_live
from trading_bot.bot.indicator_bot import IndicatorBot
from trading_bot.util.logger import get_utc_logger
from trading_bot.market.binance import BinanceMarket
import logging 

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
    bot = IndicatorBot(
        symbol="ADAUSDT",
        logger=logger,
    )
    running_live(
        bot=bot,
        market=binance,
        logger=logger,
    )
    print("#"*79)

#######
# Run #
##############################################################################

schedule.every(15).seconds.do(main)

if __name__ == "__main__":

    logger.info("Bot started. Waiting for schedule...")
    while True:
        schedule.run_pending()
        time.sleep(1)
    
##############################################################################
