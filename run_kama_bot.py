##########
# Import #
##############################################################################

import time
import schedule

from config.config import config
from flow.live import running_live
from trading_bot.bot.indicator_bot import IndicatorBot
from trading_bot.util.logger import get_utc_logger
from trading_bot.market.binance import BinanceMarket
import logging 

##########
# Static #
##############################################################################

symbol_to_check = [
    "ADAUSDT",
]

leverage = config['LEVERAGE']

logger = get_utc_logger(
    name=__name__,
    level=logging.DEBUG,
)

#############
# Functions #
##############################################################################

def main():
    for sym in symbol_to_check:
        binance = BinanceMarket(
            logger=logger,
            symbol=sym,
            leverage = leverage,
        )
        bot = IndicatorBot(
            symbol=sym,
            logger=logger,
            check_volatility = True,
            bandwidth_threshold = config['BANDWIDTH_THRESHOLD'],
        )
        try:
            running_live(
                bot=bot,
                market=binance,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"Error running live for {sym}: {e}")
        time.sleep(30)
        print("-"*79)
    print("#"*79)

#######
# Run #
##############################################################################

schedule.every(60).seconds.do(main)

if __name__ == "__main__":

    logger.info("Bot started. Waiting for schedule...")
    while True:
        schedule.run_pending()
        time.sleep(1)
    
##############################################################################
