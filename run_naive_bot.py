##########
# Import #
##############################################################################

from argparse import ArgumentParser
import logging

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

def get_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--min-wait', 
        type=int,
        default=60,
        help='Minimum wait time between requests in milliseconds'
    )
    parser.add_argument(
        '--max-wait', 
        type=int,
        default=300,
        help='Maximum wait time between requests in milliseconds'
    )
    return parser.parse_args()

##############################################################################

if __name__ == "__main__":
    args = get_args()
    bot = BinanceNaiveBot(
        logger=logger, 
        symbol="ADAUSDT",
        min_wait = args.min_wait,
        max_wait = args.max_wait,
    )
    run_naive_bot(
        bot=bot,
    )

##############################################################################
