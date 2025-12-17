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
from flow.live import profit_trailing
from trading_bot.util.logger import get_utc_logger
from util.discord import DiscordNotify

##########
# Static #
##############################################################################

discord_alert = DiscordNotify(webhook_url=os.environ["DISCORD_ALERT_WEBHOOK_URL"])

leverage = config['LEVERAGE']

logger = get_utc_logger(
    name=__name__,
    level=logging.DEBUG,
)

client = Client(os.environ["BINANCE_API_KEY"], os.environ["BINANCE_SECRET_KEY"])

#############
# Functions #
##############################################################################

def main():
    try:
        profit_trailing(
            client=client,
            logger=logger,
            leverage=leverage,
            trailing_levels = [
                (2, 1.5), (3, 2), (5, 4), (7, 5), (9, 7), (11, 9),
                (15, 12), (20, 18), (25, 20), (30, 25), (40, 35),
                (50, 45), (60, 55), (70, 65), (80, 75), (90, 85),
            ]
        )
    
    except Exception as e:
        logger.error(f"Error in profit_trailing: {e}", exc_info=True)
        discord_alert.sent_message(
            message=f":warning: Error in profit_trailing: {e}",
            username="profit_trailing"
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
