##########
# Import #
##############################################################################

import logging
import time
import os
import schedule

from binance.client import Client
from util.discord import DiscordNotify

from config.config import config
from flow.live import ensure_stop_loss
from trading_bot.util.binance import check_weight_usage
from trading_bot.util.logger import get_utc_logger

##########
# Static #
##############################################################################

leverage = config['LEVERAGE']

logger = get_utc_logger(
    name=__name__,
    level=logging.DEBUG,
)

client = Client(os.environ["BINANCE_API_KEY"], os.environ["BINANCE_SECRET_KEY"])

discord_alert = DiscordNotify(webhook_url=os.environ["DISCORD_ALERT_WEBHOOK_URL"])

#############
# Functions #
##############################################################################

def main():
    logger.info("Run ensure stop loss...")
    try:
        ensure_stop_loss(
            client=client,
            logger=logger,
            stop_loss_percent=3.0,
            leverage=leverage
        )   
        raise BufferError("Test error for discord alert")
    except Exception as e:
        logger.error(f"Error in ensure_stop_loss: {e}", exc_info=True)
        discord_alert.sent_message(
            message=f":warning: Error in ensure_stop_loss: {e}",
            username="ensure_stop_loss"
        )
    check_weight_usage(logger)
    logger.info("###################### END ONE LOOP ######################")

##########
# Flows #
##############################################################################

schedule.every(5).seconds.do(main)

if __name__ == "__main__":

    while True:
        schedule.run_pending()
        time.sleep(1)

##############################################################################
