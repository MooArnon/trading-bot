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
            # (peak_roi, stop_roi) → ห่าง → % give back
            
            # ไม่ trail จนกว่ากำไร >= 3% (มากกว่า risk 2.5%)
            (3.0, 0.0),      # → 3.0  → breakeven
            (4.0, 2.0),      # → 2.0  → 50%
            (5.0, 2.5),      # → 2.5  → 50%
            (6.0, 3.5),      # → 2.5  → 42%
            (7.0, 4.2),      # → 2.8  → 40%
            (8.0, 5.0),      # → 3.0  → 37%
            (10.0, 6.5),     # → 3.5  → 35%
            (12.0, 8.0),     # → 4.0  → 33%
            (15.0, 10.0),    # → 5.0  → 33%
            (20.0, 14.0),    # → 6.0  → 30%
            (25.0, 18.0),    # → 7.0  → 28%
            (30.0, 22.0),    # → 8.0  → 27%
            (40.0, 30.0),    # → 10.0 → 25%
            (50.0, 38.0),    # → 12.0 → 24%
            (60.0, 46.0),    # → 14.0 → 23%
            (70.0, 54.0),    # → 16.0 → 23%
            (80.0, 62.0),    # → 18.0 → 22%
            (90.0, 70.0),    # → 20.0 → 22%
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
