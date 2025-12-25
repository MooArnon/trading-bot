##########
# Import #
##############################################################################

import os
import time
import schedule
import traceback

from config.config import config
from flow.live import running_live
from trading_bot.bot.llm_bot import LLMBot
from trading_bot.util.logger import get_utc_logger
from trading_bot.market.binance import BinanceMarket
import logging 
from util.discord import DiscordNotify


##########
# Static #
##############################################################################

discord_alert = DiscordNotify(webhook_url=os.environ["DISCORD_ALERT_WEBHOOK_URL"])
discord_notify = DiscordNotify(webhook_url=os.environ["DISCORD_NOTIFY_WEBHOOK_URL"])

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
            notify_object=discord_notify,
        )
        bot = LLMBot(
            symbol=sym,
            logger=logger,
        )
        try:
            running_live(
                bot=bot,
                market=binance,
                logger=logger,
                grain="15m",
                get_data_limit=1000,
            )
        except Exception as e:
            discord_alert.sent_message(
                message=f":warning: Error running live for {sym}: {e}",
                username="live_bot"
            )
            logger.error(f"Error running live for {sym}: {e}")
            traceback.print_exc()
        time.sleep(30)
        print("-"*79)
    print("#"*79)

#######
# Run #
##############################################################################

schedule.every(60 * 15).seconds.do(main)

if __name__ == "__main__":
    main()

    logger.info("Bot started. Waiting for schedule...")
    while True:
        schedule.run_pending()
        time.sleep(1)
    
##############################################################################
