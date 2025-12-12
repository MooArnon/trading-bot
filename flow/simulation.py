##########
# Import #
##############################################################################

import datetime
import random
import time

import pandas as pd

from trading_bot.bot.__base import BaseBot

##########
# Flows #
##############################################################################

def running_simulation(start_time: str, bot: BaseBot) -> None:
    current_time = datetime.datetime.strptime(
        start_time, 
        '%Y-%m-%d %H:%M:%S',
    )
    try:
        while True:
            # 1. Generate a random second between 1 and 30
            # The user requested '01' to '30'
            random_second = random.randint(1, 30)
            
            # 2. Create the new timestamp by replacing the 
            # 'second' component of our incrementing time with 
            # the random second.
            timestamp_step = current_time.replace(
                second=random_second
            )
            timestamp_step_str = timestamp_step.strftime('%Y-%m-%d %H:%M:%S')
            timestamp_truncated = bot.truncate_to_quarter_hour(
                timestamp_step
            )

            # 3. Format and print the timestamp
            # %S handles the leading zero (e.g., 5 becomes '05')
            bot.logger.debug(
                f"start: {timestamp_step_str} | " \
                    f"trunc_timestamp:  {timestamp_truncated}"
            )
            
            # 4. Increment the base time for the next loop
            # This ensures the timestamp "increments" 
            # (e.g., minutes/hours will change)
            current_time += datetime.timedelta(minutes=15)
            
            data: pd.DataFrame = bot.get_data(
                symbol=bot.symbol, 
                target_date=timestamp_truncated, 
            )
            
            feature = bot.trasform(
                data=data
            )
            
            signal = bot.generate_signal(
                df=feature
            )
            print(signal)
            bot.logger.info(
                f"Generated signal at {timestamp_step_str}: {signal['signal'].values[-1]}"
            )
            # 5. Wait for a second before printing the next one
            time.sleep(1)
        
            
    except KeyboardInterrupt:
        print("\nTimestamp generation stopped.")

##############################################################################
