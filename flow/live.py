##########
# Import #
##############################################################################

import datetime
import random
import os
import time

import pandas as pd
from binance.helpers import round_step_size
from binance.client import Client
from binance.exceptions import BinanceAPIException

from trading_bot.bot.__base import BaseBot
from trading_bot.market.__base import BaseMarket

##########
# Flows #
##############################################################################

def running_live(bot: BaseBot, market: BaseMarket, logger, grain: str = '1m') -> None:
    current_time = datetime.datetime.now(tz=datetime.timezone.utc)
    logger.info(f"Starting live mode at {current_time}")
    try:
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
        if grain == "1m":
            timestamp_truncated = truncate_to_1_minute(
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

        signal_mapper = bot.get_signal_mapper()
        signal = signal_mapper[signal.iloc[-1]['signal']]
        logger.info(f"Price is {data['Close'].iloc[-1]} | KAMA : {data['KAMA'].iloc[-1]}| Signal: {signal}")
        
        market.open_order_flow(
            signal=signal,
        )
        
    except KeyboardInterrupt:
        logger.info("\nTimestamp generation stopped.")

##############################################################################

def profit_trailing(
    symbol: str,
    logger,
    leverage: int,
    trailing_levels: list = [
        (2, 1.5), (3, 2), (5, 3.5), (7, 5), (9, 7), (11, 9),
        (15, 12), (20, 18), (25, 20), (30, 25), (40, 35),
        (50, 45), (60, 55), (70, 65), (80, 75), (90, 85),
    ]
) -> None:
    """
    Dynamically adjust stop-loss based on ROI thresholds.
    """
    client = Client(os.environ["BINANCE_API_KEY"], os.environ["BINANCE_SECRET_KEY"])

    try:
        # 1. Get position info
        # Note: We use client here, not a new Client() instance
        position_info = client.futures_position_information(symbol=symbol)
        
        # Safe check for empty position data
        if not position_info or float(position_info[0]['positionAmt']) == 0:
            logger.info(f"No position information for {symbol}. Exiting.")
            return

        # Extract data
        position_data = position_info[0]
        position_amt = float(position_data['positionAmt'])
        entry_price = float(position_data['entryPrice'])
        unrealized_profit = float(position_data['unRealizedProfit'])
        
        # Calculate ROI % manually to be safe (unRealizedProfit / Initial Margin)
        # Initial Margin = (Entry Price * Quantity) / Leverage
        initial_margin = (entry_price * abs(position_amt)) / leverage
        if initial_margin == 0: 
            return # Avoid division by zero
        
        # ROI as a percentage (e.g., 5.0 for 5%)
        roi = (unrealized_profit / initial_margin) * 100 

        # 2. Determine side
        side = "LONG" if position_amt > 0 else "SHORT"

        # 3. Get Precision Info (CRITICAL FIX)
        # We need to know the allowed tick size and step size for this symbol
        exchange_info = client.futures_exchange_info()
        symbol_info = next((item for item in exchange_info['symbols'] if item['symbol'] == symbol), None)
        
        if not symbol_info:
            logger.error(f"Could not find symbol info for {symbol}")
            return
            
        tick_size = float(symbol_info['filters'][0]['tickSize']) # Price precision
        step_size = float(symbol_info['filters'][1]['stepSize']) # Quantity precision

        logger.info(
            f"Side: {side} | Entry: {entry_price} | ROI: {roi:.2f}% | "
            f"Precision - Price: {tick_size}, Qty: {step_size}"
        )

        if roi <= 0:
            return

        # 4. Check existing SL orders
        open_orders = client.futures_get_open_orders(symbol=symbol)
        current_sl_order = None
        for order in open_orders:
            if order.get('type') in ['STOP_MARKET', 'STOP'] and order.get('reduceOnly') is True:
                current_sl_order = order
                break

        # 5. Determine trigger level
        trailing_levels_sorted = sorted(trailing_levels, key=lambda x: x[0])
        triggered_level = None
        
        for (profit_trigger, stop_loss_level) in trailing_levels_sorted:
            if roi >= profit_trigger:
                triggered_level = (profit_trigger, stop_loss_level)
            else:
                break

        if not triggered_level:
            return

        # 6. Compute New Stop Loss Price
        stop_loss_target_roi = triggered_level[1]
        price_movement_pct = stop_loss_target_roi / leverage

        if side == "LONG":
            raw_sl_price = entry_price * (1 + (price_movement_pct / 100))
        else:
            raw_sl_price = entry_price * (1 - (price_movement_pct / 100))

        # 7. Round Price using Binance helper (CRITICAL FIX)
        new_sl_price = round_step_size(raw_sl_price, tick_size)

        # 8. Check against existing SL (Don't move backwards)
        if current_sl_order:
            current_sl_price = float(current_sl_order.get('stopPrice', 0))
            
            # Allow a tiny tolerance for floating point comparison
            if abs(new_sl_price - current_sl_price) < tick_size:
                return # Same price, ignore

            if side == "LONG" and new_sl_price < current_sl_price:
                logger.info(f"New SL {new_sl_price} < Current {current_sl_price}. Ignoring to protect profit.")
                return
            elif side == "SHORT" and new_sl_price > current_sl_price:
                logger.info(f"New SL {new_sl_price} > Current {current_sl_price}. Ignoring to protect profit.")
                return

        logger.info(f"Updating SL to lock {stop_loss_target_roi}% ROI. Price: {new_sl_price}")

        # 9. Cancel Existing SL
        if current_sl_order:
            try:
                client.futures_cancel_order(symbol=symbol, orderId=current_sl_order['orderId'])
            except Exception as e:
                logger.warning(f"Failed to cancel old SL: {e}")

        # 10. Place New SL Order
        # Round the quantity to the valid step size
        qty_to_close = round_step_size(abs(position_amt), step_size)
        
        sl_side = Client.SIDE_SELL if side == "LONG" else Client.SIDE_BUY

        try:
            cancel_algo_order(client, symbol, 'STOP_MARKET', logger)
            order_info = client.futures_create_order(
                symbol=symbol,
                side=sl_side,
                type='STOP_MARKET',
                stopPrice=new_sl_price,     # Standard endpoint uses stopPrice
                closePosition=False,        # We use reduceOnly instead
                quantity=qty_to_close,      # Explicit rounded quantity
                reduceOnly=True,
                timeInForce='GTC'
            )
            logger.info(f"SUCCESS: Trailing SL updated to {new_sl_price}")
            
            
        except BinanceAPIException as e:
            if "Order would immediately trigger" in str(e):
                logger.warning(f"SL Price {new_sl_price} is too close to current price. Skipping.")
            else:
                logger.error(f"Error placing SL: {e}")
                
    except Exception as e:
        logger.error(f"Critical error in profit_trailing: {e}", exc_info=True)
        
##########################################################################

def ensure_stop_loss(
        symbol: str,
        client,
        leverage,
        logger,
        stop_loss_percent: int = 5,
        max_stop_loss_percent: int = 15,
) -> None:
        """
        Check open positions and add stop loss

        Parameters
        ----------
        symbol : str
            Target symbol
        stop_loss_percent : int
            Initial percentage of stop loss threshold, default is 5
        max_stop_loss_percent : int
            Maximum percentage for stop loss, default is 15 
            (to prevent endless loop)
        leverage: int 
            default is 20
        """

        # Get the current position for the symbol
        positions_info = client.futures_position_information(symbol=symbol)
        
        if not positions_info or float(positions_info[0]['positionAmt']) == 0:
            logger.info(f"No position information for {symbol}. Exiting.")
            return
        
        is_opened_order = check_open_algo_order(
            client, symbol, 'STOP_MARKET', logger
        )
        if is_opened_order is True:
            logger.info(f"Stop-loss Algo order already exists for {symbol}. Exiting.")
            return

        positions_info = positions_info[0]
        
        position_amt = 0.0
        entry_price = 0.0
        position_amt = float(positions_info['positionAmt'])
        entry_price = float(positions_info['entryPrice'])

        # If no open position, exit the function
        if position_amt == 0.0:
            logger.info(f"No open position for {symbol}")
            return

        logger.info(
            f"Open position for {symbol}: {position_amt} at entry price {entry_price}"
        )

        # Check open orders to see if there's an existing stop-loss order
        open_orders = client.futures_get_open_orders(symbol=symbol)
        has_stop_loss = False
        for order in open_orders:
            if order['type'] == 'STOP_MARKET':
                has_stop_loss = True
                logger.info(f"Stop-loss already exists for {symbol}")
                break

        # If a stop-loss exists, return without doing anything
        if has_stop_loss:
            return

        # If no stop-loss, calculate and create one with retry logic
        # Calculate stop-loss price based on position type (long/short)
        while stop_loss_percent <= max_stop_loss_percent:
            try:
                
                # Long position
                if position_amt > 0: 
                    stop_loss_price = entry_price * (1 - (stop_loss_percent / 100)/leverage)
                    stop_loss_side = Client.SIDE_SELL
                
                # Short position
                else: 
                    stop_loss_price = entry_price * (1 + (stop_loss_percent / 100)/leverage )
                    stop_loss_side = Client.SIDE_BUY

                # Round stop-loss price to correct precision
                stop_loss_price = round_down_to_precision(stop_loss_price)
                logger.info(
                    f"Attempting to create stop-loss at {stop_loss_price} with {stop_loss_percent}% threshold for {symbol}"
                )

                # Place stop-loss order
                cancel_algo_order(client, symbol, 'STOP_MARKET', logger)
                stop_loss_order = client.futures_create_order(
                    symbol=symbol,
                    side=stop_loss_side,
                    type='STOP_MARKET',
                    stopPrice=stop_loss_price,
                    quantity=abs(position_amt)
                )
                logger.info(f"Stop-loss created successfully: {stop_loss_order}")
                
                # Exit the loop after successful order
                break  

            except BinanceAPIException as e:
                
                # Check if the error is due to "Order would immediately trigger"
                if "Order would immediately trigger" in str(e):
                    logger.warning(
                        f"Stop-loss failed: {e}. Increasing stop-loss percent to {stop_loss_percent + 0.5}%"
                    )
                    
                    # Increment stop loss by 0.5% and retry
                    stop_loss_percent += 0.5 
                else:
                    logger.error(f"Error creating stop-loss: {e}")
                    
                    # Break loop for other API errors
                    break  

        if stop_loss_percent > max_stop_loss_percent:
            logger.error(
                f"Failed to create stop-loss for {symbol} after reaching max stop-loss percent of {max_stop_loss_percent}%."
            )
            
##########################################################################

def truncate_to_1_minute(dt: datetime.datetime) -> datetime.datetime:
        """
        Truncates the given datetime's minutes down to the nearest
        quarter-hour (00, 15, 30, or 45).
        
        Seconds and microseconds are also reset to 0.
        """
        # Use integer division to find the 15-minute block
        # For example:
        # 14 // 15 = 0  -> 0 * 15 = 0
        # 25 // 15 = 1  -> 1 * 15 = 15
        # 46 // 15 = 3  -> 3 * 15 = 45
        new_minute = (dt.minute // 1) * 1
        
        # Return a new datetime object with 
        # truncated minutes, seconds, and microseconds
        return dt.replace(minute=new_minute, second=0, microsecond=0)

##############################################################################

def round_down_to_precision(value, is_quantity=False):
    """
    Mock method: Rounds a price or quantity value down to the required precision 
    (based on the symbol's filters). Crucial for successful API calls.
    """
    if is_quantity:
        # BTCUSDT often has 3 decimal places for quantity
        return float(f"{value:.4f}")
    else:
        # BTCUSDT often has 2 decimal places for price
        return float(f"{value:.4f}")

##############################################################################

def get_roi(symbol: str, all_positions_data: list) -> float:
        """
        Calculate the percentage ROI for the open futures position on the given symbol.

        The formula calculates the Return on Initial Margin, which includes leverage.
        ROI = (Unrealized PnL / Initial Margin) * 100 
        (This is already calculated by Binance as 'unRealizedProfit' and 'positionInitialMargin'.)

        We will use the direct price change method you implemented, 
        but ensure we find the correct position dictionary first.

        Parameters
        ----------
        symbol : str
            Futures symbol (e.g., "ADAUSDT")
        all_positions_data : list
            The full list of position information returned by client.get_position_risk().

        Returns
        -------
        float
            The ROI in percentage. Returns 0.0 if there is no open position.
        """
        
        # 1. Find the specific position dictionary for the target symbol
        position_info = next((p for p in all_positions_data if p.get('symbol') == symbol), None)

        roi = float(position_info.get('unRealizedProfit', 0)) / \
                           float(position_info.get('positionInitialMargin', 0)) * 100 

        return roi

##############################################################################

def cancel_algo_take_profits(client, symbol):
    print(f"--- Searching for ALGO Take Profits on {symbol} ---")
    # 1. Fetch all open Algo orders
    try:
        response = client.futures_get_open_algo_orders(symbol=symbol)
        # Handle if response is a list or a dict containing 'orders'
        algo_orders = response.get('orders', []) if isinstance(response, dict) else response
    except Exception as e:
        print(f"Error fetching orders: {e}")
        return

    tp_found = False

    # 2. Iterate and check for Take Profit types
    for order in algo_orders:
        # Check if it is a Take Profit (handles both MARKET and LIMIT TPs)
        if 'TAKE_PROFIT' in order.get('orderType', ''):
            algo_id = order['algoId']
            print(f"Found TP Algo Order: {algo_id}. Cancelling...")
            
            try:
                client.futures_cancel_algo_order(
                    symbol=symbol,
                    algoId=algo_id
                )
                print(f" -> Cancelled TP: {algo_id}")
                tp_found = True
            except Exception as e:
                print(f" -> Failed to cancel {algo_id}: {e}")

    if not tp_found:
        print("No Algo Take Profit orders found.")

##############################################################################

def cancel_algo_order(client, symbol, type_to_cancel, logger):
    logger.info(f"--- Searching for ALGO Take {type_to_cancel} on {symbol} ---")
    # 1. Fetch all open Algo orders
    try:
        response = client.futures_get_open_algo_orders(symbol=symbol)
        # Handle if response is a list or a dict containing 'orders'
        algo_orders = response.get('orders', []) if isinstance(response, dict) else response
    except Exception as e:
        logger.info(f"Error fetching orders: {e}")
        return

    type_to_cancel_found = False

    # 2. Iterate and check
    for order in algo_orders:

        if type_to_cancel in order.get('orderType', ''):
            algo_id = order['algoId']
            logger.info(f"Found {type_to_cancel} Algo Order: {algo_id}. Cancelling...")
            
            try:
                client.futures_cancel_algo_order(
                    symbol=symbol,
                    algoId=algo_id
                )
                logger.info(f" -> Cancelled TP: {algo_id}")
                type_to_cancel_found = True
            except Exception as e:
                logger.info(f" -> Failed to cancel {algo_id}: {e}")

    if not type_to_cancel_found:
        logger.info(f"No Algo {type_to_cancel} orders found.")

##############################################################################

def check_open_algo_order(client, symbol, type_to_cancel, logger):
    logger.info(f"--- Searching for ALGO {type_to_cancel} on {symbol} ---")
    # 1. Fetch all open Algo orders
    try:
        response = client.futures_get_open_algo_orders(symbol=symbol)
        # Handle if response is a list or a dict containing 'orders'
        algo_orders = response.get('orders', []) if isinstance(response, dict) else response
    except Exception as e:
        logger.info(f"Error fetching orders: {e}")
        return

    # 2. Iterate and check
    for order in algo_orders:

        if type_to_cancel in order.get('orderType', ''):
            algo_id = order['algoId']
            logger.info(f"Found {type_to_cancel} Algo Order: {algo_id}.")
            
            return True
        
    return False
            
##############################################################################
