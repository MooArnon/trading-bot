import os
from flow.utils import get_secret

config = {}

trading_bot_secrets = get_secret("trading-bot")
secrets_dict = eval(trading_bot_secrets)
os.environ["BINANCE_API_KEY"] = secrets_dict['BINANCE_API_KEY']
os.environ["BINANCE_SECRET_KEY"] = secrets_dict['BINANCE_SECRET_KEY']

config['BINANCE_API_KEY'] = secrets_dict['BINANCE_API_KEY']
config['BINANCE_SECRET_KEY'] = secrets_dict['BINANCE_SECRET_KEY']
config['LEVERAGE'] = int(os.environ["LEVERAGE"])
