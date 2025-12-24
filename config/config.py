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

config['BANDWIDTH_THRESHOLD'] = float(os.getenv("BANDWIDTH_THRESHOLD", 0.003)) 

config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", None)

config['LLM_CONFIDENCE_PERCENTAGE_THRESHOLD'] = int(os.getenv("LLM_CONFIDENCE_PERCENTAGE_THRESHOLD", 75))