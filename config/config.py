import os
config = {}

config['BINANCE_API_KEY'] = os.environ['BINANCE_API_KEY']
config['BINANCE_SECRET_KEY'] = os.environ['BINANCE_SECRET_KEY']
config['LEVERAGE'] = int(
    os.environ["LEVERAGE"]
)
