# Use this code snippet in your app.
# If you need more information about configurations
# or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developer/language/python/

from flow.live import get_my_open_positions
import os
from binance.client import Client
client = Client(os.environ["BINANCE_API_KEY"], os.environ["BINANCE_SECRET_KEY"])

if __name__ == "__main__":
    contracts = get_my_open_positions(client=client)
    print(contracts)
