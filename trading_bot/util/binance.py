##########
# Import #
##############################################################################

import time
import requests

##########
# Static #
##############################################################################

BASE_URL = "https://fapi.binance.com"

############
# Function #
##############################################################################

def check_weight_usage():
    try:
        # We use a low-weight endpoint just to check headers
        response = requests.get(f"{BASE_URL}/fapi/v1/time")
        
        # Parse Headers
        used_weight = int(response.headers.get('x-mbx-used-weight-1m', 0))
        
        print(f"Current Weight Used: {used_weight} / 2400")
        
        # Safety Threshold (Stop at 2300 to be safe)
        if used_weight > 2300:
            print("WARNING: Approaching limit. Sleeping for 60 seconds...")
            time.sleep(60)
            
    except Exception as e:
        print(f"Error checking limits: {e}")

##############################################################################


