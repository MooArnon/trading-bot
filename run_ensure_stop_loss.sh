#!/bin/bash

# Define the interval (in seconds) between runs
INTERVAL=10

echo "Starting iterative execution of run_ensure_stop_loss.py every $INTERVAL seconds. Press Ctrl+C to stop."
echo "---"

while true
do
    # Use python3 to ensure you're using the correct interpreter
    python3 run_ensure_stop_loss.py

    # Check if the last command (python3) succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: The script failed to run. Exiting loop."
        break
    fi

    # Wait for the specified interval before the next run
    sleep $INTERVAL

    echo "---"
done