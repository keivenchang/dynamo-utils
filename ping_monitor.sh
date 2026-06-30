#!/bin/bash

# Ping monitoring script that runs forever
# Prints date, pings www.yahoo.com, and waits 15 seconds

while true; do
    echo "$(date): $(ping -c 1 www.yahoo.com | grep 'bytes from')"
    sleep 15
done
