#!/bin/bash
# Fix ownership of directories that may get modified by Docker containers
# This script should be run periodically via cron

# Get the actual user (not root, even if running with sudo)
if [ -n "$SUDO_USER" ]; then
    TARGET_USER="$SUDO_USER"
else
    TARGET_USER="$USER"
fi

echo "$(date): Fixing ownership for user: $TARGET_USER"

# Fix ownership of common directories
sudo chown -R "$TARGET_USER":"$TARGET_USER" \
    "$HOME/.cache" \
    "$HOME/.cargo" \
    "$HOME/nvidia" \
    2>/dev/null

echo "$(date): Ownership fix completed"

