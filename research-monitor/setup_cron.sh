#!/bin/bash
# Sets up a daily cron job for Research Monitor.
# Runs at 08:00 local time every day.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="$(which python3)"
LOG_FILE="$SCRIPT_DIR/data/cron.log"

mkdir -p "$SCRIPT_DIR/data"

CRON_CMD="0 8 * * * cd $SCRIPT_DIR && $PYTHON_PATH run.py >> $LOG_FILE 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "research-monitor"; then
    echo "Cron job already exists. Updating..."
    crontab -l 2>/dev/null | grep -v "research-monitor" | { cat; echo "$CRON_CMD  # research-monitor"; } | crontab -
else
    echo "Adding new cron job..."
    (crontab -l 2>/dev/null; echo "$CRON_CMD  # research-monitor") | crontab -
fi

echo "✅ Cron job configured:"
echo "   Schedule: Every day at 08:00"
echo "   Log file: $LOG_FILE"
echo ""
echo "Current crontab:"
crontab -l | grep "research-monitor"
