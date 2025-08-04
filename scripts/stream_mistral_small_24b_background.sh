#!/bin/bash
set -euo pipefail

# Background streaming script for Mistral-Small-24B-Base-2501
# Follows CLAUDE.md safety principles

LOG_FILE="mistral_small_24b_streaming_$(date +%Y%m%d_%H%M%S).log"
SCRIPT_PATH="/home/shingai/sort/deployments/scripts/stream_mistral_small_24b_to_s3.py"

echo "Starting Mistral-Small-24B-Base-2501 background streaming..."
echo "Script: $SCRIPT_PATH"
echo "Log: $LOG_FILE"

# Run in background with nohup
nohup python3 "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Background process started with PID: $PID"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Check process status with: ps aux | grep $PID"

# Save PID for monitoring
echo "$PID" > mistral_streaming.pid

echo "Stream initiated successfully. Check log file for progress."