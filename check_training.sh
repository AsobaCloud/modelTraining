#!/bin/bash
set -euo pipefail

# Training Status Checker with Run Selection
echo "🔍 Training Status Checker"
echo "========================="

BUCKET="asoba-llm-cache"
REGION="us-east-1"

# Get list of training runs
echo "📋 Available training runs:"
echo

# List runs with numbers
aws s3 ls s3://${BUCKET}/training-runs/ --region ${REGION} | \
    grep "PRE " | \
    sed 's/.*PRE //' | \
    sed 's/\///' | \
    sort -r | \
    head -20 | \
    nl -w3 -s') '

echo
echo "🔢 Enter the number of the run to check (1-20), or 'q' to quit:"
read -r selection

if [[ "$selection" == "q" || "$selection" == "Q" ]]; then
    echo "👋 Goodbye!"
    exit 0
fi

# Validate input is a number
if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt 20 ]; then
    echo "❌ Invalid selection. Please enter a number between 1 and 20."
    exit 1
fi

# Get the selected run ID
RUN_ID=$(aws s3 ls s3://${BUCKET}/training-runs/ --region ${REGION} | \
    grep "PRE " | \
    sed 's/.*PRE //' | \
    sed 's/\///' | \
    sort -r | \
    head -20 | \
    sed -n "${selection}p")

if [[ -z "$RUN_ID" ]]; then
    echo "❌ Could not find run ID for selection $selection"
    exit 1
fi

echo
echo "🎯 Selected run: $RUN_ID"
echo "🔍 Checking training status..."
echo

# Run the monitoring script
python3 scripts/monitoring/monitor.py \
    --run-id "$RUN_ID" \
    --bucket "$BUCKET" \
    --region "$REGION" \
    --once

echo
echo "✅ Monitoring check complete!"