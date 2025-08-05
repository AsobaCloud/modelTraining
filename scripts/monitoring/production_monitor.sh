#!/bin/bash
# Production monitoring launcher with Slack alerts
# Usage: ./production_monitor.sh <run-id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load from environment or .env file
# Check multiple locations for .env
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    export $(cat "$SCRIPT_DIR/../../.env" | grep -v '^#' | xargs)
elif [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for Slack webhook
if [ -z "${SLACK_WEBHOOK_URL:-}" ]; then
    echo "ERROR: SLACK_WEBHOOK_URL not set"
    echo ""
    echo "Please set it in one of these ways:"
    echo "1. Create .env file with: SLACK_WEBHOOK_URL=https://hooks.slack.com/..."
    echo "2. Export environment variable: export SLACK_WEBHOOK_URL=https://hooks.slack.com/..."
    echo ""
    echo "To get a webhook URL:"
    echo "- Go to your Slack workspace settings"
    echo "- Add an 'Incoming Webhook' app"
    echo "- Copy the webhook URL"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <run-id> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 mistral-20250804-171621                    # Single check"
    echo "  $0 mistral-20250804-171621 --watch            # Continuous monitoring"
    echo "  $0 mistral-20250804-171621 --watch --interval 30  # Custom interval"
    exit 1
fi

RUN_ID="$1"
shift

# Default to continuous monitoring if no options provided
if [ $# -eq 0 ]; then
    set -- --watch --interval 60
fi

echo "🚀 Starting production monitoring for: $RUN_ID"
echo "📱 Slack alerts enabled"
echo "⏱️  Options: $*"
echo ""

python3 "$SCRIPT_DIR/monitor.py" \
    --run-id "$RUN_ID" \
    --slack-webhook "$SLACK_WEBHOOK_URL" \
    "$@"