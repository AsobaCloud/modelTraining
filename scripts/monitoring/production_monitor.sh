#!/bin/bash
# Production monitoring launcher with Slack alerts
# Usage: ./production_monitor.sh <run-id>

set -euo pipefail

SLACK_WEBHOOK="https://hooks.slack.com/services/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"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    --slack-webhook "$SLACK_WEBHOOK" \
    "$@"