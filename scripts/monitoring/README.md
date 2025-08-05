# Production Training Monitor

Real-time monitoring system for LLM training pipelines with failure detection and actionable alerts.

## Overview

This monitoring system provides production-grade observability for long-running training jobs. It detects silent failures, stuck processes, and training errors that traditional monitoring misses.

### Key Features

- **Silent Failure Detection**: Dual heartbeat monitoring (metadata + progress)
- **Actionable Alerts**: Slack notifications with specific remediation steps
- **Error Sentinels**: Automatic error capture via S3 `_error` files
- **State Machine**: Clear RUNNING → SUCCEEDED/FAILED/TIMED_OUT transitions
- **Rich Context**: Instance ID, phase, duration, last logs in alerts

## Quick Start

### Monitor an existing run:
```bash
# Basic monitoring (no alerts)
python3 scripts/monitoring/monitor.py --run-id mistral-20250804-171621

# With Slack alerts
./scripts/monitoring/production_monitor.sh mistral-20250804-171621

# Custom interval (default 60s)
./scripts/monitoring/production_monitor.sh mistral-20250804-171621 --interval 30
```

### Single status check:
```bash
python3 scripts/monitoring/monitor.py --run-id mistral-20250804-171621 --once
```

## Alert Examples

### TIMED_OUT Alert
```
🔥 Mistral Training TIMED_OUT
Run ID: mistral-20250804-171621
Status: TIMED_OUT
Reason: Both heartbeats stale (>10m)
🔧 Action Required: SSH to instance and check: training.log, nvidia-smi, disk space
Instance: i-01fa5b57d64c6196a
Phase: training
Duration: 127 minutes
Last Log: [2025-08-04 19:23:45] Step 1500/5000, loss: 2.134, lr: 0.0001
📊 S3 Logs: View in S3 Console
```

### FAILED Alert (CUDA OOM)
```
🔥 Mistral Training FAILED
Run ID: mistral-20250804-182347
Status: FAILED
Reason: Error: CUDA out of memory
🔧 Action Required: Reduce batch size or sequence length
Instance: i-01fa5b57d64c6196a
Phase: training
Duration: 45 minutes
Last Log: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
📊 S3 Logs: View in S3 Console
```

## How It Works

### Heartbeat Files

The monitor tracks two heartbeat files in S3:

1. **metadata.json** - Overall run status
   ```json
   {
     "run_id": "mistral-20250804-171621",
     "status": "running",
     "phase": "training",
     "last_update": "2025-08-04T19:23:45Z",
     "instance_id": "i-01fa5b57d64c6196a"
   }
   ```

2. **progress.json** - Training metrics
   ```json
   {
     "step": 1500,
     "total_steps": 5000,
     "loss": 2.134,
     "learning_rate": 0.0001,
     "last_updated": "2025-08-04T19:23:45Z"
   }
   ```

### Failure Detection

| Condition | Detection | Alert |
|-----------|-----------|-------|
| Both heartbeats stale (>10min) | TIMED_OUT | SSH and check logs |
| `_error` sentinel exists | FAILED | Check error details |
| `_complete` marker exists | SUCCEEDED | Training complete |
| No heartbeat files | TIMED_OUT | Never started |
| One heartbeat stale | WARNING | Degraded state |

### Error Sentinels

Training scripts automatically write error details to S3:

- **Location**: `s3://bucket/training-runs/{run_id}/_error`
- **Content**: Error message and stack trace
- **Trigger**: Any unhandled exception or explicit error

## Configuration

### Environment Variables
```bash
# Optional - defaults shown
export MONITORING_BUCKET="asoba-llm-cache"
export MONITORING_REGION="us-east-1"
export STALE_MINUTES=10
```

### Slack Webhook Setup

1. Create an Incoming Webhook in your Slack workspace
2. Set up your webhook URL using one of these methods:

   **Option 1: Environment File (Recommended)**
   ```bash
   cp .env.example .env
   # Edit .env and add your webhook:
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   ```

   **Option 2: Environment Variable**
   ```bash
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
   ./scripts/monitoring/production_monitor.sh mistral-20250804-171621
   ```

   **Option 3: Pass directly to monitor.py**
   ```bash
   python3 scripts/monitoring/monitor.py \
     --run-id mistral-20250804-171621 \
     --slack-webhook "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
   ```

### SNS Integration (Alternative to Slack)
```bash
python3 scripts/monitoring/monitor.py \
  --run-id mistral-20250804-171621 \
  --sns-topic arn:aws:sns:us-east-1:123456789012:training-alerts
```

## Integration with Training Scripts

### Shell Scripts
```bash
# Error handling with sentinel
error() {
    local msg="ERROR: $1"
    echo "$msg"
    
    # Write error sentinel
    echo "$1" | aws s3 cp - s3://$BUCKET/$PREFIX/$RUN_ID/_error
    exit 1
}
```

### Python Scripts
```python
# Global setup
current_run_id = "mistral-20250804-171621"
s3_bucket = "asoba-llm-cache"

def handle_error(error_msg: str):
    """Handle errors with monitoring integration"""
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=f"training-runs/{current_run_id}/_error",
        Body=error_msg
    )
    sys.exit(1)

# On successful completion
atexit.register(write_completion_marker)
```

## Testing

Run the test suite:
```bash
cd scripts/monitoring
python3 test_monitor.py -v
```

Test Slack alerts:
```python
from core import AlertManager

alert = AlertManager(slack_webhook="your-webhook-url")
alert.send("test-run", "FAILED", "Test alert", 
          {"instance_id": "i-test", "phase": "testing"})
```

## Troubleshooting

### No alerts received
- Check webhook URL is correct
- Verify S3 permissions for reading heartbeat files
- Check logs: alerts only trigger on state transitions

### False timeouts
- Increase `--stale-minutes` if setup takes longer
- Ensure training script updates heartbeats regularly
- Check instance has S3 write permissions

### Missing context in alerts
- Verify metadata.json contains required fields
- Check S3 paths match expected structure
- Ensure training logs are being uploaded

## S3 Structure for Monitoring

The monitor tracks files in this structure:

```
s3://asoba-llm-cache/training-runs/{run-id}/
├── metadata.json         # Run configuration and status
├── progress.json         # Training metrics (optional)
├── logs/
│   └── training_log_latest.txt
├── _error               # Error sentinel (if failed)
└── _complete            # Completion marker (if succeeded)
```

### Quick S3 Commands

```bash
# Find your run ID
aws s3 ls s3://asoba-llm-cache/training-runs/ | grep mistral | tail -5

# Check run status
aws s3 ls s3://asoba-llm-cache/training-runs/mistral-20250804-171621/

# View current metadata
aws s3 cp s3://asoba-llm-cache/training-runs/mistral-20250804-171621/metadata.json - | jq .

# Check for errors
aws s3 cp s3://asoba-llm-cache/training-runs/mistral-20250804-171621/_error -

# View latest logs
aws s3 cp s3://asoba-llm-cache/training-runs/mistral-20250804-171621/logs/training_log_latest.txt - | tail -20
```

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────┐
│ Training Script │────▶│ S3 Heartbeats│◀────│  Monitor   │
└─────────────────┘     └──────────────┘     └─────┬──────┘
                               │                     │
                               ▼                     ▼
                        ┌─────────────┐       ┌────────────┐
                        │Error/Success│       │Slack/SNS   │
                        │  Sentinels  │       │  Alerts    │
                        └─────────────┘       └────────────┘
```

## Files

- `core.py` - Core monitoring engine
- `monitor.py` - CLI interface
- `production_monitor.sh` - Quick launcher with Slack
- `test_monitor.py` - Test suite with moto S3 mocking
- `requirements.txt` - Python dependencies