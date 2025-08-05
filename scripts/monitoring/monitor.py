#!/usr/bin/env python3
"""
Production Training Monitor CLI
Real-time monitoring with failure detection and alerts
"""

import argparse
import time
import logging
from core import TrainingMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Production training monitor with failure detection")
    parser.add_argument("--run-id", required=True, help="Training run ID to monitor")
    parser.add_argument("--bucket", default="asoba-llm-cache", help="S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--stale-minutes", type=int, default=10, 
                       help="Minutes before heartbeat considered stale")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Check interval in seconds")
    parser.add_argument("--slack-webhook", help="Slack webhook URL for alerts")
    parser.add_argument("--sns-topic", help="SNS topic ARN for alerts")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = TrainingMonitor(args.bucket, args.region, args.stale_minutes)
    
    # Configure alerts if provided
    if args.slack_webhook or args.sns_topic:
        monitor.configure_alerts(args.slack_webhook, args.sns_topic)
        logger.info("Alerts configured")
    
    if args.once:
        # Single check
        state = monitor.check_and_alert(args.run_id)
        print(f"Run {args.run_id}: {state.status}")
        print(f"Detail: {state.detail}")
        if state.last_meta:
            print(f"Last metadata update: {state.last_meta}")
        if state.last_progress:
            print(f"Last progress update: {state.last_progress}")
    else:
        # Continuous monitoring
        logger.info(f"Starting continuous monitoring of {args.run_id}")
        logger.info(f"Check interval: {args.interval}s, stale threshold: {args.stale_minutes}m")
        
        try:
            while True:
                state = monitor.check_and_alert(args.run_id)
                
                status_emoji = {
                    "RUNNING": "🔄",
                    "SUCCEEDED": "✅", 
                    "FAILED": "❌",
                    "TIMED_OUT": "⏰"
                }.get(state.status, "❓")
                
                print(f"\n{status_emoji} [{time.strftime('%H:%M:%S')}] {args.run_id}: {state.status}")
                print(f"   {state.detail}")
                
                if state.last_meta:
                    print(f"   Metadata: {state.last_meta.strftime('%H:%M:%S')}")
                if state.last_progress:
                    print(f"   Progress: {state.last_progress.strftime('%H:%M:%S')}")
                
                # Exit if terminal state reached
                if state.status in ["SUCCEEDED", "FAILED", "TIMED_OUT"]:
                    logger.info(f"Run reached terminal state: {state.status}")
                    break
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")


if __name__ == "__main__":
    main()
