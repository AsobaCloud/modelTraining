#!/usr/bin/env python3
"""
Quick test of Slack alert functionality
"""

import sys
import os
sys.path.append('scripts/monitoring')

from core import AlertManager

def test_slack_alert():
    webhook = "https://hooks.slack.com/services/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"
    
    alert_manager = AlertManager(slack_webhook=webhook)
    
    # Test alert
    success = alert_manager.send(
        run_id="test-run-123",
        new_state="FAILED", 
        reason="Testing Slack integration for Mistral training monitoring"
    )
    
    if success:
        print("✅ Slack alert sent successfully!")
        print("Check your Slack channel for the test message.")
    else:
        print("❌ Failed to send Slack alert")
        
    return success

if __name__ == "__main__":
    test_slack_alert()