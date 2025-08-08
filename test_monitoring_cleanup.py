#!/usr/bin/env python3
"""
Test script to verify monitoring system cleanup works
"""

import sys
import boto3
from datetime import datetime
from pathlib import Path

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts" / "mistral"))

def test_monitoring_cleanup():
    """Test the monitoring cleanup function"""
    
    # Import the fixed training script
    try:
        from train_mistral_simple_validated import cleanup_monitoring_state
        import train_mistral_simple_validated as train_module
    except ImportError as e:
        print(f"❌ Failed to import training script: {e}")
        return False
    
    # Set up test run ID
    test_run_id = f"test-cleanup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    bucket = "asoba-llm-cache"
    
    print(f"🧪 Testing monitoring cleanup with run ID: {test_run_id}")
    
    # Set globals for the function
    train_module.current_run_id = test_run_id
    train_module.s3_bucket = bucket
    
    # Create some fake monitoring state to clean up
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    test_keys = [
        f"training-runs/{test_run_id}/_error",
        f"training-runs/{test_run_id}/_complete",
        f"training-runs/{test_run_id}/progress.json"
    ]
    
    # Create test state
    for key in test_keys:
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=f"test data for {key}"
            )
            print(f"📝 Created test state: {key}")
        except Exception as e:
            print(f"⚠️  Failed to create test state {key}: {e}")
    
    # Verify test state exists
    existing_keys = []
    for key in test_keys:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            existing_keys.append(key)
            print(f"✅ Confirmed test state exists: {key}")
        except:
            print(f"⚠️  Test state doesn't exist: {key}")
    
    # Run cleanup function
    print(f"🧹 Running cleanup_monitoring_state()...")
    try:
        cleanup_monitoring_state()
        print("✅ Cleanup function completed without error")
    except Exception as e:
        print(f"❌ Cleanup function failed: {e}")
        return False
    
    # Verify cleanup worked
    cleaned_count = 0
    for key in test_keys:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            print(f"❌ State still exists after cleanup: {key}")
        except:
            cleaned_count += 1
            print(f"✅ Successfully cleaned: {key}")
    
    success = cleaned_count == len(test_keys)
    
    if success:
        print(f"🎉 Monitoring cleanup test PASSED - cleaned {cleaned_count}/{len(test_keys)} keys")
    else:
        print(f"❌ Monitoring cleanup test FAILED - only cleaned {cleaned_count}/{len(test_keys)} keys")
    
    return success

if __name__ == "__main__":
    success = test_monitoring_cleanup()
    sys.exit(0 if success else 1)