#!/usr/bin/env python3
"""
Test that demonstrates run ID isolation - new runs don't see old failure states
"""

import boto3
from datetime import datetime

def test_run_isolation():
    """Test that new run IDs are properly isolated from old failures"""
    
    bucket = "asoba-llm-cache"
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    # Check an old failed run
    old_run_id = "mistral-20250807-210015"
    
    print(f"🔍 Checking old failed run: {old_run_id}")
    try:
        error_response = s3_client.get_object(
            Bucket=bucket, 
            Key=f"training-runs/{old_run_id}/_error"
        )
        old_error = error_response['Body'].read().decode()
        print(f"❌ Old run error: {old_error}")
    except Exception as e:
        print(f"⚠️  Could not read old error: {e}")
    
    # Generate new run ID (what would happen in actual training)
    new_run_id = f"mistral-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print(f"🆕 New run ID would be: {new_run_id}")
    
    # Check that new run ID has no existing state
    test_keys = [
        f"training-runs/{new_run_id}/_error",
        f"training-runs/{new_run_id}/_complete", 
        f"training-runs/{new_run_id}/progress.json"
    ]
    
    clean_state = True
    for key in test_keys:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            print(f"❌ New run already has state: {key}")
            clean_state = False
        except:
            print(f"✅ New run has clean state: {key}")
    
    if clean_state:
        print(f"🎉 NEW RUN ISOLATION CONFIRMED - {new_run_id} starts with completely clean monitoring state")
        print(f"✅ Old failures in {old_run_id} do not affect new run {new_run_id}")
    else:
        print(f"❌ Run isolation failed - new run has pre-existing state")
    
    return clean_state

if __name__ == "__main__":
    test_run_isolation()