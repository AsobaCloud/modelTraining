#!/usr/bin/env python3
"""
Test script for heartbeat monitoring system
"""

import time
import sys
from pathlib import Path

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "shared"))
from heartbeat_manager import HeartbeatManager, ProgressTracker

def test_heartbeat_system():
    """Test the heartbeat monitoring system"""
    
    # Test configuration
    s3_bucket = "asoba-llm-cache"
    run_id = "test-heartbeat-20250805-143000"
    s3_key = f"training-runs/{run_id}/metadata.json"
    
    print(f"Testing heartbeat system for run: {run_id}")
    print(f"S3 location: s3://{s3_bucket}/{s3_key}")
    
    # Initialize heartbeat manager
    heartbeat = HeartbeatManager(s3_bucket, s3_key, interval_seconds=30)
    
    try:
        # Start heartbeat
        heartbeat.start()
        print("✅ Heartbeat started")
        
        # Simulate data preparation phases
        phases = [
            ("data_prep", "downloading_policy", "corpus_federal"),
            ("data_prep", "downloading_policy", "econ_theory"),
            ("data_prep", "downloading_policy", "financial_metrics"),
            ("data_prep", "extracting_archives", "operatives"),
            ("data_prep", "processing", "combining_datasets"),
            ("data_prep", "uploading", "datasets_to_s3")
        ]
        
        for i, (phase, sub_phase, operation) in enumerate(phases):
            print(f"Phase {i+1}/{len(phases)}: {phase}/{sub_phase} - {operation}")
            
            # Update phase
            heartbeat.update_phase(phase, sub_phase, operation)
            
            # Update progress
            heartbeat.update_progress(
                current_phase=f"{i+1}/{len(phases)}",
                completion_percentage=round((i+1) / len(phases) * 100, 1)
            )
            
            # Simulate work
            time.sleep(10)
        
        # Mark as completed
        heartbeat.set_status("completed")
        heartbeat.update_phase("data_prep", "completed", "success")
        
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        heartbeat.set_status("failed")
        heartbeat.update_phase("data_prep", "error", str(e))
        
    finally:
        # Stop heartbeat
        heartbeat.stop()
        print("✅ Heartbeat stopped")

def test_progress_tracker():
    """Test the progress tracker utility"""
    
    s3_bucket = "asoba-llm-cache"
    run_id = "test-progress-20250805-143100"
    s3_key = f"training-runs/{run_id}/metadata.json"
    
    print(f"Testing progress tracker for run: {run_id}")
    
    heartbeat = HeartbeatManager(s3_bucket, s3_key, interval_seconds=15)
    
    try:
        heartbeat.start()
        heartbeat.update_phase("data_prep", "downloading_policy", "testing")
        
        # Test progress tracker
        total_files = 100
        with ProgressTracker(heartbeat, "test_download", total_files) as tracker:
            for i in range(0, total_files + 1, 10):
                tracker.update(completed=i)
                print(f"Progress: {i}/{total_files} files")
                time.sleep(2)
        
        heartbeat.set_status("completed")
        print("✅ Progress tracker test completed")
        
    except Exception as e:
        print(f"❌ Progress tracker test failed: {e}")
        heartbeat.set_status("failed")
        
    finally:
        heartbeat.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test heartbeat monitoring system")
    parser.add_argument("--test", choices=["heartbeat", "progress", "both"], default="both",
                       help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test in ["heartbeat", "both"]:
        print("=== Testing Heartbeat System ===")
        test_heartbeat_system()
        print()
    
    if args.test in ["progress", "both"]:
        print("=== Testing Progress Tracker ===")
        test_progress_tracker()
        print()
    
    print("All tests completed!")