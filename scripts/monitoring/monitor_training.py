#!/usr/bin/env python3
"""
Training Progress Monitor - User Interface
Simple script to monitor training progress from S3 without SSH access.
"""

import sys
import time
import json
import argparse
from datetime import datetime
from training_monitor import TrainingViewer


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress from S3")
    parser.add_argument("--bucket", default="asoba-llm-cache", help="S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--run-id", help="Specific run ID to monitor")
    parser.add_argument("--list", action="store_true", help="List all training runs")
    parser.add_argument("--watch", action="store_true", help="Watch progress in real-time")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    viewer = TrainingViewer(args.bucket, args.region)
    
    if args.list:
        list_all_runs(viewer)
    elif args.run_id:
        if args.watch:
            watch_run_progress(viewer, args.run_id, args.refresh)
        else:
            show_single_run(viewer, args.run_id)
    else:
        interactive_mode(viewer, args.refresh)


def list_all_runs(viewer):
    """List all available training runs"""
    print("🔍 Fetching training runs from S3...")
    runs = viewer.list_runs()
    
    if not runs:
        print("❌ No training runs found")
        return
    
    print(f"\n📊 Found {len(runs)} training run(s):")
    print("=" * 80)
    
    for i, run_id in enumerate(runs, 1):
        # Get metadata if available
        metadata = get_run_metadata(viewer, run_id)
        progress = viewer.get_run_progress(run_id)
        
        status = "UNKNOWN"
        if metadata:
            status = metadata.get("status", "RUNNING")
        elif progress:
            status = "RUNNING"
        
        # Check if completed
        if check_run_completed(viewer, run_id):
            status = "COMPLETED"
        
        print(f"[{i}] {run_id}")
        print(f"    Status: {status}")
        if metadata:
            start_time = metadata.get("start_time", "Unknown")
            if start_time != "Unknown":
                try:
                    dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    start_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                except:
                    pass
            print(f"    Started: {start_time}")
            print(f"    Instance: {metadata.get('instance_id', 'Unknown')}")
        
        if progress and status == "RUNNING":
            print(f"    Progress: {viewer.format_progress(progress)}")
        
        print()


def show_single_run(viewer, run_id):
    """Show detailed progress for a single run"""
    print(f"📊 Training Run: {run_id}")
    print("=" * 80)
    
    # Get metadata
    metadata = get_run_metadata(viewer, run_id)
    if metadata:
        print(f"Model: {metadata.get('base_model', 'Unknown')}")
        print(f"Started: {metadata.get('start_time', 'Unknown')}")
        print(f"Instance: {metadata.get('instance_id', 'Unknown')}")
        print()
    
    # Get current progress
    progress = viewer.get_run_progress(run_id)
    if progress:
        print("Current Progress:")
        print(viewer.format_progress(progress))
        print(f"Last Updated: {progress.get('last_updated', 'Unknown')}")
        
        if progress.get('estimated_eta_hours'):
            eta = progress['estimated_eta_hours']
            if eta < 1:
                eta_str = f"{int(eta * 60)} minutes"
            else:
                eta_str = f"{eta:.1f} hours"
            print(f"Estimated Time Remaining: {eta_str}")
        
        # Check if model was uploaded
        if progress.get('model_uploaded'):
            print(f"\n✅ Model uploaded to: {progress.get('model_s3_path', 'Unknown')}")
    else:
        print("❌ No progress data available")
    
    # Check completion status
    if check_run_completed(viewer, run_id):
        print("\n✅ Training completed!")
        completion_info = get_completion_info(viewer, run_id)
        if completion_info:
            print(f"Final metrics: {completion_info}")


def watch_run_progress(viewer, run_id, refresh_interval):
    """Watch run progress in real-time"""
    print(f"👀 Watching training run: {run_id}")
    print(f"🔄 Refreshing every {refresh_interval} seconds (Ctrl+C to stop)")
    print("=" * 80)
    
    try:
        while True:
            # Clear screen (simple version)
            print("\033[2J\033[H", end="")
            
            print(f"👀 Watching: {run_id} - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)
            
            progress = viewer.get_run_progress(run_id)
            if progress:
                print(viewer.format_progress(progress))
                print(f"Loss: {progress.get('loss', 'N/A')}")
                print(f"Learning Rate: {progress.get('learning_rate', 'N/A')}")
                print(f"Last Updated: {progress.get('last_updated', 'Unknown')}")
            else:
                print("❌ No progress data available")
            
            # Check if completed
            if check_run_completed(viewer, run_id):
                print("\n✅ Training completed!")
                break
            
            print(f"\n🔄 Next refresh in {refresh_interval}s... (Ctrl+C to stop)")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")


def interactive_mode(viewer, refresh_interval):
    """Interactive mode to select and monitor runs"""
    while True:
        print("\n🚀 Training Progress Monitor")
        print("=" * 80)
        
        runs = viewer.list_runs()
        if not runs:
            print("❌ No training runs found")
            return
        
        print("Available training runs:")
        for i, run_id in enumerate(runs, 1):
            status = "RUNNING"
            if check_run_completed(viewer, run_id):
                status = "COMPLETED"
            print(f"[{i}] {run_id} ({status})")
        
        print("\nOptions:")
        print("- Enter run number to view details")
        print("- Enter run number + 'w' to watch (e.g., '1w')")
        print("- 'r' to refresh list")
        print("- 'q' to quit")
        
        choice = input("\nChoice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'r':
            continue
        elif choice.endswith('w') and choice[:-1].isdigit():
            # Watch mode
            run_num = int(choice[:-1])
            if 1 <= run_num <= len(runs):
                watch_run_progress(viewer, runs[run_num - 1], refresh_interval)
        elif choice.isdigit():
            # Show details
            run_num = int(choice)
            if 1 <= run_num <= len(runs):
                show_single_run(viewer, runs[run_num - 1])
                input("\nPress Enter to continue...")
        else:
            print("❌ Invalid choice")


def get_run_metadata(viewer, run_id):
    """Get metadata for a run"""
    import subprocess
    s3_path = f"s3://{viewer.s3_bucket}/logs/training-runs/{run_id}/metadata.json"
    cmd = f"aws s3 cp {s3_path} - --region {viewer.region}"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    
    return None


def check_run_completed(viewer, run_id):
    """Check if a run is completed"""
    import subprocess
    s3_path = f"s3://{viewer.s3_bucket}/logs/training-runs/{run_id}/_complete"
    cmd = f"aws s3 ls {s3_path} --region {viewer.region}"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def get_completion_info(viewer, run_id):
    """Get completion information"""
    import subprocess
    s3_path = f"s3://{viewer.s3_bucket}/logs/training-runs/{run_id}/_complete"
    cmd = f"aws s3 cp {s3_path} - --region {viewer.region}"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return None


if __name__ == "__main__":
    main()