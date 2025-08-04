#!/usr/bin/env python3
"""
S3 Model Uploader - Handles trained model uploads with proper naming conventions
Integrates with the monitoring system to track upload progress
"""

import os
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path


def get_model_name_from_config(output_dir):
    """Extract model name from training configuration"""
    # Map output directory patterns to standardized names
    name_mapping = {
        'qwen3_14b_iac_verbosity_sft': 'qwen3-14b-iac-verbosity-qlora',
        'qwen3_14b_verbosity_pc_lora': 'qwen3-14b-verbosity-qlora',
        'qwen3_14b_verbosity_sft': 'qwen3-14b-verbosity-qlora',
        'qwen3_14b_iac_sft': 'qwen3-14b-iac-qlora',
    }
    
    # Check for exact match first
    for pattern, name in name_mapping.items():
        if pattern in output_dir:
            return name
    
    # Default naming convention: replace underscores with hyphens, add -qlora
    base_name = output_dir.replace('_', '-').lower()
    if not base_name.endswith('-qlora'):
        base_name += '-qlora'
    
    return base_name


def upload_model_to_s3(output_dir, run_id=None, s3_bucket='asoba-llm-cache', 
                      region='us-east-1', monitoring_dir=None):
    """Upload trained model to S3 with proper naming conventions"""
    
    if not os.path.exists(output_dir):
        print(f"ERROR: Output directory {output_dir} does not exist")
        return False
    
    # Get standardized model name
    model_name = get_model_name_from_config(output_dir)
    
    # S3 destination following naming conventions
    s3_destination = f"s3://{s3_bucket}/trained-models/{model_name}/"
    
    print(f"📦 Uploading model to S3...")
    print(f"   Source: {output_dir}")
    print(f"   Destination: {s3_destination}")
    
    # Update monitoring status if available
    if monitoring_dir and run_id:
        status = {
            "phase": "uploading_model",
            "model_name": model_name,
            "s3_destination": s3_destination,
            "started_at": datetime.utcnow().isoformat() + "Z"
        }
        
        status_file = Path(monitoring_dir) / "upload_status.json"
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    # Perform S3 sync
    cmd = f"aws s3 sync {output_dir} {s3_destination} --region {region}"
    
    print("🚀 Starting upload...")
    start_time = datetime.utcnow()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Upload completed in {duration:.1f} seconds")
        print(f"📍 Model available at: {s3_destination}")
        
        # Update monitoring with completion
        if monitoring_dir and run_id:
            status["completed_at"] = end_time.isoformat() + "Z"
            status["duration_seconds"] = duration
            status["success"] = True
            
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            # Also update main progress
            progress_file = Path(monitoring_dir) / "progress.json"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                
                progress["model_uploaded"] = True
                progress["model_s3_path"] = s3_destination
                progress["last_updated"] = datetime.utcnow().isoformat() + "Z"
                
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
        
        # Create a model manifest for tracking
        manifest = {
            "model_name": model_name,
            "s3_path": s3_destination,
            "uploaded_at": end_time.isoformat() + "Z",
            "run_id": run_id,
            "output_dir": output_dir,
            "files": []
        }
        
        # List uploaded files
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                file_size = os.path.getsize(os.path.join(root, file))
                manifest["files"].append({
                    "path": rel_path,
                    "size_bytes": file_size
                })
        
        # Save manifest locally
        manifest_file = Path(output_dir) / "s3_upload_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        
        # Update monitoring with failure
        if monitoring_dir and run_id:
            status["completed_at"] = datetime.utcnow().isoformat() + "Z"
            status["success"] = False
            status["error"] = str(e)
            
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload trained model to S3")
    parser.add_argument("output_dir", help="Directory containing the trained model")
    parser.add_argument("--run-id", help="Training run ID for monitoring")
    parser.add_argument("--monitoring-dir", help="Monitoring directory path")
    parser.add_argument("--bucket", default="asoba-llm-cache", help="S3 bucket name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    
    args = parser.parse_args()
    
    success = upload_model_to_s3(
        args.output_dir,
        run_id=args.run_id,
        s3_bucket=args.bucket,
        region=args.region,
        monitoring_dir=args.monitoring_dir
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()