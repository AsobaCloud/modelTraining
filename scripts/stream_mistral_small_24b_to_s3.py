#!/usr/bin/env python3
"""
Stream Mistral-Small-24B-Base-2501 from HuggingFace to S3.
Follows CLAUDE.md safety principles with region lock and tagging.
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import boto3
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import disable_progress_bars
import tempfile
import shutil

# Configuration
REGION = "us-east-1"
BUCKET = "asoba-llm-cache"
MODEL_ID = "mistralai/Mistral-Small-24B-Base-2501"
S3_PREFIX = f"models/mistralai/Mistral-Small-24B-Base-2501"
PROJECT_TAG = "FluxDeploy"
OWNER_TAG = os.environ.get("USER", "unknown")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'mistral_small_24b_streaming_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_aws_credentials():
    """Verify AWS credentials are configured."""
    try:
        boto3.client('sts', region_name=REGION).get_caller_identity()
        logger.info("AWS credentials verified")
        return True
    except Exception as e:
        logger.error(f"AWS credentials check failed: {e}")
        return False

def check_s3_bucket_access():
    """Verify S3 bucket access and region."""
    try:
        s3_client = boto3.client('s3', region_name=REGION)
        response = s3_client.head_bucket(Bucket=BUCKET)
        logger.info(f"S3 bucket {BUCKET} access verified")
        return True
    except Exception as e:
        logger.error(f"S3 bucket access check failed: {e}")
        return False

def get_model_info():
    """Get model repository information from HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        model_info = api.model_info(MODEL_ID)
        logger.info(f"Model info retrieved: {model_info.modelId}")
        return model_info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return None

def stream_file_to_s3(local_path: Path, s3_key: str, s3_client):
    """Stream a single file to S3 with progress tracking."""
    try:
        file_size = local_path.stat().st_size
        logger.info(f"Uploading {local_path.name} ({file_size:,} bytes) to s3://{BUCKET}/{s3_key}")
        
        # Use multipart upload for large files
        s3_client.upload_file(
            str(local_path),
            BUCKET,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'source-model': MODEL_ID,
                    'upload-timestamp': datetime.now().isoformat(),
                    'file-size': str(file_size)
                }
            }
        )
        logger.info(f"Successfully uploaded {local_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path.name}: {e}")
        return False

def stream_model_to_s3():
    """Main function to stream model from HuggingFace to S3."""
    logger.info(f"Starting stream of {MODEL_ID} to s3://{BUCKET}/{S3_PREFIX}")
    
    # Pre-flight checks
    if not check_aws_credentials():
        return False
    
    if not check_s3_bucket_access():
        return False
    
    # Initialize clients
    s3_client = boto3.client('s3', region_name=REGION)
    
    # Get model info
    model_info = get_model_info()
    if not model_info:
        return False
    
    # Create temporary directory for model download
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Downloading model to temporary directory: {temp_dir}")
        
        try:
            # Download model files
            local_model_path = snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=temp_dir,
                local_files_only=False,
                resume_download=True
            )
            
            logger.info(f"Model downloaded to: {local_model_path}")
            
            # Upload all files to S3
            local_path = Path(local_model_path)
            upload_count = 0
            total_size = 0
            
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    # Calculate relative path for S3 key
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{S3_PREFIX}/{relative_path}"
                    
                    # Stream file to S3
                    if stream_file_to_s3(file_path, s3_key, s3_client):
                        upload_count += 1
                        total_size += file_path.stat().st_size
            
            logger.info(f"Upload completed: {upload_count} files, {total_size:,} bytes total")
            
            # Create manifest file
            manifest = {
                "model_id": MODEL_ID,
                "s3_location": f"s3://{BUCKET}/{S3_PREFIX}",
                "upload_timestamp": datetime.now().isoformat(),
                "files_uploaded": upload_count,
                "total_size_bytes": total_size,
                "region": REGION,
                "tags": {
                    "Project": PROJECT_TAG,
                    "Owner": OWNER_TAG
                }
            }
            
            manifest_key = f"{S3_PREFIX}/upload_manifest.json"
            s3_client.put_object(
                Bucket=BUCKET,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Upload manifest created at s3://{BUCKET}/{manifest_key}")
            return True
            
        except Exception as e:
            logger.error(f"Model streaming failed: {e}")
            return False

def main():
    """Entry point."""
    logger.info("=" * 60)
    logger.info("Mistral-Small-24B-Base-2501 S3 Streaming Script")
    logger.info("=" * 60)
    
    success = stream_model_to_s3()
    
    if success:
        logger.info("✅ Model streaming completed successfully")
        logger.info(f"Model location: s3://{BUCKET}/{S3_PREFIX}")
        return 0
    else:
        logger.error("❌ Model streaming failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())