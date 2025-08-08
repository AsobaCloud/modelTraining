#!/usr/bin/env python3
"""
Direct Google Drive Downloader for PVF-Dataset
Handles the virus scan confirmation and streams directly to S3
"""

import requests
import boto3
import re
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3", region_name="us-east-1")

def download_pvf_dataset_from_gdrive():
    """Download the PVF-10.zip dataset directly from Google Drive"""
    
    # From the virus scan page we captured
    file_id = "1SQq0hETXi8I3Kdq9tDAEVyZgIsRCbOah"
    filename = "PVF-10.zip"
    expected_size_mb = 157
    
    logger.info(f"Downloading {filename} ({expected_size_mb}MB) from Google Drive")
    
    # Use the direct download URL with confirmation
    download_url = "https://drive.usercontent.google.com/download"
    params = {
        'id': file_id,
        'export': 'download',
        'confirm': 't'
    }
    
    session = requests.Session()
    
    try:
        # Stream the download
        with session.get(download_url, params=params, stream=True) as response:
            response.raise_for_status()
            
            # Check if we're getting actual file content
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                logger.error("Still getting HTML response, download may have failed")
                # Read first part to check
                chunk = next(response.iter_content(1024))
                if b'<html' in chunk:
                    logger.error("Got HTML instead of ZIP file")
                    return False
            
            # Get file size
            content_length = response.headers.get('content-length')
            if content_length:
                file_size_mb = int(content_length) / (1024 * 1024)
                logger.info(f"Actual file size: {file_size_mb:.1f} MB")
            
            # Stream directly to S3
            s3_key = f"pv-ops-and-maintenance/pvf-dataset/dataset/{filename}"
            
            logger.info("Starting S3 upload...")
            
            # Use multipart upload for large file
            multipart_upload = s3_client.create_multipart_upload(
                Bucket="policy-database",
                Key=s3_key,
                ContentType="application/zip"
            )
            upload_id = multipart_upload['UploadId']
            
            parts = []
            part_number = 1
            chunk_size = 100 * 1024 * 1024  # 100MB chunks
            
            try:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        logger.info(f"Uploading part {part_number}...")
                        
                        part_response = s3_client.upload_part(
                            Bucket="policy-database",
                            Key=s3_key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=chunk
                        )
                        
                        parts.append({
                            'ETag': part_response['ETag'],
                            'PartNumber': part_number
                        })
                        
                        part_number += 1
                
                # Complete multipart upload
                s3_client.complete_multipart_upload(
                    Bucket="policy-database",
                    Key=s3_key,
                    UploadId=upload_id,
                    MultipartUpload={'Parts': parts}
                )
                
                logger.info(f"✓ Successfully uploaded {filename} to S3")
                return True
                
            except Exception as e:
                # Abort multipart upload on error
                s3_client.abort_multipart_upload(
                    Bucket="policy-database",
                    Key=s3_key,
                    UploadId=upload_id
                )
                raise e
                
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return False

def check_download_exists():
    """Check if PVF-10.zip already exists in S3"""
    try:
        response = s3_client.head_object(
            Bucket="policy-database", 
            Key="pv-ops-and-maintenance/pvf-dataset/dataset/PVF-10.zip"
        )
        size_mb = response['ContentLength'] / (1024 * 1024)
        logger.info(f"PVF-10.zip already exists in S3 ({size_mb:.1f} MB)")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise

if __name__ == "__main__":
    if not check_download_exists():
        success = download_pvf_dataset_from_gdrive()
        if success:
            print("✅ PVF Dataset download complete!")
        else:
            print("❌ PVF Dataset download failed")
    else:
        print("✅ PVF Dataset already exists in S3")