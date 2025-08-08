#!/usr/bin/env python3
"""
PVF-Dataset Direct Streaming Pipeline
Streams photovoltaic fault dataset directly from GitHub and external sources to S3
without local downloading
"""

import os
import sys
import json
import time
import logging
import boto3
import requests
from datetime import datetime
from typing import Dict, List, Optional, Iterator
from botocore.exceptions import ClientError
from urllib.parse import urlparse, parse_qs
import re
from threading import Lock

# Configuration
POLICY_BUCKET = "policy-database"
PVF_DATASET_PATH = "pv-ops-and-maintenance/pvf-dataset"
REPO_URL = "https://github.com/wangbobby1026/PVF-Dataset"
GITHUB_API_BASE = "https://api.github.com/repos/wangbobby1026/PVF-Dataset"

# Streaming configuration
CHUNK_SIZE = 8192  # 8KB chunks for streaming
MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100MB threshold for multipart
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB max file size
PROGRESS_FILE = "pvf_streaming_progress.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3", region_name="us-east-1")
progress_lock = Lock()

class PVFDatasetStreamer:
    """Streams PVF-Dataset directly to S3 without local storage"""
    
    def __init__(self, policy_bucket: str = POLICY_BUCKET):
        self.policy_bucket = policy_bucket
        self.progress = self.load_progress()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def load_progress(self) -> Dict:
        """Load streaming progress from file"""
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return {
            "repository_metadata_processed": False,
            "code_files_processed": [],
            "external_datasets_processed": [],
            "google_drive_files_processed": [],
            "baidu_pan_files_processed": [],
            "total_bytes_streamed": 0,
            "last_updated": None
        }
    
    def save_progress(self):
        """Save current progress"""
        with progress_lock:
            self.progress["last_updated"] = datetime.now().isoformat()
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(self.progress, f, indent=2)
    
    def stream_repository_data(self):
        """Main method to stream all PVF-Dataset data"""
        logger.info("🚀 Starting PVF-Dataset direct streaming pipeline")
        start_time = time.time()
        
        try:
            # Step 1: Stream repository metadata and code
            self._stream_repository_metadata()
            self._stream_repository_files()
            
            # Step 2: Extract and stream external dataset links
            self._stream_external_datasets()
            
        except KeyboardInterrupt:
            logger.info("⚠️ Streaming interrupted by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {str(e)}")
            raise
        finally:
            elapsed = time.time() - start_time
            logger.info(f"🎉 PVF-Dataset streaming complete!")
            logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
            logger.info(f"📊 Total data: {self.progress['total_bytes_streamed'] / 1024 / 1024:.1f} MB")
            logger.info(f"📁 Location: s3://{POLICY_BUCKET}/{PVF_DATASET_PATH}/")
    
    def _stream_repository_metadata(self):
        """Stream repository metadata from GitHub API"""
        if self.progress.get("repository_metadata_processed"):
            logger.info("Repository metadata already processed, skipping...")
            return
            
        logger.info("--- Streaming repository metadata ---")
        
        try:
            # Get repository info
            repo_response = self.session.get(f"{GITHUB_API_BASE}")
            repo_response.raise_for_status()
            repo_data = repo_response.json()
            
            # Get commit history
            commits_response = self.session.get(f"{GITHUB_API_BASE}/commits")
            commits_response.raise_for_status()
            commits_data = commits_response.json()
            
            # Create comprehensive metadata
            metadata = {
                "repository_name": "PVF-Dataset",
                "full_name": repo_data.get("full_name"),
                "description": repo_data.get("description"),
                "url": repo_data.get("html_url"),
                "clone_url": repo_data.get("clone_url"),
                "created_at": repo_data.get("created_at"),
                "updated_at": repo_data.get("updated_at"),
                "size": repo_data.get("size"),
                "language": repo_data.get("language"),
                "topics": repo_data.get("topics", []),
                "default_branch": repo_data.get("default_branch"),
                "commits_count": len(commits_data),
                "recent_commits": commits_data[:10],  # Last 10 commits
                "dataset_info": {
                    "total_images": 5579,
                    "fault_classes": 10,
                    "repairable_faults": 5,
                    "irreparable_faults": 5,
                    "data_source": "8 power plants",
                    "image_type": "UAV thermal infrared",
                    "resolution": "high-resolution"
                },
                "external_data_sources": {
                    "baidu_pan": "Available with extract codes",
                    "google_drive": "Direct file links"
                },
                "collection_date": datetime.now().isoformat()
            }
            
            # Stream metadata to S3
            metadata_key = f"{PVF_DATASET_PATH}/repository_metadata/repo_info.json"
            self._stream_content_to_s3(
                json.dumps(metadata, indent=2).encode('utf-8'),
                metadata_key,
                'application/json'
            )
            
            logger.info("✓ Repository metadata processed")
            self.progress["repository_metadata_processed"] = True
            self.save_progress()
            
        except Exception as e:
            logger.error(f"Failed to process repository metadata: {str(e)}")
    
    def _stream_repository_files(self):
        """Stream code files and documentation from repository"""
        logger.info("--- Streaming repository files ---")
        
        try:
            # Get repository tree from master branch (where the code is)
            tree_response = self.session.get(f"{GITHUB_API_BASE}/git/trees/master?recursive=1")
            tree_response.raise_for_status()
            tree_data = tree_response.json()
            
            for item in tree_data.get("tree", []):
                if item["type"] == "blob":  # Files only, not directories
                    file_path = item["path"]
                    
                    if file_path in self.progress["code_files_processed"]:
                        continue
                    
                    try:
                        self._stream_repository_file(file_path, item["sha"])
                        self.progress["code_files_processed"].append(file_path)
                        
                        if len(self.progress["code_files_processed"]) % 5 == 0:
                            self.save_progress()
                            
                    except Exception as e:
                        logger.error(f"Failed to stream file {file_path}: {str(e)}")
                        continue
            
            logger.info(f"✓ Processed {len(self.progress['code_files_processed'])} repository files")
            
        except Exception as e:
            logger.error(f"Failed to stream repository files: {str(e)}")
    
    def _stream_repository_file(self, file_path: str, sha: str):
        """Stream a single repository file directly to S3"""
        try:
            # Get file content via API
            file_response = self.session.get(f"{GITHUB_API_BASE}/git/blobs/{sha}")
            file_response.raise_for_status()
            blob_data = file_response.json()
            
            # Decode content (GitHub API returns base64)
            import base64
            if blob_data.get("encoding") == "base64":
                content = base64.b64decode(blob_data["content"])
            else:
                content = blob_data["content"].encode('utf-8')
            
            # Create file metadata
            file_metadata = {
                "file_path": file_path,
                "sha": sha,
                "size": blob_data.get("size", len(content)),
                "encoding": blob_data.get("encoding"),
                "source": "GitHub Repository",
                "collection_date": datetime.now().isoformat()
            }
            
            # Determine content type
            content_type = self._get_content_type(file_path)
            
            # Stream to S3
            s3_key = f"{PVF_DATASET_PATH}/repository_files/{file_path}"
            self._stream_content_to_s3(content, s3_key, content_type)
            
            # Stream metadata
            metadata_key = f"{PVF_DATASET_PATH}/metadata/files/{file_path}.json"
            self._stream_content_to_s3(
                json.dumps(file_metadata, indent=2).encode('utf-8'),
                metadata_key,
                'application/json'
            )
            
            logger.debug(f"✓ Streamed repository file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to stream repository file {file_path}: {str(e)}")
    
    def _stream_external_datasets(self):
        """Stream external dataset files from Google Drive and Baidu Pan"""
        logger.info("--- Streaming external datasets ---")
        
        # Extract and attempt to download from Google Drive links
        self._download_google_drive_datasets()
        
        external_sources_info = {
            "google_drive": {
                "description": "Full PVF-Dataset with 5,579 high-resolution thermal infrared images",
                "access_method": "Direct download link provided in repository README",
                "fault_classes": [
                    "Repairable: Shadowing, Soiling, Module Broken, Diode Bypass, Hot Spot",
                    "Irreparable: Burn Mark, Snail Trail, Discoloration, Corrosion, Delamination"
                ],
                "image_format": "Thermal infrared images from UAV",
                "source_plants": 8,
                "total_images": 5579,
                "note": "Access requires manual download due to authentication requirements"
            },
            "baidu_pan": {
                "description": "Alternative access to the same PVF-Dataset",
                "access_method": "Baidu Pan sharing links with extract codes",
                "extract_codes": "Multiple codes provided in repository README",
                "note": "Requires Baidu account for access"
            },
            "data_structure": {
                "format": "Organized by fault type directories",
                "image_specifications": "High-resolution thermal infrared",
                "labeling": "Fine-grained 10-class fault classification",
                "applications": ["Fault detection", "Classification", "UAV-based inspection"],
                "research_paper": "Applied Energy 2024 publication"
            }
        }
        
        # Stream external sources documentation
        external_key = f"{PVF_DATASET_PATH}/external_data_sources/data_access_info.json"
        enhanced_info = {
            "external_sources": external_sources_info,
            "streaming_note": "Full dataset requires manual access due to authentication requirements",
            "repository_content": "Code, models, and documentation streamed successfully",
            "collection_date": datetime.now().isoformat()
        }
        
        self._stream_content_to_s3(
            json.dumps(enhanced_info, indent=2).encode('utf-8'),
            external_key,
            'application/json'
        )
        
        logger.info("✓ External dataset sources documented")
        
        # Attempt to extract direct links from README if possible
        try:
            readme_response = self.session.get(f"https://raw.githubusercontent.com/wangbobby1026/PVF-Dataset/main/README.md")
            if readme_response.status_code == 200:
                readme_content = readme_response.text
                
                # Extract any direct download links
                google_drive_links = re.findall(r'https://drive\.google\.com/[^\s\)]+', readme_content)
                baidu_links = re.findall(r'https://pan\.baidu\.com/[^\s\)]+', readme_content)
                
                if google_drive_links or baidu_links:
                    links_info = {
                        "google_drive_links": google_drive_links,
                        "baidu_pan_links": baidu_links,
                        "readme_content": readme_content,
                        "extraction_date": datetime.now().isoformat()
                    }
                    
                    links_key = f"{PVF_DATASET_PATH}/external_data_sources/extracted_links.json"
                    self._stream_content_to_s3(
                        json.dumps(links_info, indent=2).encode('utf-8'),
                        links_key,
                        'application/json'
                    )
                    
                    logger.info(f"✓ Extracted {len(google_drive_links)} Google Drive and {len(baidu_links)} Baidu Pan links")
                
        except Exception as e:
            logger.error(f"Failed to extract external links: {str(e)}")
    
    def _download_google_drive_datasets(self):
        """Download datasets directly from Google Drive"""
        logger.info("--- Downloading Google Drive datasets ---")
        
        try:
            # Get README to extract Google Drive links
            readme_response = self.session.get(f"https://raw.githubusercontent.com/wangbobby1026/PVF-Dataset/main/README.md")
            if readme_response.status_code != 200:
                logger.error("Failed to fetch README for Google Drive links")
                return
                
            readme_content = readme_response.text
            
            # Extract Google Drive file IDs from sharing links
            # Pattern: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
            gdrive_pattern = r'https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)/view'
            file_ids = re.findall(gdrive_pattern, readme_content)
            
            logger.info(f"Found {len(file_ids)} Google Drive file IDs")
            
            for i, file_id in enumerate(file_ids):
                if file_id in self.progress["google_drive_files_processed"]:
                    logger.info(f"Google Drive file {file_id} already processed, skipping...")
                    continue
                    
                try:
                    # Convert sharing link to direct download link
                    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    
                    # Get file info first
                    head_response = self.session.head(download_url, allow_redirects=True)
                    
                    # Determine filename from headers or use generic name
                    filename = f"pvf_dataset_file_{i+1}"
                    if 'Content-Disposition' in head_response.headers:
                        cd_header = head_response.headers['Content-Disposition']
                        filename_match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', cd_header)
                        if filename_match:
                            filename = filename_match.group(1).strip('"\'')
                    
                    # Check file size
                    content_length = head_response.headers.get('content-length')
                    if content_length:
                        file_size_mb = int(content_length) / (1024 * 1024)
                        logger.info(f"Downloading {filename} ({file_size_mb:.1f} MB)")
                        
                        # Skip extremely large files (>2GB)
                        if file_size_mb > 2048:
                            logger.warning(f"File too large ({file_size_mb:.1f} MB), skipping: {filename}")
                            continue
                    else:
                        logger.info(f"Downloading {filename} (size unknown)")
                    
                    # Download and stream to S3
                    s3_key = f"{PVF_DATASET_PATH}/dataset/{filename}"
                    
                    # Handle large file downloads with confirmation token
                    success = self._download_large_gdrive_file(file_id, s3_key, filename)
                    
                    if success:
                        self.progress["google_drive_files_processed"].append(file_id)
                        logger.info(f"✓ Downloaded: {filename}")
                        self.save_progress()
                    else:
                        logger.error(f"Failed to download: {filename}")
                        
                except Exception as e:
                    logger.error(f"Failed to process Google Drive file {file_id}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to download Google Drive datasets: {str(e)}")
    
    def _download_large_gdrive_file(self, file_id: str, s3_key: str, filename: str) -> bool:
        """Download large files from Google Drive with virus scan bypass"""
        try:
            # First attempt - normal download
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            response = self.session.get(download_url, stream=True)
            
            # Check if we need to confirm download (large file warning)
            if 'confirm=' in response.text or 'virus scan warning' in response.text.lower():
                # Extract confirmation token
                confirm_token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        confirm_token = value
                        break
                
                if not confirm_token:
                    # Try to extract from response text
                    confirm_match = re.search(r'confirm=([^&]+)', response.text)
                    if confirm_match:
                        confirm_token = confirm_match.group(1)
                
                if confirm_token:
                    # Use confirmation token for large file
                    confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = self.session.get(confirm_url, stream=True)
                else:
                    logger.warning(f"Could not find confirmation token for {filename}")
            
            response.raise_for_status()
            
            # Determine content type
            content_type = response.headers.get('content-type', 'application/octet-stream')
            if filename.endswith('.zip'):
                content_type = 'application/zip'
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                content_type = 'application/gzip'
            
            # Stream directly to S3
            return self._stream_response_to_s3(response, s3_key, content_type)
            
        except Exception as e:
            logger.error(f"Failed to download large Google Drive file {file_id}: {str(e)}")
            return False
    
    def _stream_response_to_s3(self, response, s3_key: str, content_type: str) -> bool:
        """Stream HTTP response directly to S3"""
        try:
            # Check if file already exists
            try:
                s3_client.head_object(Bucket=self.policy_bucket, Key=s3_key)
                logger.debug(f"Already exists in S3: {s3_key}")
                return True
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
            
            # Get content length for multipart decision
            content_length = response.headers.get('content-length')
            if content_length:
                content_length = int(content_length)
                
                # Use multipart for large files
                if content_length > MULTIPART_THRESHOLD:
                    return self._multipart_stream_to_s3(response, s3_key, content_type)
            
            # Simple upload for smaller files
            file_content = response.content
            s3_client.put_object(
                Bucket=self.policy_bucket,
                Key=s3_key,
                Body=file_content,
                ContentType=content_type
            )
            
            self.progress["total_bytes_streamed"] += len(file_content)
            logger.debug(f"✓ Streamed to S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stream response to S3 {s3_key}: {str(e)}")
            return False
    
    def _get_content_type(self, file_path: str) -> str:
        """Determine content type based on file extension"""
        extension = os.path.splitext(file_path.lower())[1]
        content_types = {
            '.py': 'text/x-python',
            '.md': 'text/markdown',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.yml': 'text/yaml',
            '.yaml': 'text/yaml',
            '.csv': 'text/csv',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.pdf': 'application/pdf'
        }
        return content_types.get(extension, 'application/octet-stream')
    
    def _stream_content_to_s3(self, content: bytes, s3_key: str, content_type: str) -> bool:
        """Stream content directly to S3"""
        try:
            # Check if already exists
            try:
                s3_client.head_object(Bucket=self.policy_bucket, Key=s3_key)
                logger.debug(f"Already exists: {s3_key}")
                return False
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
            
            # Upload content
            s3_client.put_object(
                Bucket=self.policy_bucket,
                Key=s3_key,
                Body=content,
                ContentType=content_type
            )
            
            # Update progress
            self.progress["total_bytes_streamed"] += len(content)
            logger.debug(f"✓ Streamed to S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stream to S3 {s3_key}: {str(e)}")
            return False
    
    def _stream_url_to_s3(self, url: str, s3_key: str, content_type: str = 'application/octet-stream') -> bool:
        """Stream URL content directly to S3 without local storage"""
        try:
            # Check if already exists
            try:
                s3_client.head_object(Bucket=self.policy_bucket, Key=s3_key)
                logger.debug(f"Already exists: {s3_key}")
                return False
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
            
            # Stream from URL
            with self.session.get(url, stream=True) as response:
                response.raise_for_status()
                
                # Get content length if available
                content_length = response.headers.get('content-length')
                if content_length:
                    content_length = int(content_length)
                    if content_length > MAX_FILE_SIZE:
                        logger.warning(f"File too large ({content_length} bytes): {url}")
                        return False
                
                # For large files, use multipart upload
                if content_length and content_length > MULTIPART_THRESHOLD:
                    return self._multipart_stream_to_s3(response, s3_key, content_type)
                else:
                    # Simple upload for smaller files
                    s3_client.put_object(
                        Bucket=self.policy_bucket,
                        Key=s3_key,
                        Body=response.content,
                        ContentType=content_type
                    )
            
            self.progress["total_bytes_streamed"] += content_length or len(response.content)
            logger.debug(f"✓ Streamed URL to S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stream URL to S3 {url} -> {s3_key}: {str(e)}")
            return False
    
    def _multipart_stream_to_s3(self, response, s3_key: str, content_type: str) -> bool:
        """Handle multipart upload for large files"""
        try:
            # Initialize multipart upload
            multipart = s3_client.create_multipart_upload(
                Bucket=self.policy_bucket,
                Key=s3_key,
                ContentType=content_type
            )
            upload_id = multipart['UploadId']
            
            parts = []
            part_number = 1
            
            # Stream in chunks
            for chunk in response.iter_content(chunk_size=MULTIPART_THRESHOLD):
                if chunk:
                    part_response = s3_client.upload_part(
                        Bucket=self.policy_bucket,
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
                    logger.debug(f"Uploaded part {part_number-1} for {s3_key}")
            
            # Complete multipart upload
            s3_client.complete_multipart_upload(
                Bucket=self.policy_bucket,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            return True
            
        except Exception as e:
            # Abort multipart upload on error
            try:
                s3_client.abort_multipart_upload(
                    Bucket=self.policy_bucket,
                    Key=s3_key,
                    UploadId=upload_id
                )
            except:
                pass
            
            logger.error(f"Multipart upload failed for {s3_key}: {str(e)}")
            return False

def main():
    """Main entry point"""
    streamer = PVFDatasetStreamer()
    streamer.stream_repository_data()

if __name__ == "__main__":
    main()