#!/usr/bin/env python3
"""
InfraredSolarModules Repository Streaming Pipeline
Streams RaptorMaps infrared solar module dataset to policy database
"""

import os
import sys
import json
import time
import logging
import boto3
import git
import zipfile
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from botocore.exceptions import ClientError
from PIL import Image
import hashlib

# Configuration
POLICY_BUCKET = "policy-database"
PV_OPS_PATH = "pv-ops-and-maintenance"
REPO_URL = "https://github.com/RaptorMaps/InfraredSolarModules.git"
TEMP_DIR = "/tmp/infrared_solar_modules_staging"

# Progress tracking
PROGRESS_FILE = "infrared_streaming_progress.json"
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3", region_name="us-east-1")

class InfraredSolarModulesStreamer:
    """Streams InfraredSolarModules repository data"""
    
    def __init__(self, policy_bucket: str = POLICY_BUCKET):
        self.policy_bucket = policy_bucket
        self.progress = self.load_progress()
        self.temp_dir = TEMP_DIR
        
    def load_progress(self) -> Dict:
        """Load streaming progress from file"""
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return {
            "repo_cloned": False,
            "git_metadata_processed": False,
            "documentation_processed": False,
            "dataset_extracted": False,
            "images_processed_by_class": {},
            "total_images_processed": 0,
            "metadata_processed": False,
            "last_updated": None
        }
    
    def save_progress(self):
        """Save current progress"""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def stream_repository_data(self):
        """Main method to stream all repository data"""
        logger.info("🚀 Starting InfraredSolarModules repository streaming")
        start_time = time.time()
        
        try:
            # Step 1: Clone repository
            repo_path = self._clone_repository()
            
            # Step 2: Stream git metadata and history
            self._stream_git_metadata(repo_path)
            
            # Step 3: Stream documentation files
            self._stream_documentation(repo_path)
            
            # Step 4: Extract and stream dataset
            self._extract_and_stream_dataset(repo_path)
            
        except KeyboardInterrupt:
            logger.info("⚠️ Streaming interrupted by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {str(e)}")
            raise
        finally:
            # Cleanup
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                
            elapsed = time.time() - start_time
            logger.info(f"🎉 InfraredSolarModules streaming complete!")
            logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
            logger.info(f"📊 Total images: {self.progress['total_images_processed']}")
            logger.info(f"📁 Location: s3://{POLICY_BUCKET}/{PV_OPS_PATH}/")
    
    def _clone_repository(self) -> str:
        """Clone the GitHub repository"""
        if self.progress.get("repo_cloned"):
            logger.info("Repository already cloned, skipping...")
            return self.temp_dir
            
        logger.info("--- Cloning InfraredSolarModules repository ---")
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        try:
            repo = git.Repo.clone_from(REPO_URL, self.temp_dir)
            logger.info(f"✓ Repository cloned to: {self.temp_dir}")
            
            self.progress["repo_cloned"] = True
            self.save_progress()
            
            return self.temp_dir
            
        except Exception as e:
            logger.error(f"Failed to clone repository: {str(e)}")
            raise
    
    def _stream_git_metadata(self, repo_path: str):
        """Stream git metadata and history"""
        if self.progress.get("git_metadata_processed"):
            logger.info("Git metadata already processed, skipping...")
            return
            
        logger.info("--- Streaming git metadata ---")
        
        try:
            repo = git.Repo(repo_path)
            
            # Collect repository metadata
            repo_metadata = {
                "repository_url": REPO_URL,
                "current_commit": repo.head.commit.hexsha,
                "current_branch": repo.active_branch.name,
                "total_commits": len(list(repo.iter_commits())),
                "remote_urls": [remote.url for remote in repo.remotes],
                "collection_date": datetime.now().isoformat()
            }
            
            # Collect commit history
            commits = []
            for commit in repo.iter_commits():
                commit_info = {
                    "sha": commit.hexsha,
                    "author_name": commit.author.name,
                    "author_email": commit.author.email,
                    "committed_date": datetime.fromtimestamp(commit.committed_date).isoformat(),
                    "message": commit.message.strip(),
                    "stats": {
                        "total_files": commit.stats.total,
                        "insertions": commit.stats.total.get("insertions", 0),
                        "deletions": commit.stats.total.get("deletions", 0)
                    }
                }
                commits.append(commit_info)
            
            repo_metadata["commit_history"] = commits
            
            # Upload git metadata
            git_metadata_key = f"{PV_OPS_PATH}/repository_metadata/git_history.json"
            self._upload_to_s3(
                json.dumps(repo_metadata, indent=2).encode('utf-8'),
                git_metadata_key,
                'application/json'
            )
            
            logger.info(f"✓ Processed git metadata: {len(commits)} commits")
            self.progress["git_metadata_processed"] = True
            self.save_progress()
            
        except Exception as e:
            logger.error(f"Failed to process git metadata: {str(e)}")
    
    def _stream_documentation(self, repo_path: str):
        """Stream documentation files"""
        if self.progress.get("documentation_processed"):
            logger.info("Documentation already processed, skipping...")
            return
            
        logger.info("--- Streaming documentation ---")
        
        doc_files = ["README.md", "LICENSE"]
        
        for doc_file in doc_files:
            file_path = os.path.join(repo_path, doc_file)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create enhanced document with metadata
                    doc_metadata = {
                        "filename": doc_file,
                        "source_repository": REPO_URL,
                        "file_size_bytes": len(content.encode('utf-8')),
                        "collection_date": datetime.now().isoformat(),
                        "content": content
                    }
                    
                    doc_key = f"{PV_OPS_PATH}/documentation/{doc_file}.json"
                    self._upload_to_s3(
                        json.dumps(doc_metadata, indent=2).encode('utf-8'),
                        doc_key,
                        'application/json'
                    )
                    
                    logger.info(f"✓ Processed documentation: {doc_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {doc_file}: {str(e)}")
        
        self.progress["documentation_processed"] = True
        self.save_progress()
    
    def _extract_and_stream_dataset(self, repo_path: str):
        """Extract and stream the infrared solar modules dataset"""
        if self.progress.get("dataset_extracted"):
            logger.info("Dataset already extracted, skipping...")
            return
            
        logger.info("--- Extracting and streaming dataset ---")
        
        zip_path = os.path.join(repo_path, "2020-02-14_InfraredSolarModules.zip")
        
        if not os.path.exists(zip_path):
            logger.error(f"Dataset zip file not found: {zip_path}")
            return
        
        extract_dir = os.path.join(self.temp_dir, "extracted_dataset")
        
        try:
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info("✓ Dataset extracted successfully")
            
            # Find the extracted InfraredSolarModules directory
            dataset_dir = os.path.join(extract_dir, "InfraredSolarModules")
            
            if os.path.exists(dataset_dir):
                # Process metadata first
                self._process_dataset_metadata(dataset_dir)
                
                # Process images by class
                self._process_dataset_images(dataset_dir)
            else:
                logger.error(f"Dataset directory not found: {dataset_dir}")
            
            self.progress["dataset_extracted"] = True
            self.save_progress()
            
        except Exception as e:
            logger.error(f"Failed to extract dataset: {str(e)}")
    
    def _process_dataset_metadata(self, dataset_dir: str):
        """Process the module metadata JSON file"""
        if self.progress.get("metadata_processed"):
            logger.info("Metadata already processed, skipping...")
            return
            
        metadata_file = os.path.join(dataset_dir, "module_metadata.json")
        
        if not os.path.exists(metadata_file):
            logger.error(f"Metadata file not found: {metadata_file}")
            return
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Analyze metadata structure
            classes_count = {}
            for image_id, image_info in metadata.items():
                anomaly_class = image_info.get("anomaly_class", "Unknown")
                classes_count[anomaly_class] = classes_count.get(anomaly_class, 0) + 1
            
            # Create enhanced metadata with statistics
            enhanced_metadata = {
                "dataset_name": "InfraredSolarModules",
                "source_repository": REPO_URL,
                "total_images": len(metadata),
                "classes": classes_count,
                "collection_date": datetime.now().isoformat(),
                "original_metadata": metadata
            }
            
            # Upload metadata
            metadata_key = f"{PV_OPS_PATH}/dataset/metadata/module_metadata.json"
            self._upload_to_s3(
                json.dumps(enhanced_metadata, indent=2).encode('utf-8'),
                metadata_key,
                'application/json'
            )
            
            logger.info(f"✓ Processed metadata: {len(metadata)} images across {len(classes_count)} classes")
            self.progress["metadata_processed"] = True
            self.save_progress()
            
        except Exception as e:
            logger.error(f"Failed to process metadata: {str(e)}")
    
    def _process_dataset_images(self, dataset_dir: str):
        """Process and stream dataset images organized by class"""
        logger.info("--- Processing dataset images ---")
        
        images_dir = os.path.join(dataset_dir, "images")
        metadata_file = os.path.join(dataset_dir, "module_metadata.json")
        
        if not os.path.exists(images_dir) or not os.path.exists(metadata_file):
            logger.error("Images directory or metadata file not found")
            return
        
        try:
            # Load metadata to organize by class
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Group images by anomaly class
            images_by_class = {}
            for image_id, image_info in metadata.items():
                anomaly_class = image_info.get("anomaly_class", "Unknown")
                if anomaly_class not in images_by_class:
                    images_by_class[anomaly_class] = []
                images_by_class[anomaly_class].append({
                    "image_id": image_id,
                    "filepath": image_info.get("image_filepath", ""),
                    "anomaly_class": anomaly_class
                })
            
            # Process each class
            for anomaly_class, class_images in images_by_class.items():
                if anomaly_class in self.progress["images_processed_by_class"]:
                    logger.info(f"Class {anomaly_class} already processed, skipping...")
                    continue
                
                logger.info(f"Processing class: {anomaly_class} ({len(class_images)} images)")
                
                processed_count = 0
                for image_info in class_images:
                    try:
                        self._process_single_image(images_dir, image_info, anomaly_class)
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            logger.info(f"  Progress: {processed_count}/{len(class_images)} images in {anomaly_class}")
                            
                    except Exception as e:
                        logger.error(f"Failed to process image {image_info['image_id']}: {str(e)}")
                        continue
                
                self.progress["images_processed_by_class"][anomaly_class] = processed_count
                self.progress["total_images_processed"] += processed_count
                logger.info(f"✓ Completed class {anomaly_class}: {processed_count} images")
                self.save_progress()
                
        except Exception as e:
            logger.error(f"Failed to process images: {str(e)}")
    
    def _process_single_image(self, images_dir: str, image_info: Dict, anomaly_class: str):
        """Process and upload a single image"""
        image_id = image_info["image_id"]
        image_filename = f"{image_id}.jpg"  # Standard format from metadata
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            logger.debug(f"Image file not found: {image_path}")
            return
        
        try:
            # Read and validate image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Verify it's a valid image
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_format = img.format
            except Exception as e:
                logger.warning(f"Invalid image {image_id}: {str(e)}")
                return
            
            # Create image metadata
            image_metadata = {
                "image_id": image_id,
                "anomaly_class": anomaly_class,
                "filename": image_filename,
                "width": width,
                "height": height,
                "format": image_format,
                "file_size_bytes": len(image_data),
                "source_dataset": "InfraredSolarModules",
                "collection_date": datetime.now().isoformat()
            }
            
            # Upload image
            image_key = f"{PV_OPS_PATH}/dataset/images/{anomaly_class}/{image_filename}"
            self._upload_to_s3(image_data, image_key, f'image/{image_format.lower()}')
            
            # Upload image metadata
            metadata_key = f"{PV_OPS_PATH}/dataset/metadata/images/{anomaly_class}/{image_id}_metadata.json"
            self._upload_to_s3(
                json.dumps(image_metadata, indent=2).encode('utf-8'),
                metadata_key,
                'application/json'
            )
            
        except Exception as e:
            logger.error(f"Failed to process image {image_id}: {str(e)}")
    
    def _upload_to_s3(self, content: bytes, s3_key: str, content_type: str) -> bool:
        """Upload content to S3 with existence check"""
        try:
            # Check if already exists
            try:
                s3_client.head_object(Bucket=self.policy_bucket, Key=s3_key)
                logger.debug(f"Already exists: {s3_key}")
                return False
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
            
            # Upload new content
            s3_client.put_object(
                Bucket=self.policy_bucket,
                Key=s3_key,
                Body=content,
                ContentType=content_type
            )
            logger.debug(f"✓ Uploaded: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"S3 upload failed for {s3_key}: {str(e)}")
            return False

def main():
    """Main entry point"""
    streamer = InfraredSolarModulesStreamer()
    streamer.stream_repository_data()

if __name__ == "__main__":
    main()