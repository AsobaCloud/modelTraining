#!/usr/bin/env python3
"""
Figshare Adult Content Dataset Downloader
Downloads and processes the authorized adult content dataset from Figshare
"""

import os
import requests
import zipfile
import boto3
import hashlib
import json
from pathlib import Path
from PIL import Image
import io
from datetime import datetime
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import tempfile
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FigshareDatasetProcessor:
    def __init__(self, s3_bucket: str, s3_prefix: str = "", region: str = "us-east-1"):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.existing_hashes = set()
        self.temp_dir = tempfile.mkdtemp()
        
    def load_existing_hashes(self):
        """Load existing image hashes to avoid duplicates"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key='dataset_index.json'
            )
            dataset_index = json.loads(response['Body'].read())
            self.existing_hashes = set(dataset_index.get('hashes', []))
            logger.info(f"Loaded {len(self.existing_hashes)} existing image hashes")
        except Exception as e:
            logger.info("No existing dataset index found, starting fresh")
    
    def download_figshare_dataset(self, url: str) -> str:
        """Download the Figshare dataset zip file"""
        logger.info(f"Downloading dataset from {url}")
        
        zip_path = os.path.join(self.temp_dir, "figshare_adult_dataset.zip")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
                            logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Download complete: {zip_path}")
        return zip_path
    
    def extract_dataset(self, zip_path: str) -> str:
        """Extract the zip file and return extraction directory"""
        extract_dir = os.path.join(self.temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find image files
        image_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_files)} image files in dataset")
        return extract_dir, image_files
    
    def get_image_hash(self, image_path: str) -> Optional[str]:
        """Generate SHA256 hash of image file"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {image_path}: {e}")
            return None
    
    def validate_and_process_image(self, image_path: str) -> Optional[Dict]:
        """Validate image and prepare metadata"""
        try:
            # Check if image is valid
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Basic quality checks
                if width < 256 or height < 256:
                    logger.debug(f"Skipping low resolution image: {width}x{height}")
                    return None
                
                # Check aspect ratio
                aspect_ratio = width / height
                if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                    logger.debug(f"Skipping extreme aspect ratio: {aspect_ratio:.2f}")
                    return None
                
                # Get file size
                file_size = os.path.getsize(image_path)
                if file_size > 50 * 1024 * 1024:  # 50MB limit
                    logger.debug(f"Skipping large file: {file_size / (1024*1024):.1f}MB")
                    return None
                
                # Generate hash
                image_hash = self.get_image_hash(image_path)
                if not image_hash:
                    return None
                
                # Check for duplicates
                if image_hash in self.existing_hashes:
                    logger.debug(f"Skipping duplicate image: {image_hash[:8]}...")
                    return None
                
                # Prepare metadata
                filename = f"figshare_{image_hash[:12]}.jpg"
                
                metadata = {
                    'source': 'figshare_adult_dataset',
                    'source_url': 'https://figshare.com/articles/dataset/Adult_content_dataset/13456484',
                    'original_filename': os.path.basename(image_path),
                    'category': 'adult_general',
                    'width': width,
                    'height': height,
                    'file_size': file_size,
                    'hash': image_hash,
                    'collection_date': datetime.utcnow().isoformat()
                }
                
                return {
                    'image_path': image_path,
                    'filename': filename,
                    'metadata': metadata,
                    'hash': image_hash
                }
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def upload_image_to_s3(self, image_data: Dict) -> bool:
        """Upload processed image to S3"""
        try:
            # Read image file
            with open(image_data['image_path'], 'rb') as f:
                image_bytes = f.read()
            
            # Upload to S3
            s3_key = os.path.join(self.s3_prefix, image_data['filename']) if self.s3_prefix else image_data['filename']
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=image_bytes,
                ContentType='image/jpeg',
                Metadata={k: str(v) for k, v in image_data['metadata'].items()}
            )
            
            # Add to existing hashes
            self.existing_hashes.add(image_data['hash'])
            
            logger.info(f"Uploaded: {image_data['filename']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {image_data['filename']}: {e}")
            return False
    
    def process_dataset(self, image_files: List[str], max_workers: int = 10, max_images: int = None):
        """Process all images in the dataset"""
        logger.info(f"Processing {len(image_files)} images with {max_workers} workers")
        
        if max_images:
            image_files = image_files[:max_images]
            logger.info(f"Limited to first {max_images} images")
        
        successful_uploads = 0
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit validation tasks
            validation_futures = {
                executor.submit(self.validate_and_process_image, img_path): img_path
                for img_path in image_files
            }
            
            # Process validation results and submit upload tasks
            upload_futures = {}
            for future in as_completed(validation_futures):
                processed_count += 1
                result = future.result()
                
                if result:
                    # Submit upload task
                    upload_future = executor.submit(self.upload_image_to_s3, result)
                    upload_futures[upload_future] = result['filename']
                
                if processed_count % 100 == 0:
                    logger.info(f"Validated {processed_count}/{len(image_files)} images")
            
            # Wait for uploads to complete
            for future in as_completed(upload_futures):
                if future.result():
                    successful_uploads += 1
                    
                if successful_uploads % 50 == 0:
                    logger.info(f"Uploaded {successful_uploads} images")
        
        logger.info(f"Processing complete: {successful_uploads} images uploaded from {len(image_files)} total")
        return successful_uploads
    
    def update_dataset_index(self, new_uploads: int):
        """Update the dataset index with new images"""
        try:
            # Try to load existing index
            try:
                response = self.s3_client.get_object(
                    Bucket=self.s3_bucket,
                    Key='dataset_index.json'
                )
                dataset_index = json.loads(response['Body'].read())
            except:
                dataset_index = {'images': {}, 'hashes': []}
            
            # Update with new hashes
            dataset_index['hashes'] = list(self.existing_hashes)
            dataset_index['last_updated'] = datetime.utcnow().isoformat()
            dataset_index['total_images'] = len(self.existing_hashes)
            dataset_index['figshare_uploads'] = new_uploads
            
            # Save updated index
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key='dataset_index.json',
                Body=json.dumps(dataset_index, indent=2),
                ContentType='application/json'
            )
            
            logger.info("Dataset index updated")
            
        except Exception as e:
            logger.error(f"Failed to update dataset index: {e}")
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
    
    def run(self, figshare_url: str, max_workers: int = 10, max_images: int = None):
        """Main processing pipeline"""
        try:
            # Load existing dataset info
            self.load_existing_hashes()
            
            # Download dataset
            zip_path = self.download_figshare_dataset(figshare_url)
            
            # Extract dataset
            extract_dir, image_files = self.extract_dataset(zip_path)
            
            # Process images
            uploaded_count = self.process_dataset(image_files, max_workers, max_images)
            
            # Update index
            self.update_dataset_index(uploaded_count)
            
            logger.info(f"Figshare dataset processing complete: {uploaded_count} new images added")
            
            return uploaded_count
            
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Download and process Figshare adult content dataset')
    parser.add_argument('--bucket', default='flux-dev-nsfw', help='S3 bucket name')
    parser.add_argument('--prefix', default='', help='S3 prefix for organizing images')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--workers', type=int, default=10, help='Number of processing workers')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process')
    parser.add_argument('--url', default='https://figshare.com/ndownloader/files/25843427', 
                       help='Figshare download URL')
    
    args = parser.parse_args()
    
    processor = FigshareDatasetProcessor(
        s3_bucket=args.bucket,
        s3_prefix=args.prefix,
        region=args.region
    )
    
    uploaded_count = processor.run(
        figshare_url=args.url,
        max_workers=args.workers,
        max_images=args.max_images
    )
    
    print(f"\nFigshare dataset processing complete!")
    print(f"New images uploaded: {uploaded_count}")
    print(f"Check bucket status: aws s3 ls s3://{args.bucket}/ --region {args.region} --recursive | wc -l")

if __name__ == '__main__':
    main()