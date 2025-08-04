#!/usr/bin/env python3
"""
BabeSource High-Quality Image Scraper
Implements the plan from babesource_scraping_plan.md with TDD methodology
"""

import os
import time
import requests
import boto3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
import json
from datetime import datetime
import argparse
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BabeSourceScraper:
    """High-quality image scraper for BabeSource galleries"""
    
    def __init__(self, s3_bucket: str, region: str = "us-east-1"):
        self.s3_bucket = s3_bucket
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.existing_hashes = set()
        self.stats = {
            'galleries_processed': 0,
            'images_downloaded': 0,
            'images_uploaded': 0,
            'images_skipped': 0,
            'errors': 0
        }
        
        # Load existing dataset to avoid duplicates
        self.load_existing_dataset()
    
    def load_existing_dataset(self):
        """Load existing image hashes to avoid duplicates"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key='dataset_index.json'
            )
            dataset_index = json.loads(response['Body'].read())
            self.existing_hashes = set(dataset_index.get('hashes', []))
            logger.info(f"Loaded {len(self.existing_hashes)} existing image hashes")
        except:
            logger.info("No existing dataset index found, starting fresh")
    
    def extract_gallery_urls(self, listing_url: str, max_galleries: int = 100) -> List[str]:
        """Extract gallery URLs from a listing page"""
        gallery_urls = []
        
        try:
            response = self.session.get(listing_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all gallery links
            gallery_links = soup.find_all('a', href=re.compile(r'/galleries/'))
            
            for link in gallery_links[:max_galleries]:
                href = link.get('href')
                if href:
                    full_url = urljoin(listing_url, href)
                    gallery_urls.append(full_url)
            
            logger.info(f"Found {len(gallery_urls)} gallery URLs from {listing_url}")
            
        except Exception as e:
            logger.error(f"Error extracting gallery URLs from {listing_url}: {e}")
            self.stats['errors'] += 1
        
        return gallery_urls
    
    def extract_gallery_id(self, gallery_url: str) -> Optional[str]:
        """Extract gallery ID from gallery page by finding image links"""
        try:
            response = self.session.get(gallery_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for direct photo links to extract gallery ID
            photo_links = soup.find_all('a', href=re.compile(r'media\.babesource\.com/galleries/'))
            
            for link in photo_links:
                href = link.get('href')
                if href:
                    # Extract gallery ID from URL pattern
                    # https://media.babesource.com/galleries/687231342db80/01.jpg
                    match = re.search(r'/galleries/([a-f0-9]+)/', href)
                    if match:
                        gallery_id = match.group(1)
                        logger.info(f"Extracted gallery ID: {gallery_id} from {gallery_url}")
                        return gallery_id
            
            logger.warning(f"No gallery ID found in {gallery_url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting gallery ID from {gallery_url}: {e}")
            self.stats['errors'] += 1
            return None
    
    def generate_image_urls(self, gallery_id: str, max_images: int = 20) -> List[str]:
        """Generate complete set of image URLs for a gallery"""
        base_url = f"https://media.babesource.com/galleries/{gallery_id}"
        image_urls = []
        
        for i in range(1, max_images + 1):
            # Format: 01.jpg, 02.jpg, etc.
            image_num = f"{i:02d}"
            image_url = f"{base_url}/{image_num}.jpg"
            image_urls.append(image_url)
        
        return image_urls
    
    def validate_image_quality(self, image_data: bytes, min_resolution: Tuple[int, int] = (512, 512)) -> bool:
        """Validate image meets quality requirements"""
        try:
            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
            
            min_width, min_height = min_resolution
            if width < min_width or height < min_height:
                return False
            
            # Check file size (should be substantial for high-res)
            file_size_mb = len(image_data) / (1024 * 1024)
            if file_size_mb < 0.1:  # Less than 100KB probably low quality
                return False
            
            # Check format
            if img.format not in ['JPEG', 'PNG']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False
    
    def download_and_validate_image(self, image_url: str) -> Optional[bytes]:
        """Download image and validate quality"""
        try:
            response = self.session.get(image_url, timeout=30)
            
            # Check if image exists (404 means we've reached end of gallery)
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            
            # Verify content type
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                return None
            
            image_data = response.content
            
            # Validate quality
            if self.validate_image_quality(image_data):
                return image_data
            else:
                logger.info(f"Image quality validation failed for {image_url}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {image_url}: {e}")
            return None
    
    def upload_image_to_s3(self, image_data: bytes, filename: str, metadata: Dict) -> bool:
        """Upload image to S3 with metadata"""
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=filename,
                Body=image_data,
                ContentType='image/jpeg',
                Metadata=metadata
            )
            return True
        except Exception as e:
            logger.error(f"Error uploading {filename} to S3: {e}")
            return False
    
    def process_gallery(self, gallery_url: str, max_images: int = 20) -> Dict:
        """Process a complete gallery and download all valid images"""
        result = {
            'gallery_url': gallery_url,
            'gallery_id': None,
            'success_count': 0,
            'total_attempted': 0,
            'errors': []
        }
        
        try:
            # Extract gallery ID
            gallery_id = self.extract_gallery_id(gallery_url)
            if not gallery_id:
                result['errors'].append("Could not extract gallery ID")
                return result
            
            result['gallery_id'] = gallery_id
            
            # Generate image URLs
            image_urls = self.generate_image_urls(gallery_id, max_images)
            result['total_attempted'] = len(image_urls)
            
            # Process each image
            for i, image_url in enumerate(image_urls):
                try:
                    # Download and validate
                    image_data = self.download_and_validate_image(image_url)
                    
                    if image_data is None:
                        # If we get 404, we've reached the end of the gallery
                        if i > 0:  # Only if we got at least one image
                            break
                        continue
                    
                    # Check for duplicates
                    image_hash = hashlib.sha256(image_data).hexdigest()
                    if image_hash in self.existing_hashes:
                        logger.info(f"Duplicate image skipped: {image_hash[:8]}")
                        self.stats['images_skipped'] += 1
                        continue
                    
                    # Generate filename
                    image_num = f"{i+1:02d}"
                    filename = f"babesource_{gallery_id}_{image_num}.jpg"
                    
                    # Prepare metadata
                    metadata = {
                        'source': 'babesource',
                        'gallery_id': gallery_id,
                        'gallery_url': gallery_url,
                        'image_number': str(i+1),
                        'hash': image_hash
                    }
                    
                    # Upload to S3
                    if self.upload_image_to_s3(image_data, filename, metadata):
                        self.existing_hashes.add(image_hash)
                        result['success_count'] += 1
                        self.stats['images_uploaded'] += 1
                        logger.info(f"Successfully uploaded: {filename}")
                    else:
                        result['errors'].append(f"Upload failed for image {i+1}")
                    
                    self.stats['images_downloaded'] += 1
                    
                    # Respectful delay
                    time.sleep(1.0)
                    
                except Exception as e:
                    error_msg = f"Error processing image {i+1}: {str(e)}"
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
            
            self.stats['galleries_processed'] += 1
            logger.info(f"Gallery {gallery_id} processed: {result['success_count']}/{result['total_attempted']} images")
            
        except Exception as e:
            error_msg = f"Error processing gallery {gallery_url}: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
            self.stats['errors'] += 1
        
        return result
    
    def process_listing_page(self, listing_url: str, max_galleries: int = 50, max_images_per_gallery: int = 20) -> Dict:
        """Process an entire listing page of galleries"""
        logger.info(f"Processing listing page: {listing_url}")
        
        # Extract gallery URLs
        gallery_urls = self.extract_gallery_urls(listing_url, max_galleries)
        
        results = {
            'listing_url': listing_url,
            'total_galleries': len(gallery_urls),
            'successful_galleries': 0,
            'total_images': 0,
            'gallery_results': []
        }
        
        # Process each gallery
        for gallery_url in gallery_urls:
            try:
                gallery_result = self.process_gallery(gallery_url, max_images_per_gallery)
                results['gallery_results'].append(gallery_result)
                
                if gallery_result['success_count'] > 0:
                    results['successful_galleries'] += 1
                    results['total_images'] += gallery_result['success_count']
                
                # Respectful delay between galleries
                time.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error processing gallery {gallery_url}: {e}")
                self.stats['errors'] += 1
        
        logger.info(f"Listing complete: {results['successful_galleries']}/{results['total_galleries']} galleries, {results['total_images']} images")
        return results
    
    def get_stats(self) -> Dict:
        """Get current scraping statistics"""
        return self.stats.copy()

def main():
    """Main function for BabeSource scraping"""
    parser = argparse.ArgumentParser(description='BabeSource high-quality image scraper')
    parser.add_argument('--bucket', default='flux-dev-nsfw', help='S3 bucket name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--listing-urls', nargs='+', required=True, help='Listing page URLs to process')
    parser.add_argument('--max-galleries', type=int, default=50, help='Max galleries per listing page')
    parser.add_argument('--max-images', type=int, default=20, help='Max images per gallery')
    parser.add_argument('--test-mode', action='store_true', help='Test mode - process only 3 galleries')
    
    args = parser.parse_args()
    
    scraper = BabeSourceScraper(s3_bucket=args.bucket, region=args.region)
    
    if args.test_mode:
        max_galleries = 3
        max_images = 5
        logger.info("Running in TEST MODE - limited processing")
    else:
        max_galleries = args.max_galleries
        max_images = args.max_images
    
    all_results = []
    
    for listing_url in args.listing_urls:
        try:
            results = scraper.process_listing_page(listing_url, max_galleries, max_images)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error processing listing {listing_url}: {e}")
    
    # Print final statistics
    stats = scraper.get_stats()
    print("\n" + "="*50)
    print("BABESOURCE SCRAPING COMPLETE")
    print("="*50)
    print(f"Galleries processed: {stats['galleries_processed']}")
    print(f"Images downloaded: {stats['images_downloaded']}")
    print(f"Images uploaded: {stats['images_uploaded']}")
    print(f"Images skipped (duplicates): {stats['images_skipped']}")
    print(f"Errors encountered: {stats['errors']}")
    
    total_images = sum(r['total_images'] for r in all_results)
    print(f"Total new images collected: {total_images}")

if __name__ == '__main__':
    main()