#!/usr/bin/env python3
"""
ImageFap High-Quality Image Scraper
Designed to extract full-resolution images from ImageFap galleries
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

class ImageFapScraper:
    """High-quality image scraper for ImageFap galleries"""
    
    def __init__(self, s3_bucket: str, region: str = "us-east-1"):
        self.s3_bucket = s3_bucket
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.session = requests.Session()
        
        # More robust headers for ImageFap
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Configure session for better reliability
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self.existing_hashes = set()
        self.stats = {
            'galleries_processed': 0,
            'images_downloaded': 0,
            'images_uploaded': 0,
            'images_skipped': 0,
            'errors': 0
        }
        
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
    
    def extract_gallery_id(self, gallery_url: str) -> Optional[str]:
        """Extract gallery ID from ImageFap URL"""
        # Pattern: https://www.imagefap.com/pictures/13323821/Title
        match = re.search(r'/pictures/(\d+)/', gallery_url)
        if match:
            return match.group(1)
        return None
    
    def get_gallery_page_urls(self, gallery_url: str) -> List[str]:
        """Get all page URLs for a multi-page gallery"""
        page_urls = [gallery_url]
        
        try:
            response = self.session.get(gallery_url, timeout=45)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for pagination links
            pagination_links = soup.find_all('a', href=re.compile(r'page=\d+'))
            
            if pagination_links:
                # Extract all page numbers
                page_numbers = set()
                for link in pagination_links:
                    href = link.get('href')
                    match = re.search(r'page=(\d+)', href)
                    if match:
                        page_numbers.add(int(match.group(1)))
                
                # Generate URLs for all pages
                base_url = gallery_url.split('?')[0]  # Remove existing params
                for page_num in sorted(page_numbers):
                    if page_num > 1:  # Skip page 1 as we already have it
                        page_url = f"{base_url}?gid={self.extract_gallery_id(gallery_url)}&page={page_num}&view=0"
                        page_urls.append(page_url)
            
            logger.info(f"Found {len(page_urls)} pages for gallery")
            
        except Exception as e:
            logger.error(f"Error getting gallery pages: {e}")
        
        return page_urls
    
    def extract_image_urls_from_page(self, page_url: str) -> List[str]:
        """Extract image URLs from a gallery page"""
        image_urls = []
        
        try:
            response = self.session.get(page_url, timeout=45)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Method 1: Look for gallery table structure
            gallery_table = soup.find('table', id='gallery')
            if gallery_table:
                # Find thumbnail links that lead to full images
                thumb_links = gallery_table.find_all('a', href=re.compile(r'/photo/'))
                
                for link in thumb_links:
                    href = link.get('href')
                    if href:
                        # Convert relative URL to absolute
                        full_url = urljoin(page_url, href)
                        image_urls.append(full_url)
            
            # Method 2: Look for direct image links in content
            content_div = soup.find('div', id='content')
            if content_div:
                img_tags = content_div.find_all('img')
                for img in img_tags:
                    src = img.get('src')
                    if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png']):
                        # Skip thumbnails, look for full-size
                        if 'thumb' not in src.lower():
                            full_url = urljoin(page_url, src)
                            image_urls.append(full_url)
            
            logger.info(f"Found {len(image_urls)} image URLs on page: {page_url}")
            
        except Exception as e:
            logger.error(f"Error extracting images from {page_url}: {e}")
            self.stats['errors'] += 1
        
        return image_urls
    
    def get_full_image_url(self, photo_page_url: str) -> Optional[str]:
        """Get full-resolution image URL from photo page"""
        try:
            response = self.session.get(photo_page_url, timeout=45)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for the main image - ImageFap typically has id='mainPhoto'
            main_img = soup.find('img', id='mainPhoto')
            if main_img:
                src = main_img.get('src')
                if src:
                    return urljoin(photo_page_url, src)
            
            # Alternative: look for largest image
            all_imgs = soup.find_all('img')
            largest_img = None
            largest_size = 0
            
            for img in all_imgs:
                src = img.get('src')
                if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    # Skip thumbnails and UI elements
                    if not any(skip in src.lower() for skip in ['thumb', 'icon', 'logo', 'banner']):
                        # Try to get image dimensions from URL or filename
                        size_estimate = len(src)  # Rough estimate
                        if size_estimate > largest_size:
                            largest_size = size_estimate
                            largest_img = src
            
            if largest_img:
                return urljoin(photo_page_url, largest_img)
            
        except Exception as e:
            logger.error(f"Error getting full image from {photo_page_url}: {e}")
        
        return None
    
    def validate_image_quality(self, image_data: bytes, min_resolution: Tuple[int, int] = (512, 512)) -> bool:
        """Validate image meets quality requirements"""
        try:
            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
            
            min_width, min_height = min_resolution
            if width < min_width or height < min_height:
                return False
            
            # Check file size
            file_size_mb = len(image_data) / (1024 * 1024)
            if file_size_mb < 0.1 or file_size_mb > 20:
                return False
            
            # Check format
            if img.format not in ['JPEG', 'PNG']:
                return False
            
            return True
            
        except Exception:
            return False
    
    def download_and_validate_image(self, image_url: str) -> Optional[bytes]:
        """Download image and validate quality"""
        try:
            response = self.session.get(image_url, timeout=45)
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
                
        except Exception as e:
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
    
    def process_gallery(self, gallery_url: str) -> Dict:
        """Process a complete ImageFap gallery"""
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
            
            # Get all page URLs for this gallery
            page_urls = self.get_gallery_page_urls(gallery_url)
            
            # Process each page
            all_image_urls = []
            for page_url in page_urls:
                # Extract photo page URLs from this gallery page
                photo_page_urls = self.extract_image_urls_from_page(page_url)
                all_image_urls.extend(photo_page_urls)
                
                # Respectful delay between pages
                time.sleep(2.0)
            
            result['total_attempted'] = len(all_image_urls)
            logger.info(f"Found {len(all_image_urls)} total images across {len(page_urls)} pages")
            
            # Process each image
            for i, photo_page_url in enumerate(all_image_urls):
                try:
                    # Get full image URL from photo page
                    full_image_url = self.get_full_image_url(photo_page_url)
                    if not full_image_url:
                        continue
                    
                    # Download and validate
                    image_data = self.download_and_validate_image(full_image_url)
                    if image_data is None:
                        continue
                    
                    # Check for duplicates
                    image_hash = hashlib.sha256(image_data).hexdigest()
                    if image_hash in self.existing_hashes:
                        logger.info(f"Duplicate image skipped: {image_hash[:8]}")
                        self.stats['images_skipped'] += 1
                        continue
                    
                    # Generate filename
                    filename = f"imagefap_{gallery_id}_{i+1:03d}.jpg"
                    
                    # Prepare metadata
                    metadata = {
                        'source': 'imagefap',
                        'gallery_id': gallery_id,
                        'gallery_url': gallery_url,
                        'photo_page_url': photo_page_url,
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
                    
                    # Respectful delay between images
                    time.sleep(1.5)
                    
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
    
    def get_stats(self) -> Dict:
        """Get current scraping statistics"""
        return self.stats.copy()

def main():
    """Main function for ImageFap scraping"""
    parser = argparse.ArgumentParser(description='ImageFap high-quality image scraper')
    parser.add_argument('--bucket', default='flux-dev-nsfw', help='S3 bucket name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--gallery-urls', nargs='+', required=True, help='Gallery URLs to process')
    parser.add_argument('--test-mode', action='store_true', help='Test mode - process only first 10 images per gallery')
    
    args = parser.parse_args()
    
    scraper = ImageFapScraper(s3_bucket=args.bucket, region=args.region)
    
    all_results = []
    
    for gallery_url in args.gallery_urls:
        try:
            logger.info(f"Processing gallery: {gallery_url}")
            result = scraper.process_gallery(gallery_url)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing gallery {gallery_url}: {e}")
    
    # Print final statistics
    stats = scraper.get_stats()
    print("\n" + "="*50)
    print("IMAGEFAP SCRAPING COMPLETE")
    print("="*50)
    print(f"Galleries processed: {stats['galleries_processed']}")
    print(f"Images downloaded: {stats['images_downloaded']}")
    print(f"Images uploaded: {stats['images_uploaded']}")
    print(f"Images skipped (duplicates): {stats['images_skipped']}")
    print(f"Errors encountered: {stats['errors']}")
    
    total_images = sum(r['success_count'] for r in all_results)
    print(f"Total new images collected: {total_images}")

if __name__ == '__main__':
    main()