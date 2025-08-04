#!/usr/bin/env python3
"""
Improved NSFW Image Collection Script for Flux Model Training
Enhanced to grab full-resolution images instead of thumbnails
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
from pathlib import Path
import re
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedScraper:
    def __init__(self, s3_bucket: str, s3_prefix: str = "", region: str = "us-east-1"):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.existing_hashes = set()
        self.existing_images = {}
        self.site_configs = self._get_improved_site_configs()
        
    def _get_improved_site_configs(self) -> Dict:
        """Enhanced site-specific configurations for full-resolution images"""
        return {
            'pornpics.com': {
                'gallery_link_selector': 'a.rel-link',  # Links to gallery pages
                'full_image_selectors': [
                    'img.main-image',  # Main gallery image
                    'img[data-original]',  # Original size image
                    'img.gallery-image',  # Gallery images
                    'img.content-image'
                ],
                'full_image_attributes': ['data-original', 'data-src', 'src'],
                'thumbnail_to_full_patterns': [
                    (r'/thumbs/(\d+)/', r'/pics/\1/'),  # Convert thumb to full
                    (r'_thumb\.', r'.')  # Remove _thumb suffix
                ],
                'min_resolution': (800, 600),
                'wait_time': 2.0
            },
            'multi.xnxx.com': {
                'gallery_link_selector': 'a.thumb',
                'full_image_selectors': [
                    'img.center_resized',
                    'img#photo',
                    'img.main-pic'
                ],
                'full_image_attributes': ['src', 'data-src'],
                'thumbnail_to_full_patterns': [
                    (r'/thumbs/', r'/pics/'),
                    (r'_t\.', r'.')
                ],
                'min_resolution': (800, 600),
                'wait_time': 3.0
            },
            'auntmia.com': {
                'gallery_link_selector': 'a.thumb-link, a.gallery-link',
                'full_image_selectors': [
                    'img.main-image',
                    'img.gallery-pic',
                    'img[data-original]'
                ],
                'full_image_attributes': ['data-original', 'data-src', 'src'],
                'thumbnail_to_full_patterns': [
                    (r'/thumb/', r'/full/'),
                    (r'_small\.', r'.')
                ],
                'min_resolution': (800, 600),
                'wait_time': 2.0
            }
        }
    
    def load_existing_dataset(self):
        """Load existing images to avoid duplicates"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key='dataset_index.json'
            )
            dataset_index = json.loads(response['Body'].read())
            self.existing_hashes = set(dataset_index.get('hashes', []))
            self.existing_images = dataset_index.get('images', {})
            logger.info(f"Loaded {len(self.existing_hashes)} existing image hashes")
        except:
            logger.info("No existing dataset index found, starting fresh")
    
    def get_full_image_url_from_thumbnail(self, thumb_url: str, site_config: Dict) -> str:
        """Convert thumbnail URL to full-resolution URL using patterns"""
        full_url = thumb_url
        
        patterns = site_config.get('thumbnail_to_full_patterns', [])
        for thumb_pattern, full_pattern in patterns:
            if re.search(thumb_pattern, thumb_url):
                full_url = re.sub(thumb_pattern, full_pattern, thumb_url)
                break
        
        return full_url
    
    def extract_full_images_from_gallery(self, gallery_url: str, site_config: Dict) -> List[Tuple[str, str]]:
        """Extract full-resolution images from a gallery page"""
        image_data = []
        
        try:
            response = self.session.get(gallery_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for full-size images using multiple selectors
            full_image_selectors = site_config.get('full_image_selectors', ['img'])
            full_image_attributes = site_config.get('full_image_attributes', ['src'])
            
            for selector in full_image_selectors:
                images = soup.select(selector)
                
                for img in images:
                    img_url = None
                    
                    # Try different attributes to find the full image
                    for attr in full_image_attributes:
                        img_url = img.get(attr)
                        if img_url:
                            break
                    
                    if img_url:
                        # Convert relative URLs to absolute
                        img_url = urljoin(gallery_url, img_url)
                        
                        # Try to convert thumbnail to full if needed
                        full_img_url = self.get_full_image_url_from_thumbnail(img_url, site_config)
                        
                        # Get alt text for potential caption
                        alt_text = img.get('alt', '')
                        
                        image_data.append((full_img_url, alt_text))
                
                # If we found images with this selector, don't try others
                if image_data:
                    break
            
            logger.info(f"Found {len(image_data)} full-resolution images in gallery: {gallery_url}")
            
        except Exception as e:
            logger.error(f"Error extracting from gallery {gallery_url}: {e}")
        
        return image_data
    
    def scrape_gallery_links(self, category_url: str, site_config: Dict, max_galleries: int = 20) -> List[str]:
        """Scrape gallery page links from category/listing pages"""
        gallery_links = []
        
        try:
            response = self.session.get(category_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find gallery links using site-specific selector
            gallery_selector = site_config.get('gallery_link_selector', 'a')
            links = soup.select(gallery_selector)
            
            for link in links[:max_galleries]:
                href = link.get('href')
                if href:
                    full_url = urljoin(category_url, href)
                    gallery_links.append(full_url)
            
            logger.info(f"Found {len(gallery_links)} gallery links on {category_url}")
            
        except Exception as e:
            logger.error(f"Error scraping gallery links from {category_url}: {e}")
        
        return gallery_links
    
    def validate_image_quality(self, image_data: bytes, min_resolution: Tuple[int, int] = (512, 512)) -> bool:
        """Validate image meets minimum quality requirements"""
        try:
            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
            
            min_width, min_height = min_resolution
            if width < min_width or height < min_height:
                logger.info(f"Image too small: {width}x{height} (min: {min_width}x{min_height})")
                return False
            
            # Check file size (should be substantial for high-res)
            file_size_mb = len(image_data) / (1024 * 1024)
            if file_size_mb < 0.1:  # Less than 100KB probably low quality
                logger.info(f"Image file too small: {file_size_mb:.2f}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False
    
    def download_and_validate_image(self, url: str, min_resolution: Tuple[int, int] = (512, 512)) -> Optional[bytes]:
        """Download image and validate it meets quality requirements"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Verify it's an image
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                logger.warning(f"Non-image content type: {content_type} for {url}")
                return None
            
            image_data = response.content
            
            # Validate quality
            if self.validate_image_quality(image_data, min_resolution):
                return image_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to download/validate {url}: {e}")
            return None
    
    def process_category(self, category_url: str, category_name: str, max_images: int = 50) -> int:
        """Process a category page and collect high-quality images"""
        site_domain = urlparse(category_url).netloc.replace('www.', '')
        site_config = self.site_configs.get(site_domain, {})
        
        if not site_config:
            logger.error(f"No configuration for site: {site_domain}")
            return 0
        
        min_resolution = site_config.get('min_resolution', (512, 512))
        wait_time = site_config.get('wait_time', 2.0)
        
        logger.info(f"Processing category: {category_name} from {site_domain}")
        
        # Step 1: Get gallery links from category page
        gallery_links = self.scrape_gallery_links(category_url, site_config)
        
        collected_count = 0
        
        # Step 2: Process each gallery
        for gallery_url in gallery_links:
            if collected_count >= max_images:
                break
            
            try:
                # Get full-resolution images from gallery
                image_urls = self.extract_full_images_from_gallery(gallery_url, site_config)
                
                # Download and process each image
                for img_url, alt_text in image_urls:
                    if collected_count >= max_images:
                        break
                    
                    # Download and validate
                    image_data = self.download_and_validate_image(img_url, min_resolution)
                    if not image_data:
                        continue
                    
                    # Check for duplicates
                    image_hash = hashlib.sha256(image_data).hexdigest()
                    if image_hash in self.existing_hashes:
                        logger.info(f"Duplicate image skipped: {image_hash[:8]}...")
                        continue
                    
                    # Generate filename and upload
                    filename = f"{category_name}_{image_hash[:12]}.jpg"
                    
                    try:
                        self.s3_client.put_object(
                            Bucket=self.s3_bucket,
                            Key=filename,
                            Body=image_data,
                            ContentType='image/jpeg'
                        )
                        
                        # Update tracking
                        self.existing_hashes.add(image_hash)
                        collected_count += 1
                        
                        logger.info(f"Uploaded high-quality image: {filename}")
                        
                    except Exception as e:
                        logger.error(f"Failed to upload {filename}: {e}")
                
                # Wait between galleries
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error processing gallery {gallery_url}: {e}")
        
        logger.info(f"Collected {collected_count} high-quality images for category: {category_name}")
        return collected_count

def main():
    """Main function to run improved scraping"""
    parser = argparse.ArgumentParser(description='Improved NSFW image scraper for high-quality dataset')
    parser.add_argument('--bucket', default='flux-dev-nsfw', help='S3 bucket name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--categories', nargs='+', default=['ebony', 'latina', 'asian'], help='Categories to collect')
    parser.add_argument('--images-per-category', type=int, default=50, help='Images per category')
    parser.add_argument('--test-mode', action='store_true', help='Test mode - process only 5 images')
    
    args = parser.parse_args()
    
    scraper = ImprovedScraper(s3_bucket=args.bucket, region=args.region)
    scraper.load_existing_dataset()
    
    # Example category URLs (you'll need to update these based on actual sites)
    category_urls = {
        'ebony': 'https://pornpics.com/galleries/ebony/',
        'latina': 'https://pornpics.com/galleries/latina/',
        'asian': 'https://pornpics.com/galleries/asian/'
    }
    
    total_collected = 0
    max_images = 5 if args.test_mode else args.images_per_category
    
    for category in args.categories:
        if category in category_urls:
            count = scraper.process_category(
                category_urls[category], 
                category, 
                max_images
            )
            total_collected += count
        else:
            logger.warning(f"No URL configured for category: {category}")
    
    logger.info(f"Total high-quality images collected: {total_collected}")

if __name__ == '__main__':
    main()