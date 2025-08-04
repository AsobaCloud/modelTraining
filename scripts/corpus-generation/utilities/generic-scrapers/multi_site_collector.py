#!/usr/bin/env python3
"""
Multi-Site NSFW Image Collection Script for Flux Model Training
Supports multiple authorized sites with category tagging and deduplication
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiSiteCollector:
    def __init__(self, s3_bucket: str, s3_prefix: str = "", region: str = "us-east-1"):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.existing_hashes = set()
        self.existing_images = {}
        self.site_configs = self._get_site_configs()
        
    def _get_site_configs(self) -> Dict:
        """Site-specific scraping configurations"""
        return {
            'pornpics.com': {
                'img_selector': 'img.rel-link, img.thumb',
                'link_selector': 'a',
                'pagination': '?page={}',
                'wait_time': 2.0
            },
            'multi.xnxx.com': {
                'img_selector': 'img.thumb',
                'link_selector': 'a.thumb',
                'pagination': '/{}',
                'wait_time': 3.0
            },
            'auntmia.com': {
                'img_selector': 'img.lazy, img[data-src]',
                'link_selector': 'a.thumb-link',
                'pagination': '?page={}',
                'wait_time': 2.0
            },
            'maturehomemadeporn.com': {
                'img_selector': 'img',
                'link_selector': 'a',
                'pagination': '/page/{}',
                'wait_time': 2.5
            }
        }
    
    def load_existing_dataset(self):
        """Load existing images with metadata to avoid duplicates"""
        try:
            # Load existing metadata file if exists
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
                
            # Also scan S3 for any images not in index
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix)
            
            s3_count = 0
            for page in pages:
                if 'Contents' in page:
                    s3_count += len(page['Contents'])
            
            logger.info(f"Found {s3_count} total objects in S3")
            
        except Exception as e:
            logger.error(f"Error loading existing dataset: {e}")
    
    def extract_category_from_url(self, url: str) -> str:
        """Extract category/tag from URL"""
        # Pattern matching for different URL structures
        patterns = [
            r'/category/(\w+)',
            r'/pics/(\w+)',
            r'/(\w+)/$',
            r'/models/([^/]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1).lower().replace('%20', '_')
        
        return 'general'
    
    def get_site_domain(self, url: str) -> str:
        """Extract site domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    
    def scrape_page(self, url: str, site_config: Dict, max_images: int = 50) -> List[Tuple[str, str]]:
        """Scrape image URLs from a page with metadata"""
        image_data = []
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find images using site-specific selectors
            img_selector = site_config.get('img_selector', 'img')
            link_selector = site_config.get('link_selector', 'a')
            
            # Try to find linked full-size images first
            links = soup.select(link_selector)
            for link in links[:max_images]:
                img = link.find('img')
                if img:
                    # Get thumbnail for reference
                    thumb_url = img.get('src') or img.get('data-src')
                    # Get full image URL from link
                    full_url = link.get('href')
                    
                    if full_url and thumb_url:
                        full_url = urljoin(url, full_url)
                        # Extract alt text for potential caption
                        alt_text = img.get('alt', '')
                        image_data.append((full_url, alt_text))
            
            # Fallback to direct image URLs if no links found
            if not image_data:
                imgs = soup.select(img_selector)
                for img in imgs[:max_images]:
                    src = img.get('src') or img.get('data-src')
                    if src:
                        full_url = urljoin(url, src)
                        alt_text = img.get('alt', '')
                        image_data.append((full_url, alt_text))
            
            logger.info(f"Found {len(image_data)} images on {url}")
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return image_data
    
    def get_image_hash(self, image_data: bytes) -> str:
        """Generate SHA256 hash of image data"""
        return hashlib.sha256(image_data).hexdigest()
    
    def download_image(self, url: str, retry_count: int = 3) -> Optional[bytes]:
        """Download image with retry logic"""
        for attempt in range(retry_count):
            try:
                # Handle different URL types
                if url.endswith(('.html', '.php')):
                    # This might be a page, not direct image
                    page_response = self.session.get(url, timeout=30)
                    soup = BeautifulSoup(page_response.content, 'html.parser')
                    # Look for main image
                    main_img = soup.find('img', class_=['main-image', 'content-image']) or soup.find('img')
                    if main_img:
                        img_url = main_img.get('src')
                        if img_url:
                            url = urljoin(url, img_url)
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Verify it's an image
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    return response.content
                else:
                    logger.warning(f"Non-image content type: {content_type} for {url}")
                    return None
                    
            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {url}: {e}")
        
        return None
    
    def process_image(self, url: str, alt_text: str, source_page: str, category: str, site: str) -> bool:
        """Download and upload a single image with metadata"""
        try:
            # Download image
            image_data = self.download_image(url)
            if not image_data:
                return False
            
            # Check hash for duplicates
            image_hash = self.get_image_hash(image_data)
            if image_hash in self.existing_hashes:
                logger.info(f"Duplicate image skipped: {image_hash[:8]}...")
                return False
            
            # Generate filename
            extension = os.path.splitext(urlparse(url).path)[1] or '.jpg'
            filename = f"{category}_{image_hash[:12]}{extension}"
            
            # Prepare metadata
            metadata = {
                'source_url': url,
                'source_page': source_page,
                'site': site,
                'category': category,
                'alt_text': alt_text[:255] if alt_text else '',
                'collection_date': datetime.utcnow().isoformat(),
                'hash': image_hash
            }
            
            # Upload to S3
            s3_key = os.path.join(self.s3_prefix, filename) if self.s3_prefix else filename
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=image_data,
                ContentType='image/jpeg',
                Metadata={k: str(v) for k, v in metadata.items()}  # S3 metadata must be strings
            )
            
            # Update tracking
            self.existing_hashes.add(image_hash)
            self.existing_images[image_hash] = {
                'filename': filename,
                'metadata': metadata
            }
            
            logger.info(f"Uploaded: {filename} ({category})")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return False
    
    def collect_from_urls(self, urls: List[str], images_per_url: int = 50, max_workers: int = 5):
        """Collect images from multiple URLs"""
        self.load_existing_dataset()
        
        all_tasks = []
        
        # Process each URL
        for url in urls:
            site_domain = self.get_site_domain(url)
            site_config = self.site_configs.get(site_domain, self.site_configs['auntmia.com'])
            category = self.extract_category_from_url(url)
            
            logger.info(f"Processing {url} (site: {site_domain}, category: {category})")
            
            # Scrape the page
            image_data = self.scrape_page(url, site_config, images_per_url)
            
            # Add to task list
            for img_url, alt_text in image_data:
                all_tasks.append((img_url, alt_text, url, category, site_domain))
            
            # Rate limiting between pages
            time.sleep(site_config.get('wait_time', 2.0))
        
        logger.info(f"Total images to process: {len(all_tasks)}")
        
        # Process images in parallel
        successful_uploads = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_image, *task): task
                for task in all_tasks
            }
            
            for future in as_completed(futures):
                if future.result():
                    successful_uploads += 1
                    if successful_uploads % 10 == 0:
                        logger.info(f"Progress: {successful_uploads}/{len(all_tasks)}")
                        # Save progress periodically
                        self.save_dataset_index()
                
                # Rate limiting
                time.sleep(0.5)
        
        logger.info(f"Collection complete. Successfully uploaded {successful_uploads} new images")
        
        # Save final dataset index
        self.save_dataset_index()
        
        # Generate summary report
        self.generate_summary_report(successful_uploads)
    
    def save_dataset_index(self):
        """Save dataset index with all metadata"""
        index_data = {
            'last_updated': datetime.utcnow().isoformat(),
            'total_images': len(self.existing_hashes),
            'hashes': list(self.existing_hashes),
            'images': self.existing_images
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key='dataset_index.json',
                Body=json.dumps(index_data, indent=2),
                ContentType='application/json'
            )
            logger.info("Dataset index saved to S3")
        except Exception as e:
            logger.error(f"Failed to save dataset index: {e}")
    
    def generate_summary_report(self, new_uploads: int):
        """Generate collection summary report"""
        # Count by category
        category_counts = {}
        for img_data in self.existing_images.values():
            category = img_data['metadata'].get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        report = {
            'collection_date': datetime.utcnow().isoformat(),
            'total_images': len(self.existing_hashes),
            'new_uploads': new_uploads,
            'categories': category_counts,
            'target_remaining': max(0, 1000 - len(self.existing_hashes))
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key='collection_report.json',
                Body=json.dumps(report, indent=2),
                ContentType='application/json'
            )
            logger.info("Collection report saved to S3")
            logger.info(f"Total dataset size: {len(self.existing_hashes)} images")
            logger.info(f"Images needed to reach 1000: {report['target_remaining']}")
            logger.info(f"Category breakdown: {json.dumps(category_counts, indent=2)}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

def main():
    parser = argparse.ArgumentParser(description='Multi-site NSFW image collector')
    parser.add_argument('--bucket', default='flux-dev-nsfw', help='S3 bucket name')
    parser.add_argument('--prefix', default='', help='S3 prefix for organizing images')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--images-per-url', type=int, default=50, help='Max images per URL')
    parser.add_argument('--workers', type=int, default=5, help='Number of download workers')
    parser.add_argument('--urls-file', help='File containing URLs to scrape')
    
    args = parser.parse_args()
    
    # Default URLs
    urls = [
        'https://www.pornpics.com/blowjob',
        'https://multi.xnxx.com/category/blowjob',
        'https://www.auntmia.com/pics/blowjob/',
        'http://maturehomemadeporn.com/blowjobs/',
        'https://www.auntmia.com/pics/doggystyle/',
        'https://www.auntmia.com/pics/anal/',
        'https://www.auntmia.com/pics/bbc/',
        'https://www.auntmia.com/pics/ebony/',
        'https://www.auntmia.com/pics/arab/',
        'https://www.auntmia.com/pics/latina/',
        'https://www.auntmia.com/pics/brazilian/',
        'https://www.auntmia.com/pics/lingerie/',
        'https://www.auntmia.com/pics/indian/',
        'https://www.auntmia.com/pics/deepthroat/',
        'https://www.auntmia.com/models/Blake%20Blossom',
        'https://www.auntmia.com/pics/reversegangbang/',
        'https://www.auntmia.com/pics/ffm/',
        'https://www.auntmia.com/pics/striptease/'
    ]
    
    # Load URLs from file if provided
    if args.urls_file:
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    
    collector = MultiSiteCollector(
        s3_bucket=args.bucket,
        s3_prefix=args.prefix,
        region=args.region
    )
    
    collector.collect_from_urls(
        urls=urls,
        images_per_url=args.images_per_url,
        max_workers=args.workers
    )

if __name__ == '__main__':
    main()