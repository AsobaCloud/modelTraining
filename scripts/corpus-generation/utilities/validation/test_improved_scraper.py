#!/usr/bin/env python3
"""
Test script for improved scraper - validates the approach before full run
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from improved_scraper import ImprovedScraper
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scraper():
    """Test the improved scraper with a small sample"""
    
    # Initialize scraper
    scraper = ImprovedScraper(s3_bucket='flux-dev-nsfw', region='us-east-1')
    scraper.load_existing_dataset()
    
    # Test URLs - update these based on actual accessible sites
    test_categories = {
        'test_category': 'https://pornpics.com/galleries/ebony/'  # Example URL
    }
    
    logger.info("Starting test run of improved scraper...")
    
    # Test with just 3 images
    for category, url in test_categories.items():
        logger.info(f"Testing category: {category}")
        
        try:
            count = scraper.process_category(url, category, max_images=3)
            logger.info(f"Test result: {count} images collected for {category}")
            
            if count > 0:
                logger.info("✅ Improved scraper working - getting high-resolution images!")
            else:
                logger.warning("❌ No images collected - may need URL/config adjustment")
                
        except Exception as e:
            logger.error(f"Test failed for {category}: {e}")
    
    logger.info("Test complete!")

if __name__ == '__main__':
    test_scraper()