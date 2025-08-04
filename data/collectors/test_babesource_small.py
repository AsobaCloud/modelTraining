#!/usr/bin/env python3
"""
Small-scale test of BabeSource scraper before full run
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from babesource_scraper import BabeSourceScraper
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_gallery():
    """Test processing a single gallery"""
    
    scraper = BabeSourceScraper(s3_bucket='flux-dev-nsfw', region='us-east-1')
    
    # Test with the specific gallery from your example
    test_gallery_url = 'https://babesource.com/galleries/blake-blossom-ar-porn-190435.html'
    
    logger.info("Testing single gallery processing...")
    logger.info(f"Gallery URL: {test_gallery_url}")
    
    try:
        # Process just this one gallery with limited images
        result = scraper.process_gallery(test_gallery_url, max_images=3)
        
        print("\n" + "="*50)
        print("SINGLE GALLERY TEST RESULTS")
        print("="*50)
        print(f"Gallery URL: {result['gallery_url']}")
        print(f"Gallery ID: {result['gallery_id']}")
        print(f"Success Count: {result['success_count']}")
        print(f"Total Attempted: {result['total_attempted']}")
        print(f"Errors: {len(result['errors'])}")
        
        if result['errors']:
            print("Error details:")
            for error in result['errors']:
                print(f"  - {error}")
        
        stats = scraper.get_stats()
        print(f"\nOverall Stats:")
        print(f"  Images downloaded: {stats['images_downloaded']}")
        print(f"  Images uploaded: {stats['images_uploaded']}")
        print(f"  Images skipped: {stats['images_skipped']}")
        print(f"  Errors: {stats['errors']}")
        
        if result['success_count'] > 0:
            print("\n✅ SUCCESS: Scraper is working and uploading images!")
            return True
        else:
            print("\n❌ ISSUE: No images were successfully processed")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        return False

def test_gallery_url_extraction():
    """Test extracting gallery URLs from listing page"""
    
    scraper = BabeSourceScraper(s3_bucket='flux-dev-nsfw', region='us-east-1')
    
    # Test with your provided URL
    test_listing_url = 'https://babesource.com/filter-content/?pornstar=9252'
    
    logger.info("Testing gallery URL extraction...")
    logger.info(f"Listing URL: {test_listing_url}")
    
    try:
        gallery_urls = scraper.extract_gallery_urls(test_listing_url, max_galleries=5)
        
        print("\n" + "="*50)
        print("GALLERY URL EXTRACTION TEST")
        print("="*50)
        print(f"Listing URL: {test_listing_url}")
        print(f"Gallery URLs found: {len(gallery_urls)}")
        
        for i, url in enumerate(gallery_urls[:5]):
            print(f"  {i+1}. {url}")
        
        if len(gallery_urls) > 0:
            print("\n✅ SUCCESS: Gallery URL extraction working!")
            return gallery_urls
        else:
            print("\n❌ ISSUE: No gallery URLs found")
            return []
            
    except Exception as e:
        print(f"\n❌ ERROR: URL extraction failed: {e}")
        return []

if __name__ == '__main__':
    print("BABESOURCE SCRAPER TEST")
    print("="*60)
    
    # Test 1: Gallery URL extraction
    gallery_urls = test_gallery_url_extraction()
    
    if gallery_urls:
        # Test 2: Single gallery processing
        success = test_single_gallery()
        
        if success:
            print("\n🎉 ALL TESTS PASSED - Ready for full scraping!")
        else:
            print("\n⚠️  URL extraction works but gallery processing has issues")
    else:
        print("\n❌ URL extraction failed - check site structure or connection")