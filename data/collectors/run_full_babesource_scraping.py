#!/usr/bin/env python3
"""
Full BabeSource scraping run with all provided URLs
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from babesource_scraper import BabeSourceScraper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run full BabeSource scraping with provided URLs"""
    
    # All the URLs you provided
    listing_urls = [
        'https://babesource.com/filter-content/?pornstar=9252',  # Blake Blossom
        'https://babesource.com/filter-content/?pornstar=3343',  # Additional performer 1
        'https://babesource.com/filter-content/?pornstar=18',    # Additional performer 2
        'https://babesource.com/pornstars/tori-black-970/',      # Tori Black
        'https://babesource.com/pornstars/emily-willis-5812/',   # Emily Willis
        'https://babesource.com/pornstars/lacy-lennon-7171/'     # Lacy Lennon
    ]
    
    print("BABESOURCE FULL SCRAPING RUN")
    print("=" * 60)
    print(f"Processing {len(listing_urls)} listing pages")
    print("Targets: 30-50 galleries per page, 10-15 images per gallery")
    print("Estimated: 1500-3000 high-quality images")
    print()
    
    # Initialize scraper
    scraper = BabeSourceScraper(s3_bucket='flux-dev-nsfw', region='us-east-1')
    
    # Process each listing page
    all_results = []
    
    for i, listing_url in enumerate(listing_urls, 1):
        print(f"\n[{i}/{len(listing_urls)}] Processing: {listing_url}")
        
        try:
            # Process with reasonable limits
            results = scraper.process_listing_page(
                listing_url=listing_url,
                max_galleries=30,  # Conservative limit per page
                max_images_per_gallery=15  # Good balance of quality vs quantity
            )
            
            all_results.append(results)
            
            print(f"  ✅ Completed: {results['successful_galleries']} galleries, {results['total_images']} images")
            
        except Exception as e:
            print(f"  ❌ Error processing {listing_url}: {e}")
            logger.error(f"Error processing {listing_url}: {e}")
    
    # Final summary
    stats = scraper.get_stats()
    total_new_images = sum(r['total_images'] for r in all_results)
    
    print("\n" + "="*60)
    print("BABESOURCE SCRAPING COMPLETE")
    print("="*60)
    print(f"Listing pages processed: {len(all_results)}")
    print(f"Total galleries processed: {stats['galleries_processed']}")
    print(f"Total images downloaded: {stats['images_downloaded']}")
    print(f"Total images uploaded: {stats['images_uploaded']}")
    print(f"Images skipped (duplicates): {stats['images_skipped']}")
    print(f"Errors encountered: {stats['errors']}")
    print(f"New images added to dataset: {total_new_images}")
    
    # Estimate final dataset size
    existing_valid = 314  # From previous validation
    estimated_total = existing_valid + total_new_images
    
    print(f"\nDataset Status:")
    print(f"Previous valid images: {existing_valid}")
    print(f"New BabeSource images: {total_new_images}")
    print(f"Estimated total dataset: {estimated_total}")
    
    if estimated_total >= 1000:
        print("🎉 TARGET ACHIEVED: 1000+ image dataset ready for training!")
    else:
        remaining = 1000 - estimated_total
        print(f"⚠️  Still need ~{remaining} more images for 1000 target")
    
    print(f"\nNext steps:")
    print("1. Run comprehensive validation on full dataset")
    print("2. Extract and process iCloud Photos zip files")
    print("3. Set up joy-caption-batch for automated captioning")
    print("4. Begin Flux model fine-tuning")

if __name__ == '__main__':
    main()