#!/usr/bin/env python3
"""
Production run script for insurance PDF collection
Focuses on reliable sources (arXiv + Semantic Scholar) to reach 100 papers
"""

import logging
from enhanced_insurance_scraper import EnhancedInsuranceScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("insurance_collection")

def main():
    """Run the full insurance paper collection"""
    logger.info("🚀 Starting full insurance paper collection (target: 100 papers)")
    
    try:
        # Initialize scraper
        scraper = EnhancedInsuranceScraper()
        
        # Run comprehensive collection
        papers = scraper.collect_comprehensive_papers(target_count=100)
        
        # Save comprehensive catalog
        scraper.save_catalog("insurance_papers_catalog_100.json")
        
        # Summary
        logger.info("="*60)
        logger.info("🎉 COLLECTION COMPLETE!")
        logger.info(f"📊 Total papers collected: {len(papers)}")
        logger.info(f"📁 S3 location: s3://{scraper.s3_bucket}/{scraper.s3_prefix}/")
        logger.info(f"📋 Catalog: insurance_papers_catalog_100.json")
        logger.info("="*60)
        
        # Print some sample papers
        if papers:
            logger.info("\n📄 Sample collected papers:")
            for i, paper in enumerate(papers[:5]):
                metadata = paper.get('metadata', {})
                title = metadata.get('title', 'Unknown')[:60]
                source = metadata.get('source', 'Unknown')
                logger.info(f"  {i+1}. {title}... [{source}]")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Collection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)