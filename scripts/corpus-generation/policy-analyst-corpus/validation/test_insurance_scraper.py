#!/usr/bin/env python3
"""
Test script for the insurance PDF scraper
"""

import logging
from enhanced_insurance_scraper import EnhancedInsuranceScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_scraper")

def test_basic_functionality():
    """Test basic scraper functionality"""
    logger.info("Testing basic scraper functionality...")
    
    try:
        # Initialize scraper
        scraper = EnhancedInsuranceScraper()
        logger.info("✅ Scraper initialized successfully")
        
        # Test S3 connection
        try:
            scraper.s3_client.head_bucket(Bucket=scraper.s3_bucket)
            logger.info("✅ S3 bucket access confirmed")
        except Exception as e:
            logger.error(f"❌ S3 bucket access failed: {str(e)}")
            return False
        
        # Test arXiv search with a simple query
        logger.info("Testing arXiv search...")
        arxiv_papers = scraper.search_arxiv_papers("insurance risk", max_results=3)
        logger.info(f"✅ Found {len(arxiv_papers)} arXiv papers")
        
        # Test Semantic Scholar search
        logger.info("Testing Semantic Scholar search...")
        semantic_papers = scraper.search_semantic_scholar("insurance risk management", max_results=3)
        logger.info(f"✅ Found {len(semantic_papers)} Semantic Scholar papers")
        
        # Test institutional sources
        logger.info("Testing institutional sources (limited)...")
        # Just test one institutional PDF to avoid downloading all
        test_doc = {
            'title': 'Test IAIS Document',
            'url': 'https://www.iaisweb.org/uploads/2022/01/191011-IAIS-ICPs-and-ComFrame-adopted-in-November-2019-Clean.pdf',
            'source': 'IAIS',
            'type': 'Standards'
        }
        
        response = scraper.session.get(test_doc['url'], timeout=30)
        if response.status_code == 200 and 'pdf' in response.headers.get('content-type', '').lower():
            logger.info("✅ Institutional PDF download test successful")
        else:
            logger.warning(f"⚠️ Institutional PDF test may have issues: {response.status_code}")
        
        logger.info("🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False

def test_small_collection():
    """Test collecting a small number of papers"""
    logger.info("Testing small collection (5 papers)...")
    
    try:
        scraper = EnhancedInsuranceScraper()
        
        # Collect just 5 papers to test the full pipeline
        papers = scraper.collect_comprehensive_papers(target_count=5)
        
        if len(papers) > 0:
            logger.info(f"✅ Successfully collected {len(papers)} papers")
            
            # Save test catalog
            scraper.save_catalog("test_insurance_catalog.json")
            logger.info("✅ Test catalog saved")
            
            return True
        else:
            logger.warning("⚠️ No papers were collected")
            return False
            
    except Exception as e:
        logger.error(f"❌ Small collection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting insurance scraper tests...")
    
    # Run basic functionality test
    if test_basic_functionality():
        logger.info("✅ Basic tests passed, proceeding with small collection test...")
        
        # Run small collection test
        if test_small_collection():
            logger.info("🎉 All tests passed! Scraper is ready for full collection.")
        else:
            logger.error("❌ Small collection test failed")
    else:
        logger.error("❌ Basic functionality tests failed")