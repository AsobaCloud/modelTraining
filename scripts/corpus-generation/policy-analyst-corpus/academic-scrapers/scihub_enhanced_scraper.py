#!/usr/bin/env python3
"""
Sci-Hub Enhanced Insurance PDF Scraper
Adds Sci-Hub fallback for paywalled sources using DOI extraction
"""

import os
import re
import json
import time
import boto3
import logging
import requests
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote
from pathlib import Path
import PyPDF2
from bs4 import BeautifulSoup

from enhanced_insurance_scraper import EnhancedInsuranceScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("scihub_enhanced_scraper")

class SciHubEnhancedScraper(EnhancedInsuranceScraper):
    """Enhanced scraper with Sci-Hub fallback for paywalled sources"""
    
    def __init__(self, s3_bucket: str = "policy-database", s3_prefix: str = "insurance"):
        super().__init__(s3_bucket, s3_prefix)
        
        # Sci-Hub mirrors (rotate through these)
        self.scihub_mirrors = [
            'https://sci-hub.se',
            'https://sci-hub.st',
            'https://sci-hub.ru',
            'https://sci-hub.ren',
            'https://sci-hub.mksa.top'
        ]
        
        # Track failed downloads for DOI extraction
        self.failed_downloads = []
        
        # Session with additional headers for Sci-Hub
        self.scihub_session = requests.Session()
        self.scihub_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract DOI from various academic URLs"""
        # Common DOI patterns
        doi_patterns = [
            r'doi\.org/(.+)',                    # Direct DOI URLs
            r'doi/(.+)',                         # Embedded DOI paths
            r'doi=([^&\s]+)',                    # DOI as parameter
            r'10\.\d{4,}/[^\s\?&]+',            # Standard DOI format
            r'/10\.\d{4,}/[^\s\?&\/]+',         # DOI in path
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                doi = match.group(1) if match.groups() else match.group(0)
                # Clean up DOI
                doi = doi.strip('/')
                if doi.startswith('10.'):
                    return doi
        
        return None
    
    def extract_doi_from_page(self, url: str) -> Optional[str]:
        """Extract DOI by scraping the page content"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for DOI in meta tags
            doi_meta = soup.find('meta', {'name': 'citation_doi'}) or \
                      soup.find('meta', {'name': 'DC.identifier'}) or \
                      soup.find('meta', {'property': 'citation_doi'})
            
            if doi_meta:
                doi = doi_meta.get('content', '')
                if doi.startswith('10.'):
                    return doi
            
            # Look for DOI in text content
            text_content = soup.get_text()
            doi_match = re.search(r'10\.\d{4,}/[^\s\<\>\"]+', text_content)
            if doi_match:
                return doi_match.group(0)
            
        except Exception as e:
            logger.debug(f"Could not extract DOI from page {url}: {str(e)}")
        
        return None
    
    def get_doi_for_failed_download(self, paper: Dict) -> Optional[str]:
        """Extract DOI for a failed download"""
        url = paper.get('pdf_url', '')
        
        # First try extracting from URL
        doi = self.extract_doi_from_url(url)
        if doi:
            return doi
        
        # Try extracting from the page
        doi = self.extract_doi_from_page(url)
        if doi:
            return doi
        
        # Look in paper metadata
        if 'doi' in paper:
            return paper['doi']
        
        return None
    
    def download_from_scihub(self, doi: str) -> Optional[bytes]:
        """Download PDF from Sci-Hub using DOI"""
        for mirror in self.scihub_mirrors:
            try:
                # Sci-Hub URL format
                scihub_url = f"{mirror}/{doi}"
                
                logger.info(f"Trying Sci-Hub: {scihub_url}")
                
                # First request to get the page
                response = self.scihub_session.get(scihub_url, timeout=30)
                
                if response.status_code != 200:
                    continue
                
                # Parse the response to find PDF link
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for PDF embed or direct link
                pdf_link = None
                
                # Check for iframe with PDF
                iframe = soup.find('iframe', {'id': 'pdf'}) or soup.find('iframe', src=re.compile(r'\.pdf'))
                if iframe:
                    pdf_link = iframe.get('src')
                
                # Check for direct PDF links
                if not pdf_link:
                    pdf_links = soup.find_all('a', href=re.compile(r'\.pdf', re.I))
                    if pdf_links:
                        pdf_link = pdf_links[0].get('href')
                
                # Check for embed tags
                if not pdf_link:
                    embed = soup.find('embed', src=re.compile(r'\.pdf', re.I))
                    if embed:
                        pdf_link = embed.get('src')
                
                if not pdf_link:
                    # Sometimes the PDF is served directly
                    if 'application/pdf' in response.headers.get('content-type', ''):
                        logger.info(f"✓ Direct PDF from Sci-Hub: {doi}")
                        return response.content
                    continue
                
                # Make PDF link absolute
                if pdf_link.startswith('//'):
                    pdf_link = 'https:' + pdf_link
                elif pdf_link.startswith('/'):
                    pdf_link = mirror + pdf_link
                elif not pdf_link.startswith('http'):
                    pdf_link = mirror + '/' + pdf_link
                
                # Download the PDF
                pdf_response = self.scihub_session.get(pdf_link, timeout=30)
                if pdf_response.status_code == 200 and 'pdf' in pdf_response.headers.get('content-type', '').lower():
                    logger.info(f"✓ Downloaded from Sci-Hub: {doi}")
                    return pdf_response.content
                
            except Exception as e:
                logger.debug(f"Sci-Hub attempt failed for {mirror}: {str(e)}")
                continue
            
            # Rate limiting between mirrors
            time.sleep(2)
        
        logger.warning(f"All Sci-Hub mirrors failed for DOI: {doi}")
        return None
    
    def download_pdf_with_scihub_fallback(self, url: str, filename: str, paper: Dict = None) -> Optional[bytes]:
        """Download PDF with Sci-Hub fallback"""
        # First try the original method
        pdf_content = self.download_pdf(url, filename)
        if pdf_content:
            return pdf_content
        
        # If original failed, try Sci-Hub
        logger.info(f"Original download failed, trying Sci-Hub fallback for: {filename[:50]}...")
        
        # Get DOI for this paper
        paper_dict = paper or {'pdf_url': url}
        doi = self.get_doi_for_failed_download(paper_dict)
        
        if not doi:
            logger.warning(f"Could not extract DOI for Sci-Hub fallback: {filename}")
            return None
        
        # Try Sci-Hub
        logger.info(f"Found DOI: {doi}, attempting Sci-Hub download...")
        return self.download_from_scihub(doi)
    
    def collect_with_scihub_fallback(self, target_count: int = 100) -> List[Dict]:
        """Main collection method with Sci-Hub fallback"""
        logger.info(f"Starting collection with Sci-Hub fallback (target: {target_count} papers)")
        
        all_papers = []
        
        # 1. Get papers from all sources (same as before)
        logger.info("Phase 1: Searching academic repositories...")
        academic_papers = super().search_academic_repositories()
        all_papers.extend(academic_papers)
        
        logger.info("Phase 2: Downloading institutional sources...")
        institutional_papers = self.search_institutional_sources()
        all_papers.extend(institutional_papers)
        
        logger.info("Phase 3: Searching additional academic sources...")
        additional_papers = self.search_additional_academic_sources()
        all_papers.extend(additional_papers)
        
        # 4. Deduplicate and rank
        unique_papers = self._deduplicate_papers(all_papers)
        sorted_papers = self._rank_papers_by_quality(unique_papers)
        
        logger.info(f"Found {len(sorted_papers)} unique papers for download")
        
        # 5. Download with Sci-Hub fallback
        successful_uploads = 0
        scihub_successes = 0
        
        for i, paper in enumerate(sorted_papers):
            if successful_uploads >= target_count:
                break
                
            logger.info(f"Processing paper {i+1}/{len(sorted_papers)}: {paper['title'][:50]}...")
            
            # Download PDF with Sci-Hub fallback
            pdf_content = None
            used_scihub = False
            
            if 'content' in paper:
                pdf_content = paper['content']
            else:
                # Try original download first
                pdf_content = self.download_pdf(paper['pdf_url'], paper['title'])
                
                # If failed, try Sci-Hub
                if not pdf_content:
                    pdf_content = self.download_from_scihub_for_paper(paper)
                    if pdf_content:
                        used_scihub = True
                        scihub_successes += 1
            
            if not pdf_content:
                continue
            
            # Extract metadata
            pdf_metadata = self.extract_pdf_metadata(pdf_content)
            
            # Add Sci-Hub flag to metadata
            combined_metadata = {
                'title': paper.get('title', ''),
                'source': paper.get('source', ''),
                'pdf_url': paper.get('pdf_url', ''),
                'collection_date': datetime.now().isoformat(),
                'scihub_used': used_scihub,
                **pdf_metadata
            }
            
            # Clean metadata for S3 (remove non-ASCII chars)
            clean_metadata = self._clean_metadata_for_s3(combined_metadata)
            
            # Generate filename and upload
            filename = self.generate_filename(paper)
            
            if self.upload_to_s3(pdf_content, filename, clean_metadata):
                successful_uploads += 1
                self.downloaded_papers.append({
                    'filename': filename,
                    's3_key': f"{self.s3_prefix}/{filename}",
                    'metadata': clean_metadata
                })
                
                source_note = " [Sci-Hub]" if used_scihub else ""
                logger.info(f"✓ Uploaded {successful_uploads}/{target_count}: {filename}{source_note}")
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"🎉 Collection complete!")
        logger.info(f"📊 Total papers: {successful_uploads}")
        logger.info(f"🔬 Sci-Hub successes: {scihub_successes}")
        
        return self.downloaded_papers
    
    def download_from_scihub_for_paper(self, paper: Dict) -> Optional[bytes]:
        """Helper to download from Sci-Hub for a specific paper"""
        doi = self.get_doi_for_failed_download(paper)
        if doi:
            return self.download_from_scihub(doi)
        return None
    
    def _clean_metadata_for_s3(self, metadata: Dict) -> Dict:
        """Clean metadata to remove non-ASCII characters for S3"""
        clean_meta = {}
        for key, value in metadata.items():
            if value is None:
                continue
            
            # Convert to string and remove non-ASCII
            str_value = str(value)
            clean_value = ''.join(char for char in str_value if ord(char) < 128)
            
            if clean_value.strip():
                clean_meta[key] = clean_value.strip()
        
        return clean_meta

def main():
    """Run collection with Sci-Hub fallback"""
    scraper = SciHubEnhancedScraper()
    
    # Collect 100 papers with Sci-Hub fallback
    papers = scraper.collect_with_scihub_fallback(target_count=100)
    
    # Save catalog
    scraper.save_catalog("insurance_papers_scihub_catalog.json")
    
    print(f"✅ Collection complete! {len(papers)} papers collected")
    print(f"📁 Location: s3://policy-database/insurance/")

if __name__ == "__main__":
    main()