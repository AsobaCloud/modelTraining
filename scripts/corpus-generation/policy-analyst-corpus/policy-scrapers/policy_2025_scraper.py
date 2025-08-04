#!/usr/bin/env python3
"""
2025 Policy & Energy Research Scraper
Following the pattern from datascraper.py with API keys
Target: s3://policy-database/corpus_7-26-2025
"""

import os
import sys
import json
import time
import logging
import requests
import boto3
import re
import hashlib
from datetime import datetime
from urllib.parse import urlencode, urljoin, urlparse, quote
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
import PyPDF2
from io import BytesIO

# -------------------------------------------------------------------------
# API CONFIGURATION - Following datascraper.py pattern
# -------------------------------------------------------------------------
GOVINFO_API_KEY = "0yyvGVuCtmK28iOU1c6jYO6w0rzXVencKGQxU5o9"
COURTLISTENER_API_KEY = "578d9ab5e03cafdc1ced7dfce158cd42ef60a2c7"

# Congress.gov API (requires registration)
# CONGRESS_API_KEY = "REGISTER_AT_CONGRESS_GOV"  # User needs to register

# S3 Configuration
S3_BUCKET_NAME = "policy-database"
S3_FOLDER_2025 = "corpus_7-26-2025"

# Track progress
PROGRESS_FILE = "policy_2025_progress.json"

# Search terms for 2025 legislation/executive orders: energy, electricity, blockchain, AI, insurance
ENERGY_KEYWORDS = [
    "energy",
    "electricity", 
    "blockchain",
    "artificial intelligence",
    "insurance"
]

# Search terms for academic papers: modeling energy production/consumption + intersection of economics and energy
RESEARCH_KEYWORDS = [
    "energy",
    "electricity", 
    "blockchain",
    "artificial intelligence",
    "insurance"
]

# Sci-Hub mirrors for aggressive fallback
SCIHUB_MIRRORS = [
    'https://sci-hub.se',
    'https://sci-hub.st',
    'https://sci-hub.ru',
    'https://sci-hub.ren',
    'https://sci-hub.mksa.top'
]

PAGE_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3", region_name="us-east-1")

# -------------------------------------------------------------------------
# PROGRESS TRACKING
# -------------------------------------------------------------------------
class ProgressTracker:
    def __init__(self, progress_file=PROGRESS_FILE):
        self.progress_file = progress_file
        self.progress = self.load_progress()
    
    def load_progress(self):
        """Load progress from file or initialize new"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "federal_legislation": {"keywords_completed": []},
            "federal_register": {"keywords_completed": []},
            "court_cases": {"queries_completed": []},
            "research_papers": {
                "arxiv_completed": [],
                "semantic_completed": []
            },
            "total_documents": 0,
            "last_updated": None
        }
    
    def save_progress(self):
        """Save current progress"""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def mark_keyword_complete(self, section, keyword):
        """Mark a keyword as completed for a section"""
        if keyword not in self.progress[section]["keywords_completed"]:
            self.progress[section]["keywords_completed"].append(keyword)
            self.save_progress()
    
    def is_keyword_complete(self, section, keyword):
        """Check if keyword was already processed"""
        return keyword in self.progress[section].get("keywords_completed", [])
    
    def mark_query_complete(self, section, query):
        """Mark a query as completed"""
        if query not in self.progress[section]["queries_completed"]:
            self.progress[section]["queries_completed"].append(query)
            self.save_progress()
    
    def is_query_complete(self, section, query):
        """Check if query was already processed"""
        return query in self.progress[section].get("queries_completed", [])
    
    def increment_documents(self, count=1):
        """Increment total document count"""
        self.progress["total_documents"] += count
        self.save_progress()

progress_tracker = ProgressTracker()

# -------------------------------------------------------------------------
# IDEMPOTENT S3 OPERATIONS
# -------------------------------------------------------------------------
def get_s3_manifest():
    """Get manifest of all files already in S3 bucket/prefix"""
    manifest = set()
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=S3_BUCKET_NAME,
            Prefix=S3_FOLDER_2025 + "/"
        )
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    manifest.add(obj['Key'])
        
        logger.info(f"S3 manifest loaded: {len(manifest)} existing files")
        return manifest
    except Exception as e:
        logger.error(f"Error loading S3 manifest: {str(e)}")
        return set()

# Global manifest for idempotency
S3_MANIFEST = get_s3_manifest()

def exists_in_s3(key: str) -> bool:
    """Check if file exists in S3 using manifest"""
    return key in S3_MANIFEST

def exists_in_s3_old(bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def sanitize_filename(key: str) -> str:
    parts = key.split("/")
    filename = parts[-1].strip()
    filename = filename.replace("\\", "_")
    filename = re.sub(r"\s+", "_", filename)
    filename = re.sub(r"[^\w\.-]", "_", filename)
    parts[-1] = filename
    return "/".join(parts)

def upload_to_s3_if_not_exists(file_content: bytes, s3_key: str, content_type: str = "application/pdf"):
    s3_key = sanitize_filename(s3_key)
    
    # Check manifest first (faster than HEAD request)
    if exists_in_s3(s3_key):
        logger.debug(f"Skipping (exists in manifest): {s3_key}")
        return False
    
    try:
        logger.info(f"Uploading to S3: {s3_key}")
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type
        )
        # Add to manifest
        S3_MANIFEST.add(s3_key)
        logger.info(f"✓ Uploaded: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {s3_key}: {e}")
        return False

def extract_doi_from_url(url: str) -> str:
    """Extract DOI from URL for Sci-Hub"""
    doi_patterns = [
        r'doi\.org/(.+)',
        r'doi/(.+)', 
        r'doi=([^&\s]+)',
        r'10\.\d{4,}/[^\s\?&]+',
    ]
    
    for pattern in doi_patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            doi = match.group(1) if match.groups() else match.group(0)
            doi = doi.strip('/')
            if doi.startswith('10.'):
                return doi
    return None

def download_from_scihub(doi: str) -> bytes:
    """Download PDF from Sci-Hub using DOI"""
    for mirror in SCIHUB_MIRRORS:
        try:
            scihub_url = f"{mirror}/{doi}"
            logger.info(f"Trying Sci-Hub: {scihub_url}")
            
            response = requests.get(scihub_url, timeout=30)
            if response.status_code != 200:
                continue
            
            # Check if direct PDF
            if 'application/pdf' in response.headers.get('content-type', ''):
                logger.info(f"✓ Direct PDF from Sci-Hub: {doi}")
                return response.content
            
            # Parse for PDF link
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for iframe with PDF
            iframe = soup.find('iframe', {'id': 'pdf'}) or soup.find('iframe', src=re.compile(r'\.pdf'))
            if iframe:
                pdf_link = iframe.get('src')
                if pdf_link:
                    if pdf_link.startswith('//'):
                        pdf_link = 'https:' + pdf_link
                    elif pdf_link.startswith('/'):
                        pdf_link = mirror + pdf_link
                    
                    pdf_response = requests.get(pdf_link, timeout=30)
                    if pdf_response.status_code == 200:
                        logger.info(f"✓ Downloaded from Sci-Hub: {doi}")
                        return pdf_response.content
            
        except Exception as e:
            logger.debug(f"Sci-Hub {mirror} failed: {str(e)}")
            continue
        
        time.sleep(2)
    
    logger.warning(f"All Sci-Hub mirrors failed for DOI: {doi}")
    return None

def download_pdf_with_scihub_fallback(url: str, title: str) -> bytes:
    """Download PDF with Sci-Hub fallback"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if 'pdf' in response.headers.get('content-type', '').lower():
            return response.content
        else:
            logger.warning(f"URL doesn't return PDF content: {url}")
    except Exception as e:
        logger.error(f"Direct download failed for {url}: {str(e)}")
    
    # Try Sci-Hub fallback
    doi = extract_doi_from_url(url)
    if doi:
        logger.info(f"Trying Sci-Hub fallback for: {title[:30]}...")
        return download_from_scihub(doi)
    
    return None

# -------------------------------------------------------------------------
# PART 1: FEDERAL LEGISLATION (GovInfo API) - COMPLETE PAGINATION
# -------------------------------------------------------------------------
def fetch_2025_federal_legislation():
    logger.info("=== FEDERAL LEGISLATION: Starting ===")
    
    search_url = f"https://api.govinfo.gov/search?api_key={GOVINFO_API_KEY}"
    total_processed = 0
    
    for keyword in ENERGY_KEYWORDS:
        # Skip if already completed
        if progress_tracker.is_keyword_complete("federal_legislation", keyword):
            logger.info(f"Skipping completed keyword: {keyword}")
            continue
            
        logger.info(f"Processing keyword: {keyword}")
        keyword_count = 0
        offset_mark = "*"
        
        while True:  # Keep going until no more results
            payload = {
                "query": f"{keyword} AND dateIssued:[2025-01-01 TO 2025-12-31]",
                "pageSize": PAGE_SIZE,
                "offsetMark": offset_mark,
                "sorts": [{"field": "dateIssued", "sortOrder": "DESC"}],
                "historical": False,
                "resultLevel": "default"
            }
            
            try:
                resp = requests.post(search_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"GovInfo search failed for {keyword}: {str(e)}")
                break
            
            results = data.get("results", [])
            if not results:
                logger.info(f"No more results for '{keyword}' (processed {keyword_count} documents)")
                progress_tracker.mark_keyword_complete("federal_legislation", keyword)
                break
            
            for item in results:
                package_id = item.get("packageId", "UnknownPackage")
                date_issued = item.get("dateIssued", "")
                
                # Skip non-2025 documents (client-side filtering)
                if not date_issued.startswith("2025"):
                    logger.debug(f"Skipping non-2025 document: {package_id} ({date_issued})")
                    continue
                
                # Check if already processed
                metadata_key = f"{S3_FOLDER_2025}/federal/metadata/{package_id}.json"
                pdf_key = f"{S3_FOLDER_2025}/federal/content/{package_id}.pdf"
                
                if exists_in_s3(metadata_key) and exists_in_s3(pdf_key):
                    logger.debug(f"Already processed: {package_id}")
                    continue
                
                metadata = {
                    "title": item.get("title", "No Title"),
                    "packageId": package_id,
                    "dateIssued": item.get("dateIssued", "UnknownDate"),
                    "keyword": keyword,
                    "source": "GovInfo 2025",
                    "type": "Federal Legislation",
                    "collection_date": datetime.now().isoformat()
                }
                
                # Save metadata
                if upload_to_s3_if_not_exists(
                    json.dumps(metadata, indent=2).encode("utf-8"),
                    metadata_key,
                    "application/json"
                ):
                    keyword_count += 1
                    progress_tracker.increment_documents()
                
                # Download PDF
                pdf_link = item.get("download", {}).get("pdfLink")
                if pdf_link and not exists_in_s3(pdf_key):
                    try:
                        pdf_resp = requests.get(pdf_link, params={"api_key": GOVINFO_API_KEY})
                        pdf_resp.raise_for_status()
                        
                        if upload_to_s3_if_not_exists(pdf_resp.content, pdf_key):
                            logger.info(f"✓ Downloaded PDF: {package_id}")
                        
                    except Exception as e:
                        logger.error(f"PDF download failed for {package_id}: {str(e)}")
                
                time.sleep(0.2)
            
            # Get next page
            offset_mark = data.get("offsetMark")
            if not offset_mark:
                logger.info(f"Completed all pages for '{keyword}' ({keyword_count} documents)")
                progress_tracker.mark_keyword_complete("federal_legislation", keyword)
                break
            
            logger.info(f"Progress: {keyword} - {keyword_count} documents so far...")
            time.sleep(1)
        
        total_processed += keyword_count
    
    logger.info(f"=== FEDERAL LEGISLATION: Complete ({total_processed} total documents) ===")

# -------------------------------------------------------------------------
# PART 2: FEDERAL REGISTER - COMPLETE PAGINATION
# -------------------------------------------------------------------------
def fetch_2025_federal_register():
    logger.info("=== FEDERAL REGISTER: Starting ===")
    
    api_url = "https://www.federalregister.gov/api/v1/documents"
    total_processed = 0
    
    for term in ENERGY_KEYWORDS:
        if progress_tracker.is_keyword_complete("federal_register", term):
            logger.info(f"Skipping completed term: {term}")
            continue
            
        logger.info(f"Processing term: {term}")
        term_count = 0
        page = 1
        
        while True:  # Keep going until no more results
            params = {
                'conditions[term]': term,
                'conditions[publication_date][gte]': '2025-01-01',
                'conditions[publication_date][lte]': '2025-12-31',
                'per_page': 100,
                'page': page,
                'order': 'relevance',
                'fields[]': ['title', 'pdf_url', 'html_url', 'publication_date', 'agencies', 'document_number']
            }
        
            try:
                response = requests.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Federal Register API error: {str(e)}")
                break
            
            results = data.get('results', [])
            if not results:
                logger.info(f"No more results for '{term}' (processed {term_count} documents)")
                progress_tracker.mark_keyword_complete("federal_register", term)
                break
            
            for result in results:
                pdf_url = result.get('pdf_url')
                if not pdf_url:
                    continue
                
                # Skip non-2025 documents (client-side filtering)
                pub_date = result.get('publication_date', '')
                if not pub_date.startswith('2025'):
                    logger.debug(f"Skipping non-2025 regulation: {pub_date}")
                    continue
                
                doc_number = result.get('document_number', '')
                doc_id = doc_number if doc_number else hashlib.md5(pdf_url.encode()).hexdigest()
                
                # Check if already processed
                metadata_key = f"{S3_FOLDER_2025}/regulations/metadata/{doc_id}.json"
                pdf_key = f"{S3_FOLDER_2025}/regulations/content/{doc_id}.pdf"
                
                if exists_in_s3(metadata_key) and exists_in_s3(pdf_key):
                    logger.debug(f"Already processed: {doc_id}")
                    continue
                
                metadata = {
                    'title': result.get('title', 'No Title'),
                    'document_number': doc_number,
                    'pdf_url': pdf_url,
                    'source': 'Federal Register 2025',
                    'type': 'Federal Regulation',
                    'publication_date': result.get('publication_date'),
                    'agencies': result.get('agencies', []),
                    'search_term': term,
                    'collection_date': datetime.now().isoformat()
                }
                
                # Save metadata
                if upload_to_s3_if_not_exists(
                    json.dumps(metadata, indent=2).encode("utf-8"),
                    metadata_key,
                    "application/json"
                ):
                    term_count += 1
                    progress_tracker.increment_documents()
                
                # Download PDF
                if not exists_in_s3(pdf_key):
                    try:
                        pdf_resp = requests.get(pdf_url)
                        pdf_resp.raise_for_status()
                        
                        if upload_to_s3_if_not_exists(pdf_resp.content, pdf_key):
                            logger.info(f"✓ Downloaded regulation: {doc_id}")
                        
                    except Exception as e:
                        logger.error(f"PDF download failed for {doc_id}: {str(e)}")
                
                time.sleep(0.5)
            
            # Check if more pages exist
            total_pages = data.get('total_pages', 1)
            if page >= total_pages:
                logger.info(f"Completed all {total_pages} pages for '{term}' ({term_count} documents)")
                progress_tracker.mark_keyword_complete("federal_register", term)
                break
            
            page += 1
            logger.info(f"Progress: {term} - page {page}/{total_pages} ({term_count} documents so far)")
            time.sleep(1)
        
        total_processed += term_count
    
    logger.info(f"=== FEDERAL REGISTER: Complete ({total_processed} total documents) ===")

# -------------------------------------------------------------------------
# PART 3: COURT CASES (CourtListener API)
# -------------------------------------------------------------------------
def fetch_2025_energy_court_cases():
    """Search for 2025 court cases related to energy/AI"""
    logger.info("=== 2025 COURT CASES: Starting search ===")
    
    base_url = "https://www.courtlistener.com/api/rest/v3/search/"
    queries = [
        "energy AND 2025",
        "artificial intelligence AND regulation",
        "blockchain AND cryptocurrency",
        "renewable energy AND policy",
        "electric vehicle AND infrastructure"
    ]
    
    headers = {
        "Authorization": f"Token {COURTLISTENER_API_KEY}",
        "Accept": "application/json",
        "User-Agent": "Policy2025Scraper/1.0"
    }
    
    for query in queries:
        logger.info(f"Searching court cases for: {query}")
        
        params = {
            "q": query,
            "page": 1,
            "page_size": 50,
            "filed_after": "2025-01-01"
        }
        
        while True:
            try:
                resp = requests.get(base_url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"CourtListener search failed: {str(e)}")
                break
            
            results = data.get("results", [])
            if not results:
                break
            
            logger.info(f"Found {len(results)} court cases for: {query}")
            
            for case in results:
                case_id = case.get("id") or hashlib.md5(str(case).encode()).hexdigest()
                
                case_metadata = {
                    **case,
                    "search_query": query,
                    "collection_date": datetime.now().isoformat(),
                    "source": "CourtListener 2025"
                }
                
                s3_key = f"{S3_FOLDER_2025}/court-cases/{case_id}.json"
                upload_to_s3_if_not_exists(
                    json.dumps(case_metadata, indent=2).encode("utf-8"),
                    s3_key,
                    "application/json"
                )
            
            # Pagination
            if not data.get("next"):
                break
            params["page"] += 1
            time.sleep(1)
    
    logger.info("=== 2025 COURT CASES: Complete ===")

# -------------------------------------------------------------------------
# PART 4: ENERGY RESEARCH PAPERS (arXiv + Semantic Scholar + Sci-Hub)
# -------------------------------------------------------------------------
def search_arxiv_energy_papers():
    """Search arXiv for energy economics and modeling papers"""
    logger.info("=== ARXIV ENERGY RESEARCH: Starting search ===")
    
    papers = []
    
    for term in RESEARCH_KEYWORDS:
        logger.info(f"Searching arXiv for: {term}")
        
        try:
            api_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{term}',
                'start': 0,
                'max_results': 20,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                
                pdf_link = None
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_link = link.get('href')
                        break
                
                if title_elem is not None and pdf_link:
                    paper = {
                        'title': title_elem.text.strip(),
                        'abstract': summary_elem.text.strip() if summary_elem is not None else '',
                        'pdf_url': pdf_link,
                        'source': 'arXiv',
                        'type': 'Research Paper',
                        'search_term': term
                    }
                    papers.append(paper)
            
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"arXiv search failed for {term}: {str(e)}")
    
    logger.info(f"Found {len(papers)} arXiv papers")
    return papers

def search_semantic_scholar_energy():
    """Search Semantic Scholar for open access energy papers"""
    logger.info("=== SEMANTIC SCHOLAR: Starting search ===")
    
    papers = []
    
    for term in RESEARCH_KEYWORDS[:15]:  # Limit to avoid rate limits
        logger.info(f"Searching Semantic Scholar for: {term}")
        
        try:
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': term,
                'limit': 25,
                'fields': 'title,abstract,year,openAccessPdf,authors',
                'yearFilter': '2020-2025'
            }
            
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for paper_data in data.get('data', []):
                open_access = paper_data.get('openAccessPdf')
                if open_access and open_access.get('url'):
                    paper = {
                        'title': paper_data.get('title', ''),
                        'abstract': paper_data.get('abstract', ''),
                        'pdf_url': open_access['url'],
                        'source': 'Semantic Scholar',
                        'type': 'Research Paper', 
                        'year': paper_data.get('year', ''),
                        'authors': [author.get('name', '') for author in paper_data.get('authors', [])],
                        'search_term': term
                    }
                    papers.append(paper)
            
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Semantic Scholar search failed for {term}: {str(e)}")
    
    logger.info(f"Found {len(papers)} Semantic Scholar papers")
    return papers

def download_and_upload_research_papers():
    """Download research papers with Sci-Hub fallback"""
    logger.info("=== DOWNLOADING RESEARCH PAPERS ===")
    
    # Get papers from multiple sources
    arxiv_papers = search_arxiv_energy_papers()
    semantic_papers = search_semantic_scholar_energy()
    
    all_papers = arxiv_papers + semantic_papers
    
    # Remove duplicates by title
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title_key = paper['title'].lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_papers.append(paper)
    
    logger.info(f"Total unique papers to download: {len(unique_papers)}")
    
    successful_downloads = 0
    scihub_successes = 0
    
    for i, paper in enumerate(unique_papers):
        logger.info(f"Processing paper {i+1}/{len(unique_papers)}: {paper['title'][:50]}...")
        
        # Download PDF with Sci-Hub fallback
        pdf_content = download_pdf_with_scihub_fallback(paper['pdf_url'], paper['title'])
        
        if not pdf_content:
            logger.warning(f"Failed to download: {paper['title'][:50]}...")
            continue
        
        # Check if Sci-Hub was used (simple heuristic)
        used_scihub = False
        if any(mirror.replace('https://', '') in paper['pdf_url'] for mirror in SCIHUB_MIRRORS):
            used_scihub = True
            scihub_successes += 1
        
        # Generate filename
        clean_title = re.sub(r'[^\w\s-]', '', paper['title'])
        clean_title = re.sub(r'\s+', '_', clean_title)[:50]
        content_hash = hashlib.md5(paper['pdf_url'].encode()).hexdigest()[:8]
        filename = f"{clean_title}_{content_hash}.pdf"
        
        # Create metadata
        metadata = {
            'title': paper['title'],
            'source': paper['source'],
            'type': paper['type'],
            'pdf_url': paper['pdf_url'],
            'search_term': paper.get('search_term', ''),
            'year': paper.get('year', ''),
            'authors': paper.get('authors', []),
            'scihub_used': used_scihub,
            'collection_date': datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_key = f"{S3_FOLDER_2025}/research/metadata/{filename}.json"
        upload_to_s3_if_not_exists(
            json.dumps(metadata, indent=2).encode("utf-8"),
            metadata_key,
            "application/json"
        )
        
        # Save PDF
        pdf_key = f"{S3_FOLDER_2025}/research/content/{filename}"
        upload_to_s3_if_not_exists(pdf_content, pdf_key)
        
        successful_downloads += 1
        source_note = " [Sci-Hub]" if used_scihub else ""
        logger.info(f"✓ Downloaded {successful_downloads}: {filename}{source_note}")
        
        time.sleep(1)
    
    logger.info(f"=== RESEARCH DOWNLOAD COMPLETE ===")
    logger.info(f"Total papers downloaded: {successful_downloads}")
    logger.info(f"Sci-Hub successes: {scihub_successes}")

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
def main():
    logger.info("🚀 Starting 2025 Policy & Energy Research Collection (COMPLETE VERSION)")
    logger.info(f"Target S3 location: s3://{S3_BUCKET_NAME}/{S3_FOLDER_2025}/")
    logger.info(f"Current progress: {progress_tracker.progress['total_documents']} documents")
    
    start_time = time.time()
    
    try:
        # Phase 1: Government documents
        logger.info("\n📋 Phase 1: Federal legislation...")
        fetch_2025_federal_legislation()
        
        logger.info("\n📋 Phase 2: Federal regulations...")
        fetch_2025_federal_register()
        
        logger.info("\n📋 Phase 3: Court cases...")
        fetch_2025_energy_court_cases()
        
        # Phase 2: Research papers
        logger.info("\n📋 Phase 4: Research papers...")
        download_and_upload_research_papers()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Collection interrupted by user")
        logger.info(f"Progress saved. Resume by running the script again.")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {str(e)}")
        raise
    finally:
        elapsed = time.time() - start_time
        logger.info(f"\n🎉 Collection session complete!")
        logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
        logger.info(f"📊 Total documents collected: {progress_tracker.progress['total_documents']}")
        logger.info(f"📁 Location: s3://{S3_BUCKET_NAME}/{S3_FOLDER_2025}/")
        
        if os.path.exists(PROGRESS_FILE):
            logger.info(f"💾 Progress saved to: {PROGRESS_FILE}")

if __name__ == "__main__":
    main()