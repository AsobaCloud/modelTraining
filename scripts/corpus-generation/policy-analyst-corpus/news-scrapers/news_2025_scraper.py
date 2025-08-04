#!/usr/bin/env python3
"""
2025 News Scraper
Target: All news available from 2025 on energy/AI/blockchain topics with full article content
Destination: s3://policy-database/corpus_news_7-27-2025-ytd
Following pattern from existing scrapers with aggressive fallback strategies
"""

import os
import re
import json
import time
import boto3
import logging
import requests
import hashlib
from datetime import datetime, date
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, quote
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("news_2025_scraper")

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

# S3 Configuration
S3_BUCKET_NAME = "policy-database"
# Generate datestamped folder name
today = datetime.now().strftime("%Y-%m-%d")
S3_FOLDER_NEWS = f"news/{today}"

# Track progress
PROGRESS_FILE = "news_2025_progress.json"

# Search keywords - comprehensive energy, AI, blockchain, and finance terms
NEWS_KEYWORDS = [
    # Core topics
    "energy", "electricity", "blockchain", "artificial intelligence", "AI", "insurance",
    
    # Energy technologies
    "renewable energy", "solar power", "wind energy", "battery storage",
    "smart grid", "microgrid", "electric vehicles", "capacity market",
    "demand response", "carbon pricing", "carbon tax", "feed-in tariff",
    "grid reliability", "transmission planning", "levelized cost of energy", 
    "power purchase agreement", "green bond", "ESG investment",
    
    # Insurance/Risk
    "catastrophe modeling", "exposure data", "reinsurance", "underwriting", 
    "climate risk",
    
    # Technology
    "cybersecurity", "digital twin", "predictive analytics",
    
    # Major agencies and regulatory bodies
    "Federal Energy Regulatory Commission", "FERC",
    "North American Electric Reliability Corporation", "NERC",
    "Department of Energy", "DOE",
    "Environmental Protection Agency", "EPA",
    "National Renewable Energy Laboratory", "NREL",
    "International Energy Agency", "IEA",
    "Commodity Futures Trading Commission", "CFTC",
    "Insurance Regulatory and Development Authority", "IRDAI",
    "Standard & Poor's", "Moody's", "Fitch",
    "Bloomberg", "Reuters"
]

# News sources for 2025 content - Actually working RSS feeds
NEWS_SOURCES = {
    'rss_feeds': [
        # BBC News (confirmed working)
        'https://feeds.bbci.co.uk/news/rss.xml',
        'https://feeds.bbci.co.uk/news/world/rss.xml',
        'https://feeds.bbci.co.uk/news/business/rss.xml',
        'https://feeds.bbci.co.uk/news/technology/rss.xml',
        'https://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
        
        # CNN (trying common working formats)
        'http://rss.cnn.com/rss/cnn_topstories.rss',
        'http://rss.cnn.com/rss/edition.rss',
        'http://rss.cnn.com/rss/cnn_world.rss',
        
        # Guardian (confirmed working)
        'https://www.theguardian.com/world/rss',
        'https://www.theguardian.com/business/rss',
        'https://www.theguardian.com/technology/rss',
        'https://www.theguardian.com/environment/rss',
        
        # Al Jazeera
        'https://www.aljazeera.com/xml/rss/all.xml',
        
        # Financial/Business
        'https://www.marketwatch.com/rss/topstories',
        'https://feeds.finance.yahoo.com/rss/2.0/headline',
        
        # Tech/Industry
        'https://feeds.arstechnica.com/arstechnica/index',
        'https://techcrunch.com/feed/',
        'https://www.wired.com/feed/rss',
        'https://feeds.feedburner.com/venturebeat/SZYF',
        
        # Academic/Research
        'https://rss.arxiv.org/rss/econ',
        'https://rss.arxiv.org/rss/cs.AI',
        'https://rss.arxiv.org/rss/cs.CL',
        
        # Energy/Policy focused
        'https://www.energy.gov/rss/all.xml',
        'https://www.whitehouse.gov/feed/',
        
        # International
        'https://feeds.cfr.org/feeds/site/current.xml'
    ],
    'news_apis': [
        'https://newsapi.org/v2/everything',  # Requires API key
        'https://api.nytimes.com/svc/search/v2/articlesearch.json'  # Requires API key
    ],
    'direct_scraping': [
        'https://www.reuters.com/technology/',
        'https://www.reuters.com/business/energy/',
        'https://techcrunch.com/',
        'https://www.theverge.com/',
        'https://arstechnica.com/',
        'https://www.wired.com/',
        'https://www.coindesk.com/',
        'https://cointelegraph.com/'
    ]
}

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
            "rss_feeds": {"feeds_completed": []},
            "direct_scraping": {"sources_completed": []},
            "total_articles": 0,
            "last_updated": None
        }
    
    def save_progress(self):
        """Save current progress"""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def mark_feed_complete(self, feed_url):
        """Mark a feed as completed"""
        if feed_url not in self.progress["rss_feeds"]["feeds_completed"]:
            self.progress["rss_feeds"]["feeds_completed"].append(feed_url)
            self.save_progress()
    
    def is_feed_complete(self, feed_url):
        """Check if feed was already processed"""
        return feed_url in self.progress["rss_feeds"].get("feeds_completed", [])
    
    def mark_source_complete(self, source_url):
        """Mark a source as completed"""
        if source_url not in self.progress["direct_scraping"]["sources_completed"]:
            self.progress["direct_scraping"]["sources_completed"].append(source_url)
            self.save_progress()
    
    def is_source_complete(self, source_url):
        """Check if source was already processed"""
        return source_url in self.progress["direct_scraping"].get("sources_completed", [])
    
    def increment_articles(self, count=1):
        """Increment total article count"""
        self.progress["total_articles"] += count
        self.save_progress()

progress_tracker = ProgressTracker()

# -------------------------------------------------------------------------
# IDEMPOTENT S3 OPERATIONS
# -------------------------------------------------------------------------
def get_s3_manifest():
    """Get manifest of all files already in S3 bucket/prefix"""
    manifest = set()
    article_urls = set()  # Track URLs we've already processed
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=S3_BUCKET_NAME,
            Prefix=S3_FOLDER_NEWS + "/"
        )
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    manifest.add(obj['Key'])
                    
                    # Extract URLs from metadata files for URL-based deduplication
                    if obj['Key'].endswith('.json') and '/metadata/' in obj['Key']:
                        try:
                            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=obj['Key'])
                            metadata = json.loads(response['Body'].read().decode('utf-8'))
                            if 'url' in metadata:
                                article_urls.add(metadata['url'])
                        except Exception as e:
                            logger.debug(f"Could not extract URL from {obj['Key']}: {e}")
        
        logger.info(f"S3 manifest loaded: {len(manifest)} existing files, {len(article_urls)} unique article URLs")
        return manifest, article_urls
    except Exception as e:
        logger.error(f"Error loading S3 manifest: {str(e)}")
        return set(), set()

# Global manifest for idempotency
S3_MANIFEST, S3_PROCESSED_URLS = get_s3_manifest()

def exists_in_s3(key: str) -> bool:
    """Check if file exists in S3 using manifest"""
    return key in S3_MANIFEST

def url_already_processed(url: str) -> bool:
    """Check if URL was already processed (idempotency across runs)"""
    return url in S3_PROCESSED_URLS

def add_processed_url(url: str):
    """Add URL to processed set"""
    S3_PROCESSED_URLS.add(url)

def sanitize_filename(key: str) -> str:
    parts = key.split("/")
    filename = parts[-1].strip()
    filename = filename.replace("\\", "_")
    filename = re.sub(r"\s+", "_", filename)
    filename = re.sub(r"[^\w\.-]", "_", filename)
    parts[-1] = filename
    return "/".join(parts)

def upload_to_s3_if_not_exists(file_content: bytes, s3_key: str, content_type: str = "text/html"):
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

# -------------------------------------------------------------------------
# NEWS EXTRACTION UTILITIES
# -------------------------------------------------------------------------
def try_archive_fallback(url: str) -> Optional[str]:
    """Try to get article content from archive.is"""
    try:
        # Try to find existing archive
        archive_search_url = f"https://archive.today/newest/{url}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(archive_search_url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for archived version link
            archive_links = soup.find_all('a', href=re.compile(r'archive\.today|archive\.is'))
            if archive_links:
                archive_url = archive_links[0]['href']
                if not archive_url.startswith('http'):
                    archive_url = 'https://archive.today' + archive_url
                
                logger.info(f"Found archive version: {archive_url}")
                return extract_full_article_content(archive_url)
        
        # If no existing archive, try to create one
        logger.info(f"Attempting to archive: {url}")
        archive_create_url = "https://archive.today/submit/"
        data = {'url': url}
        
        response = requests.post(archive_create_url, data=data, headers=headers, timeout=60)
        if response.status_code == 200:
            # Archive creation initiated, but we won't wait for completion
            logger.info(f"Archive creation initiated for: {url}")
        
        return None
        
    except Exception as e:
        logger.debug(f"Archive.is fallback failed for {url}: {str(e)}")
        return None

def is_2025_article(article_date_str: str) -> bool:
    """Check if article is from 2025 or recent (since we might not have exact dates)"""
    if not article_date_str:
        # If no date, assume it's recent and include it
        return True
    
    # Try to extract year from date string
    year_match = re.search(r'202[5-9]', article_date_str)
    if year_match:
        year = int(year_match.group())
        return year >= 2025
    
    # Look for 2025 indicators in various formats
    if '2025' in article_date_str:
        return True
    
    # If we can't determine the year, assume it's recent and include it
    return True

def extract_full_article_content(url: str) -> Optional[str]:
    """Extract full article content from URL with archive.is fallback"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'ads']):
            element.decompose()
        
        # Try multiple selectors for article content
        content_selectors = [
            'article',
            '[data-module="ArticleBody"]',
            '.article-body',
            '.story-body',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '.article-content'
        ]
        
        article_content = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                article_content = content_element.get_text(strip=True)
                if len(article_content) > 200:  # Ensure we got substantial content
                    break
        
        if not article_content:
            # Fallback: get all paragraph text
            paragraphs = soup.find_all('p')
            article_content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        return article_content if len(article_content) > 100 else None
        
    except Exception as e:
        logger.debug(f"Direct extraction failed for {url}: {str(e)}")
        
        # Try archive.is fallback
        logger.info(f"Trying archive.is fallback for: {url}")
        return try_archive_fallback(url)

def matches_keywords(text: str) -> bool:
    """Check if text contains any of our keywords"""
    if not text:
        return False
    
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in NEWS_KEYWORDS)

# -------------------------------------------------------------------------
# RSS FEED PROCESSING
# -------------------------------------------------------------------------
def process_rss_feeds():
    """Process RSS feeds for 2025 news articles"""
    logger.info("=== RSS FEEDS: Starting ===")
    total_processed = 0
    
    for feed_url in NEWS_SOURCES['rss_feeds']:
        if progress_tracker.is_feed_complete(feed_url):
            logger.info(f"Skipping completed feed: {feed_url}")
            continue
            
        logger.info(f"Processing RSS feed: {feed_url}")
        feed_count = 0
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(feed_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Try different parsing methods
            soup = None
            items = []
            
            # Method 1: XML parser
            try:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                if not items:
                    items = soup.find_all('entry')  # Atom feeds
            except:
                pass
            
            # Method 2: HTML parser fallback
            if not items:
                try:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    items = soup.find_all('item')
                    if not items:
                        items = soup.find_all('entry')  # Atom feeds
                except:
                    pass
            
            # Method 3: lxml parser fallback
            if not items:
                try:
                    soup = BeautifulSoup(response.content, 'lxml-xml')
                    items = soup.find_all('item')
                    if not items:
                        items = soup.find_all('entry')  # Atom feeds
                except:
                    pass
            
            for item in items:
                try:
                    title = item.find('title').get_text() if item.find('title') else 'No Title'
                    
                    # Handle different link formats (RSS vs Atom)
                    link = None
                    if item.find('link'):
                        link_elem = item.find('link')
                        if link_elem.get('href'):  # Atom format
                            link = link_elem.get('href')
                        else:  # RSS format
                            link = link_elem.get_text()
                    
                    # Handle different date formats
                    pub_date = ''
                    if item.find('pubDate'):
                        pub_date = item.find('pubDate').get_text()
                    elif item.find('published'):  # Atom format
                        pub_date = item.find('published').get_text()
                    elif item.find('updated'):  # Atom format
                        pub_date = item.find('updated').get_text()
                    
                    # Handle different description formats
                    description = ''
                    if item.find('description'):
                        description = item.find('description').get_text()
                    elif item.find('summary'):  # Atom format
                        description = item.find('summary').get_text()
                    elif item.find('content'):  # Atom format
                        description = item.find('content').get_text()
                    
                    if not link:
                        continue
                    
                    # Check for URL-based deduplication first (fastest check)
                    if url_already_processed(link):
                        logger.debug(f"URL already processed: {link}")
                        continue
                    
                    # Check if 2025 article - for debugging let's see what we're filtering
                    if not is_2025_article(pub_date):
                        logger.debug(f"Filtering out non-2025 article: {title[:50]}... (date: {pub_date})")
                        continue
                    
                    # Check if matches keywords
                    combined_text = title + ' ' + description
                    if not matches_keywords(combined_text):
                        logger.debug(f"Filtering out article (no keywords): {title[:50]}... (text: {combined_text[:100]}...)")
                        continue
                    
                    # Generate unique ID
                    article_id = hashlib.md5(link.encode()).hexdigest()
                    
                    # Check if already processed by file existence (backup check)
                    metadata_key = f"{S3_FOLDER_NEWS}/rss/metadata/{article_id}.json"
                    content_key = f"{S3_FOLDER_NEWS}/rss/content/{article_id}.html"
                    
                    if exists_in_s3(metadata_key) and exists_in_s3(content_key):
                        logger.debug(f"Already processed by file check: {article_id}")
                        add_processed_url(link)  # Update our URL cache
                        continue
                    
                    # Extract full article content
                    full_content = extract_full_article_content(link)
                    if not full_content:
                        logger.warning(f"Could not extract content from: {link}")
                        continue
                    
                    # Create metadata
                    metadata = {
                        'title': title,
                        'url': link,
                        'pub_date': pub_date,
                        'description': description,
                        'source': 'RSS Feed',
                        'feed_url': feed_url,
                        'content_length': len(full_content),
                        'collection_date': datetime.now().isoformat()
                    }
                    
                    # Save metadata
                    if upload_to_s3_if_not_exists(
                        json.dumps(metadata, indent=2).encode("utf-8"),
                        metadata_key,
                        "application/json"
                    ):
                        # Save full content
                        if upload_to_s3_if_not_exists(full_content.encode('utf-8'), content_key):
                            feed_count += 1
                            progress_tracker.increment_articles()
                            add_processed_url(link)  # Track URL for future idempotency
                            logger.info(f"✓ Saved article: {title[:50]}...")
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Error processing RSS item: {str(e)}")
                    continue
            
            progress_tracker.mark_feed_complete(feed_url)
            total_processed += feed_count
            logger.info(f"Completed feed: {feed_url} ({feed_count} articles)")
            
        except Exception as e:
            logger.error(f"Error processing RSS feed {feed_url}: {str(e)}")
        
        time.sleep(2)  # Rate limiting between feeds
    
    logger.info(f"=== RSS FEEDS: Complete ({total_processed} total articles) ===")

# -------------------------------------------------------------------------
# DIRECT WEBSITE SCRAPING
# -------------------------------------------------------------------------
def scrape_website_articles(base_url: str, max_articles: int = 50):
    """Scrape articles directly from news websites"""
    if progress_tracker.is_source_complete(base_url):
        logger.info(f"Skipping completed source: {base_url}")
        return 0
        
    logger.info(f"Scraping website: {base_url}")
    articles_found = 0
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(base_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links
        article_selectors = [
            'a[href*="/article/"]',
            'a[href*="/news/"]', 
            'a[href*="/story/"]',
            'a[href*="/post/"]',
            'a[href*="/blog/"]',
            '.article-link a',
            '.story-link a',
            '.headline a',
            'h1 a', 'h2 a', 'h3 a'
        ]
        
        article_links = set()
        for selector in article_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    if href.startswith('/'):
                        href = urljoin(base_url, href)
                    if href.startswith('http') and len(href) > 10:
                        article_links.add(href)
        
        logger.info(f"Found {len(article_links)} potential articles on {base_url}")
        
        for article_url in list(article_links)[:max_articles]:
            try:
                # Check for URL-based deduplication first (fastest check)
                if url_already_processed(article_url):
                    logger.debug(f"URL already processed: {article_url}")
                    continue
                
                # Generate unique ID
                article_id = hashlib.md5(article_url.encode()).hexdigest()
                
                # Check if already processed by file existence (backup check)
                metadata_key = f"{S3_FOLDER_NEWS}/direct/metadata/{article_id}.json"
                content_key = f"{S3_FOLDER_NEWS}/direct/content/{article_id}.html"
                
                if exists_in_s3(metadata_key) and exists_in_s3(content_key):
                    logger.debug(f"Already processed by file check: {article_id}")
                    add_processed_url(article_url)  # Update our URL cache
                    continue
                
                # Get article page
                article_response = requests.get(article_url, headers=headers, timeout=30)
                article_response.raise_for_status()
                
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                
                # Extract title
                title_element = article_soup.find('title') or article_soup.find('h1')
                title = title_element.get_text().strip() if title_element else 'No Title'
                
                # Check if matches keywords
                if not matches_keywords(title):
                    continue
                
                # Extract date (try multiple selectors)
                date_selectors = [
                    '[datetime]',
                    '.publish-date',
                    '.article-date', 
                    '.post-date',
                    'time'
                ]
                
                article_date = None
                for selector in date_selectors:
                    date_element = article_soup.select_one(selector)
                    if date_element:
                        article_date = date_element.get('datetime') or date_element.get_text()
                        break
                
                # Check if 2025 article
                if article_date and not is_2025_article(article_date):
                    continue
                
                # Extract full content
                full_content = extract_full_article_content(article_url)
                if not full_content:
                    continue
                
                # Check content for keywords too
                if not matches_keywords(full_content):
                    continue
                
                # Create metadata
                metadata = {
                    'title': title,
                    'url': article_url,
                    'date': article_date or 'Unknown',
                    'source': 'Direct Scraping',
                    'base_url': base_url,
                    'content_length': len(full_content),
                    'collection_date': datetime.now().isoformat()
                }
                
                # Save metadata
                if upload_to_s3_if_not_exists(
                    json.dumps(metadata, indent=2).encode("utf-8"),
                    metadata_key,
                    "application/json"
                ):
                    # Save full content
                    if upload_to_s3_if_not_exists(full_content.encode('utf-8'), content_key):
                        articles_found += 1
                        progress_tracker.increment_articles()
                        add_processed_url(article_url)  # Track URL for future idempotency
                        logger.info(f"✓ Scraped article: {title[:50]}...")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"Error scraping article {article_url}: {str(e)}")
                continue
        
        progress_tracker.mark_source_complete(base_url)
        
    except Exception as e:
        logger.error(f"Error scraping website {base_url}: {str(e)}")
        
    return articles_found

def process_direct_scraping():
    """Process direct website scraping"""
    logger.info("=== DIRECT SCRAPING: Starting ===")
    total_processed = 0
    
    for source_url in NEWS_SOURCES['direct_scraping']:
        articles_count = scrape_website_articles(source_url)
        total_processed += articles_count
        logger.info(f"Completed source: {source_url} ({articles_count} articles)")
        time.sleep(3)  # Rate limiting between sites
    
    logger.info(f"=== DIRECT SCRAPING: Complete ({total_processed} total articles) ===")

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
def main():
    logger.info("🚀 Starting 2025 News Collection")
    logger.info(f"Target S3 location: s3://{S3_BUCKET_NAME}/{S3_FOLDER_NEWS}/")
    logger.info(f"Current progress: {progress_tracker.progress['total_articles']} articles")
    
    start_time = time.time()
    
    try:
        # Phase 1: RSS Feeds
        logger.info("\n📰 Phase 1: RSS feeds...")
        process_rss_feeds()
        
        # Phase 2: Direct scraping
        logger.info("\n🌐 Phase 2: Direct website scraping...")
        process_direct_scraping()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Collection interrupted by user")
        logger.info(f"Progress saved. Resume by running the script again.")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {str(e)}")
        raise
    finally:
        elapsed = time.time() - start_time
        logger.info(f"\n🎉 News collection session complete!")
        logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
        logger.info(f"📊 Total articles collected: {progress_tracker.progress['total_articles']}")
        logger.info(f"📁 Location: s3://{S3_BUCKET_NAME}/{S3_FOLDER_NEWS}/")
        
        if os.path.exists(PROGRESS_FILE):
            logger.info(f"💾 Progress saved to: {PROGRESS_FILE}")

if __name__ == "__main__":
    main()