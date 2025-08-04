#!/usr/bin/env python3
"""
Government Officials Roster Scraper
Target: Complete roster of all government officials with contact information
Scope: All branches at federal and state level
Following pattern from existing scrapers with comprehensive coverage
"""

import os
import re
import json
import time
import boto3
import logging
import requests
import hashlib
import yaml
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, quote
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("government_officials_scraper")

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

# S3 Configuration
S3_BUCKET_NAME = "policy-database"
S3_FOLDER_OFFICIALS = "government_officials_roster"

# Track progress
PROGRESS_FILE = "government_officials_progress.json"

# Government sources for official rosters
GOVERNMENT_SOURCES = {
    'federal': {
        'executive': [
            'https://www.whitehouse.gov/administration/',
            'https://www.whitehouse.gov/administration/cabinet/',
            'https://www.senate.gov/general/contact_information/senators_cfm.cfm',
            'https://clerk.house.gov/members',
            'https://www.supremecourt.gov/about/biographies.aspx'
        ],
        'congress_api': [
            'https://api.propublica.org/congress/v1/118/senate/members.json',
            'https://api.propublica.org/congress/v1/118/house/members.json'
        ],
        'congress_backup': [
            'https://www.govtrack.us/api/v2/person?limit=600',
            'https://raw.githubusercontent.com/unitedstates/congress-legislators/main/legislators-current.yaml'
        ],
        'agencies': [
            'https://www.usa.gov/agency-index',
            'https://www.gsa.gov/about-us/organization/federal-acquisition-service/office-of-systems-management/integrated-technology-services/government-contact-database'
        ]
    },
    'state': {
        'governors': [
            'https://www.nga.org/governors/',
            'https://ballotpedia.org/List_of_current_governors'
        ],
        'state_legislators': [
            'https://www.ncsl.org/about-state-legislatures/ncsl-state-legislatures-magazine',
            'https://ballotpedia.org/State_legislatures'
        ],
        'state_apis': {
            'california': 'https://leginfo.legislature.ca.gov/faces/memberSearch.xhtml',
            'texas': 'https://capitol.texas.gov/Members/Members.aspx',
            'florida': 'https://www.flsenate.gov/Senators/',
            'new_york': 'https://www.nysenate.gov/senators',
            'pennsylvania': 'https://www.legis.state.pa.us/cfdocs/legis/home/member_information/',
            'illinois': 'https://www.ilga.gov/senate/',
            'ohio': 'https://www.ohiosenate.gov/senators',
            'georgia': 'https://www.senate.ga.gov/senators/',
            'north_carolina': 'https://www.ncleg.gov/Members/MemberTable/Senate',
            'michigan': 'https://www.senate.michigan.gov/senators.html'
        }
    },
    'judicial': {
        'federal_courts': [
            'https://www.uscourts.gov/judges-judgeships/judicial-vacancies',
            'https://www.fjc.gov/history/judges'
        ],
        'state_courts': [
            'https://ballotpedia.org/State_supreme_courts'
        ]
    }
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
            "federal_executive": {"sources_completed": []},
            "federal_congress": {"sources_completed": []},
            "federal_agencies": {"sources_completed": []},
            "state_governors": {"sources_completed": []},
            "state_legislators": {"sources_completed": []},
            "judicial": {"sources_completed": []},
            "total_officials": 0,
            "last_updated": None
        }
    
    def save_progress(self):
        """Save current progress"""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def mark_source_complete(self, category, source_url):
        """Mark a source as completed"""
        if source_url not in self.progress[category]["sources_completed"]:
            self.progress[category]["sources_completed"].append(source_url)
            self.save_progress()
    
    def is_source_complete(self, category, source_url):
        """Check if source was already processed"""
        return source_url in self.progress[category].get("sources_completed", [])
    
    def increment_officials(self, count=1):
        """Increment total officials count"""
        self.progress["total_officials"] += count
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
            Prefix=S3_FOLDER_OFFICIALS + "/"
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

def sanitize_filename(key: str) -> str:
    parts = key.split("/")
    filename = parts[-1].strip()
    filename = filename.replace("\\", "_")
    filename = re.sub(r"\s+", "_", filename)
    filename = re.sub(r"[^\w\.-]", "_", filename)
    parts[-1] = filename
    return "/".join(parts)

def upload_to_s3_if_not_exists(file_content: bytes, s3_key: str, content_type: str = "application/json"):
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
# OFFICIAL EXTRACTION UTILITIES
# -------------------------------------------------------------------------
def extract_contact_info(text: str) -> Dict:
    """Extract contact information from text"""
    contact_info = {}
    
    # Email patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        contact_info['emails'] = emails
    
    # Phone patterns
    phone_patterns = [
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{3}-\d{3}-\d{4}',
        r'\(\d{3}\)\s?\d{3}-\d{4}'
    ]
    phones = []
    for pattern in phone_patterns:
        phones.extend(re.findall(pattern, text))
    if phones:
        contact_info['phones'] = list(set(phones))
    
    # Address patterns (basic)
    address_pattern = r'\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct),?\s*[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}'
    addresses = re.findall(address_pattern, text)
    if addresses:
        contact_info['addresses'] = addresses
    
    return contact_info

def standardize_official_info(name: str, title: str, contact_info: Dict, source: str, additional_info: Dict = None) -> Dict:
    """Standardize official information format"""
    official = {
        'name': name.strip(),
        'title': title.strip(),
        'contact_info': contact_info,
        'source': source,
        'collection_date': datetime.now().isoformat()
    }
    
    if additional_info:
        official.update(additional_info)
    
    return official

# -------------------------------------------------------------------------
# FEDERAL EXECUTIVE BRANCH
# -------------------------------------------------------------------------
def scrape_federal_executive():
    """Scrape federal executive branch officials"""
    logger.info("=== FEDERAL EXECUTIVE: Starting ===")
    total_processed = 0
    
    for source_url in GOVERNMENT_SOURCES['federal']['executive']:
        if progress_tracker.is_source_complete("federal_executive", source_url):
            logger.info(f"Skipping completed source: {source_url}")
            continue
            
        logger.info(f"Processing executive source: {source_url}")
        source_count = 0
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(source_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Different selectors for different sites
            if 'whitehouse.gov' in source_url:
                officials = soup.find_all(['div', 'article'], class_=re.compile(r'(bio|member|official|staff)', re.I))
            elif 'senate.gov' in source_url:
                officials = soup.find_all('tr') + soup.find_all('div', class_=re.compile(r'senator|member'))
            elif 'house.gov' in source_url:
                officials = soup.find_all(['div', 'li'], class_=re.compile(r'member|representative'))
            elif 'supremecourt.gov' in source_url:
                officials = soup.find_all(['div', 'section'], class_=re.compile(r'justice|bio'))
            else:
                # Generic approach
                officials = soup.find_all(['div', 'li', 'tr'], class_=re.compile(r'(official|member|bio|staff)', re.I))
            
            for official_element in officials:
                try:
                    # Extract name
                    name_element = official_element.find(['h1', 'h2', 'h3', 'h4', 'strong', 'b']) or \
                                 official_element.find('a', href=re.compile(r'(bio|profile|member)'))
                    
                    if not name_element:
                        continue
                    
                    name = name_element.get_text().strip()
                    if len(name) < 3 or len(name) > 100:  # Basic validation
                        continue
                    
                    # Extract title/position
                    title_element = official_element.find(text=re.compile(r'(Secretary|Director|Justice|Senator|Representative|President|Vice|Chief|Administrator)', re.I))
                    title = title_element.strip() if title_element else 'Official'
                    
                    # Extract contact information
                    contact_text = official_element.get_text()
                    contact_info = extract_contact_info(contact_text)
                    
                    # Look for additional links
                    bio_link = official_element.find('a', href=re.compile(r'(bio|profile|contact)'))
                    if bio_link:
                        bio_url = bio_link.get('href')
                        if bio_url and not bio_url.startswith('http'):
                            bio_url = urljoin(source_url, bio_url)
                        contact_info['bio_url'] = bio_url
                    
                    # Generate unique ID
                    official_id = hashlib.md5(f"{name}_{title}_{source_url}".encode()).hexdigest()
                    
                    # Check if already processed
                    official_key = f"{S3_FOLDER_OFFICIALS}/federal/executive/{official_id}.json"
                    
                    if exists_in_s3(official_key):
                        logger.debug(f"Already processed: {official_id}")
                        continue
                    
                    # Create standardized official record
                    official_data = standardize_official_info(
                        name=name,
                        title=title,
                        contact_info=contact_info,
                        source=f"Federal Executive - {source_url}",
                        additional_info={
                            'branch': 'Executive',
                            'level': 'Federal',
                            'source_url': source_url
                        }
                    )
                    
                    # Save official data
                    if upload_to_s3_if_not_exists(
                        json.dumps(official_data, indent=2).encode("utf-8"),
                        official_key,
                        "application/json"
                    ):
                        source_count += 1
                        progress_tracker.increment_officials()
                        logger.info(f"✓ Saved official: {name} - {title}")
                    
                except Exception as e:
                    logger.debug(f"Error processing official element: {str(e)}")
                    continue
            
            progress_tracker.mark_source_complete("federal_executive", source_url)
            total_processed += source_count
            logger.info(f"Completed source: {source_url} ({source_count} officials)")
            
        except Exception as e:
            logger.error(f"Error processing executive source {source_url}: {str(e)}")
        
        time.sleep(2)  # Rate limiting
    
    logger.info(f"=== FEDERAL EXECUTIVE: Complete ({total_processed} total officials) ===")

# -------------------------------------------------------------------------
# FEDERAL CONGRESS - API-BASED WITH PROPER STATE DATA
# -------------------------------------------------------------------------
def scrape_federal_congress():
    """Scrape federal congress officials using API sources with proper state data"""
    logger.info("=== FEDERAL CONGRESS: Starting with API sources ===")
    total_processed = 0
    
    # First try API sources that provide structured data with states
    api_sources = GOVERNMENT_SOURCES['federal']['congress_api'] + GOVERNMENT_SOURCES['federal']['congress_backup']
    
    for source_url in api_sources:
        if progress_tracker.is_source_complete("federal_congress", source_url):
            logger.info(f"Skipping completed source: {source_url}")
            continue
            
        logger.info(f"Processing congress API source: {source_url}")
        source_count = 0
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Handle ProPublica API (requires API key, try without first)
            if 'propublica.org' in source_url:
                # Skip ProPublica for now as it requires API key
                logger.info("Skipping ProPublica API (requires API key)")
                continue
            
            response = requests.get(source_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Handle YAML vs JSON based on URL
            if source_url.endswith('.yaml'):
                data = yaml.safe_load(response.content.decode('utf-8'))
            else:
                data = response.json()
            
            # Process based on API structure
            if 'govtrack.us' in source_url:
                # GovTrack API structure
                objects = data.get('objects', [])
                
                for person in objects:
                    try:
                        # Extract basic info
                        name = f"{person.get('firstname', '')} {person.get('lastname', '')}".strip()
                        if not name or len(name) < 3:
                            continue
                        
                        # Get current role
                        roles = person.get('roles', [])
                        current_role = None
                        for role in roles:
                            if role.get('current'):
                                current_role = role
                                break
                        
                        if not current_role:
                            continue
                        
                        # Extract state and chamber info
                        state = current_role.get('state', 'Unknown')
                        role_type = current_role.get('role_type', '')
                        
                        if role_type == 'senator':
                            title = 'Senator'
                            chamber = 'Senate'
                        elif role_type == 'representative':
                            title = 'Representative'
                            chamber = 'House'
                        else:
                            title = 'Member of Congress'
                            chamber = 'Congress'
                        
                        # Create contact info
                        contact_info = {
                            'govtrack_id': person.get('id', ''),
                            'bioguide_id': person.get('bioguideid', ''),
                            'thomas_id': person.get('thomasid', ''),
                            'website': person.get('website', ''),
                            'twitter': person.get('twitterid', '')
                        }
                        
                        # Generate unique ID
                        official_id = hashlib.md5(f"{name}_{title}_{state}".encode()).hexdigest()
                        
                        # Check if already processed
                        official_key = f"{S3_FOLDER_OFFICIALS}/federal/congress/{official_id}.json"
                        
                        if exists_in_s3(official_key):
                            logger.debug(f"Already processed: {official_id}")
                            continue
                        
                        # Create standardized official record
                        official_data = standardize_official_info(
                            name=name,
                            title=title,
                            contact_info=contact_info,
                            source=f"GovTrack API",
                            additional_info={
                                'branch': 'Legislative',
                                'level': 'Federal',
                                'chamber': chamber,
                                'state': state,
                                'district': current_role.get('district', ''),
                                'party': current_role.get('party', ''),
                                'start_date': current_role.get('startdate', ''),
                                'end_date': current_role.get('enddate', ''),
                                'source_url': source_url
                            }
                        )
                        
                        # Save official data
                        if upload_to_s3_if_not_exists(
                            json.dumps(official_data, indent=2).encode("utf-8"),
                            official_key,
                            "application/json"
                        ):
                            source_count += 1
                            progress_tracker.increment_officials()
                            logger.info(f"✓ Saved official: {name} - {title} ({state})")
                    
                    except Exception as e:
                        logger.debug(f"Error processing GovTrack person: {str(e)}")
                        continue
            
            elif 'github.com' in source_url or 'githubusercontent.com' in source_url:
                # United States project YAML structure
                if source_url.endswith('.yaml'):
                    legislators = yaml.safe_load(response.content.decode('utf-8'))
                else:
                    legislators = data if isinstance(data, list) else []
                
                for legislator in legislators:
                    try:
                        # Extract name
                        name_info = legislator.get('name', {})
                        name = f"{name_info.get('first', '')} {name_info.get('last', '')}".strip()
                        if not name or len(name) < 3:
                            continue
                        
                        # Get current terms
                        terms = legislator.get('terms', [])
                        if not terms:
                            continue
                        
                        # Get most recent term
                        current_term = terms[-1]
                        term_type = current_term.get('type', '')
                        state = current_term.get('state', 'Unknown')
                        
                        if term_type == 'sen':
                            title = 'Senator'
                            chamber = 'Senate'
                        elif term_type == 'rep':
                            title = 'Representative'
                            chamber = 'House'
                        else:
                            continue
                        
                        # Create contact info from IDs
                        bio = legislator.get('bio', {})
                        ids = legislator.get('id', {})
                        
                        contact_info = {
                            'bioguide_id': ids.get('bioguide', ''),
                            'thomas_id': ids.get('thomas', ''),
                            'govtrack_id': ids.get('govtrack', ''),
                            'birthday': bio.get('birthday', ''),
                            'gender': bio.get('gender', '')
                        }
                        
                        # Generate unique ID
                        official_id = hashlib.md5(f"{name}_{title}_{state}".encode()).hexdigest()
                        
                        # Check if already processed
                        official_key = f"{S3_FOLDER_OFFICIALS}/federal/congress/{official_id}.json"
                        
                        if exists_in_s3(official_key):
                            logger.debug(f"Already processed: {official_id}")
                            continue
                        
                        # Create standardized official record
                        official_data = standardize_official_info(
                            name=name,
                            title=title,
                            contact_info=contact_info,
                            source=f"United States Project",
                            additional_info={
                                'branch': 'Legislative',
                                'level': 'Federal',
                                'chamber': chamber,
                                'state': state,
                                'district': current_term.get('district', ''),
                                'party': current_term.get('party', ''),
                                'start_date': current_term.get('start', ''),
                                'end_date': current_term.get('end', ''),
                                'source_url': source_url
                            }
                        )
                        
                        # Save official data
                        if upload_to_s3_if_not_exists(
                            json.dumps(official_data, indent=2).encode("utf-8"),
                            official_key,
                            "application/json"
                        ):
                            source_count += 1
                            progress_tracker.increment_officials()
                            logger.info(f"✓ Saved official: {name} - {title} ({state})")
                    
                    except Exception as e:
                        logger.debug(f"Error processing United States legislator: {str(e)}")
                        continue
            
            progress_tracker.mark_source_complete("federal_congress", source_url)
            total_processed += source_count
            logger.info(f"Completed source: {source_url} ({source_count} officials)")
            
        except Exception as e:
            logger.error(f"Error processing congress API source {source_url}: {str(e)}")
        
        time.sleep(2)  # Rate limiting
    
    logger.info(f"=== FEDERAL CONGRESS: Complete ({total_processed} total officials) ===")

# -------------------------------------------------------------------------
# STATE OFFICIALS
# -------------------------------------------------------------------------
def scrape_state_governors():
    """Scrape state governors"""
    logger.info("=== STATE GOVERNORS: Starting ===")
    total_processed = 0
    
    for source_url in GOVERNMENT_SOURCES['state']['governors']:
        if progress_tracker.is_source_complete("state_governors", source_url):
            logger.info(f"Skipping completed source: {source_url}")
            continue
            
        logger.info(f"Processing governors source: {source_url}")
        source_count = 0
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(source_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for governor listings
            governor_selectors = [
                '.governor',
                '[data-state]',
                'tr',
                '.state-official',
                '.member'
            ]
            
            governors = []
            for selector in governor_selectors:
                found_governors = soup.select(selector)
                if found_governors and len(found_governors) >= 10:  # Likely a listing
                    governors = found_governors
                    break
            
            for governor_element in governors[:60]:  # Max 50 states + territories
                try:
                    # Extract name
                    name_element = governor_element.find(['h3', 'h4', 'strong', 'a'])
                    if not name_element:
                        continue
                    
                    name = name_element.get_text().strip()
                    if len(name) < 5 or len(name) > 50:
                        continue
                    
                    # Extract state
                    state_patterns = [
                        r'\b([A-Z][a-z]+ ?[A-Z]?[a-z]*)\b',  # State names
                        r'\b([A-Z]{2})\b'  # State codes
                    ]
                    
                    state = 'Unknown'
                    governor_text = governor_element.get_text()
                    for pattern in state_patterns:
                        state_match = re.search(pattern, governor_text)
                        if state_match:
                            state = state_match.group(1)
                            break
                    
                    # Extract contact information
                    contact_text = governor_element.get_text()
                    contact_info = extract_contact_info(contact_text)
                    
                    # Look for governor page link
                    gov_link = governor_element.find('a')
                    if gov_link:
                        gov_url = gov_link.get('href')
                        if gov_url and not gov_url.startswith('http'):
                            gov_url = urljoin(source_url, gov_url)
                        contact_info['official_page'] = gov_url
                    
                    # Generate unique ID
                    official_id = hashlib.md5(f"{name}_Governor_{state}".encode()).hexdigest()
                    
                    # Check if already processed
                    official_key = f"{S3_FOLDER_OFFICIALS}/state/governors/{official_id}.json"
                    
                    if exists_in_s3(official_key):
                        logger.debug(f"Already processed: {official_id}")
                        continue
                    
                    # Create standardized official record
                    official_data = standardize_official_info(
                        name=name,
                        title="Governor",
                        contact_info=contact_info,
                        source=f"State Governors - {source_url}",
                        additional_info={
                            'branch': 'Executive',
                            'level': 'State',
                            'state': state,
                            'source_url': source_url
                        }
                    )
                    
                    # Save official data
                    if upload_to_s3_if_not_exists(
                        json.dumps(official_data, indent=2).encode("utf-8"),
                        official_key,
                        "application/json"
                    ):
                        source_count += 1
                        progress_tracker.increment_officials()
                        logger.info(f"✓ Saved governor: {name} - {state}")
                    
                except Exception as e:
                    logger.debug(f"Error processing governor: {str(e)}")
                    continue
            
            progress_tracker.mark_source_complete("state_governors", source_url)
            total_processed += source_count
            logger.info(f"Completed source: {source_url} ({source_count} governors)")
            
        except Exception as e:
            logger.error(f"Error processing governors source {source_url}: {str(e)}")
        
        time.sleep(2)  # Rate limiting
    
    logger.info(f"=== STATE GOVERNORS: Complete ({total_processed} total governors) ===")

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
def main():
    logger.info("🚀 Starting Government Officials Roster Collection")
    logger.info(f"Target S3 location: s3://{S3_BUCKET_NAME}/{S3_FOLDER_OFFICIALS}/")
    logger.info(f"Current progress: {progress_tracker.progress['total_officials']} officials")
    
    start_time = time.time()
    
    try:
        # Phase 1: Federal Executive Branch
        logger.info("\n🏛️ Phase 1: Federal executive branch...")
        scrape_federal_executive()
        
        # Phase 2: Federal Congress
        logger.info("\n🏛️ Phase 2: Federal congress...")
        scrape_federal_congress()
        
        # Phase 3: State Governors
        logger.info("\n🏛️ Phase 3: State governors...")
        scrape_state_governors()
        
        # TODO: Add more phases for state legislators, federal agencies, judicial branch
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Collection interrupted by user")
        logger.info(f"Progress saved. Resume by running the script again.")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {str(e)}")
        raise
    finally:
        elapsed = time.time() - start_time
        logger.info(f"\n🎉 Government officials collection session complete!")
        logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
        logger.info(f"📊 Total officials collected: {progress_tracker.progress['total_officials']}")
        logger.info(f"📁 Location: s3://{S3_BUCKET_NAME}/{S3_FOLDER_OFFICIALS}/")
        
        if os.path.exists(PROGRESS_FILE):
            logger.info(f"💾 Progress saved to: {PROGRESS_FILE}")

if __name__ == "__main__":
    main()