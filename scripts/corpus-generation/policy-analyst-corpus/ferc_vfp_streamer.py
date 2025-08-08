#!/usr/bin/env python3
"""
FERC Form 1 VFP Data Streaming Pipeline
Streams Visual FoxPro database files from FERC into policy-database structure
"""

import os
import sys
import json
import time
import logging
import requests
import boto3
import zipfile
import tempfile
import hashlib
from datetime import datetime
from urllib.parse import urljoin
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from botocore.exceptions import ClientError

try:
    from dbfread import DBF
except ImportError:
    print("Installing required dbfread library...")
    os.system("pip install dbfread")
    from dbfread import DBF

# Configuration
S3_BUCKET_NAME = "policy-database"
FERC_BASE_PATH = "usa/federal-legislation-executive-courts/federal-agencies/ferc-nerc-cip-epa-other-agencies"
FERC_URL = "https://www.ferc.gov/general-information-0/electric-industry-forms/form-1-1-f-3-q-electric-historical-vfp-data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3", region_name="us-east-1")

class FERCDataStreamer:
    """Streams FERC Form 1 VFP data into policy database"""
    
    def __init__(self, bucket_name: str = S3_BUCKET_NAME):
        self.bucket_name = bucket_name
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def discover_vfp_files(self) -> List[Dict[str, str]]:
        """Get known FERC VFP files from 1994-2021"""
        vfp_files = []
        
        # Known FERC Form 1 files from the table
        years = list(range(1994, 2022))  # 1994-2021
        
        for year in years:
            file_info = {
                'url': f'https://forms.ferc.gov/f1allyears/f1_{year}.zip',
                'filename': f'f1_{year}.zip',
                'description': f'FERC Form 1 {year} All Respondents VFP Database',
                'year': str(year)
            }
            vfp_files.append(file_info)
        
        logger.info(f"Prepared {len(vfp_files)} VFP files (1994-2021)")
        return vfp_files
    
    def _extract_year(self, text: str) -> Optional[str]:
        """Extract year from file description"""
        import re
        year_match = re.search(r'20\d{2}', text)
        return year_match.group(0) if year_match else None
    
    def stream_vfp_file(self, file_info: Dict[str, str]) -> Iterator[Dict]:
        """Stream and process a single VFP file"""
        logger.info(f"Processing VFP file: {file_info['filename']}")
        
        # Download zip file
        try:
            response = self.session.get(file_info['url'], stream=True)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile() as temp_zip:
                # Stream download to avoid memory issues
                for chunk in response.iter_content(chunk_size=8192):
                    temp_zip.write(chunk)
                temp_zip.flush()
                
                # Extract and process VFP files
                yield from self._process_zip_file(temp_zip.name, file_info)
                
        except Exception as e:
            logger.error(f"Failed to stream {file_info['filename']}: {str(e)}")
    
    def _process_zip_file(self, zip_path: str, file_info: Dict[str, str]) -> Iterator[Dict]:
        """Extract and process VFP files from zip archive"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            dbf_files = [f for f in zip_ref.namelist() if f.lower().endswith('.dbf')]
            
            logger.info(f"Found {len(dbf_files)} DBF files in {file_info['filename']}")
            
            for dbf_filename in dbf_files:
                try:
                    # Extract DBF file to temp location
                    with tempfile.NamedTemporaryFile(suffix='.dbf') as temp_dbf:
                        with zip_ref.open(dbf_filename) as dbf_in_zip:
                            temp_dbf.write(dbf_in_zip.read())
                        temp_dbf.flush()
                        
                        # Process DBF file
                        yield from self._process_dbf_file(
                            temp_dbf.name, 
                            dbf_filename, 
                            file_info
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to process {dbf_filename}: {str(e)}")
                    continue
    
    def _process_dbf_file(self, dbf_path: str, dbf_filename: str, file_info: Dict[str, str]) -> Iterator[Dict]:
        """Process individual DBF file and yield records"""
        try:
            table = DBF(dbf_path, encoding='latin-1')
            
            # Generate metadata
            metadata = {
                'source_file': file_info['filename'],
                'dbf_table': dbf_filename,
                'year': file_info.get('year'),
                'description': file_info.get('description'),
                'source': 'FERC Form 1 VFP Data',
                'field_names': table.field_names,
                'record_count': len(table),
                'collection_date': datetime.now().isoformat()
            }
            
            # Upload metadata
            metadata_key = f"{FERC_BASE_PATH}/form-1/metadata/{dbf_filename}.json"
            self._upload_to_s3(
                json.dumps(metadata, indent=2).encode('utf-8'),
                metadata_key,
                'application/json'
            )
            
            # Stream records in batches
            batch_size = 1000
            batch = []
            batch_num = 0
            
            for record in table:
                # Convert record to JSON-serializable format
                clean_record = {}
                for key, value in record.items():
                    if value is not None:
                        clean_record[key] = str(value)
                
                batch.append(clean_record)
                
                if len(batch) >= batch_size:
                    yield from self._upload_batch(batch, dbf_filename, batch_num, file_info)
                    batch = []
                    batch_num += 1
            
            # Upload remaining records
            if batch:
                yield from self._upload_batch(batch, dbf_filename, batch_num, file_info)
                
            logger.info(f"Completed processing {dbf_filename}: {len(table)} records")
            
        except Exception as e:
            logger.error(f"Failed to process DBF {dbf_filename}: {str(e)}")
    
    def _upload_batch(self, batch: List[Dict], dbf_filename: str, batch_num: int, file_info: Dict[str, str]) -> Iterator[Dict]:
        """Upload a batch of records to S3"""
        try:
            batch_data = {
                'source_info': file_info,
                'dbf_table': dbf_filename,
                'batch_number': batch_num,
                'records': batch,
                'upload_timestamp': datetime.now().isoformat()
            }
            
            # Create S3 key for this batch
            safe_filename = dbf_filename.replace('.dbf', '').replace('.DBF', '')
            batch_key = f"{FERC_BASE_PATH}/form-1/data/{safe_filename}/batch_{batch_num:06d}.json"
            
            success = self._upload_to_s3(
                json.dumps(batch_data, indent=2).encode('utf-8'),
                batch_key,
                'application/json'
            )
            
            if success:
                yield {
                    'batch_key': batch_key,
                    'record_count': len(batch),
                    'dbf_table': dbf_filename
                }
                
        except Exception as e:
            logger.error(f"Failed to upload batch {batch_num} for {dbf_filename}: {str(e)}")
    
    def _upload_to_s3(self, content: bytes, s3_key: str, content_type: str) -> bool:
        """Upload content to S3 with existence check"""
        try:
            # Check if already exists
            try:
                s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                logger.debug(f"Already exists: {s3_key}")
                return False
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
            
            # Upload new content
            s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType=content_type
            )
            logger.info(f"✓ Uploaded: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"S3 upload failed for {s3_key}: {str(e)}")
            return False
    
    def stream_all_ferc_data(self):
        """Main method to stream all FERC VFP data"""
        logger.info("🚀 Starting FERC Form 1 VFP data streaming")
        
        # Discover available files
        vfp_files = self.discover_vfp_files()
        if not vfp_files:
            logger.warning("No VFP files discovered - check FERC website access")
            return
        
        total_processed = 0
        start_time = time.time()
        
        try:
            for file_info in vfp_files:
                logger.info(f"Processing: {file_info['description']}")
                
                for batch_result in self.stream_vfp_file(file_info):
                    total_processed += batch_result['record_count']
                    logger.info(f"Progress: {total_processed} records processed")
                
                time.sleep(1)  # Rate limiting
                
        except KeyboardInterrupt:
            logger.info("⚠️ Streaming interrupted by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {str(e)}")
            raise
        finally:
            elapsed = time.time() - start_time
            logger.info(f"🎉 FERC streaming complete!")
            logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
            logger.info(f"📊 Total records: {total_processed}")
            logger.info(f"📁 Location: s3://{S3_BUCKET_NAME}/{FERC_BASE_PATH}/form-1/")

def main():
    """Main entry point"""
    streamer = FERCDataStreamer()
    streamer.stream_all_ferc_data()

if __name__ == "__main__":
    main()