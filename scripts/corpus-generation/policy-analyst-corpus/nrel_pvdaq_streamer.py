#!/usr/bin/env python3
"""
NREL PVDAQ Data Streaming Pipeline
Streams photovoltaic data from NREL's public OEDI data lake into policy-database
"""

import os
import sys
import json
import time
import logging
import boto3
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from botocore.exceptions import ClientError
import concurrent.futures
from threading import Lock

# Configuration
POLICY_BUCKET = "policy-database"
NREL_BASE_PATH = "usa/federal-legislation-executive-courts/federal-agencies/nrel"
OEDI_BUCKET = "oedi-data-lake"
OEDI_PREFIX = "pvdaq/csv"

# Progress tracking
PROGRESS_FILE = "nrel_streaming_progress.json"
BATCH_SIZE = 10000  # Records per batch for large CSV files
MAX_WORKERS = 4  # Parallel processing threads

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# S3 clients
s3_client = boto3.client("s3", region_name="us-east-1")
progress_lock = Lock()

class NRELDataStreamer:
    """Streams NREL PVDAQ data from OEDI to policy database"""
    
    def __init__(self, policy_bucket: str = POLICY_BUCKET):
        self.policy_bucket = policy_bucket
        self.progress = self.load_progress()
        
    def load_progress(self) -> Dict:
        """Load streaming progress from file"""
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return {
            "systems_processed": [],
            "metadata_files_processed": [],
            "systems_csv_processed": False,
            "total_records": 0,
            "last_updated": None
        }
    
    def save_progress(self):
        """Save current progress"""
        with progress_lock:
            self.progress["last_updated"] = datetime.now().isoformat()
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(self.progress, f, indent=2)
    
    def discover_systems(self) -> List[str]:
        """Discover all available PV system IDs"""
        try:
            logger.info("Discovering available PV systems...")
            
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=OEDI_BUCKET,
                Prefix=f"{OEDI_PREFIX}/pvdata/",
                Delimiter="/"
            )
            
            systems = []
            for page in page_iterator:
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        prefix_path = prefix['Prefix']
                        # Extract system_id from path like 'pvdaq/csv/pvdata/system_id=12345/'
                        if 'system_id=' in prefix_path:
                            system_id = prefix_path.split('system_id=')[1].rstrip('/')
                            systems.append(system_id)
            
            logger.info(f"Discovered {len(systems)} PV systems")
            return sorted(systems)
            
        except Exception as e:
            logger.error(f"Failed to discover systems: {str(e)}")
            return []
    
    def stream_systems_metadata(self):
        """Stream systems.csv files with overall system information"""
        logger.info("=== STREAMING SYSTEMS METADATA ===")
        
        if self.progress.get("systems_csv_processed"):
            logger.info("Systems CSV already processed, skipping...")
            return
        
        try:
            # Get latest systems file
            response = s3_client.list_objects_v2(
                Bucket=OEDI_BUCKET,
                Prefix=f"{OEDI_PREFIX}/systems",
                MaxKeys=10
            )
            
            systems_files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.csv') and 'systems' in obj['Key']:
                        systems_files.append({
                            'key': obj['Key'],
                            'modified': obj['LastModified'],
                            'size': obj['Size']
                        })
            
            # Sort by modification date, get latest
            systems_files.sort(key=lambda x: x['modified'], reverse=True)
            
            for systems_file in systems_files:
                logger.info(f"Processing systems file: {systems_file['key']}")
                
                # Download and process CSV
                csv_obj = s3_client.get_object(Bucket=OEDI_BUCKET, Key=systems_file['key'])
                df = pd.read_csv(csv_obj['Body'])
                
                # Upload to policy database
                metadata = {
                    'source_file': systems_file['key'],
                    'source': 'NREL PVDAQ Systems Registry',
                    'record_count': len(df),
                    'columns': list(df.columns),
                    'file_size_bytes': systems_file['size'],
                    'last_modified': systems_file['modified'].isoformat(),
                    'collection_date': datetime.now().isoformat()
                }
                
                # Save metadata
                filename = os.path.basename(systems_file['key'])
                metadata_key = f"{NREL_BASE_PATH}/systems/metadata/{filename}.json"
                self._upload_to_s3(
                    json.dumps(metadata, indent=2).encode('utf-8'),
                    metadata_key,
                    'application/json'
                )
                
                # Save CSV data in batches
                for batch_num, batch_df in enumerate(self._batch_dataframe(df, BATCH_SIZE)):
                    # Create JSON-serializable source_info
                    json_safe_source_info = {
                        'key': systems_file['key'],
                        'modified': systems_file['modified'].isoformat(),
                        'size': systems_file['size']
                    }
                    
                    batch_data = {
                        'source_info': json_safe_source_info,
                        'batch_number': batch_num,
                        'records': batch_df.to_dict('records'),
                        'upload_timestamp': datetime.now().isoformat()
                    }
                    
                    batch_key = f"{NREL_BASE_PATH}/systems/data/{filename}/batch_{batch_num:06d}.json"
                    self._upload_to_s3(
                        json.dumps(batch_data, indent=2).encode('utf-8'),
                        batch_key,
                        'application/json'
                    )
                
                self.progress["total_records"] += len(df)
                logger.info(f"✓ Processed {filename}: {len(df)} system records")
            
            self.progress["systems_csv_processed"] = True
            self.save_progress()
            
        except Exception as e:
            logger.error(f"Failed to stream systems metadata: {str(e)}")
    
    def stream_system_metadata(self, system_ids: List[str]):
        """Stream individual system metadata JSON files"""
        logger.info("=== STREAMING SYSTEM METADATA FILES ===")
        
        processed_count = 0
        for system_id in system_ids:
            if system_id in self.progress["metadata_files_processed"]:
                continue
                
            try:
                # Get system metadata JSON
                metadata_key = f"{OEDI_PREFIX}/system_metadata/{system_id}_system_metadata.json"
                
                try:
                    response = s3_client.get_object(Bucket=OEDI_BUCKET, Key=metadata_key)
                    metadata_json = json.loads(response['Body'].read())
                    
                    # Upload to policy database
                    policy_key = f"{NREL_BASE_PATH}/system_metadata/{system_id}_metadata.json"
                    enhanced_metadata = {
                        'system_id': system_id,
                        'source': 'NREL PVDAQ System Metadata',
                        'original_key': metadata_key,
                        'collection_date': datetime.now().isoformat(),
                        'metadata': metadata_json
                    }
                    
                    self._upload_to_s3(
                        json.dumps(enhanced_metadata, indent=2).encode('utf-8'),
                        policy_key,
                        'application/json'
                    )
                    
                    processed_count += 1
                    self.progress["metadata_files_processed"].append(system_id)
                    
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} system metadata files")
                        self.save_progress()
                        
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        logger.debug(f"No metadata file for system {system_id}")
                    else:
                        raise
                        
            except Exception as e:
                logger.error(f"Failed to process metadata for system {system_id}: {str(e)}")
        
        logger.info(f"✓ Processed {processed_count} system metadata files")
        self.save_progress()
    
    def stream_pv_data(self, system_ids: List[str], max_systems: Optional[int] = None):
        """Stream PV time-series data for systems"""
        logger.info("=== STREAMING PV TIME-SERIES DATA ===")
        
        if max_systems:
            system_ids = system_ids[:max_systems]
            logger.info(f"Limiting to first {max_systems} systems")
        
        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for system_id in system_ids:
                if system_id not in self.progress["systems_processed"]:
                    future = executor.submit(self._process_system_data, system_id)
                    futures.append((system_id, future))
            
            # Process results
            for system_id, future in futures:
                try:
                    records_processed = future.result()
                    self.progress["systems_processed"].append(system_id)
                    self.progress["total_records"] += records_processed
                    
                    logger.info(f"✓ Completed system {system_id}: {records_processed} records")
                    self.save_progress()
                    
                except Exception as e:
                    logger.error(f"Failed to process system {system_id}: {str(e)}")
    
    def _process_system_data(self, system_id: str) -> int:
        """Process all data for a single PV system"""
        total_records = 0
        system_prefix = f"{OEDI_PREFIX}/pvdata/system_id={system_id}/"
        
        try:
            # List all CSV files for this system
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=OEDI_BUCKET,
                Prefix=system_prefix
            )
            
            csv_files = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.csv'):
                            csv_files.append(obj)
            
            logger.debug(f"System {system_id}: Found {len(csv_files)} CSV files")
            
            # Process each CSV file
            for csv_file in csv_files:
                csv_key = csv_file['Key']
                
                try:
                    # Download and process CSV
                    csv_obj = s3_client.get_object(Bucket=OEDI_BUCKET, Key=csv_key)
                    df = pd.read_csv(csv_obj['Body'])
                    
                    # Extract date info from path for organization
                    path_parts = csv_key.split('/')
                    year = None
                    month = None
                    day = None
                    
                    for part in path_parts:
                        if part.startswith('year='):
                            year = part.split('=')[1]
                        elif part.startswith('month='):
                            month = part.split('=')[1]
                        elif part.startswith('day='):
                            day = part.split('=')[1]
                    
                    # Create organized path
                    policy_path = f"{NREL_BASE_PATH}/pvdata/system_id={system_id}"
                    if year:
                        policy_path += f"/year={year}"
                        if month:
                            policy_path += f"/month={month}"
                            if day:
                                policy_path += f"/day={day}"
                    
                    # Save metadata
                    filename = os.path.basename(csv_key)
                    metadata = {
                        'system_id': system_id,
                        'source_file': csv_key,
                        'source': 'NREL PVDAQ Time Series Data',
                        'year': year,
                        'month': month,
                        'day': day,
                        'record_count': len(df),
                        'columns': list(df.columns),
                        'file_size_bytes': csv_file['Size'],
                        'collection_date': datetime.now().isoformat()
                    }
                    
                    metadata_key = f"{policy_path}/metadata/{filename}.json"
                    self._upload_to_s3(
                        json.dumps(metadata, indent=2).encode('utf-8'),
                        metadata_key,
                        'application/json'
                    )
                    
                    # Save data in batches
                    for batch_num, batch_df in enumerate(self._batch_dataframe(df, BATCH_SIZE)):
                        batch_data = {
                            'system_id': system_id,
                            'source_file': csv_key,
                            'date_info': {'year': year, 'month': month, 'day': day},
                            'batch_number': batch_num,
                            'records': batch_df.to_dict('records'),
                            'upload_timestamp': datetime.now().isoformat()
                        }
                        
                        batch_key = f"{policy_path}/data/{filename}/batch_{batch_num:06d}.json"
                        self._upload_to_s3(
                            json.dumps(batch_data, indent=2).encode('utf-8'),
                            batch_key,
                            'application/json'
                        )
                    
                    total_records += len(df)
                    
                except Exception as e:
                    logger.error(f"Failed to process CSV {csv_key}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to process system {system_id}: {str(e)}")
        
        return total_records
    
    def _batch_dataframe(self, df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
        """Yield dataframe in batches"""
        for i in range(0, len(df), batch_size):
            yield df.iloc[i:i + batch_size]
    
    def _upload_to_s3(self, content: bytes, s3_key: str, content_type: str) -> bool:
        """Upload content to S3 with existence check"""
        try:
            # Check if already exists
            try:
                s3_client.head_object(Bucket=self.policy_bucket, Key=s3_key)
                logger.debug(f"Already exists: {s3_key}")
                return False
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
            
            # Upload new content
            s3_client.put_object(
                Bucket=self.policy_bucket,
                Key=s3_key,
                Body=content,
                ContentType=content_type
            )
            logger.debug(f"✓ Uploaded: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"S3 upload failed for {s3_key}: {str(e)}")
            return False
    
    def stream_all_nrel_data(self, max_systems: Optional[int] = None):
        """Main method to stream all NREL PVDAQ data"""
        logger.info("🚀 Starting NREL PVDAQ data streaming")
        start_time = time.time()
        
        try:
            # Step 1: Stream systems metadata files
            self.stream_systems_metadata()
            
            # Step 2: Discover available systems
            system_ids = self.discover_systems()
            if not system_ids:
                logger.warning("No systems discovered")
                return
            
            # Step 3: Stream individual system metadata
            self.stream_system_metadata(system_ids)
            
            # Step 4: Stream PV time-series data
            self.stream_pv_data(system_ids, max_systems)
            
        except KeyboardInterrupt:
            logger.info("⚠️ Streaming interrupted by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {str(e)}")
            raise
        finally:
            elapsed = time.time() - start_time
            logger.info(f"🎉 NREL streaming complete!")
            logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
            logger.info(f"📊 Total records: {self.progress['total_records']}")
            logger.info(f"📊 Systems processed: {len(self.progress['systems_processed'])}")
            logger.info(f"📁 Location: s3://{POLICY_BUCKET}/{NREL_BASE_PATH}/")

def main():
    """Main entry point"""
    streamer = NRELDataStreamer()
    # Limit to 10 systems for initial test
    streamer.stream_all_nrel_data(max_systems=10)

if __name__ == "__main__":
    main()