#!/usr/bin/env python3
"""
NREL ComStock Data Streaming Pipeline
Streams commercial building stock data from NREL's OEDI data lake into policy-database
"""

import os
import sys
import json
import time
import logging
import boto3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Set
from botocore.exceptions import ClientError
import concurrent.futures
from threading import Lock
import io

# Configuration
POLICY_BUCKET = "policy-database"
NREL_BASE_PATH = "usa/federal-legislation-executive-courts/federal-agencies/nrel"
COMSTOCK_PATH = f"{NREL_BASE_PATH}/comstock"
OEDI_BUCKET = "oedi-data-lake"
COMSTOCK_PREFIX = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/comstock_amy2018_release_2"

# Selective processing configuration
SAMPLE_STATES = ["CA", "TX", "NY", "FL", "IL", "WA", "CO", "MA"]  # Representative states
MAX_BUILDINGS_PER_STATE = 50  # Limit individual building files
UPGRADE_SCENARIOS = [0, 1, 2]  # Focus on baseline + key upgrade scenarios

# Progress tracking
PROGRESS_FILE = "comstock_streaming_progress.json"
BATCH_SIZE = 5000  # Records per batch
MAX_WORKERS = 3  # Conservative for large files

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

s3_client = boto3.client("s3", region_name="us-east-1")
progress_lock = Lock()

class ComStockStreamer:
    """Streams NREL ComStock building stock data"""
    
    def __init__(self, policy_bucket: str = POLICY_BUCKET):
        self.policy_bucket = policy_bucket
        self.progress = self.load_progress()
        
    def load_progress(self) -> Dict:
        """Load streaming progress from file"""
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return {
            "phase_1_completed": False,
            "phase_2_states_completed": [],
            "phase_3_completed": False,
            "root_files_processed": [],
            "metadata_aggregates_processed": [],
            "individual_buildings_processed": [],
            "total_records": 0,
            "last_updated": None
        }
    
    def save_progress(self):
        """Save current progress"""
        with progress_lock:
            self.progress["last_updated"] = datetime.now().isoformat()
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(self.progress, f, indent=2)
    
    def stream_phase_1_metadata(self):
        """Phase 1: Stream core metadata and data dictionaries"""
        logger.info("=== PHASE 1: STREAMING CORE METADATA ===")
        
        if self.progress.get("phase_1_completed"):
            logger.info("Phase 1 already completed, skipping...")
            return
        
        try:
            # Step 1: Stream root-level files
            self._stream_root_files()
            
            # Step 2: Stream metadata aggregates
            self._stream_metadata_aggregates()
            
            self.progress["phase_1_completed"] = True
            self.save_progress()
            logger.info("✓ Phase 1 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {str(e)}")
            raise
    
    def _stream_root_files(self):
        """Stream root-level metadata files"""
        logger.info("--- Streaming root metadata files ---")
        
        root_files = [
            "data_dictionary.tsv",
            "enumeration_dictionary.tsv", 
            "measure_name_crosswalk.csv",
            "upgrades_lookup.json",
            "batch_state.json"
        ]
        
        for filename in root_files:
            if filename in self.progress["root_files_processed"]:
                logger.debug(f"Already processed: {filename}")
                continue
                
            try:
                source_key = f"{COMSTOCK_PREFIX}/{filename}"
                
                # Check if file exists
                try:
                    response = s3_client.get_object(Bucket=OEDI_BUCKET, Key=source_key)
                    content = response['Body'].read()
                    
                    # Determine file type and process accordingly
                    if filename.endswith('.tsv'):
                        df = pd.read_csv(io.BytesIO(content), sep='\t')
                        self._process_tabular_file(df, filename, source_key, 'TSV')
                    elif filename.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(content))
                        self._process_tabular_file(df, filename, source_key, 'CSV')
                    elif filename.endswith('.json'):
                        json_data = json.loads(content.decode('utf-8'))
                        self._process_json_file(json_data, filename, source_key)
                    
                    self.progress["root_files_processed"].append(filename)
                    logger.info(f"✓ Processed root file: {filename}")
                    
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        logger.warning(f"File not found: {filename}")
                    else:
                        raise
                        
            except Exception as e:
                logger.error(f"Failed to process root file {filename}: {str(e)}")
                continue
        
        self.save_progress()
    
    def _process_tabular_file(self, df: pd.DataFrame, filename: str, source_key: str, file_type: str):
        """Process CSV/TSV files"""
        # Create metadata
        metadata = {
            'filename': filename,
            'source_key': source_key,
            'source': f'NREL ComStock {file_type} Data',
            'file_type': file_type,
            'record_count': len(df),
            'columns': list(df.columns),
            'collection_date': datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_key = f"{COMSTOCK_PATH}/metadata/root_files/{filename}.json"
        self._upload_to_s3(
            json.dumps(metadata, indent=2).encode('utf-8'),
            metadata_key,
            'application/json'
        )
        
        # Save data in batches
        for batch_num, batch_df in enumerate(self._batch_dataframe(df, BATCH_SIZE)):
            batch_data = {
                'filename': filename,
                'source_key': source_key,
                'batch_number': batch_num,
                'records': batch_df.to_dict('records'),
                'upload_timestamp': datetime.now().isoformat()
            }
            
            batch_key = f"{COMSTOCK_PATH}/data/root_files/{filename}/batch_{batch_num:06d}.json"
            self._upload_to_s3(
                json.dumps(batch_data, indent=2).encode('utf-8'),
                batch_key,
                'application/json'
            )
        
        self.progress["total_records"] += len(df)
    
    def _process_json_file(self, json_data: dict, filename: str, source_key: str):
        """Process JSON files"""
        enhanced_data = {
            'filename': filename,
            'source_key': source_key,
            'source': 'NREL ComStock JSON Data',
            'collection_date': datetime.now().isoformat(),
            'data': json_data
        }
        
        json_key = f"{COMSTOCK_PATH}/data/root_files/{filename}"
        self._upload_to_s3(
            json.dumps(enhanced_data, indent=2).encode('utf-8'),
            json_key,
            'application/json'
        )
    
    def _stream_metadata_aggregates(self):
        """Stream aggregated metadata and annual results"""
        logger.info("--- Streaming metadata aggregates ---")
        
        aggregate_paths = [
            "metadata_and_annual_results_aggregates/national/",
            "metadata_and_annual_results_aggregates/by_state/"
        ]
        
        for agg_path in aggregate_paths:
            if agg_path in self.progress["metadata_aggregates_processed"]:
                continue
                
            try:
                full_prefix = f"{COMSTOCK_PREFIX}/{agg_path}"
                self._stream_directory_parquet_files(full_prefix, agg_path)
                
                self.progress["metadata_aggregates_processed"].append(agg_path)
                logger.info(f"✓ Processed aggregate path: {agg_path}")
                
            except Exception as e:
                logger.error(f"Failed to process aggregate path {agg_path}: {str(e)}")
                continue
        
        self.save_progress()
    
    def _stream_directory_parquet_files(self, s3_prefix: str, logical_path: str, max_files: Optional[int] = None):
        """Stream parquet files from a directory"""
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=OEDI_BUCKET, Prefix=s3_prefix)
            
            parquet_files = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.parquet'):
                            parquet_files.append(obj)
            
            if max_files:
                parquet_files = parquet_files[:max_files]
            
            logger.info(f"Found {len(parquet_files)} parquet files in {logical_path}")
            
            for parquet_file in parquet_files:
                try:
                    self._process_parquet_file(parquet_file['Key'], logical_path)
                except Exception as e:
                    logger.error(f"Failed to process parquet file {parquet_file['Key']}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to list directory {s3_prefix}: {str(e)}")
    
    def _process_parquet_file(self, s3_key: str, logical_path: str):
        """Process individual parquet file"""
        try:
            # Download parquet file
            response = s3_client.get_object(Bucket=OEDI_BUCKET, Key=s3_key)
            df = pd.read_parquet(io.BytesIO(response['Body'].read()))
            
            filename = os.path.basename(s3_key)
            
            # Create metadata
            metadata = {
                'filename': filename,
                'source_key': s3_key,
                'logical_path': logical_path,
                'source': 'NREL ComStock Parquet Data',
                'record_count': len(df),
                'columns': list(df.columns),
                'file_size_bytes': response['ContentLength'],
                'collection_date': datetime.now().isoformat()
            }
            
            # Save metadata
            safe_path = logical_path.replace('/', '_')
            metadata_key = f"{COMSTOCK_PATH}/metadata/{safe_path}/{filename}.json"
            self._upload_to_s3(
                json.dumps(metadata, indent=2).encode('utf-8'),
                metadata_key,
                'application/json'
            )
            
            # Save data in batches
            for batch_num, batch_df in enumerate(self._batch_dataframe(df, BATCH_SIZE)):
                batch_data = {
                    'source_file': s3_key,
                    'logical_path': logical_path,
                    'batch_number': batch_num,
                    'records': batch_df.to_dict('records'),
                    'upload_timestamp': datetime.now().isoformat()
                }
                
                batch_key = f"{COMSTOCK_PATH}/data/{safe_path}/{filename}/batch_{batch_num:06d}.json"
                self._upload_to_s3(
                    json.dumps(batch_data, indent=2).encode('utf-8'),
                    batch_key,
                    'application/json'
                )
            
            self.progress["total_records"] += len(df)
            logger.debug(f"✓ Processed parquet: {filename} ({len(df)} records)")
            
        except Exception as e:
            logger.error(f"Failed to process parquet file {s3_key}: {str(e)}")
    
    def stream_phase_2_selective_sampling(self):
        """Phase 2: Stream selective sample of individual building data"""
        logger.info("=== PHASE 2: STREAMING SELECTIVE BUILDING DATA ===")
        
        for state in SAMPLE_STATES:
            if state in self.progress["phase_2_states_completed"]:
                logger.info(f"State {state} already processed, skipping...")
                continue
                
            try:
                logger.info(f"Processing state: {state}")
                self._stream_state_building_data(state)
                
                self.progress["phase_2_states_completed"].append(state)
                self.save_progress()
                
            except Exception as e:
                logger.error(f"Failed to process state {state}: {str(e)}")
                continue
    
    def _stream_state_building_data(self, state: str):
        """Stream building data for a specific state"""
        for upgrade in UPGRADE_SCENARIOS:
            prefix = f"{COMSTOCK_PREFIX}/timeseries_individual_buildings/by_state/upgrade={upgrade}/state={state}/"
            
            try:
                # List available parquet files
                paginator = s3_client.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(Bucket=OEDI_BUCKET, Prefix=prefix)
                
                parquet_files = []
                for page in page_iterator:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['Key'].endswith('.parquet'):
                                parquet_files.append(obj['Key'])
                
                # Sample subset of buildings
                sampled_files = parquet_files[:MAX_BUILDINGS_PER_STATE]
                logger.info(f"Sampling {len(sampled_files)}/{len(parquet_files)} buildings for {state} upgrade {upgrade}")
                
                for parquet_key in sampled_files:
                    logical_path = f"individual_buildings/state={state}/upgrade={upgrade}"
                    self._process_parquet_file(parquet_key, logical_path)
                    
            except Exception as e:
                logger.error(f"Failed to process {state} upgrade {upgrade}: {str(e)}")
                continue
    
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
    
    def stream_all_comstock_data(self, phases: List[int] = [1]):
        """Main method to stream ComStock data"""
        logger.info("🚀 Starting NREL ComStock data streaming")
        start_time = time.time()
        
        try:
            if 1 in phases:
                self.stream_phase_1_metadata()
            
            if 2 in phases:
                self.stream_phase_2_selective_sampling()
            
        except KeyboardInterrupt:
            logger.info("⚠️ Streaming interrupted by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {str(e)}")
            raise
        finally:
            elapsed = time.time() - start_time
            logger.info(f"🎉 ComStock streaming complete!")
            logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
            logger.info(f"📊 Total records: {self.progress['total_records']}")
            logger.info(f"📁 Location: s3://{POLICY_BUCKET}/{COMSTOCK_PATH}/")

def main():
    """Main entry point"""
    streamer = ComStockStreamer()
    # Start with Phase 1 only
    streamer.stream_all_comstock_data(phases=[1])

if __name__ == "__main__":
    main()