#!/usr/bin/env python3
"""
Regenerate policy datasets from cleaned S3 data
Excludes processed/failed folders except from financial-metrics
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Policy sources with proper exclusions
POLICY_SOURCES = [
    {
        "name": "econ_theory",
        "s3_path": "s3://policy-database/econ-theory/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "financial_metrics", 
        "s3_path": "s3://policy-database/financial-metrics/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["failed/*", "*/failed/*"]  # Keep processed for this one
    },
    {
        "name": "government_officials",
        "s3_path": "s3://policy-database/government-officials/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "international_relations",
        "s3_path": "s3://policy-database/international-relations/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "national_security",
        "s3_path": "s3://policy-database/national-security/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "policy_debate",
        "s3_path": "s3://policy-database/policy-debate/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "public_welfare",
        "s3_path": "s3://policy-database/public-welfare/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "behavioral_economics",
        "s3_path": "s3://policy-database/behavioral-economics/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*", "news/*", "*/news/*"]
    }
]

def download_and_process_policy_data(work_dir: str):
    """Download and process policy data with proper exclusions"""
    
    policy_dir = Path(work_dir) / "policy_clean"
    policy_dir.mkdir(parents=True, exist_ok=True)
    
    all_entries = []
    
    for source in POLICY_SOURCES:
        logger.info(f"Processing {source['name']}...")
        
        source_dir = policy_dir / source['name']
        source_dir.mkdir(exist_ok=True)
        
        # Build AWS CLI command
        cmd = [
            "aws", "s3", "sync",
            source['s3_path'],
            str(source_dir),
            "--region", "us-east-1",
            "--no-progress"
        ]
        
        # Add excludes first
        for exclude in source.get("excludes", []):
            cmd.extend(["--exclude", exclude])
        
        # Then add includes
        cmd.append("--exclude")
        cmd.append("*")
        for include in source["includes"]:
            cmd.extend(["--include", include])
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Download failed for {source['name']}: {result.stderr}")
            continue
        
        # Process downloaded files
        source_entries = []
        
        # Process JSON/JSONL files
        for pattern in ["*.json", "*.jsonl"]:
            for file_path in source_dir.rglob(pattern):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.suffix == '.jsonl':
                            for line in f:
                                if line.strip():
                                    entry = json.loads(line)
                                    if 'text' in entry and entry['text'].strip():
                                        source_entries.append(entry)
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and 'text' in item and item['text'].strip():
                                        source_entries.append(item)
                            elif isinstance(data, dict) and 'text' in data and data['text'].strip():
                                source_entries.append(data)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        
        # Process text files
        for file_path in source_dir.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        source_entries.append({
                            "text": content,
                            "source": str(file_path.relative_to(source_dir)),
                            "domain": "policy_analysis"
                        })
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(f"Collected {len(source_entries)} entries from {source['name']}")
        
        # Save source-specific file
        output_file = Path(work_dir) / f"policy_{source['name']}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in source_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        all_entries.extend(source_entries)
    
    # Save combined policy dataset
    combined_file = Path(work_dir) / "policy_combined.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Total policy entries: {len(all_entries)}")
    logger.info(f"Saved to: {combined_file}")
    
    return str(combined_file), len(all_entries)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate policy datasets")
    parser.add_argument("--work-dir", default="/mnt/training/data_prep", help="Working directory")
    args = parser.parse_args()
    
    logger.info("Starting policy data regeneration...")
    policy_file, count = download_and_process_policy_data(args.work_dir)
    logger.info(f"Completed: {count} policy entries in {policy_file}")

if __name__ == "__main__":
    main()