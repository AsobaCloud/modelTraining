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
from hashlib import sha1

# Setup logging FIRST (so we can safely log in import fallbacks)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import PDF processing (log safely now that logger exists)
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    logger.info("PyPDF2 not available - PDF files will be skipped")
    HAS_PDF = False

# Policy sources with actual S3 paths and proper exclusions
POLICY_SOURCES = [
    {
        "name": "usa_policy",
        "s3_path": "s3://policy-database/usa/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "south_africa_policy", 
        "s3_path": "s3://policy-database/south_africa/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "econ_theory",
        "s3_path": "s3://policy-database/econ-theory/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "federal_bills",
        "s3_path": "s3://policy-database/corpus_7-26-2025/federal/content/",
        "includes": ["*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "insurance_policy",
        "s3_path": "s3://policy-database/insurance/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
    },
    {
        "name": "financial_metrics",
        "s3_path": "s3://policy-database/financial-metrics/processed/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["failed/*", "*/failed/*"]  # Only exclude failed, keep processed content
    },
    {
        "name": "july_news",
        "s3_path": "s3://policy-database/news/2025-07-30/",
        "includes": ["*.jsonl", "*.json", "*.pdf", "*.txt"],
        "excludes": ["processed/*", "failed/*", "*/processed/*", "*/failed/*"]
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
        
        # Get all files and apply max_files limit if specified
        all_files = []
        for pattern in source["includes"]:
            files = list(source_dir.rglob(pattern))
            all_files.extend(files)
        
        # Apply max_files limit if specified
        if "max_files" in source and len(all_files) > source["max_files"]:
            logger.info(f"Limiting {source['name']} to {source['max_files']} files (found {len(all_files)})")
            all_files = all_files[:source["max_files"]]
        
        # Process JSON/JSONL files
        json_files = [f for f in all_files if f.suffix in ['.json', '.jsonl']]
        for file_path in json_files:
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
        txt_files = [f for f in all_files if f.suffix == '.txt']
        for file_path in txt_files:
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
        
        # Process PDF files if PyPDF2 is available
        if HAS_PDF:
            pdf_files = [f for f in all_files if f.suffix == '.pdf']
            for file_path in pdf_files[:100]:  # Limit PDFs per source to avoid memory issues
                try:
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text_content = ""
                        # Extract text from first 5 pages max
                        for page_num, page in enumerate(reader.pages[:5]):
                            text_content += page.extract_text() + "\n"
                        
                        if text_content.strip():
                            source_entries.append({
                                "text": text_content.strip(),
                                "source": str(file_path.relative_to(source_dir)),
                                "domain": "policy_analysis"
                            })
                except Exception as e:
                    logger.warning(f"Error processing PDF {file_path}: {e}")
        else:
            logger.info(f"Skipping {len([f for f in all_files if f.suffix == '.pdf'])} PDF files (PyPDF2 not available)")
        
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