#!/usr/bin/env python3
"""
Dataset utilities for Mistral training pipeline
Implements dataset splitting functionality missing from monolithic script
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def split_dataset(input_file: str, train_file: str, val_file: str, 
                 validation_split: float = 0.2, seed: int = 42) -> bool:
    """
    Split a JSONL dataset into training and validation sets.
    
    Args:
        input_file: Path to input JSONL file
        train_file: Path to output training JSONL file  
        val_file: Path to output validation JSONL file
        validation_split: Fraction of data for validation (0.0-1.0)
        seed: Random seed for reproducible splits
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate inputs
        if not (0.0 <= validation_split <= 1.0):
            logger.error(f"validation_split must be between 0.0 and 1.0, got {validation_split}")
            return False
        
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_file}")
            return False
        
        # Load all data
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue
        
        if not data:
            logger.error("No valid data found in input file")
            return False
        
        logger.info(f"Loaded {len(data)} entries from {input_file}")
        
        # Shuffle data for random split
        random.seed(seed)
        random.shuffle(data)
        
        # Calculate split point
        val_size = int(len(data) * validation_split)
        train_size = len(data) - val_size
        
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        logger.info(f"Split: {len(train_data)} training, {len(val_data)} validation")
        
        # Write training set
        train_path = Path(train_file)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for entry in train_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Write validation set  
        val_path = Path(val_file)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for entry in val_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Successfully created train/val split: {train_file}, {val_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        return False

def validate_dataset_format(file_path: str) -> bool:
    """
    Validate that a JSONL file has proper format for training.
    
    Args:
        file_path: Path to JSONL file to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        valid_entries = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    entry = json.loads(line)
                    
                    # Check required fields for training
                    if not isinstance(entry, dict):
                        logger.error(f"Line {line_num}: Entry must be a JSON object")
                        return False
                    
                    # Must have text content
                    if 'text' not in entry or not entry['text'].strip():
                        logger.error(f"Line {line_num}: Missing or empty 'text' field")
                        return False
                    
                    valid_entries += 1
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: Invalid JSON - {e}")
                    return False
        
        if valid_entries == 0:
            logger.error("No valid entries found")
            return False
        
        logger.info(f"Dataset validation passed: {valid_entries} valid entries")
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False

def combine_datasets(input_files: List[str], output_file: str) -> bool:
    """
    Combine multiple JSONL files into a single dataset.
    
    Args:
        input_files: List of input JSONL file paths
        output_file: Path to output combined JSONL file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        combined_data = []
        
        for input_file in input_files:
            input_path = Path(input_file)
            if not input_path.exists():
                logger.warning(f"Input file does not exist, skipping: {input_file}")
                continue
                
            logger.info(f"Reading {input_file}")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            combined_data.append(entry)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in {input_file}:{line_num}: {e}")
                            continue
        
        if not combined_data:
            logger.error("No valid data found in any input files")
            return False
        
        # Write combined dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in combined_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Combined {len(combined_data)} entries into {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to combine datasets: {e}")
        return False