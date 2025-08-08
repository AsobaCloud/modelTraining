# ACTUAL STATE OF THE TRAINING PIPELINE - AUGUST 7, 2025

## Current Status: TRAINING SUCCESSFULLY

### Active Training Run
- **Run ID**: mistral-20250807-180708
- **Instance**: West coast g5.4xlarge (34.217.17.67)
- **Status**: Training in progress with normalized dataset
- **Memory Usage**: ~11.6GB during tokenization (down from 25.9GB after fixes)

## What Actually Exists (Verified Working)

### Datasets
- **Normalized comprehensive dataset**: 89,948 entries (train) + 10,323 (val)
  - Location: `s3://asoba-llm-cache/datasets/mistral-verbosity/normalized_train.jsonl` and `normalized_val.jsonl`
  - Successfully training with this dataset after normalization
  - All entries have only 'text' field (column inconsistency fixed)

- **Original datasets**: 51,617 entries (41,294 train + 10,323 val)
  - Location: `s3://asoba-llm-cache/datasets/mistral-verbosity/final_train.jsonl` and `final_val.jsonl`
  - Composition: Mostly State Dept cables (operatives) + minimal policy data

### Models
- **Base model**: Mistral-7B-v0.3 (uncensored)
  - Location: `s3://asoba-llm-cache/models/mistralai/Mistral-7B-v0.3/`
  - Successfully deployed to west coast instance
  - Using for current training run

### Official Mistral OOM Fix Applied
**Problem**: Training was getting OOM killed at ~53% tokenization, using 27.8GB RAM

**Root Cause**: Batch size too large for memory - effective batch was `4 x 1024 = 4096` sequence length

**Solution** (Official Mistral best practice):
1. **Reduce batch size**: `per_device_train_batch_size=1` (was 4)
2. **Reduce sequence length**: `max_seq_length=512` (was 1024) 
3. **Increase gradient accumulation**: `gradient_accumulation_steps=16` (was 4)
4. **Net effect**: Sequence batch reduced from 4096 to 512 (8x reduction)

**Source**: Official mistral-finetune repo: "If you encounter CUDA OOM, reduce batch size. batch_size = seq_len x batch_size"

### Working Training Configuration
- **Script**: `/home/shingai/sort/llm-training/scripts/mistral/train_mistral_simple.py` (with official OOM fix)
- **Model path**: `/mnt/training/models/mistralai/Mistral-7B-v0.3`
- **Output**: `/mnt/training/mistral_output`
- **Batch size**: 1 (official Mistral OOM fix)
- **Sequence length**: 512 tokens (official Mistral OOM fix)
- **Gradient accumulation**: 16 steps (maintains effective batch size)
- **Quantization**: 4-bit with BitsAndBytesConfig

### Instances
- **West coast g5.4xlarge**: 34.217.17.67 - Currently training Mistral
- **East coast g5.2xlarge**: 54.197.142.172 - Has processed policy data

### What Was Fixed
1. ✅ Dataset normalization - removed inconsistent columns
2. ✅ Memory optimization - reordered model loading after tokenization
3. ✅ Subprocess crashes - disabled multiprocessing for stability
4. ✅ Model deployment - using resolve_model.sh to get from S3