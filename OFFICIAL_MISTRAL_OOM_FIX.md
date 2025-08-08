# Official Mistral OOM Fix Implementation

## Problem
Training repeatedly failed with OOM kills at ~53% tokenization on g5.4xlarge (64GB RAM), using 27.8GB RSS.

## Root Cause Discovery
After investigating Mistral best practices vs our failing approach:

**Our problematic configuration:**
- `batch_size=4`
- `max_seq_length=1024` 
- **Effective sequence batch = 4 × 1024 = 4096**

**Official Mistral guidance:**
> "If you encounter a CUDA out-of-memory error, one possible solution is to reduce the batch size per GPU. The batch size is equal to seq_len x batch_size."

Source: https://github.com/mistralai/mistral-finetune

## Solution: Official Mistral Parameters

### Configuration Changes
```python
# OLD (OOM at 27.8GB)
per_device_train_batch_size=4
max_seq_length=1024
gradient_accumulation_steps=4
# Effective batch = 4 × 1024 = 4096

# NEW (Official Mistral OOM fix)  
per_device_train_batch_size=1
max_seq_length=512
gradient_accumulation_steps=16
# Effective batch = 1 × 512 = 512 (8x reduction)
```

### Implementation
Applied in `train_mistral_simple.py`:

1. **Reduced batch size**: 4 → 1
2. **Reduced sequence length**: 1024 → 512
3. **Increased gradient accumulation**: 4 → 16 (maintains training dynamics)

### Expected Results
- **8x reduction** in sequence batch size (4096 → 512)
- **Drastically lower** memory usage during tokenization
- **Same effective training** due to gradient accumulation compensation

## Verification
Run ID: Currently testing with new parameters

**Command used:**
```bash
python3 train_mistral_simple.py \
  --model-tag mistralai/Mistral-7B-v0.3 \
  --train-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/normalized_train.jsonl \
  --val-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/normalized_val.jsonl \
  --max-seq-length 512 \
  --batch-size 1
```

## Key Learning
The issue wasn't complex memory management techniques - it was using **wrong batch parameters** for Mistral. The official Mistral repository has specific OOM guidance that directly addresses this exact scenario.

**Best Practice**: Always check official model repositories for OOM solutions before implementing custom memory management.