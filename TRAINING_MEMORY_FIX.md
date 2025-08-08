# Mistral Training Memory Optimization Fix

## Problem
Training was repeatedly getting OOM killed at ~53% tokenization on g5.4xlarge (64GB RAM), using 27.8GB RSS.

## Root Cause Analysis

### Memory Components During Tokenization
| Component | Memory Usage |
|-----------|--------------|
| Mistral-7B 4-bit model weights | 8-10 GB |
| Raw JSON dataset in memory | 6-8 GB |
| Tokenization working set | 10-12 GB |
| Python/HF caches | 1-2 GB |
| **Total Peak** | **~27 GB** |

The issue: The script loaded BOTH the full model AND the raw dataset before starting tokenization.

## Solution: Reorder Operations

### Before (OOM at 27.8GB)
```python
1. Load raw dataset → 6-8GB
2. Load full model → +8-10GB (total: 14-18GB)
3. Start tokenization → +10-12GB (total: 27GB → OOM)
```

### After (Stable at 11.6GB)
```python
1. Load raw dataset → 6-8GB
2. Load tokenizer ONLY → +0.5GB (total: 7-9GB)
3. Tokenize with memory_safe_map() → peaks at 11.6GB
4. Free raw dataset (gc.collect())
5. Load full model → back to ~26GB but tokenization done
```

## Implementation Details

### Key Changes in train_mistral_simple.py

1. **Memory-safe mapping function**:
```python
def memory_safe_map(dataset, func, **kwargs):
    return dataset.map(
        func,
        batched=True,
        batch_size=512,              # Small chunks
        writer_batch_size=512,       # Flush immediately
        keep_in_memory=False,        # Disk-backed
        load_from_cache_file=False,
        num_proc=1,                  # Avoid subprocess crashes
        **kwargs
    )
```

2. **Reordered operations**:
```python
# Load tokenizer ONLY
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Tokenize datasets
train_dataset = memory_safe_map(...)
val_dataset = memory_safe_map(...)

# Free raw data
del train_file, val_file
gc.collect()

# NOW load model
model, _ = setup_model_and_tokenizer(model_path)
```

3. **Removed unnecessary copying**:
```python
# Before: tokenized["labels"] = tokenized["input_ids"].copy()
# After:  tokenized["labels"] = tokenized["input_ids"]
```

## Results

- **Peak tokenization memory**: 27.8GB → 11.6GB (58% reduction)
- **Training status**: Successfully running on 89,948 examples
- **No more OOM kills** at tokenization phase

## Monitoring

Check memory during training:
```bash
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 \
  "ps aux | grep train_mistral_simple | grep -v grep"
```

Current run: `mistral-20250807-180708` on west coast g5.4xlarge