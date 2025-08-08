# Mistral Training Plan - Option A Implementation

## Current Status
✅ **Completed:**
- Created `prepare_arrow.py` preprocessing script
- Modified `train_mistral_simple.py` to support `TOKENIZED_DIR` env var
- Identified root cause of OOM: 27.8GB RAM usage during tokenization
- Upgraded monitoring system with 60s heartbeats and log streaming

❌ **Current Issue:**  
Preprocessing crashed with multiprocessing error: "One of the subprocesses has abruptly died during map operation"

## Next Steps

### Step 1: Fix Preprocessing (Immediate)
**Problem:** Multiprocessing crash during tokenization
**Solution:** Disable multiprocessing and run single-threaded

```bash
# On the g5.4xlarge instance
cd /mnt/training/mistral_training
source mistral_venv/bin/activate

# Run with single process to avoid crashes
TRAIN_JSONL=/mnt/training/mistral_final_output/normalized_train.jsonl \
VAL_JSONL=/mnt/training/mistral_final_output/normalized_val.jsonl \
OUT_DIR=/mnt/training/tokenized_data \
BASE_MODEL=/mnt/training/models/mistralai/Mistral-7B-v0.3 \
BLOCK_SIZE=512 \
NUMPROC=1 \
python3 prepare_arrow.py
```

**Expected:** 
- Single-threaded processing will be slower but reliable
- Should create `/mnt/training/tokenized_data/` with Arrow files
- Will take ~30-60 minutes for 90K examples

### Step 2: Run Training with Pre-tokenized Data
Once preprocessing completes successfully:

```bash
cd /mnt/training/mistral_training
source mistral_venv/bin/activate

# Use pre-tokenized data (no streaming, no runtime tokenization)
TOKENIZED_DIR=/mnt/training/tokenized_data \
python3 train_mistral_simple.py \
  --model-name /mnt/training/models/mistralai/Mistral-7B-v0.3 \
  --output-dir /mnt/training/mistral_final_output \
  --batch-size 1 \
  --max-seq-length 512 \
  --run-id mistral-20250808-arrow \
  --local-files-only
```

**Expected Benefits:**
- No runtime tokenization = no OOM during data loading
- Memory-mapped Arrow files = minimal RAM usage
- Already packed into 512-token blocks = efficient training
- Should use <10GB RAM instead of 27.8GB

### Step 3: Monitor Training Success
**New Monitoring Features:**
- 60-second heartbeats (was 5 minutes)
- Progress updates every 20 steps (was 100)
- Log streaming to S3 every 30 seconds
- Enhanced error reporting with full context

**Monitoring Commands:**
```bash
# Check real-time status
aws s3 ls s3://asoba-llm-cache/training-runs/mistral-20250808-arrow/ --region us-west-2

# View latest logs
aws s3 cp s3://asoba-llm-cache/training-runs/mistral-20250808-arrow/logs/training_log_latest.txt - --region us-west-2

# Check GPU utilization
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "nvidia-smi"
```

## Fallback Plan (Option B)
If preprocessing continues to fail:

### Alternative: Streaming with Tiny Buffer
```bash
# Skip TOKENIZED_DIR to use streaming approach with fixed settings
cd /mnt/training/mistral_training
source mistral_venv/bin/activate

python3 train_mistral_simple.py \
  --train-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/normalized_train.jsonl \
  --val-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/normalized_val.jsonl \
  --model-name /mnt/training/models/mistralai/Mistral-7B-v0.3 \
  --output-dir /mnt/training/mistral_final_output \
  --batch-size 1 \
  --max-seq-length 512 \
  --run-id mistral-20250808-streaming \
  --local-files-only
```

**Key Changes Made:**
- `shuffle(buffer_size=1000)` instead of 10,000
- `dataloader_num_workers=0` instead of 4
- Lazy tokenization still applies

## Success Criteria
**Training Considered Successful When:**
1. ✅ Process starts without OOM during data loading
2. ✅ GPU utilization reaches 80%+ during training
3. ✅ Heartbeat monitoring shows "training/active" status
4. ✅ Loss decreases over first 100 steps
5. ✅ Memory usage stays below 50GB RAM

## Technical Details

### Why Option A Should Work
- **Pre-tokenized data**: No runtime tokenization memory spikes
- **Memory-mapped Arrow**: Reads directly from disk, minimal RAM
- **Fixed block packing**: All sequences exactly 512 tokens
- **No shuffle buffer**: Data pre-shuffled during preprocessing
- **Single worker**: No multiprocess memory multiplication

### Resource Usage Estimate
- **RAM**: 8-12GB (model) + 2-4GB (data) = ~15GB total
- **GPU**: 23GB A10G for model + gradients + optimizer
- **Disk**: ~2GB for tokenized data + model outputs

### Training Timeline
- **Preprocessing**: 30-60 minutes (one-time cost)
- **Training**: 2-3 hours for 3 epochs
- **Total**: 3-4 hours end-to-end

## Files Modified
1. `scripts/mistral/prepare_arrow.py` - New preprocessing script
2. `scripts/mistral/train_mistral_simple.py` - Added TOKENIZED_DIR support
3. Training config: `dataloader_num_workers=0` for reliability

## Next Action Required
Execute Step 1 with single-process preprocessing to resolve the multiprocessing crash.