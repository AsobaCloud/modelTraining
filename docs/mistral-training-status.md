# Mistral Training Pipeline - Current Status & Context

## Quick Start for LLM Context
**Instance**: `i-01fa5b57d64c6196a` (54.197.142.172)  
**SSH**: `ssh -i config/mistral-base.pem ubuntu@54.197.142.172`  
**Goal**: Fine-tune Mistral-7B-v0.3 on policy and operatives data using QLoRA

## Current Status: Training In Progress ✅ 

### What's Been Completed ✅
1. **Infrastructure Setup**
   - G5.2xlarge instance running with 24GB GPU (NVIDIA A10G)
   - 300GB EBS volume mounted at `/mnt/training`
   - Model downloaded: `/mnt/training/models/mistral-7b-v0.3`
   - Environment configured with HuggingFace tokens
   - Fixed disk space issues and dependency conflicts

2. **Data Validation & Cleanup**
   - Identified and removed garbage data (behavioral-economics/news)
   - Generated 50,000 validated operatives entries
   - Regenerated clean policy datasets from ALL source folders including financial-networks
   - Fixed "Missing or empty 'text' field" errors
   - **Uploaded datasets to S3**: 41,294 train + 10,323 validation samples

3. **Testing Framework**
   - Built comprehensive pre-flight validation (35+ checks)
   - Implemented heartbeat monitoring for long operations
   - Created failure matrix documentation
   - Added anti-mock testing enforcement

4. **Training Launch** 🚀
   - **TRAINING ACTIVE**: Mistral-7B fine-tuning started successfully
   - Configuration: QLoRA 4-bit quantization, batch_size=1, max_seq_length=512
   - Trainable params: 167M (2.26% of total 7.4B parameters)
   - Est. completion: ~15-16 hours (started Aug 6, 07:49 UTC)

### What's Currently Running 🔄
- **Active Training Process**: PID 613610 on instance i-01fa5b57d64c6196a
- **Progress Monitoring**: S3://asoba-llm-cache/training-runs/mistral-production-*
- **Log Location**: `/mnt/training/mistral_training/training_production.log`
- **Model Output**: `/mnt/training/outputs/` → S3://asoba-llm-cache/mistral-finetuned/

### What's Next 📋
1. **Monitor training progress** (~15 hours remaining)
2. **Validate trained model** performance and outputs  
3. **Upload final model** to S3 for deployment
4. **Performance evaluation** on test data

## Critical Context for Continuation

### Directory Structure
```
/mnt/training/
├── mistral_training/       # Main training scripts
│   ├── prepare_mistral_dataset.py
│   ├── train_mistral.py
│   └── .env               # From /home/shingai/sort/deployments/
├── data_prep/             # Data preparation workspace
│   ├── operatives_dataset.jsonl  # 50k validated entries
│   ├── policy_dataset.jsonl      # Regenerated clean data
│   └── pipeline_state.json       # Processing state
└── models/
    └── mistral-7b-v0.3/  # Base model
```

### S3 Structure
```
s3://policy-database/      # Source data
├── corpus_federal/
├── econ-theory/
├── financial-metrics/     # Include processed/failed folders
├── financial-networks/    # Include all subfolders
├── insurance/
├── government_officials_roster/
├── usa/congressional-research/
└── operatives/           # Load LAST, 50k limit

s3://asoba-llm-cache/     # Outputs
├── datasets/mistral-verbosity/
└── training-runs/        # Monitoring metadata
```

### Key Scripts & Commands

#### 1. Data Preparation (Current Stage)
```bash
cd /mnt/training/mistral_training
python prepare_mistral_dataset.py \
    --bucket policy-database \
    --output-dir /mnt/training/data_prep \
    --s3-output asoba-llm-cache/datasets/mistral-verbosity
```

#### 2. Pre-flight Validation
```bash
cd /home/shingai/sort/llm-training
python scripts/preflight_check.py \
    --instance-id i-01fa5b57d64c6196a \
    --instance-ip 54.197.142.172 \
    --ssh-key config/mistral-base.pem
```

#### 3. Training Execution (Next Step)
```bash
cd /mnt/training/mistral_training
python train_mistral.py \
    --model-path /mnt/training/models/mistral-7b-v0.3 \
    --dataset-path /mnt/training/data_prep \
    --output-dir /mnt/training/outputs \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

#### 4. Monitoring
```bash
# Check heartbeat status
aws s3api head-object \
    --bucket asoba-llm-cache \
    --key training-runs/mistral-$(date +%Y%m%d)/heartbeat.json \
    --region us-east-1
```

## Known Issues & Solutions

### Issue 1: Data Validation Failures
**Symptom**: "Missing or empty 'text' field"  
**Cause**: Garbage data in source folders  
**Solution**: Clean data, exclude processed/failed folders except financial-metrics

### Issue 2: Incomplete Policy Data
**Symptom**: Only operatives data being used  
**Cause**: Policy processing marked complete with 0 entries  
**Solution**: Remove pipeline_state.json and regenerate

### Issue 3: Monitoring Timeouts
**Symptom**: TIMED_OUT status during long operations  
**Cause**: Downloads not updating heartbeat  
**Solution**: HeartbeatManager integration in prepare_mistral_dataset.py

## Critical Configuration

### Environment Variables (.env)
```bash
HF_TOKEN=<huggingface_token>
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=asoba-llm-cache
MONITORING_PREFIX=training-runs/
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Training Parameters (ACTIVE CONFIG)
- **Model**: Mistral-7B-v0.3
- **Method**: QLoRA (4-bit quantization with BitsAndBytes)
- **GPU Memory**: 24GB NVIDIA A10G
- **Batch Size**: 1 (memory optimized)
- **Max Sequence Length**: 512 (memory optimized)
- **Trainable Parameters**: 167,772,160 (2.26% of total)
- **Total Training Steps**: 30,969
- **Estimated Training Time**: 15-16 hours

## Data Processing Rules
1. **Policy Folders First**: Process all policy data before operatives
2. **Operatives Last**: Load with 50,000 file limit
3. **Exclusions**: Skip processed/failed folders except in financial-metrics
4. **Format**: JSONL with 'text' field required
5. **Validation**: Each entry must have non-empty text field

## Testing Requirements (per CLAUDE.md)
- **Integration Tests**: Named `test_int_*`, hit real boundaries
- **E2E Tests**: Named `test_e2e_*`, run actual entrypoints
- **No Internal Mocks**: Only mock true I/O boundaries
- **Pre-flight**: Run before any deployment
- **Heartbeat**: Monitor all operations >3 minutes

## Recovery Procedures

### If Training Fails
1. Check logs: `/mnt/training/mistral_training/logs/`
2. Verify GPU: `nvidia-smi`
3. Check disk: `df -h /mnt/training`
4. Review heartbeat: Check S3 metadata for last update
5. Run pre-flight: Validate all components

### If Data Issues
1. Check pipeline state: `cat /mnt/training/data_prep/pipeline_state.json`
2. Validate datasets: Run validation script
3. Review S3 sources: List bucket contents
4. Clean and regenerate: Remove state file, rerun prep

## Contact & Resources
- **Repository**: /home/shingai/sort/llm-training
- **Previous Context**: See git commits a1c3391, 4167e66
- **Testing Framework**: scripts/preflight_check.py
- **Monitoring**: scripts/shared/heartbeat_manager.py
- **Documentation**: docs/failure-matrix.md

## CRITICAL: Training Process Management

### ✅ CORRECTED: PyTorch DataLoader Worker Process Model
**Training creates expected worker processes - this is normal PyTorch behavior.**

#### Process Architecture
- **1 Trainer Process (Session Leader)**: Main training logic, model updates, checkpointing
- **4 DataLoader Workers**: Parallel data loading, preprocessing, batching
- **Total Expected**: 5 processes (1 trainer + 4 workers)

The training configuration sets `dataloader_num_workers: 4` in `get_training_config()`, which spawns 4 worker processes to parallelize data loading. This is standard PyTorch optimization, not a bug.

### Correct Process Detection & Management

#### 1. **Count Only Trainer Processes (Session Leaders)**
```bash
# Count trainers only - should return 1 for normal operation
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 \
  "ps -eo pid,sid,cmd | awk '/train_mistral_simple\.py/ && \$1==\$2 {print \$1}' | wc -l"
```

#### 2. **View Complete Process Hierarchy**
```bash
# Show trainer and its workers with process relationships
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 \
  "ps -eo pid,ppid,sid,etimes,cmd | awk '/train_mistral_simple\.py/ {print}' | sort -k4,4nr"
```

Expected output format:
```
  PID  PPID   SID ELAPSED CMD
636471     1 636469    0:18 python3 /mnt/training/mistral_training/train_mistral_simple.py ...  [TRAINER]
636501 636471 636469    0:15 python3 /mnt/training/mistral_training/train_mistral_simple.py ...  [WORKER 1]  
636502 636471 636469    0:15 python3 /mnt/training/mistral_training/train_mistral_simple.py ...  [WORKER 2]
636503 636471 636469    0:15 python3 /mnt/training/mistral_training/train_mistral_simple.py ...  [WORKER 3]
636504 636471 636469    0:15 python3 /mnt/training/mistral_training/train_mistral_simple.py ...  [WORKER 4]
```

#### 3. **Safe Duplicate Trainer Cleanup**
Only use when multiple **trainers** (session leaders) are detected:

```bash
# Transfer and run improved cleanup script
scp -i config/mistral-base.pem safe_kill_duplicates.sh ubuntu@54.197.142.172:/tmp/
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 "tr -d '\r' < /tmp/safe_kill_duplicates.sh > /tmp/safe_kill.sh && chmod +x /tmp/safe_kill.sh"

# Always dry-run first
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 "/tmp/safe_kill.sh --dry-run"

# Execute only if multiple trainers confirmed
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 "/tmp/safe_kill.sh --force"
```

**Key Improvement**: Script now preserves all worker processes of the kept trainer while eliminating duplicate trainers and their workers.

### Training Progress Monitoring

#### 1. **Verify Normal Process Count**
```bash
# Count trainer processes (should be 1)
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 \
  "ps -eo pid,sid,cmd | awk '/train_mistral_simple\.py/ && \$1==\$2 {print \$1}' | wc -l"

# Count all training-related processes (should be 5: 1 trainer + 4 workers)
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 "pgrep -cf train_mistral_simple.py"
```

**Normal Output**: Trainers=1, Total=5

#### 2. **Monitor Training Progress** 
```bash
# Real-time log monitoring
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 "tail -f /home/ubuntu/logs/mistral_train.log"

# Parse latest step progress
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 \
  "tac /home/ubuntu/logs/mistral_train.log | grep -m1 -Eo '[0-9]+/[0-9]{4,}' || echo 'No progress found'"
```

#### 3. **S3 Heartbeat Monitoring** (Real Progress via HeartbeatCallback)
```bash
# Check heartbeat metadata (includes real step/loss data)
aws s3api get-object \
  --bucket asoba-llm-cache \
  --key training-runs/mistral-production-20250806-163706/metadata.json \
  --region us-east-1 /tmp/heartbeat.json && cat /tmp/heartbeat.json | jq '.'
```

#### 4. **Checkpoint Recovery Readiness**
```bash
# List available checkpoints
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 \
  "ls -1dt /mnt/training/outputs/*/checkpoint-* 2>/dev/null | head -5"

# Verify trainer state for resume capability  
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 \
  "find /mnt/training/outputs -name trainer_state.json -exec echo 'Found: {}' \;"
```

### Fixed Launch Command (Prevents Duplicates)

```bash
# Use singleton lock to prevent duplicate launches
ssh -i config/mistral-base.pem ubuntu@54.197.142.172 "
exec 200>/var/lock/mistral_train.lock
flock -n 200 || { echo 'training already running'; exit 1; }
RUN_ID='mistral-production-20250806-025008'
OUTDIR='/mnt/training/outputs/\${RUN_ID}'
nohup python3 /mnt/training/mistral_training/train_mistral_simple.py \
  --train-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/final_train.jsonl \
  --val-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/final_val.jsonl \
  --model-name /mnt/training/models/mistral-7b-v0.3 \
  --output-dir \"\$OUTDIR\" \
  --s3-bucket asoba-llm-cache \
  --s3-prefix mistral-finetuned \
  --batch-size 1 \
  --max-seq-length 512 \
  --run-id \"\$RUN_ID\" \
  >> /var/log/mistral_train.log 2>&1 &
"
```

## Current Status Summary  
**✅ TRAINING ACTIVE - CORRECTED PROCESS MODEL**

### What Actually Happened (Corrected Understanding)
- **Original Issue**: Misidentified normal PyTorch DataLoader workers (4 workers + 1 trainer = 5 processes) as "process multiplication bug"
- **Incorrect Action**: Killed legitimate DataLoader workers, causing training to crash after step 47
- **Root Cause**: Lack of understanding of PyTorch multiprocessing architecture
- **Current Status**: Training restarted successfully with proper process model understanding

### Key Corrections Made
1. **Process Detection**: Now counts session leaders only (`SID==PID`) to identify trainers vs workers
2. **Safe Cleanup Script**: Updated to preserve worker processes of legitimate trainers  
3. **Monitoring**: Proper distinction between trainer count (should be 1) and total processes (should be 5)
4. **Documentation**: Corrected mental model documented with accurate process hierarchy

### Current Training Details
- **Trainer PID**: 636471 (session leader)  
- **Workers**: 4 DataLoader processes (children of trainer)
- **Total Processes**: 5 (normal and expected)
- **Progress**: Model loaded, datasets processed, training in progress
- **Monitoring**: S3 heartbeat + HeartbeatCallback providing real step/loss data

### Lessons Learned  
- ✅ **PyTorch DataLoader workers are normal** - 4 workers + 1 trainer = 5 processes expected
- ✅ **Session leader detection** (`SID==PID`) distinguishes trainers from workers  
- ✅ **Process hierarchy understanding** essential for safe management
- ✅ **Never kill worker processes** of a legitimate trainer
- ❌ **Misdiagnosing normal behavior as bugs** leads to destructive actions

**Training pipeline now operating correctly with proper process model understanding.**