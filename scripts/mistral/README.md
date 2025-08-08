# Mistral Training Pipeline

Production-ready pipeline for training Mistral-7B on policy analysis data with comprehensive monitoring.

## Overview

The Mistral pipeline implements a complete training workflow with:
- Multi-source policy data collection (7 domains)
- Operatives-last data processing for optimal training
- One-shot deployment script with monitoring integration
- Production-grade failure detection and alerts
- S3-based model and dataset management

## Quick Start

### Current Dataset Status (Ready for Training)

**Instance**: i-01fa5b57d64c6196a (54.197.142.172)  
**Datasets**: ✅ Ready and validated
- **Training**: `/mnt/training/data_prep/final_train.jsonl` (41,294 entries)
- **Validation**: `/mnt/training/data_prep/final_val.jsonl` (10,323 entries)  
- **Total**: 51,617 entries (50k operatives + 1,617 clean policy data)
- **Status**: All datasets pass validation with required 'text' fields

### One-Shot Training (Recommended)

```bash
# Launch new instance and run complete pipeline
./scripts/mistral/run_mistral_training_pipeline.sh

# Use existing g5.2xlarge instance (CURRENT SETUP)
./scripts/mistral/run_mistral_training_pipeline.sh \
  --instance-id i-01fa5b57d64c6196a \
  --instance-ip 54.197.142.172 \
  --ssh-key config/mistral-base.pem \
  --use-existing-data  # Use prepared datasets
```

### Monitor Training

```bash
# With Slack alerts (recommended)
./scripts/monitoring/production_monitor.sh mistral-20250804-171621

# Basic monitoring
python3 scripts/monitoring/monitor.py --run-id mistral-20250804-171621 --watch
```

## Pipeline Architecture

### Phase 1: Data Collection

**Script**: `prepare_mistral_dataset.py` + `regenerate_policy_data.py`

Downloads and processes data from 8 policy domains with operatives loaded LAST:

```python
policy_sources = [
    {"name": "econ_theory", "s3_path": "s3://policy-database/econ-theory/"},
    {"name": "financial_metrics", "s3_path": "s3://policy-database/financial-metrics/"},
    {"name": "government_officials", "s3_path": "s3://policy-database/government-officials/"},
    {"name": "international_relations", "s3_path": "s3://policy-database/international-relations/"},
    {"name": "national_security", "s3_path": "s3://policy-database/national-security/"},
    {"name": "policy_debate", "s3_path": "s3://policy-database/policy-debate/"},
    {"name": "public_welfare", "s3_path": "s3://policy-database/public-welfare/"},
    {"name": "behavioral_economics", "s3_path": "s3://policy-database/behavioral-economics/"},
    {"name": "operatives", "s3_path": "s3://operatives-datasets/", "max_files": 50000}
]
```

**Data Integrity Features**:
- **Clean Data Processing**: Excludes `processed/` and `failed/` folders (except financial-metrics/processed)
- **Garbage Prevention**: Excludes `behavioral-economics/news/` folder containing non-policy content
- **Validation**: All entries require 'text' field with actual content
- Processes JSON, JSONL, TXT, PDF files
- Smart deduplication by content hash
- 80/20 train/validation split
- **Final Dataset**: 51,617 entries (50k operatives + 1,617 policy) that pass validation

### Phase 2: Model Training

**Script**: `train_mistral_simple.py`

Trains Mistral-7B-v0.3 with QLoRA optimization:

**Configuration**:
- **Model**: Downloaded from S3 to `/mnt/training/models/mistral-7b-v0.3`
- **Quantization**: 4-bit with BitsAndBytesConfig
- **LoRA**: r=64, alpha=16, dropout=0.1
- **Batch Size**: 1 (for g5.2xlarge stability)
- **Sequence Length**: 1024 tokens
- **Learning Rate**: 2e-4 with cosine scheduler

**Monitoring Integration**:
- Writes error sentinels on failure
- Completion markers on success
- Progress updates to S3

### Phase 3: Infrastructure

**One-Shot Script**: `run_mistral_training_pipeline.sh`

Complete pipeline automation:

1. **Instance Management**
   - Launches g5.2xlarge in us-east-1 OR uses existing
   - 300GB EBS volume at `/mnt/training`
   - Auto-configures security groups and SSH

2. **Environment Setup**
   - PyTorch 2.5.1 with CUDA 12.1
   - Transformers, PEFT, BitsAndBytes
   - Downloads model from S3 (no Hugging Face!)

3. **Monitoring Integration**
   - Posts metadata to S3 immediately
   - Dual heartbeat updates (metadata + progress)
   - Error sentinels for failure detection
   - Direct integration with production monitor

## Storage Layout

### EBS Volume (`/mnt/training/`)
```
/mnt/training/
├── models/
│   └── mistral-7b-v0.3/          # Base model from S3
├── data_prep/                    # Temporary data processing
├── mistral_output/               # Training outputs
└── mistral_training/             # Scripts and logs
```

### S3 Structure & Navigation

#### Primary Buckets

**`s3://asoba-llm-cache/`** - Main training bucket
```
├── models/
│   └── mistralai/Mistral-7B-v0.3/    # Base model (27GB)
│       ├── config.json
│       ├── model.safetensors.index.json
│       ├── model-00001-of-00003.safetensors
│       ├── model-00002-of-00003.safetensors
│       └── model-00003-of-00003.safetensors
├── datasets/mistral-verbosity/
│   ├── train_dataset.jsonl            # ~2-5GB depending on sources
│   └── val_dataset.jsonl              # ~500MB-1GB
├── models/mistral-verbosity/          # Fine-tuned models
│   └── adapter_model.bin              # LoRA weights only (~200MB)
└── training-runs/{run-id}/            # Monitoring data
    ├── metadata.json                  # Run status and config
    ├── progress.json                  # Training metrics
    ├── logs/
    │   └── training_log_latest.txt    # Last 1000 lines
    ├── _error                         # Error details (if failed)
    └── _complete                      # Success marker (if completed)
```

**`s3://policy-database/`** - Source data
```
├── corpus_7-26-2025/federal/          # Federal policy documents
├── econ-theory/                       # Economic theory papers
├── financial-metrics/                 # Financial analysis
├── healthcare-policy/                 # Healthcare regulations
├── operational-data/                  # Operational reports
├── regulatory-compliance/             # Compliance documents
└── operatives/                        # Policy operative archives
    ├── operative_batch_001.zip
    ├── operative_batch_002.tar.gz
    └── ... (loaded LAST in training)
```

#### Navigation Commands

**Browse S3 in Console**:
```bash
# Quick links (adjust run-id):
https://s3.console.aws.amazon.com/s3/buckets/asoba-llm-cache?prefix=training-runs/mistral-20250804-171621/
https://s3.console.aws.amazon.com/s3/buckets/asoba-llm-cache?prefix=datasets/mistral-verbosity/
https://s3.console.aws.amazon.com/s3/buckets/asoba-llm-cache?prefix=models/
```

**CLI Navigation**:
```bash
# List all training runs
aws s3 ls s3://asoba-llm-cache/training-runs/

# Check specific run status
aws s3 ls s3://asoba-llm-cache/training-runs/mistral-20250804-171621/

# View run metadata
aws s3 cp s3://asoba-llm-cache/training-runs/mistral-20250804-171621/metadata.json - | jq .

# Check if training completed
aws s3 ls s3://asoba-llm-cache/training-runs/mistral-20250804-171621/_complete

# View error details (if failed)
aws s3 cp s3://asoba-llm-cache/training-runs/mistral-20250804-171621/_error -

# List available datasets
aws s3 ls s3://asoba-llm-cache/datasets/mistral-verbosity/

# Check dataset sizes
aws s3 ls s3://asoba-llm-cache/datasets/mistral-verbosity/ --human-readable

# List trained models
aws s3 ls s3://asoba-llm-cache/models/mistral-verbosity/ --recursive

# Download training logs
aws s3 cp s3://asoba-llm-cache/training-runs/mistral-20250804-171621/logs/training_log_latest.txt .
```

#### Finding Your Run ID

Run IDs follow the pattern: `mistral-YYYYMMDD-HHMMSS`

1. **From pipeline output**:
   ```
   Run ID: mistral-20250804-171621
   ```

2. **From saved info file**:
   ```bash
   cat mistral_training_info.txt | grep "Run ID"
   ```

3. **List recent runs**:
   ```bash
   aws s3 ls s3://asoba-llm-cache/training-runs/ | sort -r | head -10
   ```

#### Key Files to Check

**During Training**:
- `metadata.json` - Current phase and status
- `progress.json` - Step count, loss, metrics
- `logs/training_log_latest.txt` - Recent output

**After Completion**:
- `_complete` - Success marker (exists = success)
- `_error` - Failure details (exists = failed)
- `models/mistral-verbosity/` - Final model weights

## Monitoring & Alerts

The pipeline integrates with production monitoring for:

- **Silent Failure Detection**: Dual heartbeat monitoring
- **Actionable Alerts**: Slack notifications with remediation steps
- **Error Details**: Automatic capture of failure reasons
- **Direct S3 Links**: Quick access to logs and artifacts

Example alert:
```
🔥 Mistral Training FAILED
Run ID: mistral-20250804-171621
Status: FAILED
Reason: Error: CUDA out of memory
🔧 Action Required: Reduce batch size or sequence length
Instance: i-01fa5b57d64c6196a
Phase: training
Duration: 45 minutes
Last Log: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
📊 S3 Logs: View in S3 Console
```

## Hardware Requirements

- **Instance**: g5.2xlarge (8 vCPU, 32GB RAM, 1x A10G GPU)
- **Storage**: 300GB EBS (gp3)
- **Region**: us-east-1
- **Time**: ~30min setup, 2-4 hours training

## Data Validation & Quality Assurance

### Recent Issues Fixed

#### Logger Crash During Data Preparation
**Problem**: Pipeline crashed with "NameError: name 'logger' is not defined" during PyPDF2 import fallback
**Root Cause**: Logger was used in except block before being initialized
**Solution**: Moved logging setup before all imports in `regenerate_policy_data.py`

#### Training Configuration Drift
**Problem**: Multiple config mismatches causing OOM and false success markers
**Root Cause**: Hardcoded values overriding CLI args, dtype mismatches, premature completion markers
**Solutions**:
- Honor `--batch-size` and `--max-seq-length` CLI arguments
- Switch to bf16 throughout (matches Ampere GPU + model dtype)
- Write completion markers only on actual success
- Make heartbeat imports optional

#### Monitoring Script Argument Errors
**Problem**: `monitor.py: error: unrecognized arguments: --watch`
**Root Cause**: Production monitor wrapper passed unsupported --watch flag
**Solution**: Updated `production_monitor.sh` to use supported arguments (--interval 60)

#### Data Source Updates
**Current Sources**: Updated to reflect actual S3 bucket structure
- **USA Policy**: `s3://policy-database/usa/`
- **South Africa**: `s3://policy-database/south_africa/`
- **Economic Theory**: `s3://policy-database/econ-theory/`
- **Federal Bills**: `s3://policy-database/corpus_7-26-2025/federal/content/`
- **Insurance**: `s3://policy-database/insurance/`
- **Financial Metrics**: `s3://policy-database/financial-metrics/processed/` (processed folder included)
- **Recent News**: `s3://policy-database/news/2025-07-30/`
- **Operatives**: `s3://policy-database/operatives/` (50+ classified collections)

#### Data Processing Improvements
**Script**: `regenerate_policy_data.py`
```bash
# Usage on EC2 instance:
python3 regenerate_policy_data.py --work-dir /mnt/training/data_prep
```

**Enhancements**:
- Global SHA1 hash-based deduplication across all sources
- Proper exclusion of processed/failed folders (except financial-metrics)
- Systematic text field validation
- Support for PDF extraction (first 5 pages)
- Detailed logging of duplicates and processing statistics

### Dataset Validation Commands

```bash
# Validate individual datasets
cd /mnt/training/mistral_training
python3 -c "from shared.dataset_utils import validate_dataset_format; print(validate_dataset_format('/path/to/dataset.jsonl'))"

# Check dataset sizes and content
wc -l /mnt/training/data_prep/final_*.jsonl
head -1 /mnt/training/data_prep/final_train.jsonl | jq .

# Verify no garbage content
grep -i "workout\|aluminum\|news" /mnt/training/data_prep/final_*.jsonl || echo "✅ No garbage content found"
```

## Troubleshooting

### Common Issues

1. **CUDA OOM**
   - Reduce batch size in training script
   - Current default: 1 (minimum for stability)

2. **Stuck in Setup Phase**
   - Check instance has internet access
   - Verify S3 permissions
   - Model download can take 15-30 minutes

3. **Data Preparation Fails**
   - Check S3 bucket permissions
   - Verify policy folders exist
   - Ensure sufficient EBS space

4. **Dataset Validation Failures** (NEW)
   - Use `regenerate_policy_data.py` for clean data processing
   - Verify all entries have required 'text' field
   - Check for excluded processed/failed folders

### Debug Commands

```bash
# SSH to instance
ssh -i config/mistral-base.pem ubuntu@<instance-ip>

# Check training logs
cd /mnt/training/mistral_training
tail -f training.log

# Monitor GPU
nvidia-smi -l 1

# Check S3 monitoring
aws s3 ls s3://asoba-llm-cache/training-runs/<run-id>/
```

## Advanced Configuration

### Custom Data Sources

Edit `prepare_mistral_dataset.py` to add new sources:

```python
policy_sources.append({
    "name": "custom_domain",
    "s3_path": "s3://your-bucket/path/",
    "max_files": 10000  # Optional limit
})
```

### Training Parameters

Modify in `train_mistral_simple.py`:

```python
# Batch size (memory constrained)
parser.add_argument("--batch-size", type=int, default=1)

# Sequence length
parser.add_argument("--max-seq-length", type=int, default=1024)

# In get_training_config():
"num_train_epochs": 3,
"learning_rate": 2e-4,
```

### Alert Configuration

Set Slack webhook in `production_monitor.sh`:

```bash
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

## Best Practices

1. **Always use EBS volume** - Home directory is too small
2. **Monitor from start** - Launch monitoring immediately after pipeline
3. **Check datasets exist** - Pipeline skips data prep if datasets found
4. **Use existing instances** - Saves time and preserves state
5. **Review alerts promptly** - Quick action prevents wasted GPU time

## Memory Optimization (CRITICAL)

### OOM Prevention During Tokenization

The training script has been optimized to prevent OOM kills on g5.4xlarge (64GB RAM):

#### Problem
Training would get OOM killed at ~53% tokenization, using 27.8GB RAM because:
- Effective batch size was too large: `4 batch_size × 1024 seq_len = 4096 sequence batch`
- This exceeded memory capacity during tokenization phase

#### Solution (Official Mistral Best Practice)
Following official mistral-finetune repository guidance for CUDA OOM:

1. **Reduce batch size**: `per_device_train_batch_size=1` (was 4)
2. **Reduce sequence length**: `max_seq_length=512` (was 1024)
3. **Increase gradient accumulation**: `gradient_accumulation_steps=16` (was 4)
4. **Net effect**: Sequence batch reduced from 4096 to 512 (8x reduction)

**Source**: "If you encounter a CUDA out-of-memory error, one possible solution is to reduce the batch size per GPU. The batch size is equal to seq_len x batch_size." - Official Mistral docs

### Official Mistral Configuration

- **Dataset**: 89,948 training + 10,323 validation entries (normalized)
- **Instance**: g5.4xlarge (64GB RAM, 24GB GPU)
- **Batch size**: 1 (official Mistral OOM fix)
- **Sequence length**: 512 tokens (official Mistral OOM fix)  
- **Gradient accumulation**: 16 steps (maintains effective batch size)
- **Effective batch**: 1 × 512 = 512 (vs previous 4096)
- **Expected result**: 8x reduction in memory usage during tokenization