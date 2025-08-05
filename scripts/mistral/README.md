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

### One-Shot Training (Recommended)

```bash
# Launch new instance and run complete pipeline
./scripts/mistral/run_mistral_training_pipeline.sh

# Use existing g5.2xlarge instance
./scripts/mistral/run_mistral_training_pipeline.sh \
  --instance-id i-01fa5b57d64c6196a \
  --instance-ip 54.197.142.172 \
  --ssh-key config/mistral-base.pem
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

**Script**: `prepare_mistral_dataset.py`

Downloads and processes data from 7 policy domains with operatives loaded LAST:

```python
policy_sources = [
    {"name": "corpus_federal", "s3_path": "s3://policy-database/corpus_7-26-2025/federal/"},
    {"name": "econ_theory", "s3_path": "s3://policy-database/econ-theory/"},
    {"name": "financial_metrics", "s3_path": "s3://policy-database/financial-metrics/"},
    {"name": "healthcare_policy", "s3_path": "s3://policy-database/healthcare-policy/"},
    {"name": "operational_data", "s3_path": "s3://policy-database/operational-data/"},
    {"name": "regulatory_compliance", "s3_path": "s3://policy-database/regulatory-compliance/"},
    {"name": "operatives", "s3_path": "s3://policy-database/operatives/", "max_files": 50000}
]
```

**Features**:
- Processes JSON, JSONL, TXT, CSV files
- Smart deduplication by content hash
- 80/20 train/validation split
- Outputs to S3: `s3://asoba-llm-cache/datasets/mistral-verbosity/`

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