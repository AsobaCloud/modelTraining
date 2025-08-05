# Mistral Training Pipeline - Current Working State

## Overview

The current Mistral training pipeline consists of three working scripts that implement a two-phase approach: separate data preparation and training execution.

## Golden Path Architecture

### Phase 1: Data Preparation
**Script**: `scripts/mistral/prepare_mistral_dataset.py`

**Purpose**: Downloads and processes corpus data from S3, combines with operatives data, creates train/val splits

**Key Features**:
- Downloads corpus from `s3://asoba-llm-cache/corpus/` (default bucket)
- Downloads operatives from `s3://policy-database/operatives/`
- Processes up to 50,000 PDFs by default
- Creates 80/20 train/validation split
- Outputs to S3 at `s3://{bucket}/datasets/mistral-verbosity/`
- Stateful pipeline with resumability (`pipeline_state.json`)

**Usage**:
```bash
python3 prepare_mistral_dataset.py \
  --output-bucket asoba-llm-cache \
  --max-pdfs 50000 \
  --validation-split 0.2
```

**Outputs**:
- `train_dataset.jsonl` → `s3://asoba-llm-cache/datasets/mistral-verbosity/train_dataset.jsonl`
- `val_dataset.jsonl` → `s3://asoba-llm-cache/datasets/mistral-verbosity/val_dataset.jsonl`

### Phase 2: Training Execution
**Script**: `scripts/mistral/train_mistral_simple.py`

**Purpose**: Downloads prepared datasets from S3 and executes Mistral 7B QLoRA training

**Key Features**:
- Downloads datasets from S3 paths provided as arguments
- Uses Mistral-7B-v0.3 as base model
- 4-bit quantization with BitsAndBytesConfig
- QLoRA with proven configuration (r=64, alpha=16, dropout=0.1)
- Uploads trained model back to S3

**Usage**:
```bash
python3 train_mistral_simple.py \
  --train-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/train_dataset.jsonl \
  --val-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/val_dataset.jsonl \
  --output-dir ./mistral_simple_output
```

**Configuration**:
- Batch size: 4 (adjustable)
- Sequence length: 1024 tokens
- Learning rate: 2e-4
- Target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

### Phase 3: Deployment
**Script**: `scripts/mistral/deploy_mistral_to_g5.sh`

**Purpose**: Deploys training environment to g5.2xlarge instance

**Key Features**:
- Transfers deployment package to instance
- Creates setup and monitoring scripts
- Validates environment before training
- Expects Mistral-7B-v0.3 at `/home/ubuntu/mistral-7b-v0.3`

**Usage**:
```bash
./deploy_mistral_to_g5.sh <instance_ip> <ssh_key_path>
```

**Dependencies**:
- Requires `train_mistral_verbosity_qlora.py` (different from `train_mistral_simple.py`)
- Expects `verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl` dataset file
- Uses `validate_mistral_golden_config.py` for pre-flight checks

## Current Pipeline Issues

### Inconsistencies Identified

1. **Deployment Script Mismatch**:
   - `deploy_mistral_to_g5.sh` expects `train_mistral_verbosity_qlora.py`
   - But current training script is `train_mistral_simple.py`
   - Different dataset expectations

2. **Dataset Path Conflicts**:
   - Data prep outputs to S3
   - Training script expects S3 inputs
   - Deployment script expects local JSONL file

3. **Missing Files**:
   - `train_mistral_verbosity_qlora.py` referenced in deployment but not present
   - `verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl` expected but not generated

## Working Execution Path

Based on current scripts, the working path is:

1. **Run Data Preparation**:
   ```bash
   python3 scripts/mistral/prepare_mistral_dataset.py --output-bucket asoba-llm-cache
   ```

2. **Run Training**:
   ```bash
   python3 scripts/mistral/train_mistral_simple.py \
     --train-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/train_dataset.jsonl \
     --val-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/val_dataset.jsonl
   ```

3. **Deploy** (requires manual instance setup, deployment script has mismatched dependencies)

## S3 Bucket Structure

```
s3://asoba-llm-cache/
├── corpus/                     # Input corpus files (JSONL)
├── datasets/mistral-verbosity/ # Processed training datasets
│   ├── train_dataset.jsonl
│   └── val_dataset.jsonl
└── models/mistral-verbosity/   # Trained model outputs

s3://policy-database/
└── operatives/                 # PDF archives for processing
    ├── *.zip
    └── *.tar.gz
```

## Resource Requirements

- **Data Preparation**: Any instance with AWS CLI access, ~50GB storage
- **Training**: g5.2xlarge minimum (8 vCPU, 1x A10G GPU, 32GB RAM)
- **Storage**: 100GB EBS for training artifacts
- **Time**: 2-4 hours training after data prep completion

## Next Steps for Clean Execution

1. **Verify S3 Bucket Access**: Confirm `asoba-llm-cache` and `policy-database` accessibility
2. **Run Data Preparation**: Execute corpus download and processing
3. **Launch Training Instance**: Use infrastructure scripts or manual setup
4. **Execute Training**: Run with S3 dataset paths from step 2
5. **Model Validation**: Test trained model outputs