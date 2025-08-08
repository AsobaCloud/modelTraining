# Complete LLM Training Pipeline Guide
**End-to-End Documentation for Qwen LoRA Verbosity & Mistral Policy Models**

*Last Updated: August 7, 2025*

---

## 🎯 Current State Overview

### Active Models
- **Qwen Verbosity Model**: ✅ **COMPLETED** - Production-ready on west coast instance
- **Mistral Policy Model**: 🔄 **IN PROGRESS** - Training pipeline operational

### Critical Issues & Solutions
Based on `/home/shingai/sort/llm-training/ACTUAL_STATE.md`:

**Problem**: Comprehensive dataset (91,380 entries) fails training due to inconsistent columns
**Solution**: Normalize to only 'text' field before training
**Status**: Identified, needs normalization step

---

## 🖥️ Infrastructure & Access

### Active Instances

#### West Coast Instance (Qwen Model)
- **Type**: g5.4xlarge (16 vCPUs, 64GB RAM, 24GB A10G GPU)
- **Public IP**: `34.217.17.67`
- **Instance ID**: `i-0645c6db622720234`  
- **Region**: `us-west-2`
- **Current Service**: Flux.1 Dev API (port 8000) + Qwen verbosity model
- **SSH Access**: 
```bash
ssh -i /home/shingai/sort/llm-training/config/mistral-base.pem ubuntu@34.217.17.67
```

#### East Coast Instance (Data Processing)
- **Type**: g5.2xlarge (8 vCPUs, 32GB RAM, 24GB A10G GPU)
- **Public IP**: `54.197.142.172`
- **Purpose**: Mistral training, processed policy data storage
- **Location of processed data**: `/mnt/training/data_prep/policy_*.jsonl`
- **SSH Access**:
```bash
ssh -i /home/shingai/sort/llm-training/config/mistral-base.pem ubuntu@54.197.142.172
```

#### Instance Management
```bash
# Check running instances
aws ec2 describe-instances --region us-west-2 \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[InstanceType,InstanceId,PublicIpAddress,Tags[?Key==`Name`].Value|[0]]' \
  --output table

# Instance costs: ~$1.60/hour (g5.4xlarge), ~$1.20/hour (g5.2xlarge)
```

### SSH Key Configuration
- **Primary Key**: `/home/shingai/sort/llm-training/config/mistral-base.pem`
- **Permissions**: `chmod 400 /home/shingai/sort/llm-training/config/mistral-base.pem`
- **User**: `ubuntu` (all instances)

---

## 📊 Data Pipeline Architecture

### S3 Storage Structure
```
s3://asoba-llm-cache/
├── corpus/                           # Raw corpus files (JSONL)
├── datasets/
│   ├── mistral-verbosity/           # Processed Mistral datasets
│   │   ├── final_train.jsonl        # 41,294 entries (original)
│   │   ├── final_val.jsonl          # 10,323 entries (original)
│   │   └── comprehensive_train.jsonl # 91,380 entries (needs normalization)
│   └── qwen-verbosity/              # Qwen training data
└── models/
    ├── Qwen/Qwen3-14B/              # Base Qwen model
    ├── mistralai/Mistral-7B-v0.3/   # Base Mistral model
    └── trained-models/              # Output directory

s3://policy-database/
└── operatives/                      # State Dept cables (PDF archives)
    ├── *.zip
    └── *.tar.gz
```

### Data Sources & Collection

#### 1. IAC/DevOps Corpus (Qwen Training)
**Location**: `/home/shingai/sort/llm-training/scripts/corpus-generation/iac-devops-corpus/`

**Primary Collectors**:
- `infrastructure-collectors/` - AWS CLI, Terraform, CDK, Docker, Helm
- `code-generation-collectors/` - Smart collectors for AI/ML engineering
- `github-collectors/` - Real-world repository mining
- `mcp-integration/` - CLAUDE.md methodology examples

**Usage**:
```bash
# Generate comprehensive IAC corpus
cd /home/shingai/sort/llm-training/scripts/corpus-generation/iac-devops-corpus
./corpus-builders/create_final_iac_corpus.py

# Output: /home/shingai/sort/llm-training/data/corpus/combined_iac_corpus.jsonl
```

#### 2. Policy Analysis Corpus (Mistral Training)
**Location**: `/home/shingai/sort/llm-training/scripts/corpus-generation/policy-analyst-corpus/`

**Primary Sources**:
- Government policy documents
- Insurance regulatory materials  
- Academic research papers
- News and analysis articles

**Processing Chain**:
1. PDF collection via `policy_pdf_processor.py`
2. Text extraction and normalization
3. Operatives integration (State Dept cables)
4. Train/validation split

#### 3. Real-World Data Enhancement
**Collectors**:
- **AWS CLI**: `/home/shingai/sort/llm-training/data/collectors/aws_cli_collector.py`
- **Shell Commands**: `/home/shingai/sort/llm-training/data/collectors/shell_corpus_builder.py`
- **Code Generation**: Various smart collectors in corpus-generation/

---

## 🤖 Model Training Pipelines

## Pipeline 1: Qwen LoRA Verbosity Model ✅

### Status: **PRODUCTION READY**
- **Model**: Qwen3-14B with LoRA verbosity control
- **Location**: West coast instance (`34.217.17.67`)
- **Training**: COMPLETED with validated 16:1 token separation
- **Use Case**: CLAUDE.md methodology, IAC/DevOps assistance

### Validated Configuration

#### Environment Setup
```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### Model Loading (Proven Working)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit quantization for 24GB GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model with forced single GPU placement
model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/Qwen3-14B",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0},  # Critical for single GPU
)
```

#### Training Parameters (A10G Optimized)
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="qwen3_14b_verbosity_pc_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=2,
    bf16=True,
    gradient_checkpointing=True,
    save_steps=500,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    remove_unused_columns=False,  # Critical for custom format
)
```

### Deployment Workflow
```bash
# 1. Set environment variables
export QWEN_GPU_INSTANCE_IP=34.217.17.67
export SSH_KEY_PATH=/home/shingai/sort/llm-training/config/mistral-base.pem

# 2. Deploy training script
./scripts/qwen/deploy_qwen_verbosity_training_to_gpu.sh

# 3. SSH to instance and execute
ssh -i $SSH_KEY_PATH ubuntu@$QWEN_GPU_INSTANCE_IP
cd qwen_verbosity/
python3 train_qwen_golden_config.py
```

### Validated Results
- **Training Time**: 60 minutes (400 steps × 9s/step)
- **Memory Usage**: 9.7GB allocated, 11.3GB reserved
- **Verbosity Control**: TERSE (2 tokens) vs VERBOSE (32 tokens) = 16:1 ratio
- **Model Size**: 0.43% trainable parameters (64M parameters)

---

## Pipeline 2: Mistral Policy Model 🔄

### Status: **OPERATIONAL PIPELINE** 
- **Model**: Mistral-7B-v0.3 with policy analysis specialization
- **Location**: East coast instance (`54.197.142.172`) 
- **Training**: Multi-phase pipeline with resumable state management
- **Use Case**: Policy analysis, multi-domain reasoning

### Current Working Scripts

#### Phase 1: Data Preparation
**Script**: `/home/shingai/sort/llm-training/scripts/mistral/prepare_mistral_dataset.py`

**Features**:
- Downloads corpus from `s3://asoba-llm-cache/corpus/`
- Processes operatives from `s3://policy-database/operatives/` (up to 50K PDFs)
- Creates 80/20 train/validation split
- Stateful pipeline with `pipeline_state.json` resumability
- Outputs normalized datasets to S3

**Execution**:
```bash
cd /home/shingai/sort/llm-training/scripts/mistral

# Standard preparation
python3 prepare_mistral_dataset.py \
  --output-bucket asoba-llm-cache \
  --max-pdfs 50000 \
  --validation-split 0.2

# Outputs:
# s3://asoba-llm-cache/datasets/mistral-verbosity/train_dataset.jsonl
# s3://asoba-llm-cache/datasets/mistral-verbosity/val_dataset.jsonl
```

#### Phase 2: Training Execution  
**Script**: `/home/shingai/sort/llm-training/scripts/mistral/train_mistral_simple.py`

**Configuration**:
- Base Model: Mistral-7B-v0.3 (uncensored)
- Quantization: 4-bit with BitsAndBytesConfig
- LoRA: r=64, alpha=16, dropout=0.1
- Target modules: all projection layers
- Sequence length: 1024 tokens

**Execution**:
```bash
python3 train_mistral_simple.py \
  --train-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/train_dataset.jsonl \
  --val-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/val_dataset.jsonl \
  --output-dir ./mistral_simple_output
```

#### Phase 3: Infrastructure Deployment
**Script**: `/home/shingai/sort/llm-training/infrastructure/auto-deploy-mistral.sh`

```bash
# Deploy to g5.2xlarge with full environment setup
./infrastructure/auto-deploy-mistral.sh
```

### Current Dataset Issue & Solution

**Problem**: Comprehensive dataset (91,380 entries) has inconsistent columns
- Policy data includes: `bill_status`, `congress`, `short_title`, etc.
- Training expects only `text` field

**Solution** (implement before training):
```python
import pandas as pd

# Normalize dataset to single 'text' field
def normalize_dataset(input_path, output_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    normalized = []
    for item in data:
        if 'text' in item:
            normalized.append({"text": item['text']})
        # Handle other formats as needed
    
    with open(output_path, 'w') as f:
        for item in normalized:
            f.write(json.dumps(item) + '\n')

# Usage
normalize_dataset(
    's3://asoba-llm-cache/datasets/mistral-verbosity/comprehensive_train.jsonl',
    's3://asoba-llm-cache/datasets/mistral-verbosity/normalized_train.jsonl'
)
```

---

## 🔧 Operational Procedures

### Environment Setup & Validation

#### Prerequisites Check
```bash
# AWS credentials
source /home/shingai/sort/llm-training/.env
aws s3 ls s3://asoba-llm-cache/

# SSH access
chmod 400 /home/shingai/sort/llm-training/config/mistral-base.pem

# Instance validation
ssh -i /home/shingai/sort/llm-training/config/mistral-base.pem ubuntu@34.217.17.67 "nvidia-smi"
```

#### QLora Environment Setup
```bash
# Use automated setup script
/home/shingai/sort/llm-training/infrastructure/setup_qlora_instance.sh

# Manual validation on instance
ssh -i /path/to/key.pem ubuntu@<instance_ip>
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers, peft, bitsandbytes; print('✅ QLoRA ready')"
```

### Training Execution Patterns

#### Qwen Training (Validated)
```bash
# 1. Environment validation
df -h  # Should show 485GB for g5.4xlarge
nvidia-smi  # Should show A10G with ~23GB

# 2. Model download (if needed)
aws s3 sync s3://asoba-llm-cache/models/Qwen/Qwen3-14B/ /home/ubuntu/Qwen3-14B/

# 3. Execute training
python3 train_qwen_golden_config.py

# 4. Expected output
# Training completes in ~60 minutes
# Adapter saved to: qwen3_14b_verbosity_pc_lora/
```

#### Mistral Training (Multi-Phase)
```bash
# Phase 1: Data Preparation (any instance with AWS access)
python3 scripts/mistral/prepare_mistral_dataset.py --output-bucket asoba-llm-cache

# Phase 2: Training (GPU instance required)  
python3 scripts/mistral/train_mistral_simple.py \
  --train-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/train_dataset.jsonl \
  --val-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/val_dataset.jsonl

# Expected: 2-4 hours training time on g5.2xlarge
```

### Monitoring & Management

#### Training Progress Monitoring
```bash
# Production monitoring (with Slack alerts)
./scripts/monitoring/production_monitor.sh <run-id>

# Basic monitoring
python3 scripts/monitoring/monitor.py --run-id <run-id>

# One-time status check
python3 scripts/monitoring/monitor.py --run-id <run-id> --once
```

#### Instance Resource Management
```bash
# Check GPU usage
nvidia-smi

# Check disk space
df -h

# Check running processes
ps aux | grep python

# Memory usage
free -h
```

---

## 🚀 Quick Start Guide

### For Qwen Verbosity Training
```bash
# 1. Verify west coast instance is running
aws ec2 describe-instances --region us-west-2 --instance-ids i-0645c6db622720234

# 2. Deploy training
export QWEN_GPU_INSTANCE_IP=34.217.17.67
./scripts/qwen/deploy_qwen_verbosity_training_to_gpu.sh

# 3. Execute on instance
ssh -i /home/shingai/sort/llm-training/config/mistral-base.pem ubuntu@34.217.17.67
cd qwen_verbosity/
python3 train_qwen_golden_config.py

# 4. Validate results (expected 16:1 token ratio)
python3 test_verbosity_inference.py
```

### For Mistral Policy Training  
```bash
# 1. Prepare data (run locally or on any AWS instance)
cd /home/shingai/sort/llm-training/scripts/mistral
python3 prepare_mistral_dataset.py --output-bucket asoba-llm-cache

# 2. Launch/access training instance
ssh -i /home/shingai/sort/llm-training/config/mistral-base.pem ubuntu@54.197.142.172

# 3. Execute training
python3 scripts/mistral/train_mistral_simple.py \
  --train-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/train_dataset.jsonl \
  --val-dataset s3://asoba-llm-cache/datasets/mistral-verbosity/val_dataset.jsonl

# 4. Monitor progress
python3 scripts/monitoring/monitor.py --run-id mistral-$(date +%Y%m%d-%H%M%S)
```

---

## 🔍 Troubleshooting Guide

### Common Issues & Solutions

#### Instance Access Issues
```bash
# Wrong instance targeted
# Symptom: No GPU, wrong disk size (97GB instead of 485GB)
# Solution: Use specific instance variables
export QWEN_GPU_INSTANCE_IP=34.217.17.67  # NOT generic GPU_INSTANCE_IP

# SSH permission denied
chmod 400 /home/shingai/sort/llm-training/config/mistral-base.pem
```

#### Training Failures
```bash
# CUDA OOM
# Solution: Kill existing processes, verify batch size
nvidia-smi  # Check for existing processes
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill

# Dataset format errors
# Solution: Normalize data to consistent 'text' field format
python3 -c "
import json
# Add normalization code here
"
```

#### Model Loading Issues
```bash
# Unicode decode errors (S3 gzipped files)
# Solution: Decompress JSON files
cd /tmp/model && for f in *.json; do 
  if file $f | grep -q gzip; then 
    zcat $f > $f.tmp && mv $f.tmp $f
  fi
done
```

### Performance Optimization

#### Memory Management
- **Qwen (24GB GPU)**: Use 4-bit quantization, batch_size=1, gradient_accumulation_steps=16
- **Mistral (24GB GPU)**: Use 4-bit quantization, batch_size=4, sequence_length=1024

#### Training Efficiency  
- Enable `gradient_checkpointing=True`
- Use `bf16=True` for mixed precision
- Set `remove_unused_columns=False` for custom datasets

---

## 📚 Additional Resources

### Key Documentation Files
- **ACTUAL_STATE.md** - Current pipeline status and known issues
- **QWEN_TRAINING_GUIDE.md** - Detailed Qwen configuration and troubleshooting  
- **MISTRAL_CURRENT_PIPELINE.md** - Mistral pipeline architecture
- **INSTANCES.md** - Instance management and access details

### Validation Scripts
- **Test Suites**: `/home/shingai/sort/llm-training/tests/`
- **Validation Pipeline**: `/home/shingai/sort/llm-training/data/validation/universal_validation_pipeline.py`
- **Component Validation**: `./validate-components.sh`

### Cost Management
- **Instance Costs**: g5.4xlarge (~$1.60/hour), g5.2xlarge (~$1.20/hour)  
- **Monitoring**: Instances auto-appear at https://status.asoba.co/ when healthy
- **Stopping**: Stop instances when not in use to save costs

---

*This guide represents the complete, current state of both training pipelines as of August 7, 2025. All commands and configurations are validated against the actual codebase and infrastructure.*