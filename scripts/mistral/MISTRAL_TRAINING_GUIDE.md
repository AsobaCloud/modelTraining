# Mistral Training Pipeline - Improved Version

## Overview

The improved Mistral training pipeline (`train_mistral_full_pipeline_improved.sh`) addresses the critical issues in the original script:

✅ **Proper .env file handling**  
✅ **AWS credentials validation**  
✅ **Comprehensive error handling**  
✅ **Configurable parameters**  
✅ **GPU/CUDA checks**  
✅ **Structured logging**  
✅ **Resource validation**  

## Quick Start

### 1. Setup Environment

```bash
# Copy the template and configure your settings
cp .env.template .env
nano .env  # Edit with your AWS credentials and settings
```

### 2. Configure .env file

Required settings in `.env`:
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# S3 Configuration  
S3_POLICY_DATABASE_BUCKET=your-bucket-name

# Training Configuration
MODEL_DIR=/path/to/mistral-7b-v0.3
WORKDIR=/mnt/data/policy-corpus
```

### 3. Run Training

```bash
./train_mistral_full_pipeline_improved.sh
```

## Key Improvements

### 🔐 Security & Credentials
- Proper `.env` file loading with validation
- AWS credentials verification before training starts
- S3 bucket access validation
- No hardcoded secrets or keys

### 🛡️ Error Handling  
- Comprehensive validation of all required environment variables
- AWS connectivity checks before data download
- System resource validation (disk space, GPU)
- Graceful error messages with actionable suggestions

### ⚙️ Configuration
- All hardcoded paths now configurable via environment variables
- Flexible training parameters (batch size, learning rate, epochs)
- Configurable logging directory and levels
- GPU/CUDA settings support

### 📊 Monitoring & Logging
- Structured logging with timestamps and log levels
- Color-coded console output for better visibility
- Comprehensive training metadata saved with model
- Resource usage monitoring and reporting

### 🧪 Validation
- Pre-flight system checks (disk space, GPU availability)
- Model validation with test questions after training
- Dataset quality validation before training
- Training completion verification

## File Structure

After successful training:
```
├── .env                              # Your configuration
├── .env.template                     # Configuration template
├── train_mistral_full_pipeline_improved.sh  # Improved training script
├── mistral_full_lora_adapter/        # Trained LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── training_metadata.json
├── train_dataset.jsonl              # Training data
├── val_dataset.jsonl                # Validation data
└── /var/log/mistral-training/       # Training logs
    └── pipeline_YYYYMMDD_HHMMSS.log
```

## Configuration Options

### Core Training Parameters
- `TRAINING_BATCH_SIZE`: Per-device batch size (default: 1)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation (default: 16)  
- `LEARNING_RATE`: Learning rate (default: 2e-4)
- `NUM_TRAIN_EPOCHS`: Number of epochs (default: 1)
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 1024)

### LoRA Parameters
- `LORA_R`: LoRA rank (default: 16)
- `LORA_ALPHA`: LoRA alpha (default: 32)
- `LORA_DROPOUT`: LoRA dropout (default: 0.05)

### System Requirements
- `MIN_DISK_GB`: Minimum free disk space (default: 100GB)
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `LOG_DIR`: Log file directory (default: /var/log/mistral-training)

## Usage Examples

### Basic Training
```bash
# Use default settings
./train_mistral_full_pipeline_improved.sh
```

### Custom Configuration
```bash
# Edit .env file first, then run
TRAINING_BATCH_SIZE=2 LEARNING_RATE=1e-4 ./train_mistral_full_pipeline_improved.sh
```

### GPU Selection
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=1 ./train_mistral_full_pipeline_improved.sh
```

## Troubleshooting

### Common Issues

**AWS Credentials Error**
```bash
[ERROR] AWS credentials validation failed
```
→ Check your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env`

**S3 Access Error**  
```bash
[ERROR] Cannot access S3 bucket: your-bucket-name
```
→ Verify bucket name and IAM permissions

**Insufficient Disk Space**
```bash  
[ERROR] Insufficient disk space: 50GB available, 100GB required
```
→ Free up disk space or change `MIN_DISK_GB` setting

**GPU Not Available**
```bash
[WARN] nvidia-smi not found - GPU training may not be available
```
→ Install NVIDIA drivers or run on CPU (will be much slower)

### Logs and Debugging

Training logs are saved to `/var/log/mistral-training/pipeline_YYYYMMDD_HHMMSS.log`

View real-time training progress:
```bash
tail -f /var/log/mistral-training/pipeline_*.log
```

## Performance Notes

- **GPU Required**: Training is optimized for GPU. CPU training will be extremely slow.
- **Memory**: Requires ~16GB GPU memory for g5.2xlarge instances
- **Time**: Typical training time is 2-4 hours depending on dataset size
- **Storage**: Requires 100GB+ free disk space for datasets and model files

## Security Considerations

- Never commit `.env` files to version control
- Use IAM roles with minimal required permissions
- Regularly rotate AWS access keys
- Monitor S3 bucket access logs
- Use VPC endpoints for S3 access when possible