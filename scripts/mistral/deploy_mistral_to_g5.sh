#!/bin/bash
# Deployment script for Mistral-7B golden-path training on g5.2xlarge
# Following CLAUDE.md safety principles

set -euo pipefail

# Configuration
INSTANCE_IP="${1:-}"
KEY_PATH="${2:-}"
REMOTE_USER="ubuntu"
DEPLOYMENT_DIR="/home/ubuntu/mistral_verbosity_deployment"

# Validate arguments
if [[ -z "$INSTANCE_IP" || -z "$KEY_PATH" ]]; then
    echo "Usage: $0 <instance_ip> <ssh_key_path>"
    echo "Example: $0 34.217.17.67 ~/.ssh/my-key.pem"
    exit 1
fi

# Safety check - confirm deployment
echo "=== Mistral-7B Verbosity Training Deployment ==="
echo "Target: ${REMOTE_USER}@${INSTANCE_IP}"
echo "Deployment directory: ${DEPLOYMENT_DIR}"
echo ""
read -p "Proceed with deployment? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Deployment cancelled"
    exit 0
fi

# Create deployment package
echo "Creating deployment package..."
mkdir -p mistral_deployment_package

# Copy all necessary files
cp train_mistral_verbosity_qlora.py mistral_deployment_package/
cp validate_mistral_golden_config.py mistral_deployment_package/
cp mistral_golden_config_documentation.md mistral_deployment_package/
cp verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl mistral_deployment_package/

# Create remote setup script
cat > mistral_deployment_package/setup_and_run.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "=== Mistral-7B Golden-Path Setup on g5.2xlarge ==="

# Environment setup (following golden config)
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_USE_DEEPSPEED=0
export ACCELERATE_USE_FSDP=0
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check GPU availability
echo "Checking GPU..."
nvidia-smi || { echo "ERROR: No GPU found"; exit 1; }

# Verify model directory
MODEL_DIR="/home/ubuntu/mistral-7b-v0.3"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found at $MODEL_DIR"
    echo "Please ensure Mistral-7B v0.3 is downloaded to this location"
    exit 1
fi

echo "Model directory verified: $MODEL_DIR"

# Install/verify dependencies
echo "Checking Python dependencies..."
pip3 show transformers | grep Version || pip3 install transformers==4.54.0
pip3 show torch | grep Version || pip3 install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip3 show peft | grep Version || pip3 install peft==0.16.0
pip3 show accelerate | grep Version || pip3 install accelerate==1.9.0
pip3 show bitsandbytes | grep Version || pip3 install bitsandbytes==0.46.1
pip3 show datasets | grep Version || pip3 install datasets==3.6.0
pip3 show trl | grep Version || pip3 install trl
pip3 show flask | grep Version || pip3 install flask

# Run validation first
echo ""
echo "=== Running Golden Config Validation ==="
python3 validate_mistral_golden_config.py

# Check validation result
if [[ $? -ne 0 ]]; then
    echo "ERROR: Validation failed. Please fix issues before training."
    exit 1
fi

echo ""
echo "=== Validation Passed! Ready for training ==="
echo "Next steps:"
echo "1. Run training: python3 train_mistral_verbosity_qlora.py"
echo "2. Test model: python3 train_mistral_verbosity_qlora.py --mode test"
echo "3. Start API: python3 train_mistral_verbosity_qlora.py --mode serve"
EOF

chmod +x mistral_deployment_package/setup_and_run.sh

# Create monitoring script
cat > mistral_deployment_package/monitor_training.sh << 'EOF'
#!/bin/bash
# Monitor GPU usage during training

echo "GPU Memory Monitoring (updates every 5 seconds)"
echo "Press Ctrl+C to stop"

while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
    echo ""
    echo "=== Top Processes ==="
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits | head -5
    sleep 5
done
EOF

chmod +x mistral_deployment_package/monitor_training.sh

# Transfer to instance
echo ""
echo "Transferring deployment package to instance..."
ssh -i "$KEY_PATH" "${REMOTE_USER}@${INSTANCE_IP}" "mkdir -p ${DEPLOYMENT_DIR}"

# Use rsync for efficient transfer
rsync -avz -e "ssh -i $KEY_PATH" \
    mistral_deployment_package/ \
    "${REMOTE_USER}@${INSTANCE_IP}:${DEPLOYMENT_DIR}/"

# Clean up local package
rm -rf mistral_deployment_package

echo ""
echo "=== Deployment Complete ==="
echo "Files deployed to: ${REMOTE_USER}@${INSTANCE_IP}:${DEPLOYMENT_DIR}"
echo ""
echo "To proceed with setup and validation:"
echo "1. SSH to instance: ssh -i $KEY_PATH ${REMOTE_USER}@${INSTANCE_IP}"
echo "2. Navigate to: cd ${DEPLOYMENT_DIR}"
echo "3. Run setup: ./setup_and_run.sh"
echo ""
echo "To monitor GPU during training (in separate terminal):"
echo "ssh -i $KEY_PATH ${REMOTE_USER}@${INSTANCE_IP} 'cd ${DEPLOYMENT_DIR} && ./monitor_training.sh'"