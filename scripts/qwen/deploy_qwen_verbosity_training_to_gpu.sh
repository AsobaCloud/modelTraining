#!/bin/bash
# Deploy Qwen verbosity control to g5.4xlarge instance
# Run this from LOCAL development environment

set -euo pipefail

# Load configuration from .env
source .env

# Target instance  
INSTANCE_IP="${QWEN_GPU_INSTANCE_IP}"
INSTANCE_USER="${QWEN_GPU_INSTANCE_USER}"
SSH_KEY="${SSH_KEY_PATH}"
REMOTE_DIR="/home/ubuntu/qwen_verbosity"

# SSH options
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no"

echo "=== Deploying Qwen Verbosity Control to g5.4xlarge ==="
echo "Target: ${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}"

# Create remote directory
ssh ${SSH_OPTS} ${INSTANCE_USER}@${INSTANCE_IP} "mkdir -p ${REMOTE_DIR}"

# Copy essential files
echo "Copying training and inference scripts..."
scp ${SSH_OPTS} train_qwen_golden_config.py ${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/
scp ${SSH_OPTS} decoding_presets.py ${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/
scp ${SSH_OPTS} tool_router.py ${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/
scp ${SSH_OPTS} inference.py ${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/
scp ${SSH_OPTS} production_eval.py ${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/
scp ${SSH_OPTS} improved_eval.py ${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/

# Copy training data
echo "Copying training data..."
scp ${SSH_OPTS} verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl ${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/

# Copy documentation
echo "Copying documentation..."
scp ${SSH_OPTS} qwen_golden_config_documentation.md ${INSTANCE_USER}@${INSTANCE_IP}:${REMOTE_DIR}/

echo "=== Deployment Complete ==="
echo ""
echo "Next steps on the g5.4xlarge instance:"
echo "1. SSH: ssh ${INSTANCE_USER}@${INSTANCE_IP}"
echo "2. cd ${REMOTE_DIR}"
echo "3. Run training: python3 train_qwen_golden_config.py"
echo "4. Run evaluation: python3 production_eval.py"