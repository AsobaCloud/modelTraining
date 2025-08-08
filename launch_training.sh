#!/usr/bin/env bash
# launch_training.sh - Programmatic training launcher
#
# Usage:
#   ./launch_training.sh [--model-tag TAG] [--region REGION] [--batch-size N]
#
# This script demonstrates the programmatic approach:
# 1. Model resolution happens automatically inside the training script
# 2. Dependencies are auto-deployed
# 3. Singleton lock prevents duplicates
# 4. Full monitoring and packaging

set -euo pipefail

# Default values
MODEL_TAG="mistral-7b-v0.3"
REGION="us-east-1"  
BATCH_SIZE=1
INSTANCE_IP="54.197.142.172"
SSH_KEY="config/mistral-base.pem"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --instance-ip) INSTANCE_IP="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "[INFO] Launching programmatic training:"
echo "  Model Tag: ${MODEL_TAG}"
echo "  Region: ${REGION}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Instance: ${INSTANCE_IP}"

# Generate unique run ID
RUN_ID="mistral-production-$(date +%Y%m%d-%H%M%S)"

# Deploy updated scripts to instance
echo "[INFO] Deploying scripts to instance..."
scp -i "${SSH_KEY}" scripts/resolve_model.sh "ubuntu@${INSTANCE_IP}:/mnt/training/"
scp -i "${SSH_KEY}" scripts/mistral/train_mistral_simple.py "ubuntu@${INSTANCE_IP}:/mnt/training/mistral_training/"

# Launch training with programmatic model resolution
echo "[INFO] Launching training with Run ID: ${RUN_ID}"
ssh -i "${SSH_KEY}" "ubuntu@${INSTANCE_IP}" bash -c "
  set -euo pipefail
  cd /mnt/training/mistral_training
  
  # Singleton lock to prevent duplicates
  exec 200>/var/lock/mistral_train.lock
  flock -n 200 || { echo '[ERR] Training already running'; exit 1; }
  
  # Launch training (model resolution happens automatically)
  nohup python3 train_mistral_simple.py \\
    --model-tag '${MODEL_TAG}' \\
    --region '${REGION}' \\
    --train-dataset 's3://asoba-llm-cache/datasets/mistral-verbosity/final_train.jsonl' \\
    --val-dataset 's3://asoba-llm-cache/datasets/mistral-verbosity/final_val.jsonl' \\
    --output-dir '/mnt/training/outputs/${RUN_ID}' \\
    --s3-bucket 'asoba-llm-cache' \\
    --s3-prefix 'mistral-finetuned' \\
    --batch-size ${BATCH_SIZE} \\
    --max-seq-length 512 \\
    --run-id '${RUN_ID}' \\
    >> /home/ubuntu/logs/mistral_train.log 2>&1 &
  
  echo '[OK] Training launched successfully'
  echo 'Run ID: ${RUN_ID}'
  echo 'Logs: tail -f /home/ubuntu/logs/mistral_train.log'
"

echo "[OK] Training launched! Run ID: ${RUN_ID}"
echo ""
echo "Monitor with:"
echo "  ssh -i ${SSH_KEY} ubuntu@${INSTANCE_IP} 'tail -f /home/ubuntu/logs/mistral_train.log'"
echo ""
echo "Check heartbeat:"
echo "  aws s3 cp s3://asoba-llm-cache/training-runs/${RUN_ID}/metadata.json - --region ${REGION}"