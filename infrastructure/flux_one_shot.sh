#!/usr/bin/env bash
set -euo pipefail

# PRODUCTION AMI AVAILABLE: ami-0e6b0570006182a68
# For production use, launch directly from AMI instead of running this script:
# aws ec2 run-instances --image-id ami-0e6b0570006182a68 --instance-type g5.4xlarge --key-name mistral-base

# Load environment variables
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
else
  echo "❌ .env file not found" >&2
  exit 1
fi

die() { echo "❌ $*" >&2; exit 1; }

#######################################
# 0. Configuration & Secrets
#######################################
REGION="${AWS_REGION:-us-west-2}"
INSTANCE_TYPE="g5.4xlarge"  # Updated: g5.2xlarge causes OOM, g5.4xlarge required
PROJECT_TAG="FluxDeploy"
OWNER_TAG="$(whoami)"
VOLUME_SIZE=500  # GiB
KEY_NAME="mistral-base"
REMOTE_CHECK_SCRIPT='/tmp/flux_sanity.sh'
S3_MODEL_PATH="s3://flux-model-cache-1752234728/flux-complete/"
DLAMI_OWNER="763104351884"      # AWS DLAMI publishing account
# SSH username for this DLAMI (ubuntu instead of ec2-user)
REMOTE_USER="ubuntu"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_PATH="$SCRIPT_DIR/${KEY_NAME}.pem"
[[ -f "$KEY_PATH" ]] || die "SSH key not found at $KEY_PATH"

# SSH username for this DLAMI (ubuntu instead of ec2-user)
REMOTE_USER="ubuntu"



#######################################
# 1. vCPU Quota Pre-Flight
#######################################
preflight() {
  echo "🔍 1/6  vCPU quota pre-flight…"
  local requested=8 used total
  used=$(aws ec2 describe-instances --region "$REGION" \
    --filters Name=instance-state-name,Values=pending,running Name=tag:Project,Values="$PROJECT_TAG" \
    --query 'Reservations[].Instances[].CpuOptions.CoreCount' --output text \
    | awk 'BEGIN{sum=0}!/^(None)?$/{sum+=$1}END{print sum}')
  used=${used:-0}
  total=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --region "$REGION" \
    --query 'ServiceQuota.Value' --output text 2>/dev/null || echo 64)
  [[ "$total" == "None" ]] && total=64
  if (( used + requested > total )); then
    die "vCPU limit ($total) insufficient (used=$used, req=$requested)"
  fi
}

#######################################
# 2. Locate GPU PyTorch DLAMI
#######################################
locate_dlami() {
  echo "🔍 2/6  Locating GPU PyTorch DLAMI…"
  local ssm
  ssm="/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id"
  AMI_ID=$(aws ssm get-parameter --name "$ssm" --region "$REGION" --query 'Parameter.Value' --output text) || die "SSM query failed"
  [[ "$AMI_ID" =~ ^ami- ]] || die "Invalid AMI ID: $AMI_ID"
  echo "✅ Found DLAMI: $AMI_ID"
}

#######################################
# 3. Locate & Health-Check Existing Instance
#######################################
find_existing() {
  echo "🔍 3/6  Checking for existing instance…"
  local json
  json=$(aws ec2 describe-instances --region "$REGION" \
    --filters Name=tag:Project,Values="$PROJECT_TAG" Name=instance-state-name,Values=pending,running \
    --query 'Reservations[].Instances[].{Id:InstanceId,IP:PublicIpAddress}' --output json)
  INSTANCE_ID=$(jq -r '.[0].Id // empty' <<<"$json")
  PUBLIC_IP=$(jq -r '.[0].IP // empty' <<<"$json")
  # --- detect actual key-pair name on the instance ---
INSTANCE_KEY_NAME=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].KeyName' \
  --output text)

echo "🔍 3.1/6  Instance $INSTANCE_ID uses key-pair: $INSTANCE_KEY_NAME"

if [[ "$INSTANCE_KEY_NAME" != "$KEY_NAME" ]]; then
  echo "⚠️ 3.2/6  Key-pair mismatch! Updating to use '$INSTANCE_KEY_NAME'.pem"
  KEY_NAME="$INSTANCE_KEY_NAME"
  KEY_PATH="$SCRIPT_DIR/${KEY_NAME}.pem"
  [[ -f "$KEY_PATH" ]] || die "Private key not found at $KEY_PATH for instance key-pair '$KEY_NAME'"
fi

echo "✅ 3.3/6  Using SSH key: $KEY_PATH"

  if [[ -n "$INSTANCE_ID" ]]; then
    echo "✅ Found instance $INSTANCE_ID at $PUBLIC_IP. Repairing/updating…"
    # Copy .env file for AWS credentials
    scp -o StrictHostKeyChecking=no -i "$KEY_PATH" .env ${REMOTE_USER}@"$PUBLIC_IP":/tmp/.env
    ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" ${REMOTE_USER}@"$PUBLIC_IP" bash -s "$S3_MODEL_PATH" << 'REPAIR'
set -euo pipefail
# Set up AWS credentials file
mkdir -p ~/.aws
grep AWS_ACCESS_KEY_ID /tmp/.env | cut -d= -f2 > /tmp/access_key
grep AWS_SECRET_ACCESS_KEY /tmp/.env | cut -d= -f2 > /tmp/secret_key
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $(cat /tmp/access_key)
aws_secret_access_key = $(cat /tmp/secret_key)
EOF
cat > ~/.aws/config << EOF
[default]
region = us-west-2
output = json
EOF
rm /tmp/access_key /tmp/secret_key
S3_MODEL_PATH="$1"
source /opt/pytorch/bin/activate
# Fix XFormers compatibility
pip uninstall xformers -y
pip install xformers==0.0.29.post2 --no-deps
sudo mkdir -p /opt/models/flux
sudo chown -R ubuntu:ubuntu /opt/models

# Check if model files already exist in expected structure
echo "🔍 Checking for model files in expected structure: /opt/models/flux/model_index.json"
if [[ -f "/opt/models/flux/model_index.json" ]]; then
    echo "✅ Model files already in correct structure"
else
    echo "❌ Model files not in expected structure"
    # Check if model exists in HuggingFace cache structure
    hf_model_path="/opt/models/flux/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    echo "🔍 Checking for HuggingFace cache: $hf_model_path/model_index.json"
    if [[ -f "$hf_model_path/model_index.json" ]]; then
        echo "✅ Found model in HuggingFace cache structure"
        echo "🔄 Restructuring existing model files from HuggingFace cache"
        # Copy files from HF cache to expected structure
        cp -r "$hf_model_path"/* /opt/models/flux/
        echo "🧹 Cleaning up old HuggingFace cache structure"
        # Clean up old structure
        rm -rf /opt/models/flux/models--black-forest-labs--FLUX.1-dev
        rm -rf /opt/models/flux/.locks
        rm -rf /opt/models/flux/blobs
    else
        echo "❌ No model found in HuggingFace cache either"
        echo "⬇️ Downloading model from S3"
        aws s3 sync "$S3_MODEL_PATH/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/" /opt/models/flux --region us-west-2
    fi
fi

# Verify specific files needed by health check exist
required_files=("model_index.json" "transformer/config.json" "text_encoder/config.json" "vae/config.json")
missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "/opt/models/flux/$file" ]]; then
        missing_files+=("$file")
    fi
done
if [[ ${#missing_files[@]} -eq 0 ]]; then
    echo "✅ Model files ready - all required files present"
else
    echo "❌ Model setup incomplete - missing files: ${missing_files[*]}"
    echo "Files in /opt/models/flux: $(find /opt/models/flux -type f | head -10)"
    exit 1
fi
REPAIR
    
    # Now run health check AFTER repair
    echo "🔍 Running health check after setup…"
    cat << 'EOF' > "$REMOTE_CHECK_SCRIPT"
#!/usr/bin/env bash
set -euo pipefail
source /opt/pytorch/bin/activate || exit 1

# Diagnostic collection function
collect_failure_diagnostics() {
    local failure_type="$1"
    echo "🔍 Collecting diagnostics for: $failure_type"
    
    echo "=== FAILURE DIAGNOSTICS ==="
    echo "Failure Type: $failure_type"
    echo "Time: $(date)"
    echo
    
    echo "--- Last 50 lines of health check log ---"
    tail -50 /tmp/flux_health.log 2>/dev/null || echo "No log available"
    echo
    
    echo "--- GPU Status ---"
    nvidia-smi 2>/dev/null || echo "nvidia-smi failed"
    echo
    
    echo "--- System Memory ---"
    free -h 2>/dev/null || echo "free command failed"
    echo
    
    echo "--- Recent kernel messages ---"
    sudo dmesg 2>/dev/null | tail -20 || echo "dmesg not accessible"
    echo
    
    echo "--- Python processes ---"
    ps aux | grep python 2>/dev/null || echo "No python processes"
    echo
    
    echo "=== END DIAGNOSTICS ==="
}

# Start health check in background and monitor
screen -dmS flux_health bash -c '
python - << "PY" 2>&1 | tee /tmp/flux_health.log
import importlib.util,sys,pathlib
mods=["torch","diffusers","transformers","accelerate","xformers"]
if any(importlib.util.find_spec(m) is None for m in mods): sys.exit(1)

# Test image generation
from diffusers import FluxPipeline
import torch
pipe = FluxPipeline.from_pretrained("/opt/models/flux/", torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()
pipe.safety_checker = None  # Remove NSFW filter
image = pipe("test", height=512, width=512, num_inference_steps=4).images[0]
print("✅ Image generation test passed - NSFW filter removed")
PY
echo "Exit code: $?" >> /tmp/flux_health.log
'

# Monitor the health check process with comprehensive logic
start_time=$(date +%s)
last_log_size=0
stuck_counter=0
max_runtime=900  # 15 minutes max

echo "🔍 Starting comprehensive health monitoring..."

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    # Check if process completed normally
    if ! screen -list | grep -q flux_health; then
        if tail -1 /tmp/flux_health.log 2>/dev/null | grep -q "Exit code: 0"; then
            echo "✅ Health check passed"
            exit 0
        else
            echo "❌ Health check failed"
            collect_failure_diagnostics "process_exit_failure"
            exit 1
        fi
    fi
    
    # FAST FAIL CONDITIONS (check every iteration)
    
    # 1. Check for OOM in dmesg (skip if no permission)
    if sudo dmesg 2>/dev/null | tail -20 | grep -qi "out of memory\|killed process.*python"; then
        echo "❌ OOM detected in system logs"
        collect_failure_diagnostics "oom_detected"
        screen -S flux_health -X quit 2>/dev/null || true
        exit 1
    fi
    
    # 2. Check for CUDA errors (skip if no permission)
    if sudo dmesg 2>/dev/null | tail -10 | grep -qi "cuda\|nvidia.*error"; then
        echo "❌ CUDA/driver errors detected"
        collect_failure_diagnostics "cuda_error"
        screen -S flux_health -X quit 2>/dev/null || true
        exit 1
    fi
    
    # 3. Check if Python process crashed
    if ! pgrep -f "python.*flux" > /dev/null; then
        if [[ -f /tmp/flux_health.log ]] && [[ $(wc -l < /tmp/flux_health.log) -gt 0 ]]; then
            echo "❌ Python process crashed unexpectedly"
            collect_failure_diagnostics "process_crash"
            exit 1
        fi
    fi
    
    # LOG ANALYSIS
    if [[ -f /tmp/flux_health.log ]]; then
        current_log_size=$(wc -c < /tmp/flux_health.log)
        
        # Check for error accumulation
        error_count=$(grep -c "ERROR\|WARN\|Failed\|Exception\|Traceback" /tmp/flux_health.log 2>/dev/null || echo 0)
        if [[ $error_count -gt 10 ]]; then
            echo "❌ High error count detected ($error_count errors)"
            collect_failure_diagnostics "error_accumulation"
            screen -S flux_health -X quit 2>/dev/null || true
            exit 1
        fi
        
        # Check for infinite loops (same line repeated many times)
        max_repeats=$(tail -30 /tmp/flux_health.log | sort | uniq -c | sort -nr | head -1 | awk '{print $1}' 2>/dev/null || echo 0)
        if [[ $max_repeats -gt 15 ]]; then
            echo "❌ Infinite loop detected (line repeated $max_repeats times)"
            collect_failure_diagnostics "infinite_loop"
            screen -S flux_health -X quit 2>/dev/null || true
            exit 1
        fi
        
        # Progress detection
        if [[ $current_log_size -gt $last_log_size ]]; then
            stuck_counter=0  # Reset stuck counter on progress
            echo "📊 Progress detected (log: ${current_log_size} bytes, elapsed: ${elapsed}s)"
        else
            ((stuck_counter++))
            echo "⏳ No log progress (stuck count: $stuck_counter, elapsed: ${elapsed}s)"
        fi
        
        last_log_size=$current_log_size
    else
        if [[ $elapsed -gt 120 ]]; then  # No log file after 2 minutes
            echo "❌ No log file created after 2 minutes"
            collect_failure_diagnostics "no_log_file"
            screen -S flux_health -X quit 2>/dev/null || true
            exit 1
        fi
    fi
    
    # GPU ANALYSIS
    gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0,0,0")
    gpu_mem_used=$(echo "$gpu_info" | cut -d',' -f1 | tr -d ' \n')
    gpu_mem_total=$(echo "$gpu_info" | cut -d',' -f2 | tr -d ' \n')
    gpu_util=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' \n')
    
    # Ensure numeric values (default to 0 if empty/invalid)
    gpu_mem_used=${gpu_mem_used:-0}
    gpu_mem_total=${gpu_mem_total:-0}
    gpu_util=${gpu_util:-0}
    
    # Check for GPU memory allocation failure
    if [[ $gpu_mem_used -eq 0 ]] && [[ $elapsed -gt 300 ]]; then  # No GPU memory after 5 minutes
        echo "❌ No GPU memory allocated after 5 minutes"
        collect_failure_diagnostics "no_gpu_allocation"
        screen -S flux_health -X quit 2>/dev/null || true
        exit 1
    fi
    
    # STUCK DETECTION (no progress for 3 minutes = 12 iterations of 15s)
    if [[ $stuck_counter -gt 12 ]]; then
        # But allow if GPU is actively computing
        if [[ ${gpu_util:-0} -gt 5 ]]; then
            echo "🔄 No log progress but GPU active (${gpu_util}% util) - continuing..."
            stuck_counter=0  # Reset since GPU is working
        else
            echo "❌ Process stuck (no progress for 3+ minutes, GPU idle)"
            collect_failure_diagnostics "stuck_process"
            screen -S flux_health -X quit 2>/dev/null || true
            exit 1
        fi
    fi
    
    # MAXIMUM RUNTIME CHECK (only if genuinely stuck)
    if [[ $elapsed -gt $max_runtime ]]; then
        echo "❌ Maximum runtime exceeded (${max_runtime}s) - forcing termination"
        collect_failure_diagnostics "max_runtime_exceeded"
        screen -S flux_health -X quit 2>/dev/null || true
        exit 1
    fi
    
    # Show current status
    last_log_line=$(tail -1 /tmp/flux_health.log 2>/dev/null | head -c 80 || echo "Starting...")
    echo "📊 Status: GPU ${gpu_mem_used}MB/${gpu_mem_total}MB (${gpu_util}% util) | ${last_log_line}"
    
    sleep 15
done
EOF
    chmod +x "$REMOTE_CHECK_SCRIPT"
    scp -o StrictHostKeyChecking=no -i "$KEY_PATH" "$REMOTE_CHECK_SCRIPT" ${REMOTE_USER}@"$PUBLIC_IP":$REMOTE_CHECK_SCRIPT >/dev/null
    if ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" ${REMOTE_USER}@"$PUBLIC_IP" "bash $REMOTE_CHECK_SCRIPT"; then
      echo "🎉 Deployment successful! SSH: ssh -i $KEY_PATH ${REMOTE_USER}@$PUBLIC_IP"
      exit 0
    else
      echo "❌ Health check failed after repair"
      exit 1
    fi
  fi
}

#######################################
# 4. Launch New Instance
#######################################
launch_new() {
  echo "🚀 4/6  No healthy instance. Launching new one…"
  USER_DATA=$(base64 << 'CLOUD'
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/flux-bootstrap.log) 2>&1
apt-get update && apt-get install -y git jq
source /opt/pytorch/bin/activate
pip install --upgrade pip setuptools wheel
pip install diffusers>=0.34.0 transformers>=4.44 accelerate>=0.30 safetensors>=0.8 ninja triton xformers
git clone https://github.com/black-forest-labs/flux /opt/flux
pip install -e "/opt/flux[all]"
mkdir -p /opt/models/flux
aws s3 sync $S3_MODEL_PATH /opt/models/flux --region $REGION
CLOUD
)
  local resp
  resp=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":'$VOLUME_SIZE',"VolumeType":"gp3"}}]' \
    --tag-specifications '[{"ResourceType":"instance","Tags":[{"Key":"Project","Value":"'$PROJECT_TAG'"},{"Key":"Owner","Value":"'$OWNER_TAG'"}]}]' \
    --user-data "$USER_DATA")
  INSTANCE_ID=$(jq -r '.Instances[0].InstanceId' <<<"$resp")
  echo "🔍 5/6  Waiting for instance $INSTANCE_ID to run…"
  aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
  PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "🔍 6/6  New instance ready at $PUBLIC_IP"
  echo "🎯 SSH: ssh -i $KEY_PATH ${REMOTE_USER}@$PUBLIC_IP"
}

main() {
  preflight
  locate_dlami
  find_existing
  launch_new
}

main
