#!/bin/bash

# QWEN CLAUDE.md Deployment Script
# Deploys Qwen3-14B with integrated CLAUDE.md methodology on existing Flux AMI
# 
# Usage: ./deploy_qwen_claude_md.sh [--instance-type g5.4xlarge] [--region us-east-1]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFERRED_REGIONS=("us-east-1" "us-west-2")
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.4xlarge}"
PROJECT_TAG="QwenClaude"
SERVICE_PORT="8001"

# AMI mapping for mistral-qlora-training AMI across regions
declare -A AMI_MAP=(
    ["us-east-1"]="ami-0a39335458731538a"
    ["us-west-2"]="ami-0123456789abcdef0"  # Update with actual us-west-2 AMI ID
)

# Key and security group mapping per region
declare -A KEY_MAP=(
    ["us-east-1"]="mistral-base-key"
    ["us-west-2"]="mistral-base-key"
)

declare -A SG_MAP=(
    ["us-east-1"]="default"
    ["us-west-2"]="default"
)

# Model configuration
MODEL_S3_PATH="s3://asoba-llm-cache/models/Qwen/Qwen3-14B"
CONFIG_FILES=("qwen_config.json" "qwen_claude_md_system_prompt.txt" "qwen_inference_server.py")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install it first."
    fi
    
    # Check required files
    for file in "${CONFIG_FILES[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
            error "Required file not found: $file"
        fi
    done
    
    # Check SSH key
    if [[ ! -f "$SCRIPT_DIR/config/mistral-base.pem" ]]; then
        error "SSH key not found: $SCRIPT_DIR/config/mistral-base.pem"
    fi
    
    # Check S3 model availability
    if ! aws s3 ls "$MODEL_S3_PATH/" --region "$REGION" >/dev/null 2>&1; then
        error "Qwen model not accessible in S3: $MODEL_S3_PATH"
    fi
    
    log "✅ Prerequisites validated"
}

select_region() {
    log "🌍 Selecting optimal region based on vCPU availability..."
    
    # Get instance vCPU count
    VCPU_COUNT=$(aws ec2 describe-instance-types \
        --instance-types "$INSTANCE_TYPE" \
        --region "${PREFERRED_REGIONS[0]}" \
        --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' \
        --output text)
    
    for region in "${PREFERRED_REGIONS[@]}"; do
        log "Checking capacity in $region..."
        
        # Get current usage
        USED_VCPUS=$(aws ec2 describe-instances \
            --region "$region" \
            --filters "Name=instance-state-name,Values=pending,running" \
            --query 'sum(Reservations[].Instances[].CpuOptions.CoreCount)' \
            --output text 2>/dev/null || echo "0")
        
        USED_VCPUS=${USED_VCPUS:-0}
        
        # Get quota
        QUOTA=$(aws service-quotas get-service-quota \
            --service-code ec2 \
            --quota-code L-1216C47A \
            --region "$region" \
            --query 'ServiceQuota.Value' \
            --output text 2>/dev/null || echo "1000")
        
        AVAILABLE=$((QUOTA - USED_VCPUS))
        
        log "$region: Used=$USED_VCPUS, Available=$AVAILABLE, Required=$VCPU_COUNT"
        
        if (( VCPU_COUNT <= AVAILABLE )); then
            SELECTED_REGION=$region
            AMI_ID=${AMI_MAP[$region]}
            KEY_NAME=${KEY_MAP[$region]}
            SECURITY_GROUP=${SG_MAP[$region]}
            log "✅ Selected region: $SELECTED_REGION"
            return 0
        fi
    done
    
    error "No region has sufficient vCPU capacity for $INSTANCE_TYPE"
}

check_capacity() {
    log "🔋 Verifying capacity in selected region: $SELECTED_REGION"
    log "✅ Capacity confirmed: $VCPU_COUNT vCPUs available"
}

launch_instance() {
    log "🚀 Launching $INSTANCE_TYPE instance..."
    
    # Create user data script
    USER_DATA=$(cat << 'EOF'
#!/bin/bash
set -euo pipefail

# Update system
apt-get update

# Create deployment directory
mkdir -p /home/ubuntu/qwen-claude-md
cd /home/ubuntu/qwen-claude-md

# Log setup progress
exec > >(tee -a /var/log/qwen-setup.log) 2>&1
echo "$(date): Starting Qwen CLAUDE.md setup..."

# Download model from S3 (will be done by inference server on first run)
echo "$(date): Model will be downloaded on first inference request"

# Install additional dependencies if needed
pip3 install --quiet --upgrade fastapi uvicorn

echo "$(date): Setup complete. Ready for deployment."
echo "SETUP_COMPLETE" > /home/ubuntu/qwen_setup_status.txt

EOF
    )
    
    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --count 1 \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-groups "$SECURITY_GROUP" \
        --user-data "$USER_DATA" \
        --region "$SELECTED_REGION" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=qwen-claude-md},{Key=Project,Value=$PROJECT_TAG},{Key=Owner,Value=claude},{Key=Purpose,Value=Qwen-CLAUDE-md}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    log "✅ Instance launched: $INSTANCE_ID"
    echo "$INSTANCE_ID"
}

wait_for_instance() {
    local instance_id=$1
    log "⏳ Waiting for instance to be running..."
    
    aws ec2 wait instance-running \
        --instance-ids "$instance_id" \
        --region "$SELECTED_REGION"
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --region "$SELECTED_REGION" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    log "✅ Instance running at $PUBLIC_IP"
    
    # Wait for SSH
    log "⏳ Waiting for SSH access..."
    for i in {1..30}; do
        if ssh -i "$SCRIPT_DIR/config/mistral-base.pem" \
           -o ConnectTimeout=5 \
           -o StrictHostKeyChecking=no \
           ubuntu@"$PUBLIC_IP" \
           "echo 'SSH ready'" >/dev/null 2>&1; then
            break
        fi
        sleep 10
    done
    
    log "✅ SSH access confirmed"
    echo "$PUBLIC_IP"
}

deploy_qwen_service() {
    local public_ip=$1
    log "📦 Deploying Qwen CLAUDE.md service to $public_ip..."
    
    # Upload configuration files
    for file in "${CONFIG_FILES[@]}"; do
        log "Uploading $file..."
        scp -i "$SCRIPT_DIR/config/mistral-base.pem" \
            -o StrictHostKeyChecking=no \
            "$SCRIPT_DIR/$file" \
            ubuntu@"$public_ip":/home/ubuntu/qwen-claude-md/
    done
    
    # Upload test script
    scp -i "$SCRIPT_DIR/config/mistral-base.pem" \
        -o StrictHostKeyChecking=no \
        "$SCRIPT_DIR/test_qwen_claude_md.py" \
        ubuntu@"$public_ip":/home/ubuntu/qwen-claude-md/
    
    # Create systemd service
    log "Creating systemd service..."
    ssh -i "$SCRIPT_DIR/config/mistral-base.pem" \
        -o StrictHostKeyChecking=no \
        ubuntu@"$public_ip" << 'EOF'
        
# Create service file
sudo tee /etc/systemd/system/qwen-claude-md.service > /dev/null << 'SERVICE'
[Unit]
Description=Qwen CLAUDE.md Inference Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/qwen-claude-md
Environment=PYTHONPATH=/home/ubuntu/qwen-claude-md
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/usr/bin/python3 qwen_inference_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable qwen-claude-md.service

# Set permissions
sudo chown -R ubuntu:ubuntu /home/ubuntu/qwen-claude-md
chmod +x /home/ubuntu/qwen-claude-md/qwen_inference_server.py

EOF
    
    log "✅ Service configuration complete"
}

start_service() {
    local public_ip=$1
    log "🔄 Starting Qwen CLAUDE.md service..."
    
    ssh -i "$SCRIPT_DIR/config/mistral-base.pem" \
        -o StrictHostKeyChecking=no \
        ubuntu@"$public_ip" \
        "sudo systemctl start qwen-claude-md.service"
    
    log "⏳ Waiting for service to initialize (model download may take 5-10 minutes)..."
    
    # Wait for service to be ready
    for i in {1..60}; do
        if ssh -i "$SCRIPT_DIR/config/mistral-base.pem" \
           -o StrictHostKeyChecking=no \
           ubuntu@"$public_ip" \
           "curl -s http://localhost:$SERVICE_PORT/health" >/dev/null 2>&1; then
            log "✅ Service is ready!"
            break
        fi
        
        if (( i % 10 == 0 )); then
            log "Still waiting... ($i/60)"
            # Show service status
            ssh -i "$SCRIPT_DIR/config/mistral-base.pem" \
                -o StrictHostKeyChecking=no \
                ubuntu@"$public_ip" \
                "sudo systemctl status qwen-claude-md.service --no-pager -l" || true
        fi
        
        sleep 30
    done
}

run_tests() {
    local public_ip=$1
    log "🧪 Running integration tests..."
    
    ssh -i "$SCRIPT_DIR/config/mistral-base.pem" \
        -o StrictHostKeyChecking=no \
        ubuntu@"$public_ip" << 'EOF'
cd /home/ubuntu/qwen-claude-md

# Install test dependencies
pip3 install --quiet pytest requests

# Run tests
echo "Running Qwen CLAUDE.md integration tests..."
python3 test_qwen_claude_md.py
EOF
    
    log "✅ Tests completed"
}

print_summary() {
    local instance_id=$1
    local public_ip=$2
    
    cat << EOF

🎉 Qwen CLAUDE.md Deployment Complete!

Instance Details:
  Instance ID: $instance_id
  Public IP: $public_ip
  Instance Type: $INSTANCE_TYPE
  Region: $SELECTED_REGION

Service Access:
  Health Check: http://$public_ip:$SERVICE_PORT/health
  Generate API: http://$public_ip:$SERVICE_PORT/generate
  Documentation: http://$public_ip:$SERVICE_PORT/docs

SSH Access:
  ssh -i $SCRIPT_DIR/config/mistral-base.pem ubuntu@$public_ip

Service Management:
  Status: sudo systemctl status qwen-claude-md.service
  Logs: sudo journalctl -u qwen-claude-md.service -f
  Restart: sudo systemctl restart qwen-claude-md.service

Testing:
  cd /home/ubuntu/qwen-claude-md && python3 test_qwen_claude_md.py

Example API Usage:
  curl -X POST http://$public_ip:$SERVICE_PORT/generate \\
    -H "Content-Type: application/json" \\
    -d '{"prompt": "Create a Python web API with authentication", "complexity": "medium"}'

EOF
}

main() {
    log "🚀 Starting Qwen CLAUDE.md deployment..."
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --instance-type)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            --region)
                # Override preferred regions if specific region requested
                PREFERRED_REGIONS=("$2")
                shift 2
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    check_prerequisites
    select_region
    check_capacity
    
    INSTANCE_ID=$(launch_instance)
    PUBLIC_IP=$(wait_for_instance "$INSTANCE_ID")
    
    deploy_qwen_service "$PUBLIC_IP"
    start_service "$PUBLIC_IP"
    run_tests "$PUBLIC_IP"
    
    print_summary "$INSTANCE_ID" "$PUBLIC_IP"
    
    log "🎉 Deployment successful!"
}

# Handle interruption
trap 'error "Deployment interrupted"' INT TERM

# Run main function
main "$@"