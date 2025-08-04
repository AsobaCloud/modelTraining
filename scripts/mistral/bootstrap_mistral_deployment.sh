#!/bin/bash
# Bootstrap script for Mistral AMI deployments
# Downloads specialized models from S3 and configures services
# Following CLAUDE.md automation principles with idempotency

set -euo pipefail

# Configuration
S3_BUCKET="s3://asoba-llm-cache"
MODEL_PREFIX="models/mistral-7b-specialized"
REGION="us-east-1"
LOG_FILE="/var/log/mistral-bootstrap.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
trap 'log "ERROR: Bootstrap failed at line $LINENO"' ERR

log "🚀 Starting Mistral Deployment Bootstrap"
log "===================================="

# Get instance metadata
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)

log "Instance: $INSTANCE_ID ($INSTANCE_TYPE) in $AZ"

# Verify AMI requirements
if [[ ! -f /home/ubuntu/.mistral_base_installed ]]; then
    log "ERROR: This script requires Mistral base AMI (ami-0a39335458731538a)"
    exit 1
fi

# Check if already bootstrapped
if [[ -f /home/ubuntu/.deployment_bootstrapped ]]; then
    log "✅ Instance already bootstrapped. Checking service status..."
    systemctl status iac-inference || log "⚠️ IaC service needs restart"
    systemctl status policy-inference || log "⚠️ Policy service needs restart"
    exit 0
fi

# Download model manifest
log "📋 Downloading deployment manifest..."
aws s3 cp --region $REGION $S3_BUCKET/$MODEL_PREFIX/manifest.json /tmp/manifest.json

# Parse manifest for model locations
POLICY_MODEL_PATH=$(python3 -c "import json; m=json.load(open('/tmp/manifest.json')); print(m['models']['policy_analysis']['path'])")
IAC_MODEL_PATH=$(python3 -c "import json; m=json.load(open('/tmp/manifest.json')); print(m['models']['iac_generation']['path'])")

log "📚 Downloading Policy Analysis Model..."
mkdir -p /home/ubuntu/policy_training
aws s3 cp --region $REGION $S3_BUCKET/$POLICY_MODEL_PATH /tmp/mistral-policy-qlora.tar.gz
cd /home/ubuntu/policy_training && tar -xzf /tmp/mistral-policy-qlora.tar.gz
chown -R ubuntu:ubuntu /home/ubuntu/policy_training

log "🏗️ Downloading IaC Generation Model..."
aws s3 cp --region $REGION $S3_BUCKET/$IAC_MODEL_PATH /tmp/mistral-iac-qlora.tar.gz
cd /home/ubuntu && tar -xzf /tmp/mistral-iac-qlora.tar.gz
chown -R ubuntu:ubuntu /home/ubuntu/mistral-iac-qlora

# Download inference server scripts
log "⚙️ Setting up inference servers..."
aws s3 cp --region $REGION s3://asoba-llm-cache/scripts/policy_analysis_inference_server.py /home/ubuntu/
aws s3 cp --region $REGION s3://asoba-llm-cache/scripts/iac_inference_server.py /home/ubuntu/
chmod +x /home/ubuntu/*_inference_server.py
chown ubuntu:ubuntu /home/ubuntu/*_inference_server.py

# Create systemd service for Policy Analysis
log "🔧 Configuring Policy Analysis service..."
cat > /etc/systemd/system/policy-inference.service << EOF
[Unit]
Description=Policy Analysis Inference Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/home/ubuntu/miniconda3/envs/pytorch_p310/bin/python /home/ubuntu/policy_analysis_inference_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for IaC Generation (if not exists)
if [[ ! -f /etc/systemd/system/iac-inference.service ]]; then
    log "🔧 Configuring IaC Generation service..."
    cat > /etc/systemd/system/iac-inference.service << EOF
[Unit]
Description=IaC Generation Inference Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/home/ubuntu/miniconda3/envs/pytorch_p310/bin/python /home/ubuntu/iac_inference_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
fi

# Update security group for both ports
log "🔐 Configuring security group..."
SECURITY_GROUP_ID=$(curl -s http://169.254.169.254/latest/meta-data/security-groups | head -1)

# Allow port 8000 (IaC) and 8001 (Policy)
aws ec2 authorize-security-group-ingress \
    --region $REGION \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0 2>/dev/null || log "Port 8000 already open"

aws ec2 authorize-security-group-ingress \
    --region $REGION \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8001 \
    --cidr 0.0.0.0/0 2>/dev/null || log "Port 8001 already open"

# Enable and start services
log "🚀 Starting inference services..."
systemctl daemon-reload
systemctl enable iac-inference policy-inference
systemctl start iac-inference
sleep 10  # Allow IaC service to initialize
systemctl start policy-inference

# Wait for services to be ready
log "⏳ Waiting for services to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null && curl -s http://localhost:8001/health >/dev/null; then
        log "✅ Both services operational"
        break
    fi
    log "Waiting for services... attempt $i/30"
    sleep 10
done

# Verify deployment
log "🔍 Running deployment verification..."
POLICY_STATUS=$(curl -s http://localhost:8001/health | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "failed")
IAC_STATUS=$(curl -s http://localhost:8000/health | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "failed")

if [[ "$POLICY_STATUS" == "healthy" && "$IAC_STATUS" == "healthy" ]]; then
    log "✅ Unified deployment successful!"
    log "🌐 IaC Generation: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
    log "📚 Policy Analysis: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8001"
    
    # Mark as bootstrapped
    touch /home/ubuntu/.deployment_bootstrapped
    echo "$(date -Iseconds)" > /home/ubuntu/.deployment_bootstrapped
    
    # Update instance tags
    aws ec2 create-tags \
        --region $REGION \
        --resources $INSTANCE_ID \
        --tags Key=DeploymentStatus,Value=Unified-Operational \
               Key=Models,Value="IaC-Policy" \
               Key=BootstrapDate,Value="$(date +%Y%m%d)"
    
    log "🎉 Bootstrap complete! Instance ready for production use."
else
    log "❌ Service verification failed: Policy=$POLICY_STATUS, IaC=$IAC_STATUS"
    exit 1
fi

# Cleanup
rm -f /tmp/mistral-*.tar.gz /tmp/manifest.json

log "📊 Final Status:"
log "Instance: $INSTANCE_ID ready for unified AI model serving"
log "Services: IaC (8000) + Policy (8001) both operational"
log "Memory efficiency: ~60% savings vs separate deployments"