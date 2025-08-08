#!/bin/bash
"""
Policy Analysis Training Instance Deployment
Deploy GPU instance using ami-0a39335458731538a for policy-specific Mistral training
Following CLAUDE.md safety-first cloud edition principles
"""

set -euo pipefail

# Configuration
AMI_ID="ami-0a39335458731538a"
INSTANCE_TYPE="g4dn.xlarge"
KEY_NAME="mistral-base"
REGION="us-east-1"
PROJECT_NAME="PolicyAnalyst"

# Safety checks and region configuration
check_region() {
    echo "🔍 Checking region configuration..."
    if [[ -f "/config/REGION" ]]; then
        CONFIG_REGION=$(cat /config/REGION)
        if [[ "$CONFIG_REGION" != "$REGION" ]]; then
            echo "⚠️  Region mismatch: config=$CONFIG_REGION, script=$REGION"
            read -p "Use config region $CONFIG_REGION? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                REGION="$CONFIG_REGION"
            fi
        fi
    fi
    echo "✅ Using region: $REGION"
}

# Verify AMI and instance compatibility
verify_ami_instance_compatibility() {
    echo "🔍 Verifying AMI and instance type compatibility..."
    
    # Get AMI architecture and description
    AMI_INFO=$(aws ec2 describe-images \
        --image-ids "$AMI_ID" \
        --region "$REGION" \
        --query 'Images[0].[Architecture,Description,Name]' \
        --output text)
    
    if [[ -z "$AMI_INFO" ]]; then
        echo "❌ AMI $AMI_ID not found in region $REGION"
        exit 1
    fi
    
    echo "✅ AMI verification complete:"
    echo "  AMI ID: $AMI_ID"
    echo "  Details: $AMI_INFO"
    
    # Check if instance type supports GPU (for T4 GPU requirement)
    if [[ "$INSTANCE_TYPE" =~ ^g4dn\.|^g5\.|^p3\.|^p4\. ]]; then
        echo "✅ GPU instance type verified: $INSTANCE_TYPE"
    else
        echo "❌ Instance type $INSTANCE_TYPE may not support GPU training"
        exit 1
    fi
}

# Check vCPU quota
check_vcpu_quota() {
    echo "🔍 Checking vCPU quota availability..."
    
    # Get current running instances vCPU usage
    USED_VCPUS=$(aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[].Instances[].CpuOptions.CoreCount' \
        --output text | awk '{s+=$1} END{print s+0}')
    
    echo "Current vCPU usage: $USED_VCPUS"
    
    # Get vCPU quota (On-Demand Standard instances)
    QUOTA=$(aws service-quotas get-service-quota \
        --service-code ec2 \
        --quota-code L-1216C47A \
        --region "$REGION" \
        --query 'ServiceQuota.Value' \
        --output text 2>/dev/null || echo "0")
    
    if [[ "$QUOTA" == "0" ]]; then
        echo "⚠️  Could not retrieve vCPU quota, proceeding with caution"
        return 0
    fi
    
    echo "vCPU quota: $QUOTA"
    
    # Estimate vCPUs for requested instance
    case "$INSTANCE_TYPE" in
        "g4dn.xlarge") REQUESTED_VCPUS=4 ;;
        "g4dn.2xlarge") REQUESTED_VCPUS=8 ;;
        "g5.2xlarge") REQUESTED_VCPUS=8 ;;
        *) REQUESTED_VCPUS=4 ;;  # Conservative estimate
    esac
    
    if (( USED_VCPUS + REQUESTED_VCPUS > QUOTA )); then
        echo "❌ vCPU quota exceeded: used=$USED_VCPUS + requested=$REQUESTED_VCPUS > quota=$QUOTA"
        exit 1
    fi
    
    echo "✅ vCPU quota check passed"
}

# Create security group
create_security_group() {
    echo "🔍 Creating security group for policy training..."
    
    SG_NAME="policy-training-sg"
    
    # Check if security group already exists
    SG_ID=$(aws ec2 describe-security-groups \
        --region "$REGION" \
        --filters "Name=group-name,Values=$SG_NAME" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "None")
    
    if [[ "$SG_ID" != "None" ]]; then
        echo "✅ Security group already exists: $SG_ID"
        echo "$SG_ID"
        return 0
    fi
    
    # Create new security group
    SG_ID=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Security group for policy analysis training" \
        --query 'GroupId' \
        --output text)
    
    echo "✅ Created security group: $SG_ID"
    
    # Add SSH rule (replace 0.0.0.0/0 with your IP for better security)
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0
    
    # Add inference server port
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0
    
    echo "✅ Security group rules configured"
    echo "$SG_ID"
}

# Create user data script
create_user_data_script() {
    echo "📝 Creating user data script..."
    
    cat > /tmp/policy_training_user_data.sh << 'EOF'
#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=== Policy Training Instance Setup Started ==="
echo "Timestamp: $(date)"

# Update system
apt-get update
apt-get install -y htop tree jq git curl wget unzip

# Install Python packages for PDF processing
pip3 install PyPDF2 pdfplumber boto3

# Create training directory structure
mkdir -p /home/ubuntu/policy_training
mkdir -p /home/ubuntu/data
chown -R ubuntu:ubuntu /home/ubuntu/policy_training
chown -R ubuntu:ubuntu /home/ubuntu/data

# Verify GPU availability
echo "=== GPU Verification ==="
nvidia-smi

# Verify Python environment
echo "=== Python Environment Check ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Create completion marker
echo "Instance setup completed at $(date)" > /home/ubuntu/setup_complete.txt
chown ubuntu:ubuntu /home/ubuntu/setup_complete.txt

echo "=== Policy Training Instance Setup Complete ==="
EOF
    
    echo "✅ User data script created"
}

# Launch instance
launch_instance() {
    echo "🚀 Launching policy training instance..."
    
    # Get security group ID
    SG_ID=$(create_security_group)
    
    # Create user data script
    create_user_data_script
    
    # Get default subnet ID
    SUBNET_ID=$(aws ec2 describe-subnets \
        --region "$REGION" \
        --filters "Name=default-for-az,Values=true" \
        --query 'Subnets[0].SubnetId' \
        --output text)
    
    if [[ "$SUBNET_ID" == "None" || -z "$SUBNET_ID" ]]; then
        echo "❌ No default subnet found in region $REGION"
        exit 1
    fi
    
    echo "Using subnet: $SUBNET_ID"
    
    # [CLOUD-CONFIRM] Pre-flight confirmation
    echo "=============================================="
    echo "[CLOUD-CONFIRM] Launch instance with config:"
    echo "  AMI: $AMI_ID"
    echo "  Instance Type: $INSTANCE_TYPE"
    echo "  Region: $REGION"
    echo "  Key: $KEY_NAME"
    echo "  Security Group: $SG_ID"
    echo "  Subnet: $SUBNET_ID"
    echo "=============================================="
    read -p "Proceed with launch? (yes/no): " -r CONFIRM
    
    if [[ "$CONFIRM" != "yes" ]]; then
        echo "❌ Launch cancelled by user"
        exit 1
    fi
    
    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --subnet-id "$SUBNET_ID" \
        --block-device-mappings '[{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": 100,
                "VolumeType": "gp3",
                "DeleteOnTermination": true
            }
        }]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=policy-analysis-training},{Key=Project,Value=$PROJECT_NAME},{Key=Owner,Value=$USER}]" \
        --user-data file:///tmp/policy_training_user_data.sh \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    echo "✅ Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    echo "⏳ Waiting for instance to be running..."
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo "✅ Instance is running"
    echo "Instance ID: $INSTANCE_ID"
    echo "Public IP: $PUBLIC_IP"
    
    # Save instance info
    cat > /tmp/policy_instance_info.json << EOF
{
    "instance_id": "$INSTANCE_ID",
    "public_ip": "$PUBLIC_IP",
    "instance_type": "$INSTANCE_TYPE",
    "ami_id": "$AMI_ID",
    "region": "$REGION",
    "key_name": "$KEY_NAME",
    "security_group_id": "$SG_ID",
    "launch_time": "$(date -Iseconds)",
    "project": "$PROJECT_NAME"
}
EOF
    
    echo "📊 Instance information saved to: /tmp/policy_instance_info.json"
    
    # Connection instructions
    echo ""
    echo "🔑 Connection Instructions:"
    echo "ssh -i /config/$KEY_NAME.pem ubuntu@$PUBLIC_IP"
    echo ""
    echo "⏳ Wait 2-3 minutes for user data script to complete setup"
    echo "   Check setup status: ssh -i /config/$KEY_NAME.pem ubuntu@$PUBLIC_IP 'cat setup_complete.txt'"
    echo ""
    
    return 0
}

# Dry run validation
dry_run_validation() {
    echo "🧪 Running dry-run validation..."
    
    # Test run-instances with dry-run
    aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --dry-run \
        --query 'Instances[0].InstanceId' \
        --output text 2>/dev/null || {
            echo "❌ Dry-run failed - check permissions and configuration"
            return 1
        }
    
    echo "✅ Dry-run validation passed"
}

# Main execution
main() {
    echo "🚀 Policy Analysis Training Instance Deployment"
    echo "==============================================="
    echo "AMI: $AMI_ID"
    echo "Instance Type: $INSTANCE_TYPE"
    echo "Region: $REGION"
    echo "Key: $KEY_NAME"
    echo ""
    
    # Run all safety checks
    check_region
    verify_ami_instance_compatibility
    check_vcpu_quota
    dry_run_validation
    
    # Launch instance
    launch_instance
    
    echo "🎯 Deployment Complete!"
    echo "Instance is ready for policy analysis training"
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi