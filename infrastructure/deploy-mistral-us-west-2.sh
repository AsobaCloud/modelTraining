#!/bin/bash

# Mistral Deployment Script for us-west-2 (Oregon)
# Uses your approved 8 vCPU Spot quota

set -e

# Configuration
REGION="us-west-2"
PROJECT_NAME="mistral-uswest2"
INSTANCE_TYPES=("g4dn.xlarge" "g5.xlarge")  # 4 vCPUs each, saving 4 vCPUs for SDXL
PRODUCT_CODE="prod-md5sjr2k4sdu6"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Find Ubuntu AMI for base model installation in us-west-2
find_base_ami() {
    log "Looking for Ubuntu 22.04 LTS AMI in us-west-2..."
    
    # Try to find AMI by product code
    AMI_ID=$(aws ec2 describe-images \
        --region $REGION \
        --owners 099720109477 \
        --filters "Name=product-code,Values=${PRODUCT_CODE}" \
        --query 'Images[0].ImageId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$AMI_ID" = "None" ] || [ -z "$AMI_ID" ]; then
        # Try to find by name pattern
        log "Searching by name pattern in us-west-2..."
        AMI_ID=$(aws ec2 describe-images \
            --region $REGION \
            --owners 099720109477 \
            --filters "Name=name,Values=*mistral*7b*" "Name=state,Values=available" \
            --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
            --output text 2>/dev/null || echo "None")
    fi
    
    if [ "$AMI_ID" = "None" ] || [ -z "$AMI_ID" ]; then
        error "Could not find Ubuntu AMI in us-west-2."
    fi
    
    # Get AMI details
    AMI_NAME=$(aws ec2 describe-images \
        --region $REGION \
        --image-ids $AMI_ID \
        --query 'Images[0].Name' \
        --output text)
    
    log "Found AMI: $AMI_ID"
    log "AMI Name: $AMI_NAME"
    
    export AMI_ID
}

# Setup network configuration in us-west-2
setup_network() {
    log "Setting up network configuration in us-west-2..."
    
    # Use default VPC in us-west-2
    VPC_ID=$(aws ec2 describe-vpcs \
        --region $REGION \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text)
    
    if [ "$VPC_ID" = "None" ]; then
        error "No default VPC found in us-west-2. Please create a VPC first."
    fi
    
    # Get all public subnets in default VPC (for multiple AZ fallback)
    SUBNETS=$(aws ec2 describe-subnets \
        --region $REGION \
        --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
        --query 'Subnets[*].[SubnetId,AvailabilityZone]' \
        --output text)
    
    # Store subnets for fallback
    echo "$SUBNETS" > /tmp/available_subnets_uswest2.txt
    
    log "Using VPC: $VPC_ID"
    log "Available subnets in us-west-2:"
    while read -r subnet_line; do
        if [ -n "$subnet_line" ]; then
            subnet_id=$(echo "$subnet_line" | cut -f1)
            subnet_az=$(echo "$subnet_line" | cut -f2)
            log "  - $subnet_id ($subnet_az)"
        fi
    done < /tmp/available_subnets_uswest2.txt
    
    export VPC_ID
}

# Create security group in us-west-2
setup_security_group() {
    log "Setting up security group in us-west-2..."
    
    # Get your public IP
    YOUR_IP=$(curl -s ipinfo.io/ip 2>/dev/null || echo "0.0.0.0")
    if [ "$YOUR_IP" = "0.0.0.0" ]; then
        warn "Could not determine your IP. Using 0.0.0.0/0 (not recommended for production)"
        YOUR_IP="0.0.0.0/0"
    else
        YOUR_IP="$YOUR_IP/32"
        log "Detected your IP: $YOUR_IP"
    fi
    
    # Check if security group exists
    SG_ID=$(aws ec2 describe-security-groups \
        --region $REGION \
        --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=${PROJECT_NAME}-sg" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$SG_ID" = "None" ]; then
        log "Creating security group in us-west-2..."
        SG_ID=$(aws ec2 create-security-group \
            --region $REGION \
            --group-name "${PROJECT_NAME}-sg" \
            --description "Security group for ${PROJECT_NAME} in us-west-2" \
            --vpc-id $VPC_ID \
            --query 'GroupId' \
            --output text)
        
        # Add rules for SSH and API access
        aws ec2 authorize-security-group-ingress \
            --region $REGION \
            --group-id $SG_ID \
            --protocol tcp \
            --port 22 \
            --cidr $YOUR_IP
        
        aws ec2 authorize-security-group-ingress \
            --region $REGION \
            --group-id $SG_ID \
            --protocol tcp \
            --port 8000 \
            --cidr $YOUR_IP
        
        aws ec2 authorize-security-group-ingress \
            --region $REGION \
            --group-id $SG_ID \
            --protocol tcp \
            --port 8080 \
            --cidr $YOUR_IP
        
        log "Created security group: $SG_ID"
    else
        log "Using existing security group: $SG_ID"
    fi
    
    export SG_ID
}

# Setup SSH key in us-west-2
setup_ssh_key() {
    log "Setting up SSH key in us-west-2..."
    
    SSH_KEY_NAME="${PROJECT_NAME}-key"
    
    # Check if key exists in us-west-2
    if aws ec2 describe-key-pairs \
        --region $REGION \
        --key-names $SSH_KEY_NAME &>/dev/null; then
        log "Using existing SSH key: $SSH_KEY_NAME"
    else
        log "Creating SSH key: $SSH_KEY_NAME"
        aws ec2 create-key-pair \
            --region $REGION \
            --key-name $SSH_KEY_NAME \
            --query 'KeyMaterial' \
            --output text > ~/.ssh/$SSH_KEY_NAME.pem
        chmod 600 ~/.ssh/$SSH_KEY_NAME.pem
        log "SSH key saved to ~/.ssh/$SSH_KEY_NAME.pem"
    fi
    
    export SSH_KEY_NAME
}

# Launch instance in us-west-2
launch_instance() {
    log "Launching Mistral instance in us-west-2..."
    log "Using Spot instance (you have 8 vCPU Spot quota approved)"
    
    # Check if instance already running
    EXISTING_INSTANCE=$(aws ec2 describe-instances \
        --region $REGION \
        --filters "Name=tag:Name,Values=${PROJECT_NAME}-instance" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[].Instances[0].InstanceId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$EXISTING_INSTANCE" != "None" ] && [ "$EXISTING_INSTANCE" != "" ]; then
        log "Found existing instance: $EXISTING_INSTANCE"
        INSTANCE_ID=$EXISTING_INSTANCE
        USED_INSTANCE_TYPE="existing"
    else
        # Try each instance type and availability zone combination
        INSTANCE_ID=""
        USED_INSTANCE_TYPE=""
        
        for instance_type in "${INSTANCE_TYPES[@]}"; do
            log "Trying instance type: $instance_type"
            
            while read -r subnet_line; do
                if [ -n "$subnet_line" ]; then
                    try_subnet_id=$(echo "$subnet_line" | cut -f1)
                    try_subnet_az=$(echo "$subnet_line" | cut -f2)
                    
                    log "  → Trying $instance_type in AZ: $try_subnet_az"
                    
                    INSTANCE_ID=$(aws ec2 run-instances \
                        --region $REGION \
                        --image-id $AMI_ID \
                        --instance-type $instance_type \
                        --key-name $SSH_KEY_NAME \
                        --security-group-ids $SG_ID \
                        --subnet-id $try_subnet_id \
                        --associate-public-ip-address \
                        --instance-market-options '{
                            "MarketType": "spot",
                            "SpotOptions": {
                                "MaxPrice": "1.00",
                                "SpotInstanceType": "one-time"
                            }
                        }' \
                        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PROJECT_NAME}-instance},{Key=Project,Value=${PROJECT_NAME}},{Key=InstanceType,Value=$instance_type},{Key=Region,Value=us-west-2}]" \
                        --query 'Instances[0].InstanceId' \
                        --output text 2>/dev/null || echo "")
                    
                    if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
                        log "✅ Successfully launched $instance_type in AZ: $try_subnet_az"
                        USED_INSTANCE_TYPE=$instance_type
                        break 2  # Break out of both loops
                    else
                        warn "    ❌ No $instance_type capacity in AZ: $try_subnet_az"
                    fi
                fi
            done < /tmp/available_subnets_uswest2.txt
        done
        
        if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
            error "Failed to launch instance in any availability zone in us-west-2."
        fi
        
        log "Launched instance: $INSTANCE_ID ($USED_INSTANCE_TYPE)"
        log "Waiting for instance to be running..."
        aws ec2 wait instance-running \
            --region $REGION \
            --instance-ids $INSTANCE_ID
        
        # Save instance ID
        echo $INSTANCE_ID > .mistral-uswest2-instance-id
    fi
    
    # Get instance details
    INSTANCE_INFO=$(aws ec2 describe-instances \
        --region $REGION \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].[PrivateIpAddress,PublicIpAddress,State.Name,InstanceType,Placement.AvailabilityZone]' \
        --output text)
    
    PRIVATE_IP=$(echo $INSTANCE_INFO | cut -d' ' -f1)
    PUBLIC_IP=$(echo $INSTANCE_INFO | cut -d' ' -f2)
    STATE=$(echo $INSTANCE_INFO | cut -d' ' -f3)
    ACTUAL_TYPE=$(echo $INSTANCE_INFO | cut -d' ' -f4)
    ACTUAL_AZ=$(echo $INSTANCE_INFO | cut -d' ' -f5)
    
    log "Instance Details:"
    log "  Instance ID: $INSTANCE_ID"
    log "  Instance Type: $ACTUAL_TYPE"
    log "  Availability Zone: $ACTUAL_AZ"
    log "  State: $STATE"
    log "  Private IP: $PRIVATE_IP"
    log "  Public IP: $PUBLIC_IP"
    log ""
    log "🎉 Mistral deployment successful in us-west-2!"
    log "SSH: ssh -i ~/.ssh/${SSH_KEY_NAME}.pem ubuntu@$PUBLIC_IP"
    log "API: http://$PUBLIC_IP:8000"
    log "Wait 5-15 minutes for Mistral service to start, then test:"
    log "curl http://$PUBLIC_IP:8000/health"
}

# Main execution
main() {
    log "Starting Mistral deployment in us-west-2 (Oregon)..."
    log "Using 4 vCPUs (saving 4 vCPUs for SDXL server)"
    
    find_base_ami
    setup_network
    setup_security_group
    setup_ssh_key
    launch_instance
    
    log "Deployment complete! Instance running in us-west-2."
}

main "$@"