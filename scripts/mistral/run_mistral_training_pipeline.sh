#!/bin/bash
# One-shot Mistral training pipeline script
# Executes data preparation and training on g5.2xlarge

set -euo pipefail

# Configuration
INSTANCE_TYPE="g5.2xlarge"
REGION="us-east-1"
RUN_ID="mistral-$(date +%Y%m%d-%H%M%S)"
PROJECT_NAME="mistral-training-$RUN_ID"
S3_BUCKET="asoba-llm-cache"
DATASET_PREFIX="datasets/mistral-verbosity"
MODEL_PREFIX="models/mistral-verbosity"
MONITORING_PREFIX="training-runs"
MAX_PDFS=50000
VALIDATION_SPLIT=0.2
LOG_FILE="mistral_pipeline_$RUN_ID.log"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

update_s3_status() {
    local status=$1
    local phase=${2:-"unknown"}
    local details=${3:-""}
    
    cat > /tmp/status_update.json << EOF_STATUS
{
  "run_id": "$RUN_ID",
  "status": "$status",
  "phase": "$phase",
  "last_update": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "instance_type": "$INSTANCE_TYPE",
  "region": "$REGION",
  "details": "$details"
}
EOF_STATUS
    
    aws s3 cp /tmp/status_update.json s3://$S3_BUCKET/$MONITORING_PREFIX/$RUN_ID/metadata.json --region $REGION --quiet
    rm /tmp/status_update.json
}

log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

error() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo -e "${RED}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
    echo -e "${RED}Full error context:${NC}"
    # Print last 10 lines of log for context
    tail -10 "$LOG_FILE" 2>/dev/null || true
    
    # Write error sentinel to S3 for production monitoring
    if [[ -n "${RUN_ID:-}" ]]; then
        echo "$1" | aws s3 cp - s3://$S3_BUCKET/$MONITORING_PREFIX/$RUN_ID/_error --region $REGION --quiet 2>/dev/null || true
        update_s3_status "failed" "error" "$1" 2>/dev/null || true
    fi
    
    exit 1
}

warn() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

check_command() {
    local cmd=$1
    local msg=$2
    log "Executing: $cmd"
    if ! eval "$cmd"; then
        error "$msg - Command failed: $cmd"
    fi
}

# Phase 1: Launch Training Instance
launch_instance() {
    log "Launching $INSTANCE_TYPE instance in $REGION..."
    
    # Create key pair if needed
    KEY_NAME="${PROJECT_NAME}-key"
    if ! aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &>/dev/null; then
        log "Creating SSH key pair..."
        check_command "aws ec2 create-key-pair \
            --key-name $KEY_NAME \
            --region $REGION \
            --query 'KeyMaterial' \
            --output text > ~/.ssh/$KEY_NAME.pem" \
            "Failed to create SSH key pair"
        check_command "chmod 600 ~/.ssh/$KEY_NAME.pem" "Failed to set key permissions"
    else
        log "Using existing key pair: $KEY_NAME"
    fi
    
    # Get default VPC and subnet
    VPC_ID=$(aws ec2 describe-vpcs \
        --region $REGION \
        --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' \
        --output text)
    
    SUBNET_ID=$(aws ec2 describe-subnets \
        --region $REGION \
        --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
        --query 'Subnets[0].SubnetId' \
        --output text)
    
    # Create security group
    SG_NAME="${PROJECT_NAME}-sg"
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "Security group for Mistral training" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' \
        --output text)
    
    # Add SSH access
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    # Find Ubuntu 22.04 AMI with Deep Learning
    AMI_ID=$(aws ec2 describe-images \
        --owners 898082745236 \
        --filters "Name=name,Values=Deep Learning AMI GPU TensorFlow*Ubuntu 22.04*" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --region $REGION \
        --output text)
    
    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --subnet-id $SUBNET_ID \
        --region $REGION \
        --block-device-mappings '[{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": 200,
                "VolumeType": "gp3",
                "DeleteOnTermination": true
            }
        }]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$PROJECT_NAME}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    log "Instance launched: $INSTANCE_ID"
    log "Waiting for instance to be running..."
    
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --region $REGION \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    log "Instance ready at: $PUBLIC_IP"
    echo $INSTANCE_ID > .mistral-instance-id
    echo $PUBLIC_IP > .mistral-instance-ip
    
    # Wait for SSH to be ready
    log "Waiting for SSH to be ready..."
    while ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP exit 2>/dev/null; do
        sleep 5
    done
    
    export INSTANCE_IP=$PUBLIC_IP
    export SSH_KEY=~/.ssh/$KEY_NAME.pem
}

# Phase 2: Setup Instance
setup_instance() {
    log "Setting up instance environment..."
    
    # Create setup script
    cat > /tmp/setup_instance.sh << 'EOF'
#!/bin/bash
set -euo pipefail

# Update system
sudo apt-get update

# Install Python dependencies
pip install --upgrade pip
pip install transformers==4.53.1
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install accelerate==1.8.1
pip install peft==0.16.0
pip install bitsandbytes==0.46.1
pip install datasets==4.0.0
pip install boto3
pip install safetensors

# Create working directory on EBS volume
mkdir -p /mnt/training/mistral_training
cd /mnt/training/mistral_training

# Download Mistral-7B-v0.3 model from S3 to EBS volume
echo "Downloading Mistral-7B-v0.3 model from S3 to EBS volume..."
mkdir -p /mnt/training/models/mistral-7b-v0.3
aws s3 sync s3://asoba-llm-cache/models/mistralai/Mistral-7B-v0.3/ /mnt/training/models/mistral-7b-v0.3/ --region us-east-1
echo "Model downloaded successfully from S3 to /mnt/training/models/mistral-7b-v0.3"

echo "Setup complete!"
EOF

    # Copy and run setup script
    scp -o StrictHostKeyChecking=no -i $SSH_KEY /tmp/setup_instance.sh ubuntu@$INSTANCE_IP:~/
    ssh -o StrictHostKeyChecking=no -i $SSH_KEY ubuntu@$INSTANCE_IP "chmod +x setup_instance.sh && ./setup_instance.sh"
}

# Phase 3: Deploy Training Scripts
deploy_scripts() {
    log "Deploying training scripts..."
    
    # Copy necessary scripts to EBS volume
    scp -o StrictHostKeyChecking=no -i $SSH_KEY \
        scripts/mistral/prepare_mistral_dataset.py \
        scripts/mistral/train_mistral_simple.py \
        scripts/mistral/shared/dataset_utils.py \
        ubuntu@$INSTANCE_IP:/mnt/training/mistral_training/
    
    # Create shared directory on EBS volume
    ssh -o StrictHostKeyChecking=no -i $SSH_KEY ubuntu@$INSTANCE_IP "mkdir -p /mnt/training/mistral_training/shared"
    scp -o StrictHostKeyChecking=no -i $SSH_KEY \
        scripts/mistral/shared/__init__.py \
        ubuntu@$INSTANCE_IP:/mnt/training/mistral_training/shared/
}

# Phase 4: Run Data Preparation
run_data_prep() {
    log "Checking if datasets already exist..."
    
    # Check if datasets exist in S3
    TRAIN_EXISTS=false
    VAL_EXISTS=false
    
    if aws s3 ls "s3://$S3_BUCKET/$DATASET_PREFIX/train_dataset.jsonl" --region $REGION &>/dev/null; then
        TRAIN_EXISTS=true
    fi
    
    if aws s3 ls "s3://$S3_BUCKET/$DATASET_PREFIX/val_dataset.jsonl" --region $REGION &>/dev/null; then
        VAL_EXISTS=true
    fi
    
    if [[ "$TRAIN_EXISTS" == "true" && "$VAL_EXISTS" == "true" ]]; then
        log "✅ Datasets already exist in S3, skipping data preparation"
        log "Training dataset: s3://$S3_BUCKET/$DATASET_PREFIX/train_dataset.jsonl"
        log "Validation dataset: s3://$S3_BUCKET/$DATASET_PREFIX/val_dataset.jsonl"
        log ""
        log "To force data regeneration, delete the existing datasets:"
        log "aws s3 rm s3://$S3_BUCKET/$DATASET_PREFIX/train_dataset.jsonl"
        log "aws s3 rm s3://$S3_BUCKET/$DATASET_PREFIX/val_dataset.jsonl"
        return 0
    fi
    
    log "Datasets not found in S3, running data preparation pipeline..."
    log "Expected location: s3://$S3_BUCKET/$DATASET_PREFIX/"
    
    ssh -o StrictHostKeyChecking=no -i $SSH_KEY ubuntu@$INSTANCE_IP << EOF
cd /mnt/training/mistral_training
export PYTHONPATH=/mnt/training/mistral_training:\$PYTHONPATH

echo "Starting data preparation..."
python3 prepare_mistral_dataset.py \
    --work-dir /mnt/training/data_prep \
    --output-bucket $S3_BUCKET \
    --output-prefix $DATASET_PREFIX \
    --max-pdfs $MAX_PDFS \
    --validation-split $VALIDATION_SPLIT

echo "Data preparation complete!"
EOF

    # Verify datasets were created
    log "Verifying datasets were created..."
    if ! aws s3 ls "s3://$S3_BUCKET/$DATASET_PREFIX/train_dataset.jsonl" --region $REGION &>/dev/null; then
        error "Training dataset was not created in S3"
    fi
    
    if ! aws s3 ls "s3://$S3_BUCKET/$DATASET_PREFIX/val_dataset.jsonl" --region $REGION &>/dev/null; then
        error "Validation dataset was not created in S3"
    fi
    
    log "✅ Datasets successfully created and verified in S3"
}

# Phase 5: Run Training
run_training() {
    log "Running Mistral training..."
    
    # Get dataset paths from S3
    TRAIN_DATASET="s3://$S3_BUCKET/$DATASET_PREFIX/train_dataset.jsonl"
    VAL_DATASET="s3://$S3_BUCKET/$DATASET_PREFIX/val_dataset.jsonl"
    
    log "Training dataset: $TRAIN_DATASET"
    log "Validation dataset: $VAL_DATASET"
    
    # Copy monitoring scripts to EBS volume
    log "Setting up monitoring integration..."
    scp -o StrictHostKeyChecking=no -i $SSH_KEY \
        scripts/monitoring/training_monitor.py \
        scripts/monitoring/s3_model_uploader.py \
        ubuntu@$INSTANCE_IP:/mnt/training/mistral_training/
    
    # Create training script
    cat > /tmp/run_training.sh << EOF
#!/bin/bash
cd /mnt/training/mistral_training
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Setup monitoring
RUN_ID="$RUN_ID"
S3_BUCKET="$S3_BUCKET"
MONITORING_PREFIX="$MONITORING_PREFIX"
MONITORING_DIR="monitoring_\${RUN_ID}"

mkdir -p \$MONITORING_DIR

# Create monitoring metadata
cat > \$MONITORING_DIR/metadata.json << METADATA
{
  "run_id": "\$RUN_ID",
  "base_model": "mistralai/Mistral-7B-v0.3",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "instance_id": "$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo 'unknown')",
  "status": "running",
  "instance_type": "$INSTANCE_TYPE",
  "region": "$REGION"
}
METADATA

# Initial sync to S3
aws s3 cp \$MONITORING_DIR/metadata.json s3://\$S3_BUCKET/\$MONITORING_PREFIX/\$RUN_ID/metadata.json

# Monitor script with S3 sync
cat > monitor_training.sh << 'MONITOR'
#!/bin/bash
MONITORING_DIR="monitoring_$RUN_ID"
S3_PATH="s3://$S3_BUCKET/$MONITORING_PREFIX/$RUN_ID"

# Function to sync logs to S3
sync_to_s3() {
    # Copy training log
    if [ -f training.log ]; then
        tail -1000 training.log > \$MONITORING_DIR/training_log_latest.txt
        aws s3 cp \$MONITORING_DIR/training_log_latest.txt \$S3_PATH/logs/training_log_latest.txt --quiet
    fi
    
    # Parse and upload metrics if available
    if [ -f training.log ]; then
        grep -E "loss:|eval_loss:|gpu_memory:" training.log | tail -100 > \$MONITORING_DIR/metrics.txt
        aws s3 cp \$MONITORING_DIR/metrics.txt \$S3_PATH/metrics.txt --quiet
    fi
}

# Monitoring loop
while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    echo ""
    echo "=== Training Progress ==="
    tail -20 training.log 2>/dev/null || echo "Waiting for training to start..."
    echo ""
    echo "=== S3 Monitoring ==="
    echo "Run ID: $RUN_ID"
    echo "Monitor at: https://console.aws.amazon.com/s3/buckets/$S3_BUCKET/$MONITORING_PREFIX/$RUN_ID/"
    
    # Sync to S3 every iteration
    sync_to_s3
    
    sleep 30
done
MONITOR
chmod +x monitor_training.sh

# Run training with comprehensive logging and monitoring integration
echo "Starting training with monitoring..."
nohup python3 -u train_mistral_simple.py \
    --train-dataset $TRAIN_DATASET \
    --val-dataset $VAL_DATASET \
    --output-dir /mnt/training/mistral_output \
    --s3-bucket $S3_BUCKET \
    --s3-prefix $MODEL_PREFIX \
    --batch-size 1 \
    --max-seq-length 1024 \
    --run-id $RUN_ID 2>&1 | tee training.log &

TRAINING_PID=\$!
echo "Training started! PID: \$TRAINING_PID"

# Start monitoring in background
nohup ./monitor_training.sh > monitor.log 2>&1 &
MONITOR_PID=\$!

echo "Monitoring started! PID: \$MONITOR_PID"
echo ""
echo "Training PID: \$TRAINING_PID" > process_info.txt
echo "Monitor PID: \$MONITOR_PID" >> process_info.txt
echo ""
echo "To check training: tail -f training.log"
echo "To check monitoring: tail -f monitor.log"
echo "To view GPU status: ./monitor_training.sh"
echo ""
echo "S3 monitoring URL: https://console.aws.amazon.com/s3/buckets/$S3_BUCKET/$MONITORING_PREFIX/$RUN_ID/"
EOF

    # Deploy and run training
    scp -o StrictHostKeyChecking=no -i $SSH_KEY /tmp/run_training.sh ubuntu@$INSTANCE_IP:/mnt/training/mistral_training/
    ssh -o StrictHostKeyChecking=no -i $SSH_KEY ubuntu@$INSTANCE_IP "cd /mnt/training/mistral_training && chmod +x run_training.sh && ./run_training.sh"
    
    log "Training started on instance!"
    log "SSH to monitor: ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
    log "Then run: cd /mnt/training/mistral_training && ./monitor_training.sh"
}

# Phase 6: Save instance info
save_info() {
    cat > mistral_training_info.txt << EOF
Mistral Training Instance Information
====================================
Run ID: $RUN_ID
Instance ID: $INSTANCE_ID
Public IP: $INSTANCE_IP
SSH Key: $SSH_KEY
Region: $REGION

SSH Access:
ssh -i $SSH_KEY ubuntu@$INSTANCE_IP

Monitor Training Locally:
ssh -i $SSH_KEY ubuntu@$INSTANCE_IP "cd /mnt/training/mistral_training && ./monitor_training.sh"

View Logs:
ssh -i $SSH_KEY ubuntu@$INSTANCE_IP "cd /mnt/training/mistral_training && tail -f training.log"

Monitor from S3 (no SSH needed):
python3 scripts/monitoring/monitor_training.py --run-id $RUN_ID --watch

S3 Monitoring URL:
https://console.aws.amazon.com/s3/buckets/$S3_BUCKET/$MONITORING_PREFIX/$RUN_ID/

Stop Instance (when done):
aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION

Terminate Instance (cleanup):
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION

Pipeline Log:
cat $LOG_FILE
EOF

    log "Instance information saved to: mistral_training_info.txt"
    log ""
    log "Monitor training progress without SSH:"
    log "python3 scripts/monitoring/monitor_training.py --run-id $RUN_ID --watch"
}

# Parse command line arguments
parse_args() {
    INSTANCE_ID=""
    INSTANCE_IP=""
    SSH_KEY=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --instance-id)
                INSTANCE_ID="$2"
                shift 2
                ;;
            --instance-ip)
                INSTANCE_IP="$2"
                shift 2
                ;;
            --ssh-key)
                SSH_KEY="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --instance-id ID    Use existing instance with this ID"
                echo "  --instance-ip IP    IP address of existing instance (required with --instance-id)"
                echo "  --ssh-key PATH      Path to SSH key for existing instance (required with --instance-id)"
                echo ""
                echo "Examples:"
                echo "  # Launch new instance:"
                echo "  $0"
                echo ""
                echo "  # Use existing instance:"
                echo "  $0 --instance-id i-0123456789abcdef0 --instance-ip 54.1.2.3 --ssh-key ~/.ssh/my-key.pem"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    # Validate existing instance parameters
    if [[ -n "$INSTANCE_ID" ]]; then
        if [[ -z "$INSTANCE_IP" ]] || [[ -z "$SSH_KEY" ]]; then
            error "When using --instance-id, you must also provide --instance-ip and --ssh-key"
        fi
        if [[ ! -f "$SSH_KEY" ]]; then
            error "SSH key not found: $SSH_KEY"
        fi
        USE_EXISTING_INSTANCE=true
    else
        USE_EXISTING_INSTANCE=false
    fi
}

# Main execution
main() {
    # Initialize log file
    echo "=== Mistral Training Pipeline Log ===" > "$LOG_FILE"
    echo "Run ID: $RUN_ID" >> "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "===================================" >> "$LOG_FILE"
    
    log "Starting Mistral training pipeline..."
    log "Run ID: $RUN_ID"
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install and configure AWS CLI."
    fi
    
    # Verify AWS credentials
    log "Verifying AWS credentials..."
    if ! aws sts get-caller-identity --region $REGION &>/dev/null; then
        error "AWS credentials not configured. Run 'aws configure' first."
    fi
    
    # Immediately post run metadata to S3 for monitoring
    log "Posting initial run metadata to S3..."
    cat > /tmp/initial_metadata.json << EOF_META
{
  "run_id": "$RUN_ID",
  "status": "initializing",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "phase": "setup",
  "instance_type": "$INSTANCE_TYPE",
  "region": "$REGION"
}
EOF_META
    
    # Upload immediately to S3
    aws s3 cp /tmp/initial_metadata.json s3://$S3_BUCKET/$MONITORING_PREFIX/$RUN_ID/metadata.json --region $REGION
    rm /tmp/initial_metadata.json
    log "Run metadata posted - monitoring available immediately"
    
    # Phase 1: Get instance (launch new or use existing)
    if [[ "$USE_EXISTING_INSTANCE" == "true" ]]; then
        log "Using existing instance: $INSTANCE_ID"
        log "Instance IP: $INSTANCE_IP"
        
        # Verify instance is running
        INSTANCE_STATE=$(aws ec2 describe-instances \
            --instance-ids $INSTANCE_ID \
            --region $REGION \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text 2>/dev/null || echo "error")
        
        if [[ "$INSTANCE_STATE" == "stopped" ]]; then
            log "Instance is stopped. Starting it..."
            check_command "aws ec2 start-instances --instance-ids $INSTANCE_ID --region $REGION" \
                "Failed to start instance"
            log "Waiting for instance to be running..."
            aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
            
            # Get updated IP
            INSTANCE_IP=$(aws ec2 describe-instances \
                --instance-ids $INSTANCE_ID \
                --region $REGION \
                --query 'Reservations[0].Instances[0].PublicIpAddress' \
                --output text)
            log "Instance started. New IP: $INSTANCE_IP"
        elif [[ "$INSTANCE_STATE" != "running" ]]; then
            error "Instance is in unexpected state: $INSTANCE_STATE"
        fi
        
        # Test SSH connection
        log "Testing SSH connection..."
        if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY ubuntu@$INSTANCE_IP exit 2>/dev/null; then
            error "Cannot connect to instance via SSH. Check IP and key."
        fi
        log "SSH connection successful"
    else
        launch_instance
    fi
    
    # Phase 2: Setup environment
    update_s3_status "running" "setup" "Setting up instance environment"
    setup_instance
    
    # Phase 3: Deploy scripts
    update_s3_status "running" "deploy" "Deploying training scripts"
    deploy_scripts
    
    # Phase 4: Run data preparation
    update_s3_status "running" "data_prep" "Running data preparation pipeline"
    run_data_prep
    
    # Phase 5: Start training
    update_s3_status "running" "training" "Starting model training"
    run_training
    
    # Phase 6: Save info
    save_info
    
    update_s3_status "training_started" "complete" "Pipeline started successfully, training in progress"
    log "🎉 Mistral training pipeline started successfully!"
    log "Training is running on the instance. Check mistral_training_info.txt for details."
    log ""
    if [[ "$USE_EXISTING_INSTANCE" != "true" ]]; then
        log "IMPORTANT: Remember to stop/terminate the instance when training is complete!"
    fi
}

# Parse arguments and run main
parse_args "$@"
main