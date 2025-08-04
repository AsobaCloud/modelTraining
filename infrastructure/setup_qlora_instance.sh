#!/bin/bash
# QLoRA Instance Setup Script
# Following CLAUDE.md: Automate environment setup and run TDD tests

set -euo pipefail

# Configuration
INSTANCE_IP="54.197.142.172"
KEY_PATH="/home/shingai/sort/deployments/mistral-base.pem"
INSTANCE_USER="ubuntu"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Verify connection
check_connection() {
    log "Checking SSH connection to g5.2xlarge instance..."
    
    if ! ssh -i "$KEY_PATH" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "$INSTANCE_USER@$INSTANCE_IP" "echo 'Connection successful'" 2>/dev/null; then
        error "Cannot connect to instance $INSTANCE_IP"
    fi
    
    log "✅ SSH connection verified"
}

# Validate GPU environment
validate_gpu_environment() {
    log "Validating GPU environment on instance..."
    
    ssh -i "$KEY_PATH" "$INSTANCE_USER@$INSTANCE_IP" << 'EOF'
        echo "=== GPU Environment Check ==="
        
        # Check NVIDIA driver
        if ! nvidia-smi >/dev/null 2>&1; then
            echo "❌ NVIDIA driver not working"
            exit 1
        fi
        
        echo "✅ NVIDIA driver working"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        
        # Check Python and basic packages
        echo -e "\n=== Python Environment ==="
        python3 --version
        
        if python3 -c "import torch" 2>/dev/null; then
            echo "✅ PyTorch available"
            python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
            python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
        else
            echo "❌ PyTorch not available"
        fi
        
        # Check disk space
        echo -e "\n=== Disk Space ==="
        df -h / | grep -v Filesystem
        
        # Check memory
        echo -e "\n=== System Memory ==="
        free -h
EOF
    
    log "✅ Environment validation complete"
}

# Transfer test file
transfer_test_file() {
    log "Transferring QLoRA test file to instance..."
    
    scp -i "$KEY_PATH" -o StrictHostKeyChecking=no \
        "test_qlora_setup.py" "$INSTANCE_USER@$INSTANCE_IP:~/" || \
        error "Failed to transfer test file"
    
    log "✅ Test file transferred"
}

# Install QLoRA dependencies
install_qlora_dependencies() {
    log "Installing QLoRA dependencies on instance..."
    
    ssh -i "$KEY_PATH" "$INSTANCE_USER@$INSTANCE_IP" << 'EOF'
        echo "=== Installing QLoRA Dependencies ==="
        
        # Update pip first
        python3 -m pip install --upgrade pip
        
        # Install QLoRA packages with specific versions
        echo "Installing core QLoRA packages..."
        pip install transformers>=4.36.0
        pip install accelerate>=0.25.0
        pip install peft>=0.8.0
        pip install bitsandbytes>=0.41.0
        pip install datasets>=2.16.0
        
        # Install additional dependencies
        echo "Installing additional packages..."
        pip install pytest psutil scipy
        
        # Verify installations
        echo -e "\n=== Verification ==="
        python3 -c "
import transformers, accelerate, peft, bitsandbytes, datasets
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ accelerate: {accelerate.__version__}')
print(f'✅ peft: {peft.__version__}')
print(f'✅ bitsandbytes: {bitsandbytes.__version__}')
print(f'✅ datasets: {datasets.__version__}')
"
        
        echo "✅ All QLoRA dependencies installed"
EOF
    
    log "✅ QLoRA dependencies installation complete"
}

# Run TDD tests
run_tdd_tests() {
    log "Running TDD tests (should show initial failures and first successes)..."
    
    ssh -i "$KEY_PATH" "$INSTANCE_USER@$INSTANCE_IP" << 'EOF'
        echo "=== Running QLoRA Setup Tests ==="
        
        # Run tests with verbose output
        python3 test_qlora_setup.py
EOF
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        warn "Tests passed - this is unexpected in TDD red phase!"
        warn "Some tests may need to be updated to ensure proper failure initially"
    else
        log "✅ Tests failed as expected (TDD red phase)"
        log "Next step: Implement features to make tests pass (TDD green phase)"
    fi
}

# Main execution
main() {
    log "🚀 Starting QLoRA Instance Setup (CLAUDE.md TDD approach)"
    log "Instance: $INSTANCE_IP"
    
    check_connection
    validate_gpu_environment
    transfer_test_file
    install_qlora_dependencies
    run_tdd_tests
    
    log "🎯 Setup phase complete!"
    log "Next: Implement QLoRA features to make tests pass"
    
    echo -e "\n${GREEN}=== Connection Command ===${NC}"
    echo "ssh -i $KEY_PATH $INSTANCE_USER@$INSTANCE_IP"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi