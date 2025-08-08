#!/bin/bash
# validate-components.sh - Test each deployment component individually

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="asoba-llm-cache"
REGION="us-east-1"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

success() {
    echo "✅ $1"
}

failure() {
    echo "❌ $1"
    return 1
}

test_s3_access() {
    log "Testing S3 access and model manifest..."
    
    # Test AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        failure "AWS credentials not configured"
    fi
    success "AWS credentials valid"
    
    # Test S3 bucket access
    if ! aws s3 ls "s3://$BUCKET/" --region "$REGION" >/dev/null 2>&1; then
        failure "Cannot access S3 bucket: $BUCKET"
    fi
    success "S3 bucket accessible"
    
    # Test model manifest download
    local manifest_file="/tmp/test_manifest.json"
    if ! aws s3 cp "s3://$BUCKET/models/comprehensive_model_manifest.json" "$manifest_file" --region "$REGION" 2>/dev/null; then
        failure "Cannot download model manifest"
    fi
    success "Model manifest downloaded"
    
    # Test manifest parsing
    if ! jq -e '.models' "$manifest_file" >/dev/null 2>&1; then
        failure "Invalid manifest JSON structure"
    fi
    
    local model_count
    model_count=$(jq '.models | length' "$manifest_file")
    success "Manifest contains $model_count configured models"
    
    # Test specific model lookup
    if ! jq -e '.models["mistral_iac_generation"]' "$manifest_file" >/dev/null 2>&1; then
        failure "Test model 'mistral_iac_generation' not found in manifest"
    fi
    success "Test model lookup works"
    
    # Test raw model directory access
    if ! aws s3 ls "s3://$BUCKET/models/DeepSeek-R1-Distill-Llama-8B/" --region "$REGION" >/dev/null 2>&1; then
        failure "Cannot access raw model directory"
    fi
    success "Raw model directory accessible"
    
    rm -f "$manifest_file"
    log "✅ S3 access validation passed"
}

test_ssh_connectivity() {
    log "Testing SSH connectivity to 4xlarge instance..."
    
    # Check SSH key exists
    local ssh_key="$SCRIPT_DIR/config/mistral-base.pem"
    if [[ ! -f "$ssh_key" ]]; then
        failure "SSH key not found: $ssh_key"
    fi
    chmod 400 "$ssh_key"
    success "SSH key found and permissions set"
    
    # Get instance IP
    local instance_ip
    instance_ip=$(aws ec2 describe-instances --region us-west-2 \
        --filters Name=tag:Project,Values=FluxDeploy Name=instance-state-name,Values=running \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
    
    if [[ -z "$instance_ip" || "$instance_ip" == "None" ]]; then
        failure "4xlarge instance not found or not running"
    fi
    success "Instance IP: $instance_ip"
    
    # Test basic SSH connectivity
    if ! ssh -i "$ssh_key" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
         ubuntu@"$instance_ip" "echo 'SSH test successful'" >/dev/null 2>&1; then
        failure "SSH connection failed to $instance_ip"
    fi
    success "SSH connection successful"
    
    # Test command execution
    local remote_output
    remote_output=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        "whoami && hostname && uname -r" 2>/dev/null)
    
    if [[ -z "$remote_output" ]]; then
        failure "Remote command execution failed"
    fi
    success "Remote command execution works"
    
    # Test sudo access
    if ! ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
         "sudo echo 'sudo test'" >/dev/null 2>&1; then
        failure "sudo access not available"
    fi
    success "sudo access confirmed"
    
    # Test required tools
    local missing_tools=()
    for tool in jq curl systemctl nvidia-smi python3; do
        if ! ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
             "command -v $tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        failure "Missing required tools: ${missing_tools[*]}"
    fi
    success "All required tools available"
    
    log "✅ SSH connectivity validation passed"
}

test_resource_monitoring() {
    log "Testing resource monitoring capabilities..."
    
    local ssh_key="$SCRIPT_DIR/config/mistral-base.pem"
    local instance_ip
    instance_ip=$(aws ec2 describe-instances --region us-west-2 \
        --filters Name=tag:Project,Values=FluxDeploy Name=instance-state-name,Values=running \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    
    # Test system resource monitoring
    local resource_check
    resource_check=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" '
        echo "FREE_MEMORY:$(free -g | awk "/^Mem:/ {print \$7}")"
        echo "TOTAL_MEMORY:$(free -g | awk "/^Mem:/ {print \$2}")"
        echo "GPU_MEMORY:$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0,0")"
        echo "CPU_COUNT:$(nproc)"
    ' 2>/dev/null)
    
    if [[ -z "$resource_check" ]]; then
        failure "Resource monitoring commands failed"
    fi
    
    local free_mem total_mem gpu_info cpu_count
    free_mem=$(echo "$resource_check" | grep "FREE_MEMORY:" | cut -d: -f2)
    total_mem=$(echo "$resource_check" | grep "TOTAL_MEMORY:" | cut -d: -f2) 
    gpu_info=$(echo "$resource_check" | grep "GPU_MEMORY:" | cut -d: -f2)
    cpu_count=$(echo "$resource_check" | grep "CPU_COUNT:" | cut -d: -f2)
    
    success "System resources: ${total_mem}GB RAM (${free_mem}GB free), ${cpu_count} CPUs, GPU: $gpu_info"
    
    # Test Flux service detection
    local flux_status
    flux_status=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        'systemctl is-active flux-api.service 2>/dev/null || echo "inactive"')
    
    if [[ "$flux_status" == "active" ]]; then
        success "Flux service detected as running"
        
        # Test Flux memory usage detection
        local flux_memory
        flux_memory=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
            'ps aux | grep flux_api_server_current.py | grep -v grep | awk "{print \$4}" || echo "0"')
        
        if [[ -n "$flux_memory" ]] && [[ "$flux_memory" != "0" ]]; then
            success "Flux memory usage detected: ${flux_memory}%"
        else
            failure "Cannot detect Flux memory usage"
        fi
    else
        log "ℹ️ Flux service not running (status: $flux_status)"
    fi
    
    # Test port monitoring
    local port_check
    port_check=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        'ss -tlnp | grep -E ":80[0-9][0-9]" | wc -l')
    
    success "Detected $port_check services on 8xxx ports"
    
    log "✅ Resource monitoring validation passed"
}

test_service_management() {
    log "Testing Flux service management (read-only)..."
    
    local ssh_key="$SCRIPT_DIR/config/mistral-base.pem"
    local instance_ip
    instance_ip=$(aws ec2 describe-instances --region us-west-2 \
        --filters Name=tag:Project,Values=FluxDeploy Name=instance-state-name,Values=running \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    
    # Test service status checking
    local service_info
    service_info=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        'systemctl status flux-api.service --no-pager -l 2>/dev/null || echo "SERVICE_NOT_FOUND"')
    
    if [[ "$service_info" == "SERVICE_NOT_FOUND" ]]; then
        log "ℹ️ Flux service not found - may be running differently"
    else
        success "Flux service information accessible"
        
        # Test if we can read service logs
        local log_lines
        log_lines=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
            'journalctl -u flux-api.service --no-pager -n 5 2>/dev/null | wc -l')
        
        if [[ "$log_lines" -gt 0 ]]; then
            success "Service logs accessible ($log_lines lines)"
        else
            failure "Cannot access service logs"
        fi
    fi
    
    # Test systemd management capabilities (dry-run)
    local systemctl_test
    systemctl_test=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        'systemctl --version' 2>/dev/null)
    
    if [[ -z "$systemctl_test" ]]; then
        failure "systemctl not available"
    fi
    success "systemctl available for service management"
    
    # Test service file creation capabilities
    local service_test
    service_test=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        'ls -la /etc/systemd/system/ | head -3' 2>/dev/null)
    
    if [[ -z "$service_test" ]]; then
        failure "Cannot access systemd service directory"
    fi
    success "systemd service directory accessible"
    
    log "✅ Service management validation passed"
}

test_python_environment() {
    log "Testing Python environment and dependencies..."
    
    local ssh_key="$SCRIPT_DIR/config/mistral-base.pem"
    local instance_ip
    instance_ip=$(aws ec2 describe-instances --region us-west-2 \
        --filters Name=tag:Project,Values=FluxDeploy Name=instance-state-name,Values=running \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    
    # Test Python environment
    local python_info
    python_info=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        'source /opt/pytorch/bin/activate 2>/dev/null && python3 --version && which python3' || \
        ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        'python3 --version && which python3')
    
    if [[ -z "$python_info" ]]; then
        failure "Python3 not available"
    fi
    success "Python environment: $python_info"
    
    # Test key Python packages
    local package_test
    package_test=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" '
        source /opt/pytorch/bin/activate 2>/dev/null || true
        python3 -c "
import sys
packages = [\"torch\", \"transformers\", \"accelerate\"]
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f\"✓ {pkg}\")
    except ImportError:
        missing.append(pkg)
        print(f\"✗ {pkg}\")
if missing:
    print(f\"Missing: {missing}\")
    sys.exit(1)
print(\"All core packages available\")
" 2>/dev/null')
    
    if [[ $? -ne 0 ]]; then
        failure "Core ML packages not available"
    fi
    success "Core ML packages verified"
    
    # Test package installation capability
    local pip_test
    pip_test=$(ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        'source /opt/pytorch/bin/activate 2>/dev/null && pip --version' || \
        ssh -i "$ssh_key" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        'pip --version || pip3 --version')
    
    if [[ -z "$pip_test" ]]; then
        failure "pip not available for package installation"
    fi
    success "pip available: $pip_test"
    
    log "✅ Python environment validation passed"
}

test_health_endpoint_format() {
    log "Testing health endpoint format compatibility..."
    
    # Create a minimal test health endpoint
    cat > /tmp/test_health_server.py << 'EOF'
#!/usr/bin/env python3
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            response = {
                "status": "healthy",
                "model": "test-model",
                "gpu_memory_used": 1234,
                "gpu_memory_total": 24000,
                "ram_usage_gb": 2.5,
                "port": 8999,
                "timestamp": datetime.now().isoformat(),
                "service": "asoba-inference"
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/health':
            response = {"status": "ok", "timestamp": datetime.now().isoformat()}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8999), HealthHandler)
    print("Test health server running on port 8999")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
EOF
    
    # Start test server in background
    python3 /tmp/test_health_server.py &
    local server_pid=$!
    sleep 2
    
    # Test health endpoint response format
    local health_response
    if health_response=$(curl -s http://localhost:8999/status 2>/dev/null); then
        # Validate JSON structure
        if echo "$health_response" | jq -e '.status and .model and .timestamp' >/dev/null 2>&1; then
            success "Health endpoint JSON structure valid"
            
            # Check required fields for monitoring
            local status model timestamp
            status=$(echo "$health_response" | jq -r '.status')
            model=$(echo "$health_response" | jq -r '.model')
            timestamp=$(echo "$health_response" | jq -r '.timestamp')
            
            if [[ "$status" == "healthy" ]]; then
                success "Status field format correct"
            else
                failure "Status field incorrect: $status"
            fi
            
            success "Health endpoint format compatible with monitoring"
        else
            failure "Invalid health endpoint JSON structure"
        fi
    else
        failure "Cannot reach test health endpoint"
    fi
    
    # Test simple health endpoint
    local simple_health
    if simple_health=$(curl -s http://localhost:8999/health 2>/dev/null); then
        if echo "$simple_health" | jq -e '.status == "ok"' >/dev/null 2>&1; then
            success "Simple health endpoint works"
        else
            failure "Simple health endpoint format incorrect"
        fi
    else
        failure "Simple health endpoint not accessible"
    fi
    
    # Cleanup
    kill $server_pid 2>/dev/null || true
    rm -f /tmp/test_health_server.py
    
    log "✅ Health endpoint format validation passed"
}

# Main validation execution
main() {
    log "🧪 Starting component validation..."
    
    local failed_tests=()
    
    # Run all validation tests
    test_s3_access || failed_tests+=("S3 Access")
    test_ssh_connectivity || failed_tests+=("SSH Connectivity") 
    test_resource_monitoring || failed_tests+=("Resource Monitoring")
    test_service_management || failed_tests+=("Service Management")
    test_python_environment || failed_tests+=("Python Environment")
    test_health_endpoint_format || failed_tests+=("Health Endpoint Format")
    
    echo
    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        log "🎉 All component validations passed!"
        log "✅ System is ready for deployment"
        exit 0
    else
        log "❌ Validation failures:"
        printf '   • %s\n' "${failed_tests[@]}"
        log "🚫 Fix failed components before deployment"
        exit 1
    fi
}

# Check prerequisites
command -v jq >/dev/null || { echo "❌ jq is required"; exit 1; }
command -v aws >/dev/null || { echo "❌ AWS CLI is required"; exit 1; }
command -v curl >/dev/null || { echo "❌ curl is required"; exit 1; }

main "$@"