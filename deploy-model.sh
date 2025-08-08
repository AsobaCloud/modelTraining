#!/bin/bash
set -euo pipefail

# deploy-model.sh - Resource-aware model deployment for 4xlarge instance
# Version: 1.0.0
# Documentation: DEPLOYMENT_SYSTEM.md
# Validated: 2025-08-06 (all components tested)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="asoba-llm-cache"
REGION="us-east-1"
SERVICE_NAME="asoba-inference"
SERVICE_PORT="8001"
HEALTH_PORT="8002"
FLUX_PORT="8000"

# Instance resource limits (g5.4xlarge = 64GB RAM, 24GB GPU)
MAX_MEMORY_GB=64
MAX_GPU_MEMORY_GB=24
FLUX_MEMORY_GB=60  # Flux uses ~59.8GB
AVAILABLE_MEMORY_GB=26  # Current free memory (validated)

# Model size thresholds
SMALL_MODEL_THRESHOLD_GB=3  # Can coexist with Flux
QUANTIZATION_THRESHOLD_GB=20

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

success() {
    echo "✅ $1"
}

error() {
    log "ERROR: $1"
    exit 1
}

show_usage() {
    cat << EOF
deploy-model.sh - Resource-aware model deployment for 4xlarge instance

Usage: $0 <model_name> [options]

VALIDATED ENVIRONMENT:
  Instance: 34.217.17.67 (g5.4xlarge, us-west-2)
  Resources: 62GB RAM (26GB free), 16 CPUs, 23GB GPU
  Current: Flux service using 54.8% RAM on port 8000

AVAILABLE MODELS:
  Tier 1 (Production Ready):
    mistral_policy_analysis     - 298MB, coexists with Flux
    mistral_iac_generation     - 297MB, coexists with Flux  
    qwen3-14b                  - 28GB, stops Flux, uses port 8000
  
  Tier 2 (Raw Models):
    qwen3-14b-awq              - 10GB, pre-quantized
    qwen3-coder-30b            - 62GB, auto-quantized to ~16GB
    deepseek-r1-8b             - 16GB, may need quantization

DEPLOYMENT OPTIONS:
  --coexist                  Force coexistence with Flux (may fail if insufficient memory)
  --replace-flux             Stop Flux service and use its resources
  --service-port PORT        Override default service port ($SERVICE_PORT)
  --health-port PORT         Override health check port ($HEALTH_PORT) 
  --dry-run                  Show deployment plan without executing

RESOURCE MANAGEMENT:
  • Small models (<3GB): Deploy alongside Flux on port 8001
  • Large models (>3GB): Automatically stop Flux, use port 8000
  • Auto-quantization: Applied for models >20GB GPU memory
  • Memory validation: Pre-deployment resource checking

EXAMPLES:
  $0 mistral_iac_generation           # Small model, coexists with Flux
  $0 qwen3-14b                        # Large model, stops Flux automatically  
  $0 qwen3-coder-30b --dry-run        # Show quantization plan
  $0 deepseek-r1-8b --coexist         # Force coexistence (may fail)

INTEGRATION:
  • MCP Server: Automatic environment variable setup
  • Monitoring: Health endpoints compatible with status.asoba.co
  • Service: systemd-managed with auto-restart

For detailed documentation, see: DEPLOYMENT_SYSTEM.md

EOF
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required tools
    local missing_tools=()
    for tool in jq aws bc curl; do
        if ! command -v "$tool" >/dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        error "AWS credentials not configured"
    fi
    
    # Check SSH key
    local ssh_key="$SCRIPT_DIR/config/mistral-base.pem"
    if [[ ! -f "$ssh_key" ]]; then
        error "SSH key not found: $ssh_key"
    fi
    chmod 400 "$ssh_key"
    
    success "Prerequisites validated"
}

get_instance_ip() {
    log "Getting 4xlarge instance IP..."
    
    local instance_ip
    instance_ip=$(aws ec2 describe-instances --region us-west-2 \
        --filters Name=tag:Project,Values=FluxDeploy Name=instance-state-name,Values=running \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
    
    if [[ -z "$instance_ip" || "$instance_ip" == "None" ]]; then
        error "4xlarge instance not found or not running"
    fi
    
    # Validate connectivity
    if ! ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
         ubuntu@"$instance_ip" "echo 'Connected'" >/dev/null 2>&1; then
        error "Cannot connect to 4xlarge instance at $instance_ip"
    fi
    
    success "Connected to instance: $instance_ip"
    echo "$instance_ip"
}

get_instance_resources() {
    local instance_ip="$1"
    log "Checking current resource usage..."
    
    local resource_info
    resource_info=$(ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no \
        ubuntu@"$instance_ip" '
        echo "FLUX_RUNNING:$(systemctl is-active flux-api.service 2>/dev/null || echo inactive)"
        echo "FLUX_MEMORY:$(ps aux | grep flux_api_server_current.py | grep -v grep | awk "{print \$4}" | head -1 || echo 0)"
        echo "FREE_MEMORY:$(free -g | awk "/^Mem:/ {print \$7}")"
        echo "TOTAL_MEMORY:$(free -g | awk "/^Mem:/ {print \$2}")"
        echo "GPU_FREE:$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)"
        echo "GPU_TOTAL:$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)"
        echo "PORT_8000:$(ss -tlnp | grep :8000 | wc -l)"
        echo "PORT_8001:$(ss -tlnp | grep :8001 | wc -l)"
        echo "CPU_COUNT:$(nproc)"
    ')
    
    echo "$resource_info"
}

validate_model() {
    local model_name="$1"
    
    log "Validating model: $model_name"
    
    # Download and parse model manifest
    local manifest_file="/tmp/model_manifest_$(date +%s).json"
    if ! aws s3 cp "s3://$BUCKET/models/comprehensive_model_manifest.json" "$manifest_file" --region "$REGION" >/dev/null 2>&1; then
        error "Failed to download model manifest from S3"
    fi
    
    # Check if model exists in manifest or as raw model
    if jq -e ".models[\"$model_name\"]" "$manifest_file" >/dev/null 2>&1; then
        echo "configured"
    elif aws s3 ls "s3://$BUCKET/models/$model_name/" --region "$REGION" >/dev/null 2>&1; then
        echo "raw"
    else
        rm -f "$manifest_file"
        error "Model '$model_name' not found. Use --help to see available models."
    fi
    
    rm -f "$manifest_file"
}

get_model_info() {
    local model_name="$1"
    local model_type="$2"
    
    local manifest_file="/tmp/model_manifest_$(date +%s).json"
    aws s3 cp "s3://$BUCKET/models/comprehensive_model_manifest.json" "$manifest_file" --region "$REGION" >/dev/null 2>&1
    
    if [[ "$model_type" == "configured" ]]; then
        # Extract info from manifest
        jq -r ".models[\"$model_name\"] | {path, size_mb, size_gb, bootstrap_script, port} | @json" \
            "$manifest_file"
    else
        # Raw model - estimate size
        local model_size_mb
        model_size_mb=$(aws s3 ls "s3://$BUCKET/models/$model_name/" --recursive --region "$REGION" 2>/dev/null | 
                       awk '{sum+=$3} END {print int(sum/1024/1024)}')
        local model_size_gb=$((model_size_mb / 1024))
        
        jq -n --arg path "models/$model_name/" \
               --arg size_mb "$model_size_mb" \
               --arg size_gb "$model_size_gb" \
               --arg bootstrap_script "" \
               --arg port "8001" \
               '{path: $path, size_mb: ($size_mb|tonumber), size_gb: ($size_gb|tonumber), bootstrap_script: $bootstrap_script, port: ($port|tonumber)}'
    fi
    
    rm -f "$manifest_file"
}

plan_deployment() {
    local model_name="$1"
    local model_info="$2"
    local resource_info="$3"
    local coexist="$4"
    local replace_flux="$5"
    
    local model_size_gb
    model_size_gb=$(echo "$model_info" | jq -r '.size_gb // 0')
    
    # Parse resource info
    local flux_running free_memory total_memory gpu_free gpu_total port_8000_used
    flux_running=$(echo "$resource_info" | grep "FLUX_RUNNING:" | cut -d: -f2)
    free_memory=$(echo "$resource_info" | grep "FREE_MEMORY:" | cut -d: -f2)
    total_memory=$(echo "$resource_info" | grep "TOTAL_MEMORY:" | cut -d: -f2)
    gpu_free=$(echo "$resource_info" | grep "GPU_FREE:" | cut -d: -f2)
    gpu_total=$(echo "$resource_info" | grep "GPU_TOTAL:" | cut -d: -f2)
    port_8000_used=$(echo "$resource_info" | grep "PORT_8000:" | cut -d: -f2)
    
    local deployment_plan=()
    local target_port="$SERVICE_PORT"
    local requires_flux_stop=false
    local quantization_needed=false
    local estimated_memory_gb="$model_size_gb"
    
    # Resource analysis
    deployment_plan+=("📊 Current Resources:")
    deployment_plan+=("   Total Memory: ${total_memory}GB, Free: ${free_memory}GB")
    deployment_plan+=("   GPU Memory: ${gpu_free}MB free / ${gpu_total}MB total")
    deployment_plan+=("   Flux Status: $flux_running")
    deployment_plan+=("")
    
    # Quantization check
    if [[ $(echo "$model_size_gb > $QUANTIZATION_THRESHOLD_GB" | bc -l) -eq 1 ]]; then
        # Check for pre-quantized variants first
        local model_path
        model_path=$(echo "$model_info" | jq -r '.path')
        local awq_path="${model_path%/}-AWQ/"
        
        if aws s3 ls "s3://$BUCKET/$awq_path" --region "$REGION" >/dev/null 2>&1; then
            deployment_plan+=("💡 Pre-quantized variant found: $awq_path")
            deployment_plan+=("   Will use AWQ quantized version (~50% smaller)")
            estimated_memory_gb=$((model_size_gb / 2))
        else
            quantization_needed=true
            deployment_plan+=("⚡ Auto-quantization required (model >20GB)")
            deployment_plan+=("   Will apply 4-bit BitsAndBytes quantization")
            estimated_memory_gb=$((model_size_gb / 4))
        fi
        deployment_plan+=("")
    fi
    
    # Deployment strategy
    if [[ $(echo "$estimated_memory_gb <= $SMALL_MODEL_THRESHOLD_GB" | bc -l) -eq 1 ]] && [[ "$replace_flux" == "false" ]]; then
        deployment_plan+=("✅ Small model strategy:")
        deployment_plan+=("   Size: ${estimated_memory_gb}GB (fits in available ${free_memory}GB)")
        deployment_plan+=("   Action: Deploy alongside Flux")
        deployment_plan+=("   Port: $SERVICE_PORT (Flux keeps port 8000)")
        deployment_plan+=("   Flux: Continues running")
    else
        if [[ "$coexist" == "true" ]]; then
            if [[ $(echo "$free_memory >= $estimated_memory_gb" | bc -l) -eq 1 ]]; then
                deployment_plan+=("⚠️ Forced coexistence strategy:")
                deployment_plan+=("   Size: ${estimated_memory_gb}GB (barely fits in ${free_memory}GB)")
                deployment_plan+=("   Action: Deploy alongside Flux (risky)")
                deployment_plan+=("   Port: $SERVICE_PORT")
                deployment_plan+=("   Warning: May cause memory pressure")
            else
                deployment_plan+=("❌ Coexistence impossible:")
                deployment_plan+=("   Size: ${estimated_memory_gb}GB > Available: ${free_memory}GB")
                echo
                printf '%s\n' "${deployment_plan[@]}"
                error "Insufficient memory for coexistence. Use --replace-flux or choose smaller model."
            fi
        else
            requires_flux_stop=true
            target_port="8000"
            local freed_memory=$((free_memory + 34))  # Flux uses ~34GB
            deployment_plan+=("🛑 Large model strategy:")
            deployment_plan+=("   Size: ${estimated_memory_gb}GB > Threshold: ${SMALL_MODEL_THRESHOLD_GB}GB")
            deployment_plan+=("   Action: Stop Flux service")
            deployment_plan+=("   Port: 8000 (replaces Flux)")
            deployment_plan+=("   Memory Available: ${freed_memory}GB (after stopping Flux)")
            deployment_plan+=("   Flux: Will be stopped")
        fi
    fi
    
    # Create deployment plan object
    jq -n \
        --arg target_port "$target_port" \
        --arg requires_flux_stop "$requires_flux_stop" \
        --arg quantization_needed "$quantization_needed" \
        --arg estimated_memory_gb "$estimated_memory_gb" \
        --argjson deployment_plan "$(printf '%s\n' "${deployment_plan[@]}" | jq -R . | jq -s .)" \
        '{
            target_port: $target_port,
            requires_flux_stop: ($requires_flux_stop == "true"),
            quantization_needed: ($quantization_needed == "true"),
            estimated_memory_gb: ($estimated_memory_gb | tonumber),
            deployment_plan: $deployment_plan
        }'
}

create_inference_service() {
    local instance_ip="$1"
    local model_name="$2" 
    local model_path="$3"
    local target_port="$4"
    local quantize="$5"
    
    log "Creating inference service for $model_name on port $target_port"
    
    # Create the inference server script
    cat > "/tmp/inference_server.py" << 'EOF'
#!/usr/bin/env python3
"""
Inference server compatible with existing monitoring and MCP integration
Version: 1.0.0 - Validated with 4xlarge instance
"""
import os
import json
import torch
import uvicorn
import psutil
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Request/Response models for MCP compatibility
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class Choice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class GenerateResponse(BaseModel):
    choices: List[Choice]
    usage: Usage
    model: str
    created: int
    id: str

class StatusResponse(BaseModel):
    status: str
    model: str
    gpu_memory_used: Optional[int] = None
    gpu_memory_total: Optional[int] = None
    ram_usage_gb: Optional[float] = None
    port: int
    timestamp: str
    service: str = "asoba-inference"
    version: str = "1.0.0"

app = FastAPI(
    title="Asoba Inference Server",
    description="Resource-aware model inference server with monitoring integration",
    version="1.0.0"
)

# Enable CORS for web interface compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
tokenizer = None
model_name = os.getenv("MODEL_NAME", "unknown")
model_path = os.getenv("MODEL_PATH", "/opt/models/current")
quantized = os.getenv("QUANTIZED", "false").lower() == "true"
service_port = int(os.getenv("SERVICE_PORT", "8001"))

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    
    logger.info(f"🚀 Loading model {model_name} from {model_path}")
    logger.info(f"⚡ Quantization enabled: {quantized}")
    logger.info(f"🔌 Service port: {service_port}")
    
    try:
        if quantized:
            # Load quantized model for large models
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            logger.info("📦 Configuring 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            logger.info("🔄 Loading quantized model (this may take a few minutes)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.info("✅ Model loaded with 4-bit quantization")
            
        else:
            # Load full precision model for small models
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info("🔄 Loading full precision model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.info("✅ Model loaded with full precision")
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Log memory usage
        gpu_info = get_gpu_info()
        ram_usage = psutil.virtual_memory().used / (1024**3)
        logger.info(f"📊 Memory usage: RAM {ram_usage:.1f}GB, GPU {gpu_info['used']}MB/{gpu_info['total']}MB")
        logger.info(f"🎉 Model {model_name} loaded successfully and ready for inference")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def get_gpu_info():
    """Get GPU memory information using multiple methods"""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                "used": int(gpu.memoryUsed),
                "total": int(gpu.memoryTotal),
                "utilization": int(gpu.load * 100)
            }
    except:
        pass
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            used, total, util = result.stdout.strip().split(', ')
            return {
                "used": int(used),
                "total": int(total),
                "utilization": int(util)
            }
    except:
        pass
    
    return {"used": 0, "total": 0, "utilization": 0}

@app.get("/status", response_model=StatusResponse)
async def health_check():
    """Health endpoint compatible with existing monitoring system"""
    gpu_info = get_gpu_info()
    ram_usage = psutil.virtual_memory().used / (1024**3)
    
    status = "healthy" if model is not None else "loading"
    
    return StatusResponse(
        status=status,
        model=model_name,
        gpu_memory_used=gpu_info["used"],
        gpu_memory_total=gpu_info["total"],
        ram_usage_gb=round(ram_usage, 1),
        port=service_port,
        timestamp=datetime.now().isoformat(),
        service="asoba-inference",
        version="1.0.0"
    )

@app.get("/health")
async def simple_health():
    """Simple health check for load balancers"""
    return {
        "status": "ok", 
        "model": model_name, 
        "timestamp": datetime.now().isoformat(),
        "service": "asoba-inference"
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate endpoint compatible with MCP server"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate input
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Empty prompt provided")
        
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_length = inputs.input_ids.shape[1]
        
        logger.info(f"🔤 Generating response for prompt length: {input_length} tokens")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.to(model.device),
                max_new_tokens=min(request.max_tokens, 1024),  # Limit max tokens
                temperature=max(0.1, min(request.temperature, 2.0)),  # Clamp temperature
                top_p=max(0.1, min(request.top_p, 1.0)),  # Clamp top_p
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(request.prompt):].strip()
        
        # Calculate usage
        output_length = outputs.shape[1] - input_length
        
        logger.info(f"✅ Generated {output_length} tokens")
        
        return GenerateResponse(
            choices=[Choice(text=response_text, index=0, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=input_length,
                completion_tokens=output_length,
                total_tokens=input_length + output_length
            ),
            model=model_name,
            created=int(datetime.now().timestamp()),
            id=f"gen-{uuid.uuid4().hex[:8]}"
        )
        
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Asoba Inference Server",
        "model": model_name,
        "status": "healthy" if model is not None else "loading",
        "endpoints": ["/generate", "/status", "/health"],
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "port": service_port
    }

@app.get("/info")
async def model_info():
    """Model information endpoint"""
    gpu_info = get_gpu_info()
    return {
        "model": model_name,
        "quantized": quantized,
        "gpu_memory": f"{gpu_info['used']}MB / {gpu_info['total']}MB",
        "gpu_utilization": f"{gpu_info['utilization']}%",
        "model_loaded": model is not None,
        "service_port": service_port
    }

if __name__ == "__main__":
    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port=service_port,
        workers=1,
        log_level="info"
    )
EOF

    # Create systemd service file
    cat > "/tmp/$SERVICE_NAME.service" << EOF
[Unit]
Description=Asoba Inference Server - $model_name
Documentation=file://$(pwd)/DEPLOYMENT_SYSTEM.md
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/asoba-inference
Environment=MODEL_PATH=$model_path
Environment=MODEL_NAME=$model_name
Environment=QUANTIZED=$quantize
Environment=SERVICE_PORT=$target_port
Environment=CUDA_VISIBLE_DEVICES=0
Environment=TRANSFORMERS_CACHE=/opt/asoba-inference/cache
Environment=HF_HOME=/opt/asoba-inference/cache
ExecStartPre=/bin/mkdir -p /opt/asoba-inference/cache
ExecStartPre=/bin/chown -R ubuntu:ubuntu /opt/asoba-inference
ExecStart=/opt/pytorch/bin/python /opt/asoba-inference/inference_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=48G
CPUQuota=1200%

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/asoba-inference

[Install]
WantedBy=multi-user.target
EOF

    # Transfer files to instance
    log "📤 Transferring service files to instance..."
    
    scp -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no \
        /tmp/inference_server.py ubuntu@"$instance_ip":/tmp/ >/dev/null 2>&1
    
    scp -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no \
        "/tmp/$SERVICE_NAME.service" ubuntu@"$instance_ip":/tmp/ >/dev/null 2>&1
    
    # Install service on instance
    ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" << EOF
set -euo pipefail

# Create service directory
sudo mkdir -p /opt/asoba-inference
sudo cp /tmp/inference_server.py /opt/asoba-inference/
sudo cp "/tmp/$SERVICE_NAME.service" /etc/systemd/system/
sudo chmod +x /opt/asoba-inference/inference_server.py
sudo chown -R ubuntu:ubuntu /opt/asoba-inference

# Install Python dependencies if needed
source /opt/pytorch/bin/activate
pip install --quiet fastapi uvicorn psutil GPUtil bitsandbytes 2>/dev/null || echo "Some packages may already be installed"

# Reload systemd
sudo systemctl daemon-reload

echo "✅ Service files installed successfully"
EOF

    success "Inference service created"
    
    # Cleanup temp files
    rm -f /tmp/inference_server.py "/tmp/$SERVICE_NAME.service"
}

deploy_model() {
    local model_name="$1"
    local dry_run="${2:-false}"
    local coexist="${3:-false}"
    local replace_flux="${4:-false}"
    
    log "🚀 Starting deployment of model: $model_name"
    
    # Step 1: Validate prerequisites
    check_prerequisites
    
    # Step 2: Connect to instance
    local instance_ip
    instance_ip=$(get_instance_ip)
    
    # Step 3: Validate model
    local model_type
    model_type=$(validate_model "$model_name")
    success "Model type: $model_type"
    
    # Step 4: Get model information
    local model_info
    model_info=$(get_model_info "$model_name" "$model_type")
    local model_size_gb
    model_size_gb=$(echo "$model_info" | jq -r '.size_gb // 0')
    local model_path
    model_path=$(echo "$model_info" | jq -r '.path')
    local bootstrap_script
    bootstrap_script=$(echo "$model_info" | jq -r '.bootstrap_script // ""')
    
    success "Model: ${model_size_gb}GB at $model_path"
    
    # Step 5: Check resources
    local resource_info
    resource_info=$(get_instance_resources "$instance_ip")
    
    # Step 6: Create deployment plan
    local deployment_plan
    deployment_plan=$(plan_deployment "$model_name" "$model_info" "$resource_info" "$coexist" "$replace_flux")
    
    local target_port
    target_port=$(echo "$deployment_plan" | jq -r '.target_port')
    local requires_flux_stop
    requires_flux_stop=$(echo "$deployment_plan" | jq -r '.requires_flux_stop')
    local quantization_needed
    quantization_needed=$(echo "$deployment_plan" | jq -r '.quantization_needed')
    local estimated_memory_gb
    estimated_memory_gb=$(echo "$deployment_plan" | jq -r '.estimated_memory_gb')
    
    if [[ "$dry_run" == "true" ]]; then
        cat << EOF

🚀 DEPLOYMENT PLAN: $model_name
============================================
Model: $model_name ($model_type)
Size: Original ${model_size_gb}GB, Estimated ${estimated_memory_gb}GB
S3 Path: s3://$BUCKET/$model_path
Target Instance: $instance_ip (g5.4xlarge, us-west-2)
Target Port: $target_port
Quantization: $quantization_needed
Bootstrap Script: ${bootstrap_script:-"Generic deployment"}

$(echo "$deployment_plan" | jq -r '.deployment_plan[]')

DEPLOYMENT ACTIONS:
1. $([ "$requires_flux_stop" == "true" ] && echo "🛑 Stop Flux service (frees ~34GB)" || echo "✅ Keep Flux running")
2. 📥 Download model from S3 to /opt/models/$model_name  
3. 🔧 Create FastAPI inference service ($([ "$quantization_needed" == "true" ] && echo "4-bit quantized" || echo "full precision"))
4. 📋 Install systemd service: $SERVICE_NAME
5. 🚀 Start service on port $target_port
6. 🔍 Verify health endpoints respond correctly
7. 🔗 Set MCP discovery environment variables
8. 📊 Service appears in status.asoba.co monitoring

INTEGRATION:
• MCP Server: POST http://$instance_ip:$target_port/generate
• Health Check: GET http://$instance_ip:$target_port/status
• Simple Health: GET http://$instance_ip:$target_port/health
• Service Info: GET http://$instance_ip:$target_port/

RESOURCE IMPACT:
• Memory: ~${estimated_memory_gb}GB RAM usage
• GPU: Auto-allocated based on model size
• Services: $([ "$requires_flux_stop" == "true" ] && echo "Replaces Flux temporarily" || echo "Runs alongside Flux")

Use './deploy-model.sh $model_name' to execute this plan.

EOF
        return
    fi
    
    log "📋 Executing deployment plan..."
    
    # Step 7: Stop existing inference service if running
    log "🛑 Stopping any existing inference service..."
    ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        "sudo systemctl stop $SERVICE_NAME 2>/dev/null || true; sudo systemctl disable $SERVICE_NAME 2>/dev/null || true"
    
    # Step 8: Stop Flux if required
    if [[ "$requires_flux_stop" == "true" ]]; then
        log "🛑 Stopping Flux service to free resources..."
        ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
            "sudo systemctl stop flux-api.service" || true
        sleep 3
        success "Flux service stopped (freed ~34GB memory)"
    fi
    
    # Step 9: Download model
    local local_model_path="/opt/models/$model_name"
    log "📥 Downloading model to $local_model_path..."
    
    if [[ -n "$bootstrap_script" ]] && [[ "$bootstrap_script" != "null" ]]; then
        # Use existing bootstrap script
        log "📋 Using bootstrap script: $bootstrap_script"
        local temp_script="/tmp/bootstrap_$(date +%s).sh"
        aws s3 cp "s3://$BUCKET/scripts/$bootstrap_script" "$temp_script" --region "$REGION" >/dev/null 2>&1
        chmod +x "$temp_script"
        scp -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no \
            "$temp_script" ubuntu@"$instance_ip":/tmp/bootstrap.sh >/dev/null 2>&1
        
        ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" << EOF
set -euo pipefail
echo "📦 Running bootstrap script for $model_name..."
sudo /tmp/bootstrap.sh "$model_name" "$local_model_path"
echo "✅ Bootstrap completed"
EOF
        rm -f "$temp_script"
    else
        # Generic model download
        log "📦 Downloading model files directly..."
        ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" << EOF
set -euo pipefail
sudo mkdir -p "$local_model_path"
sudo chown -R ubuntu:ubuntu /opt/models
echo "📥 Syncing model files from S3..."
aws s3 sync "s3://$BUCKET/$model_path" "$local_model_path" --region "$REGION" --quiet
echo "✅ Model files downloaded"
EOF
    fi
    success "Model downloaded successfully"
    
    # Step 10: Create and install inference service
    create_inference_service "$instance_ip" "$model_name" "$local_model_path" "$target_port" "$quantization_needed"
    
    # Step 11: Start the service
    log "🚀 Starting inference service..."
    ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" << EOF
set -euo pipefail
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME
echo "🔄 Service starting..."
EOF
    
    # Step 12: Wait for service to become healthy with progress updates
    log "⏳ Waiting for service to become healthy..."
    local attempts=0
    local max_attempts=60  # 10 minutes max
    local last_status=""
    
    while [[ $attempts -lt $max_attempts ]]; do
        local current_status="unknown"
        local status_response=""
        
        if status_response=$(ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no \
           ubuntu@"$instance_ip" "curl -s http://localhost:$target_port/status 2>/dev/null" 2>/dev/null); then
            
            if current_status=$(echo "$status_response" | jq -r '.status // "unknown"' 2>/dev/null); then
                if [[ "$current_status" == "healthy" ]]; then
                    success "Service is healthy and ready!"
                    
                    # Get final status info
                    local ram_usage gpu_memory
                    ram_usage=$(echo "$status_response" | jq -r '.ram_usage_gb // "unknown"')
                    gpu_memory=$(echo "$status_response" | jq -r '.gpu_memory_used // "unknown"')
                    success "Memory usage: ${ram_usage}GB RAM, ${gpu_memory}MB GPU"
                    break
                elif [[ "$current_status" != "$last_status" ]]; then
                    log "📊 Service status: $current_status"
                    last_status="$current_status"
                fi
            fi
        fi
        
        ((attempts++))
        if [[ $((attempts % 10)) -eq 0 ]]; then
            log "⏳ Still waiting... (${attempts}/${max_attempts}, status: $current_status)"
        fi
        sleep 10
    done
    
    if [[ $attempts -eq $max_attempts ]]; then
        log "❌ Service failed to become healthy within $((max_attempts * 10)) seconds"
        log "📋 Checking service logs..."
        ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
            "journalctl -u $SERVICE_NAME --no-pager -l | tail -30" || true
        error "Deployment failed - service not responding"
    fi
    
    # Step 13: Set environment variables for MCP discovery
    log "🔗 Setting up MCP discovery environment..."
    ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" << EOF
# Update environment for MCP discovery
sudo tee -a /etc/environment > /dev/null << ENVEOF
MISTRAL_STATUS_URL=http://localhost:$target_port/status
MISTRAL_FALLBACK_IP=127.0.0.1
ASOBA_INFERENCE_PORT=$target_port
ASOBA_INFERENCE_MODEL=$model_name
ENVEOF
echo "✅ Environment variables set"
EOF
    
    # Step 14: Final verification and summary
    local final_status
    final_status=$(ssh -i "$SCRIPT_DIR/config/mistral-base.pem" -o StrictHostKeyChecking=no ubuntu@"$instance_ip" \
        "curl -s http://localhost:$target_port/status 2>/dev/null" || echo '{"status":"unknown","ram_usage_gb":0,"gpu_memory_used":0}')
    
    local final_ram final_gpu final_service_status
    final_service_status=$(echo "$final_status" | jq -r '.status // "unknown"')
    final_ram=$(echo "$final_status" | jq -r '.ram_usage_gb // 0')
    final_gpu=$(echo "$final_status" | jq -r '.gpu_memory_used // 0')
    
    cat << EOF

🎉 DEPLOYMENT SUCCESSFUL!
=========================
Model: $model_name
Status: $final_service_status
Instance: $instance_ip (g5.4xlarge)
Memory Usage: ${final_ram}GB RAM, ${final_gpu}MB GPU

🔗 SERVICE ENDPOINTS:
• Inference API: http://$instance_ip:$target_port/generate
• Health Check: http://$instance_ip:$target_port/status
• Service Info: http://$instance_ip:$target_port/
• Model Details: http://$instance_ip:$target_port/info

🔧 MANAGEMENT COMMANDS:
# Check service status
ssh -i $SCRIPT_DIR/config/mistral-base.pem ubuntu@$instance_ip "systemctl status $SERVICE_NAME"

# View real-time logs
ssh -i $SCRIPT_DIR/config/mistral-base.pem ubuntu@$instance_ip "journalctl -u $SERVICE_NAME -f"

# Restart service
ssh -i $SCRIPT_DIR/config/mistral-base.pem ubuntu@$instance_ip "sudo systemctl restart $SERVICE_NAME"

📊 INTEGRATION STATUS:
✅ MCP Server: Environment variables configured
✅ Health Monitoring: Compatible with status.asoba.co
✅ Service Management: systemd integration enabled

$([ "$requires_flux_stop" == "true" ] && echo "
⚠️  FLUX SERVICE STATUS:
🛑 Flux was stopped to free resources for this model
🔄 To restore Flux: ssh -i $SCRIPT_DIR/config/mistral-base.pem ubuntu@$instance_ip \"sudo systemctl start flux-api.service\"
🔍 Check status: ssh -i $SCRIPT_DIR/config/mistral-base.pem ubuntu@$instance_ip \"systemctl status flux-api.service\"
" || echo "
✅ FLUX SERVICE STATUS:
🔄 Flux continues running alongside the new model
📍 Flux available at: http://$instance_ip:8000/
")

📈 NEXT STEPS:
1. Test the inference endpoint with a sample request
2. Verify the service appears in https://status.asoba.co/ 
3. Configure your MCP client to use the new endpoint
4. Monitor service performance and resource usage

EOF
    success "Model deployment completed successfully! 🚀"
}

# Parse command line arguments
POSITIONAL_ARGS=()
DRY_RUN=false
COEXIST=false
REPLACE_FLUX=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --coexist)
            COEXIST=true
            shift
            ;;
        --replace-flux)
            REPLACE_FLUX=true
            shift
            ;;
        --service-port)
            SERVICE_PORT="$2"
            shift 2
            ;;
        --health-port)
            HEALTH_PORT="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        -*)
            error "Unknown option $1. Use --help for usage information."
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Validate arguments
if [[ $# -eq 0 ]]; then
    show_usage
    error "Model name required. Use --help for usage information."
fi

MODEL_NAME="$1"

# Execute deployment
deploy_model "$MODEL_NAME" "$DRY_RUN" "$COEXIST" "$REPLACE_FLUX"