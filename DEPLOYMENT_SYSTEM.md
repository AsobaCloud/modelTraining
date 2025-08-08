# Model Deployment System Documentation

> **Status**: Implementation Complete  
> **Last Updated**: 2025-08-06  
> **Validation**: All components tested and verified ✅

## Overview

Resource-aware model deployment system for 4xlarge instance with intelligent management of existing Flux service and integration with status.asoba.co monitoring.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  deploy-model.sh│    │ 4xlarge Instance │    │ status.asoba.co │
│  (Local)        │───▶│ 34.217.17.67     │◀───│ (Monitoring)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ S3 Models       │    │ FastAPI Server   │    │ Health Checks   │
│ asoba-llm-cache │    │ Port 8001/8000   │    │ /status endpoint│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Validated Environment

**4xlarge Instance (g5.4xlarge)**:
- **IP**: 34.217.17.67 (us-west-2)
- **Resources**: 62GB RAM, 16 CPUs, ~23GB GPU memory
- **Current Usage**: Flux service (54.8% RAM, port 8000)
- **Available**: 26GB free RAM for new deployments
- **SSH**: `ssh -i config/mistral-base.pem ubuntu@34.217.17.67`

**Validated Components**:
- ✅ S3 access to asoba-llm-cache bucket
- ✅ SSH connectivity and sudo access
- ✅ Resource monitoring (memory, GPU, services)
- ✅ Service management via systemd
- ✅ Python 3.12.10 + PyTorch environment
- ✅ Health endpoint format compatibility

## Available Models

### Tier 1: Production Ready (Auto-Deploy)
| Model | Size | Memory | Coexist with Flux | Bootstrap Script |
|-------|------|--------|-------------------|------------------|
| `mistral_policy_analysis` | 298MB | <1GB | ✅ Yes | `bootstrap_mistral_deployment.sh` |
| `mistral_iac_generation` | 297MB | <1GB | ✅ Yes | `bootstrap_mistral_deployment.sh` |
| `qwen3-14b` | 28GB | ~30GB | ❌ No (stops Flux) | `bootstrap_qwen_claude_deployment.sh` |

### Tier 2: Raw Models (Generic Deploy)
| Model | Size | Memory | Auto-Quantization | Coexist with Flux |
|-------|------|--------|-------------------|-------------------|
| `qwen3-14b-awq` | ~10GB | ~12GB | Pre-quantized | ❌ No |
| `qwen3-coder-30b` | ~62GB | ~16GB | ✅ 4-bit | ❌ No |
| `deepseek-r1-8b` | ~16GB | ~18GB | Optional | ❌ No |

## Resource Management Strategy

### Small Models (<3GB)
- **Action**: Deploy on port 8001 alongside Flux
- **Memory**: Uses available 26GB free RAM
- **GPU**: Shared with Flux
- **Flux Status**: Remains running

### Large Models (>3GB)
- **Action**: Stop Flux service, deploy on port 8000
- **Memory**: Uses freed ~34GB + available 26GB = 60GB total
- **GPU**: Full 23GB available
- **Flux Status**: Stopped (can be restarted manually)

### Auto-Quantization (>20GB)
- **Trigger**: Models requiring >20GB GPU memory
- **Method**: 4-bit BitsAndBytes quantization
- **Result**: ~75% memory reduction
- **Fallback**: Check for pre-quantized AWQ variants first

## Deployment Workflow

### 1. Pre-Deployment Validation
```bash
./validate-components.sh  # Verify all systems operational
```

### 2. Deployment Planning
```bash
./deploy-model.sh <model_name> --dry-run
```
**Shows**:
- Resource requirements and availability
- Whether Flux will be stopped
- Quantization decisions
- Target ports and endpoints
- Memory impact analysis

### 3. Model Deployment
```bash
./deploy-model.sh <model_name>
```
**Process**:
1. Connect to 4xlarge instance
2. Validate model exists in S3
3. Check current resource usage
4. Stop Flux if required (large models)
5. Download model from S3
6. Create FastAPI inference service
7. Install as systemd service
8. Start service and verify health
9. Set MCP discovery environment variables

### 4. Service Verification
```bash
# Check deployment status
curl http://34.217.17.67:8001/status

# View service logs
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "journalctl -u asoba-inference -f"
```

## Service Integration

### MCP Server Compatibility
The deployed service provides OpenAI-compatible API endpoints:

**Inference Endpoint**: `POST /generate`
```json
{
  "prompt": "Your prompt here",
  "max_tokens": 512,
  "temperature": 0.7
}
```

**Response Format**:
```json
{
  "choices": [{"text": "Generated response", "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
  "model": "model-name",
  "created": 1691234567,
  "id": "gen-20250806-123456"
}
```

### Health Monitoring Integration
**Status Endpoint**: `GET /status`
```json
{
  "status": "healthy",
  "model": "model-name", 
  "gpu_memory_used": 12345,
  "gpu_memory_total": 23000,
  "ram_usage_gb": 15.2,
  "port": 8001,
  "timestamp": "2025-08-06T12:34:56",
  "service": "asoba-inference"
}
```

**Simple Health**: `GET /health`
```json
{
  "status": "ok",
  "timestamp": "2025-08-06T12:34:56"
}
```

### Environment Variables (MCP Discovery)
Automatically set on deployment:
```bash
export MISTRAL_STATUS_URL=http://localhost:8001/status
export MISTRAL_FALLBACK_IP=127.0.0.1
export ASOBA_INFERENCE_PORT=8001
```

## Service Management

### systemd Service
- **Name**: `asoba-inference.service`
- **User**: `ubuntu`
- **Working Directory**: `/opt/asoba-inference`
- **Auto-restart**: Enabled with 10-second delay
- **Resource Limits**: 32GB memory, 800% CPU

### Management Commands
```bash
# Check service status
sudo systemctl status asoba-inference

# View logs
journalctl -u asoba-inference -f

# Restart service
sudo systemctl restart asoba-inference

# Stop service
sudo systemctl stop asoba-inference

# Disable service
sudo systemctl disable asoba-inference
```

### Flux Service Management
```bash
# Check Flux status
systemctl status flux-api.service

# Start Flux (if stopped by deployment)
sudo systemctl start flux-api.service

# Stop Flux (to free resources)
sudo systemctl stop flux-api.service
```

## Monitoring Integration

### status.asoba.co Integration
- **Automatic Detection**: Services appear when health endpoint responds
- **Health Checks**: Existing monitoring queries `/status` endpoint
- **Dashboard Updates**: Real-time status reflected in web interface
- **No Configuration Required**: Uses established monitoring patterns

### Expected Behavior
1. Deploy model with health endpoint
2. Service becomes available on designated port
3. Existing monitoring detects service automatically
4. Status appears in https://status.asoba.co/ dashboard
5. Uptime tracking begins immediately

## Usage Examples

### Deploy Small Model (Coexists with Flux)
```bash
# Plan deployment
./deploy-model.sh mistral_iac_generation --dry-run

# Execute deployment
./deploy-model.sh mistral_iac_generation

# Result: Service on port 8001, Flux continues on port 8000
```

### Deploy Large Model (Replaces Flux)
```bash
# Plan deployment (shows Flux will stop)
./deploy-model.sh qwen3-14b --dry-run

# Execute deployment
./deploy-model.sh qwen3-14b

# Result: Service on port 8000, Flux stopped
# Restore Flux: ssh ubuntu@34.217.17.67 "sudo systemctl start flux-api.service"
```

### Deploy with Quantization
```bash
# Large model gets auto-quantized
./deploy-model.sh qwen3-coder-30b

# Force quantization on smaller model
./deploy-model.sh qwen3-14b --force-quantize

# Use pre-quantized variant if available
./deploy-model.sh qwen3-14b-awq
```

## Troubleshooting

### Common Issues

#### SSH Connection Failed
```bash
# Check instance status
aws ec2 describe-instances --region us-west-2 --filters Name=tag:Project,Values=FluxDeploy

# Verify SSH key permissions
chmod 400 config/mistral-base.pem

# Test basic connectivity
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "echo test"
```

#### Service Won't Start
```bash
# Check service logs
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "journalctl -u asoba-inference -n 50"

# Check GPU availability
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "nvidia-smi"

# Verify model files
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "ls -la /opt/models/"
```

#### Out of Memory Errors
```bash
# Check current memory usage
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "free -h"

# Stop Flux to free memory
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "sudo systemctl stop flux-api.service"

# Redeploy with quantization
./deploy-model.sh model_name --force-quantize
```

#### Health Check Failures
```bash
# Test endpoints manually
curl http://34.217.17.67:8001/status
curl http://34.217.17.67:8001/health

# Check service binding
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "ss -tlnp | grep :8001"
```

## Development and Testing

### Validation Script
Run before any deployment:
```bash
./validate-components.sh
```

### Safe Testing
```bash
# Always dry-run first
./deploy-model.sh model_name --dry-run

# Test with small model first
./deploy-model.sh mistral_iac_generation

# Verify monitoring integration
curl http://34.217.17.67:8001/status
```

### Cleanup
```bash
# Stop service
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "sudo systemctl stop asoba-inference"

# Remove service files
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "
  sudo systemctl disable asoba-inference 2>/dev/null || true
  sudo rm -f /etc/systemd/system/asoba-inference.service
  sudo rm -rf /opt/asoba-inference
  sudo systemctl daemon-reload
"

# Restart Flux if needed
ssh -i config/mistral-base.pem ubuntu@34.217.17.67 "sudo systemctl start flux-api.service"
```

## Security Considerations

### Access Control
- SSH key-based authentication only
- sudo access required for service management
- Services run as `ubuntu` user (non-root)
- Network access via security groups (managed separately)

### Service Isolation  
- Each model deployment replaces previous inference service
- Model files stored in separate directories under `/opt/models/`
- Service logs via systemd journald
- Resource limits prevent system exhaustion

### Monitoring Security
- Health endpoints provide system metrics (memory, GPU usage)
- No authentication on health endpoints (internal network only)
- Service discovery via environment variables
- Integration with existing monitoring infrastructure

## Cost Implications

### Instance Costs
- **g5.4xlarge**: ~$1.60/hour when running
- **Recommendation**: Stop instance when not in active use
- **Auto-scaling**: Not implemented (manual start/stop)

### Model Storage
- **S3 Storage**: Models stored in asoba-llm-cache bucket
- **Local Storage**: Models cached locally after first deployment
- **Cleanup**: Manual removal of unused models

### Resource Optimization
- **Small Models**: Deploy alongside Flux (maximize utilization)
- **Large Models**: Replace Flux temporarily (single-model focus)
- **Quantization**: Automatic memory reduction for large models

## Future Enhancements

### Planned Features
- [ ] Multi-model concurrent deployment (with memory management)
- [ ] Automatic model switching API
- [ ] Model performance benchmarking
- [ ] Cost tracking and optimization
- [ ] Auto-scaling based on usage

### Integration Opportunities
- [ ] Direct integration with AsobaCode MCP server
- [ ] Automated testing pipeline
- [ ] Model performance monitoring
- [ ] Usage analytics and reporting

---

**Last Validation**: 2025-08-06 06:59:31 UTC ✅  
**All Components Verified**: S3 access, SSH connectivity, resource monitoring, service management, Python environment, health endpoints