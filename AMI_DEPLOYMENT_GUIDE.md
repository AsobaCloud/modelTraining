# 🚀 AMI-Based AI Model Deployment Guide

## Quick Start Options

### Option 1: Unified Mistral Deployment (Recommended)
Launch instance with automatic IaC + Policy Analysis models:

```bash
aws ec2 run-instances \
  --image-id ami-0a39335458731538a \
  --instance-type g5.2xlarge \
  --key-name your-key \
  --security-group-ids sg-your-sg \
  --user-data file://scripts/mistral_ami_user_data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=Mistral-Unified-Auto},{Key=Project,Value=FluxDeploy}]'
```

**Result**: Dual-model AI deployment in ~10 minutes (60% cost savings)

### Option 2: Qwen CLAUDE.md Deployment
Launch instance for advanced reasoning with CLAUDE.md methodology:

```bash
# Download and run Qwen deployment script
aws s3 cp s3://asoba-llm-cache/scripts/bootstrap_qwen_claude_deployment.sh .
chmod +x bootstrap_qwen_claude_deployment.sh

# Works on various instance types (script defaults to g5.4xlarge but configurable)
./bootstrap_qwen_claude_deployment.sh --instance-type g5.2xlarge  # 24GB GPU - sufficient
# ./bootstrap_qwen_claude_deployment.sh --instance-type g5.4xlarge  # 48GB GPU - plenty of headroom
```

**Result**: Qwen 14B model with integrated CLAUDE.md methodology in ~15 minutes

---

## 📋 Deployment Architecture

### Base AMI Requirements
- **Mistral AMI**: `ami-0a39335458731538a` (Mistral-7B base model pre-installed)
- **Flux AMI**: `ami-0d23439fbd78468a2` (Alternative base, Flux dev model)
- **Compatible with**: Both Mistral specialized models and Qwen deployments

### Automated Bootstrap Process
1. **Model Download**: Specialized LoRA adapters from S3
2. **Service Configuration**: Systemd services for both models
3. **Security Setup**: Firewall rules for ports 8000, 8001
4. **Health Validation**: Automated service verification

---

## 🛠️ Manual Deployment Steps

If you need to run bootstrap manually on an existing instance:

```bash
# SSH to your Mistral AMI instance
ssh -i config/mistral-base.pem ubuntu@YOUR_INSTANCE_IP

# Download and run bootstrap
aws s3 cp s3://asoba-llm-cache/scripts/bootstrap_mistral_deployment.sh .
chmod +x bootstrap_mistral_deployment.sh
sudo ./bootstrap_mistral_deployment.sh
```

---

## 📊 Resource Requirements

### Instance Types
**Mistral Models:**
- **Minimum**: g4dn.xlarge (1 GPU, 16GB GPU memory) - Single model only
- **Recommended**: g5.2xlarge (1 GPU, 24GB GPU memory) - Unified deployment
- **High Load**: g5.4xlarge+ for heavy concurrent usage

**Qwen Model:**
- **Works on**: g5.2xlarge (24GB GPU memory) - Sufficient for Qwen 14B
- **Default**: g5.4xlarge (48GB GPU memory) - Extra headroom  
- **Minimum**: g4dn.2xlarge+ (32GB+ GPU memory)

### Memory Usage
- **Unified Deployment**: ~8GB GPU memory (both models)
- **Memory Efficiency**: 60% savings vs separate instances
- **Concurrent Requests**: 2-3 supported on g5.2xlarge

---

## 🌐 Service Endpoints

### Mistral Unified Deployment
After successful deployment:

- **IaC Generation**: `http://YOUR_IP:8000`
  - Health: `/health`
  - Generate: `/v1/completions`
  - Endpoints: `/generate/iac`, `/generate/terraform`, `/generate/kubernetes`

- **Policy Analysis**: `http://YOUR_IP:8001`
  - Health: `/health`
  - Analyze: `/analyze`
  - Specialized: `/analyze/economic`, `/analyze/regulatory`, `/analyze/social`

- **Web Interface**: Use existing chat interface with model dropdown

### Qwen CLAUDE.md Deployment
After successful deployment:

- **Qwen Generation**: `http://YOUR_IP:8001`
  - Health: `/health`
  - Generate: `/generate`
  - Documentation: `/docs`
  - Features: Advanced reasoning with CLAUDE.md methodology integration

---

## 🔧 Service Management

### Systemd Commands

**Mistral Services:**
```bash
# Check service status
sudo systemctl status iac-inference policy-inference

# Restart services
sudo systemctl restart iac-inference policy-inference

# View logs
sudo journalctl -u iac-inference -f
sudo journalctl -u policy-inference -f
```

**Qwen Service:**
```bash
# Check service status
sudo systemctl status qwen-claude-md.service

# Restart service
sudo systemctl restart qwen-claude-md.service

# View logs
sudo journalctl -u qwen-claude-md.service -f
```

### Health Checks

**Mistral Models:**
```bash
# Quick health verification
curl http://localhost:8000/health
curl http://localhost:8001/health

# Test IaC generation
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create Terraform for S3 bucket", "max_tokens": 200}'

# Test policy analysis
curl -X POST http://localhost:8001/analyze/economic \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Carbon tax implementation"}'
```

**Qwen Model:**
```bash
# Health check
curl http://localhost:8001/health

# Test generation with CLAUDE.md methodology
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a Python web API with authentication", "complexity": "medium"}'
```

---

## 📁 S3 Model Storage Structure

```
s3://asoba-llm-cache/
├── models/
│   ├── comprehensive_model_manifest.json # Complete deployment metadata
│   ├── mistral-7b-specialized/
│   │   ├── manifest.json                 # Mistral-specific metadata
│   │   ├── policy-analysis/
│   │   │   └── mistral-policy-qlora.tar.gz  # Policy LoRA (298MB)
│   │   └── iac-generation/
│   │       └── mistral-iac-qlora.tar.gz     # IaC LoRA (297MB)
│   └── Qwen/
│       └── Qwen3-14B/                    # Qwen model files (downloaded on-demand)
└── scripts/
    ├── bootstrap_mistral_deployment.sh   # Mistral unified bootstrap
    ├── bootstrap_qwen_claude_deployment.sh # Qwen bootstrap
    ├── mistral_ami_user_data.sh         # EC2 user data script
    ├── policy_analysis_inference_server.py
    ├── iac_inference_server.py
    ├── qwen_inference_server.py
    ├── qwen_config.json
    └── qwen_claude_md_system_prompt.txt
```

---

## 🚨 Troubleshooting

### Common Issues

**Services not starting**:
```bash
# Check GPU availability
nvidia-smi

# Verify model files
ls -la /home/ubuntu/policy_training/mistral-policy-qlora/
ls -la /home/ubuntu/mistral-iac-qlora/

# Check Python environment
/home/ubuntu/miniconda3/envs/pytorch_p310/bin/python --version
```

**Port conflicts**:
```bash
# Check what's using ports
sudo netstat -tlnp | grep -E ':800[01]'

# Kill conflicting processes
sudo pkill -f inference_server
```

**Security group issues**:
```bash
# Verify security group allows ports 8000, 8001
aws ec2 describe-security-groups --group-ids sg-YOUR-SG
```

### Log Locations
- **Bootstrap log**: `/var/log/mistral-bootstrap.log`
- **Service logs**: `journalctl -u iac-inference -u policy-inference`
- **Application logs**: Check service stdout/stderr

---

## 🎯 Performance Optimization

### GPU Memory Optimization
- Both models share the same Mistral-7B base model
- LoRA adapters add minimal overhead (~7MB each)
- 4-bit quantization reduces memory footprint

### Scaling Thresholds
- **Scale Up**: When GPU utilization > 80% consistently
- **Scale Out**: When response time > 10 seconds average
- **Instance Types**: g5.4xlarge → g5.8xlarge → multiple instances

### Response Time Optimization
- **Cold Start**: 15-25 seconds (first request)
- **Warm Requests**: 3-8 seconds typical
- **Batching**: Group similar requests for efficiency

---

## ✅ Deployment Verification Checklist

- [ ] Instance launched from correct AMI (ami-0a39335458731538a)
- [ ] Bootstrap script completed successfully
- [ ] Both services running: `systemctl status iac-inference policy-inference`
- [ ] Health endpoints responding: `/health` on both ports
- [ ] Security groups configured: ports 8000, 8001 open
- [ ] GPU memory usage reasonable: `nvidia-smi` shows ~8GB used
- [ ] Web interface shows both models in dropdown
- [ ] Test requests work for both IaC and Policy endpoints

---

## 📈 Monitoring & Maintenance

### Key Metrics to Monitor
- **GPU Utilization**: Should be 35-80% during normal operation
- **Memory Usage**: Both services ~4GB each
- **Response Times**: <10 seconds for warm requests
- **Error Rates**: <5% failed requests acceptable

### Maintenance Tasks
- **Weekly**: Review service logs for errors
- **Monthly**: Update LoRA models if retrained
- **As Needed**: Scale instance type based on load patterns

---

## 🎉 Success Criteria

**Deployment is successful when**:
1. Both inference servers respond to health checks
2. GPU memory usage is efficient (~35-40%)
3. Test requests return valid responses
4. Web interface shows both models available
5. Services auto-restart on failure

**Performance targets**:
- IaC Generation: <5 seconds per request
- Policy Analysis: <15 seconds per request
- Concurrent requests: 2-3 supported simultaneously
- Uptime: >99% with auto-restart enabled