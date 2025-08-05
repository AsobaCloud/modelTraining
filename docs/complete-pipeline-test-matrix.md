# Complete Pipeline Test Matrix

## Problem: Trial and Error Development

Currently we have **zero systematic validation** before deployment. Each run is:
1. Deploy blind
2. Wait for mysterious failure 
3. SSH forensics to find root cause
4. Apply band-aid fix
5. Repeat cycle

This is a massive waste of time and completely unprofessional.

## Solution: Comprehensive Pre-Deployment Validation

Before any pipeline deployment, we need **100% confidence** that it will work or **specific diagnostics** on what needs to be fixed.

## Complete Failure Matrix

### Phase 1: Infrastructure & Prerequisites
| Test Category | Failure Mode | Test Method | Specific Diagnostic | Fix Action |
|---------------|--------------|-------------|-------------------|------------|
| **AWS Credentials** | Invalid/expired credentials | `aws sts get-caller-identity` | "AWS credentials invalid: {error}" | Check ~/.aws/credentials |
| **S3 Access** | No read access to policy-database | `aws s3 ls s3://policy-database/` | "Cannot read policy data: {permissions}" | Fix S3 bucket policy |
| **S3 Access** | No write access to asoba-llm-cache | `aws s3 cp test.txt s3://asoba-llm-cache/test/` | "Cannot write outputs: {permissions}" | Fix S3 bucket policy |
| **EC2 Instance** | Instance doesn't exist | `aws ec2 describe-instances --instance-ids {id}` | "Instance {id} not found" | Launch instance or fix ID |
| **EC2 Instance** | Instance not running | Instance state check | "Instance {id} in state {state}" | Start instance |
| **SSH Access** | Cannot connect to instance | `ssh -i {key} ubuntu@{ip} exit` | "SSH failed: {error}" | Fix key permissions/IP |
| **EBS Volume** | /mnt/training not mounted | `df -h /mnt/training` | "Training volume not available" | Mount EBS volume |
| **Disk Space** | Insufficient space for training | `df -h /mnt/training` | "Only {GB}GB free, need {required}GB" | Resize EBS or cleanup |

### Phase 2: Environment & Dependencies  
| Test Category | Failure Mode | Test Method | Specific Diagnostic | Fix Action |
|---------------|--------------|-------------|-------------------|------------|
| **Python Environment** | Wrong Python version | `python3 --version` | "Python {version}, need 3.10+" | Install correct Python |
| **PyTorch** | Missing/wrong PyTorch | `python3 -c "import torch; print(torch.__version__)"` | "PyTorch {version} invalid" | Install PyTorch 2.5.1+cu121 |
| **CUDA** | CUDA not available | `python3 -c "import torch; print(torch.cuda.is_available())"` | "CUDA not available on instance" | Fix CUDA installation |
| **GPU Memory** | Insufficient GPU memory | `nvidia-smi` | "GPU has {GB}GB, need {required}GB" | Use larger instance type |
| **Dependencies** | Missing required packages | Import tests for each package | "Missing package: {package}" | pip install {package} |
| **Model Files** | Mistral model not available | `ls /mnt/training/models/mistral-7b-v0.3/` | "Base model not found at {path}" | Download model to EBS |

### Phase 3: Data Pipeline Validation
| Test Category | Failure Mode | Test Method | Specific Diagnostic | Fix Action |
|---------------|--------------|-------------|-------------------|------------|
| **Data Sources** | Policy folders empty/missing | Count files in each S3 folder | "Source {folder} has 0 files" | Check S3 policy data |
| **Data Format** | Invalid JSONL structure | Parse sample files | "Invalid JSON in {file}: {error}" | Fix data format |
| **Data Content** | Missing required fields | Schema validation | "Records missing 'text' field: {count}" | Fix data processing |
| **Data Volume** | Insufficient training data | Count valid records | "Only {count} valid records, need {min}" | Add more data sources |
| **Download Speed** | Network too slow for 27GB | Bandwidth test | "Speed {mbps}Mbps, will take {hours}h" | Use larger instance |
| **Operatives Limit** | PDF processing will exceed limits | Count PDF files | "Found {count} PDFs, limit is {max}" | Adjust processing limits |

### Phase 4: Training Process Validation
| Test Category | Failure Mode | Test Method | Specific Diagnostic | Fix Action |
|---------------|--------------|-------------|-------------------|------------|
| **Dataset Loading** | Cannot load train/val datasets | Load datasets in memory | "Dataset load failed: {error}" | Fix dataset format |
| **Model Loading** | Cannot load base model | Load model in memory | "Model load failed: {error}" | Fix model files |
| **Memory Requirements** | Training exceeds available memory | Calculate memory needs | "Need {GB}GB RAM, have {available}GB" | Use larger instance |
| **Batch Size** | Batch size too large for GPU | Test training step | "Batch size {size} causes OOM" | Reduce batch size |
| **Learning Rate** | Learning rate will cause divergence | Validate hyperparameters | "LR {lr} may cause instability" | Adjust hyperparameters |
| **Checkpoint Saving** | Cannot save checkpoints | Test checkpoint write | "Checkpoint save failed: {error}" | Fix permissions/space |

### Phase 5: Monitoring & Alerting  
| Test Category | Failure Mode | Test Method | Specific Diagnostic | Fix Action |
|---------------|--------------|-------------|-------------------|------------|
| **Slack Webhook** | Webhook URL invalid/expired | Send test message | "Slack webhook failed: {error}" | Update webhook URL |
| **S3 Monitoring** | Cannot write monitoring data | Write test metadata | "Monitoring S3 write failed: {error}" | Fix S3 permissions |
| **Heartbeat System** | Heartbeat manager crashes | Start/stop heartbeat test | "Heartbeat failed: {error}" | Fix heartbeat code |
| **Error Detection** | Monitoring doesn't detect failures | Inject test failures | "Failed to detect {failure_type}" | Fix monitoring logic |
| **Alert Content** | Alerts missing critical info | Validate alert format | "Alert missing {required_field}" | Fix alert template |

### Phase 6: End-to-End Integration
| Test Category | Failure Mode | Test Method | Specific Diagnostic | Fix Action |
|---------------|--------------|-------------|-------------------|------------|
| **Pipeline Orchestration** | Scripts fail to execute in sequence | Run shortened pipeline | "Failed at step {step}: {error}" | Fix script dependencies |
| **State Persistence** | Pipeline cannot resume after interruption | Test resume functionality | "Resume failed: {error}" | Fix state management |
| **Error Propagation** | Failures in one component don't stop others | Inject failures and test | "Error not propagated from {component}" | Fix error handling |
| **Cleanup** | Failed runs leave resources running | Test cleanup after failure | "Resources not cleaned: {resources}" | Fix cleanup scripts |
| **Documentation** | Instructions don't match reality | Follow docs exactly | "Step {step} in docs is wrong: {error}" | Update documentation |

## Test Implementation Strategy

### 1. Pre-Flight Validation (< 2 minutes)
Run before any deployment to catch obvious issues:
```bash
./scripts/preflight_check.py --instance-id i-xxx --ssh-key key.pem
```

**Output:**
```
🔍 INFRASTRUCTURE CHECKS
✅ AWS credentials valid
✅ S3 buckets accessible  
✅ Instance i-xxx running
✅ SSH connection established
✅ EBS volume mounted (250GB free)

🔍 ENVIRONMENT CHECKS  
✅ Python 3.10.12
✅ PyTorch 2.5.1+cu121
✅ CUDA available (24GB GPU)
✅ All dependencies installed
❌ Mistral model missing at /mnt/training/models/

🚨 DEPLOYMENT BLOCKED: 1 critical issue
   Fix: Download base model before proceeding
```

### 2. Data Pipeline Validation (< 10 minutes)
Test data processing with small sample:
```bash
./scripts/validate_data_pipeline.py --sample-size 1000
```

**Output:**
```
🔍 DATA PIPELINE VALIDATION
✅ Policy sources accessible (8/8)
✅ Sample data valid format  
✅ Text field present in 99.8% of records
❌ Validation fails on 12 records: Missing text field
✅ Processing speed: 1.2k records/minute

🚨 PIPELINE NEEDS ATTENTION: 1 issue
   Fix: Update validation logic to skip invalid records
```

### 3. Training Validation (< 5 minutes)
Test training loop with minimal data:
```bash  
./scripts/validate_training.py --quick-test
```

**Output:**
```
🔍 TRAINING VALIDATION
✅ Dataset loads successfully
✅ Model loads successfully  
✅ Forward pass works
✅ Backward pass works
✅ Checkpoint saving works
❌ Memory usage: 22.1GB (95% of available)

⚠️  TRAINING RISKY: 1 warning
   Warning: Memory usage very high, consider reducing batch size
```

### 4. Monitoring Validation (< 1 minute)
Test monitoring and alerting:
```bash
./scripts/validate_monitoring.py --run-id test-123
```

**Output:**
```
🔍 MONITORING VALIDATION
✅ Heartbeat system working
✅ S3 metadata updates working  
✅ Slack alerts delivering
✅ Error detection working
✅ Status transitions accurate

✅ MONITORING FULLY OPERATIONAL
```

### 5. Full End-to-End Test (< 30 minutes)
Complete pipeline test with real but small dataset:
```bash
./scripts/full_pipeline_test.py --quick-run
```

**Pipeline Readiness Score: 87/100**

```
🔍 FULL PIPELINE TEST RESULTS

✅ Infrastructure Setup      (20/20)
✅ Environment Validation    (15/15)  
⚠️  Data Pipeline           (12/15) - 3 warnings
✅ Training Process          (20/20)
✅ Monitoring System         (10/10)
⚠️  Error Handling          (8/10) - 2 gaps
✅ Cleanup & Documentation   (10/10)

DEPLOYMENT RECOMMENDATION: 
🟡 PROCEED WITH CAUTION
   Fix 2 error handling gaps before production use
   
ESTIMATED FULL RUN SUCCESS: 87%
```

## Success Criteria

✅ **Zero surprises** - Every failure mode has been tested  
✅ **Specific diagnostics** - Know exactly what's broken and how to fix it  
✅ **Deployment confidence** - Know success probability before starting  
✅ **Time savings** - No more trial-and-error debugging cycles  
✅ **Professional operation** - Systematic validation like real software teams  

The goal: **Never deploy blind again.**