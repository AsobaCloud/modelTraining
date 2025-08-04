# Qwen Complete Fine-tuning Strategy (Corrected)

> **Objective**: Multi-stage fine-tuning of Qwen3-14B for production-ready software engineering with CLAUDE.md methodology
> **Stage 1**: Verbosity Control with style tokens - Fix response length appropriateness
> **Stage 2**: IaC Specialization - Domain expertise enhancement  
> **Target**: Production-ready software engineering agent with appropriate verbosity and specialized knowledge

---

## Corrections from Original Plan

### What was wrong in the original

1. **Training on AWQ**: AWQ is an **inference** quantization scheme. You don't fine-tune "on AWQ." The standard low-VRAM path is **QLoRA** (bnb 4-bit) for training; **then** you optionally quantize the *merged* FP16 weights for inference (AWQ/GPTQ/Marlin).

2. **Memory/timing conflation**: The listed ~10–11 GB figure could describe **inference** with 4-bit + LoRA, not **training**. Even QLoRA training of a 14B model will need appreciably more.

3. **Context lengths and speed claims**: Claims like "Mistral 7B context 512, Qwen 14B 2048, 5–10× faster" are not credible. Modern Mistral/Qwen variants have context windows ≥8k, and a 5–10× speedup from custom code swap is not realistic.

4. **Adapter composition ambiguity**: "Use Stage 1 adapter as base for Stage 2" is vague. The safe/clean production path is: **Merge** Stage-1 LoRA → base FP16, then fine-tune, or load Stage-1 LoRA and continue training to produce a single "complete" LoRA.

5. **Evaluation is not IaC-aware**: Speed tests like "2+2" are not decision-grade. You need IaC-specific automated checks (`terraform validate`, `tflint`, `checkov`, `cfn-lint`, `kubeconform`, etc.).

6. **Serving stack**: A bespoke server often underperforms vs **vLLM** or **TGI**.

---

## 1. Strategy Overview (Corrected)

### Multi-Stage Training Approach
**Stage 1: Verbosity Control (Foundation)**
- **Problem**: Base model generates 50+ tokens for "What is 2+2?" instead of "4"
- **Solution**: QLoRA fine-tuning with control tokens `<STYLE_TERSE>` and `<STYLE_VERBOSE>`
- **Method**: BitsAndBytes 4-bit quantization + LoRA adapters
- **Control**: Prepend style tokens to user prompts for deterministic verbosity

**Stage 2: IaC Specialization**  
- **Enhancement**: Load Stage-1 LoRA and continue training on IaC corpus
- **Data Sources**: `final_enhanced_iac_corpus.jsonl` (2,193 examples)
- **Output**: Single adapter encoding both verbosity control + IaC expertise

### Assumptions (Confirmed)
- **Base model**: Qwen/Qwen3-14B-Chat (exact variant to be confirmed)
- **GPU**: NVIDIA A10G (24GB VRAM) - sufficient for QLoRA
- **OS**: Ubuntu 22.04; CUDA 12.x
- **Storage**: S3 bucket `s3://asoba-llm-cache/` exists

---

## 2. Technical Specifications (Corrected)

### Environment Setup
```bash
conda create -n qwen-iac python=3.10 -y
conda activate qwen-iac
pip install --upgrade pip
pip install "transformers>=4.43" "accelerate>=0.33" "datasets>=2.19" \
            "bitsandbytes>=0.43" "peft>=0.11" "trl>=0.9" evaluate \
            sentencepiece autoawq "vllm>=0.5.0" awscli
```

### Stage 1: Verbosity Control LoRA Configuration (QLoRA)
```python
# QLoRA configuration - correct approach
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
task_type: "CAUSAL_LM"
bias: "none"

# Base quantization: BitsAndBytes 4-bit (for training)
load_in_4bit: True
bnb_4bit_quant_type: "nf4"
bnb_4bit_compute_dtype: torch.bfloat16
```

### Stage 1: Training Parameters
```python
learning_rate: 2e-4
num_train_epochs: 1
warmup_ratio: 0.03
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: True
fp16: True
optim: "adamw_torch"
lr_scheduler_type: "cosine"
max_seq_length: 2048
packing: True
```

### Stage 2: IaC Training Parameters
```python
learning_rate: 1.5e-4  # Lower for continued training
num_train_epochs: 1
warmup_ratio: 0.03
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: True
fp16: True
max_seq_length: 3072  # Longer for IaC manifests
packing: True
```

---

## 3. Training Data Strategy (Updated)

### Stage 1: Verbosity Control Dataset
**Format**: JSONL with style control tokens
**Fields**: `system`, `instruction`, `input`, `output`, `style`

**Example**:
```json
{"system":"You are a precise assistant.",
 "instruction":"What is HTML?",
 "input":"",
 "output":"HyperText Markup Language.",
 "style":"TERSE"}
```

**Special Tokens**: Add `<STYLE_TERSE>` and `<STYLE_VERBOSE>` to tokenizer
**Minimum Size**: >1k pairs recommended (100 is too small for robust control)

### Stage 2: IaC Dataset
**Source**: `final_enhanced_iac_corpus.jsonl`
**Content**: Terraform (AWS/GCP/Azure), Helm charts, Kubernetes YAML, CloudFormation, Pulumi
**Quality**: Runnable, validated infrastructure code
**Format**: Include style tokens for appropriate verbosity level

---

## 4. Implementation Steps (Corrected)

### Phase 1: Environment Setup
**Instance**: i-0645c6db622720234 (g5.4xlarge, A10G GPU, 500GB EBS)
**Region**: us-west-2 (Flux dev server)
**SSH**: `ssh -i config/mistral-base.pem ubuntu@34.217.17.67`

```bash
# Environment setup
cd /opt/models
conda create -n qwen-iac python=3.10 -y
conda activate qwen-iac
pip install "transformers>=4.43" "accelerate>=0.33" "datasets>=2.19" \
            "bitsandbytes>=0.43" "peft>=0.11" "trl>=0.9" evaluate \
            sentencepiece autoawq "vllm>=0.5.0" awscli
```

### Phase 2: Stage 1 Training (QLoRA Verbosity Control)

**Training Script** (using TRL SFTTrainer):
```python
import os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

model_name = "Qwen/Qwen3-14B-Chat"
ds_path = "/opt/models/verbosity_training_data_v2.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
special_tokens = {"additional_special_tokens": ["<STYLE_TERSE>", "<STYLE_VERBOSE>"]}
tokenizer.add_special_tokens(special_tokens)

def format_example(ex):
    style_token = "<STYLE_TERSE>" if ex["style"].upper().startswith("TERSE") else "<STYLE_VERBOSE>"
    sys = (ex.get("system") or "").strip()
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    
    parts = []
    if sys: parts.append(f"<|system|>\n{sys}")
    if instr and inp:
        parts.append(f"<|user|>\n{style_token}\n{instr}\n\n{inp}")
    elif instr:
        parts.append(f"<|user|>\n{style_token}\n{instr}")
    else:
        parts.append(f"<|user|>\n{style_token}")
    parts.append(f"<|assistant|>\n{out}")
    return {"text":"\n\n".join(parts)}

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", load_in_4bit=True
)
model.resize_token_embeddings(len(tokenizer))

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./qwen_verbosity_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=25,
    save_steps=500,
    num_train_epochs=1,
    gradient_checkpointing=True,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    save_total_limit=2,
    report_to="none"
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=train,
    peft_config=peft_cfg, packing=True, max_seq_length=2048,
    dataset_text_field="text", args=training_args
)

trainer.train()
trainer.model.save_pretrained("./qwen_verbosity_lora")
tokenizer.save_pretrained("./qwen_verbosity_lora")
```

**Expected Output**: `./qwen_verbosity_lora/` (LoRA adapters + tokenizer with style tokens)

### Phase 3: Stage 2 Training (IaC Specialization)

**Strategy**: Load Stage-1 LoRA and continue training to produce single "complete" LoRA

```python
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, PeftModel

model_name = "Qwen/Qwen3-14B-Chat"
stage1_path = "./qwen_verbosity_lora"
ds_path = "/opt/models/final_enhanced_iac_corpus.jsonl"

tokenizer = AutoTokenizer.from_pretrained(stage1_path, use_fast=True)

def fmt(ex):
    sys = ex.get("system","You are an IaC assistant that follows best practices.").strip()
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    style = ex.get("style","VERBOSE").upper()
    style_tok = "<STYLE_VERBOSE>" if style.startswith("VERBOSE") else "<STYLE_TERSE>"
    u = f"<|user|>\n{style_tok}\n{instr}\n\n{inp}" if inp else f"<|user|>\n{style_tok}\n{instr}"
    a = f"<|assistant|>\n{out}"
    return {"text": f"<|system|>\n{sys}\n\n{u}\n\n{a}"}

base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", load_in_4bit=True)
base.resize_token_embeddings(len(tokenizer))

# Load Stage-1 LoRA and continue training
model = PeftModel.from_pretrained(base, stage1_path, is_trainable=True)

args = TrainingArguments(
    output_dir="./qwen_complete_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1.5e-4,
    warmup_ratio=0.03,
    logging_steps=25,
    save_steps=1000,
    num_train_epochs=1,
    gradient_checkpointing=True,
    fp16=True,
    lr_scheduler_type="cosine",
    save_total_limit=2,
    report_to="none"
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=train,
    peft_config=peft_cfg, packing=True, max_seq_length=3072,
    dataset_text_field="text", args=args
)

trainer.train()
trainer.model.save_pretrained("./qwen_complete_lora")
tokenizer.save_pretrained("./qwen_complete_lora")
```

**Expected Output**: `./qwen_complete_lora/` (single adapter covering verbosity + IaC)

---

## 5. Deployment (Corrected)

### Option A: vLLM Serving (Recommended)
```bash
# Serve base model + LoRA adapter
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B-Chat \
  --enable-lora \
  --lora-modules qwen_complete_lora=./qwen_complete_lora \
  --max-model-len 4096 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --port 8000
```

### Option B: AWQ for Inference (Only After Merge)
```bash
# 1. Merge LoRA into FP16 first
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base_id = "Qwen/Qwen3-14B-Chat"
lora_dir = "./qwen_complete_lora"
out = "./qwen_complete_fp16_merged"
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="float16")
model = PeftModel.from_pretrained(model, lora_dir)
model = model.merge_and_unload()
model.save_pretrained(out)
PY

# 2. AWQ quantize for inference
python - <<'PY'
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained("./qwen_complete_fp16_merged")
model.quantize(tokenizer, quant_config={"w_bit":4, "q_group_size":128})
model.save_quantized("./qwen_complete_awq_int4")
PY
```

### Systemd Service
```bash
sudo tee /etc/systemd/system/qwen.service >/dev/null <<'UNIT'
[Unit]
Description=Qwen vLLM Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/miniconda3/envs/qwen-iac/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B-Chat \
  --enable-lora \
  --lora-modules qwen_complete_lora=/home/ubuntu/qwen_complete_lora \
  --max-model-len 4096 --dtype float16 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now qwen
```

---

## 6. Validation & Metrics (Concrete)

### Verbosity Control Validation
- **Test**: Same question with `<STYLE_TERSE>` vs `<STYLE_VERBOSE>`
- **Success Criteria**: ≥3× token-count separation on 25-prompt eval set
- **Examples**:
  - "What is 2+2?" with `<STYLE_TERSE>` → "4"
  - "What is 2+2?" with `<STYLE_VERBOSE>` → detailed arithmetic explanation

### IaC Quality Validation (Automated)
```bash
# Terraform validation
terraform fmt -check && terraform init -backend=false && terraform validate
tflint --config .tflint.hcl
checkov -d .

# CloudFormation validation
cfn-lint template.yaml

# Kubernetes validation
kubeconform -strict -ignore-missing-schemas -summary -exit-on-error manifests/
```

**Success Criteria**:
- `terraform validate` pass rate ≥85% on held-out prompts
- `tflint` critical findings = 0 on pass cases
- `checkov` high-severity = 0 on pass cases
- `cfn-lint` error-free for CFN outputs
- `kubeconform` pass for Kubernetes outputs

### Performance Metrics
- **Latency**: Report p50/p90 for short prompts (≤32 tokens output) and long prompts (≥512)
- **Method**: Constant `max_new_tokens`, `temperature=0`, same server settings
- **Baseline**: Compare against current setup, not speculative numbers

---

## 7. Risk Mitigation (Concrete)

### Training Risks
- **OOM during training**: Reduce `max_seq_length`, increase grad accum, enable `gradient_checkpointing`
- **Catastrophic style drift**: Keep Stage-1 dataset in Stage-2 mix at 5-10% to preserve control tokens
- **Training instability**: Fixed seed (42), monitor loss curves, early stopping on plateau

### Deployment Risks
- **Inference regressions after AWQ**: Compare FP16 vs AWQ output parity on 100-prompt suite
- **vLLM compatibility**: Test LoRA loading before production deployment
- **Memory pressure**: Monitor GPU utilization, implement request queuing if needed

---

## 8. Success Metrics (Measurable)

### Technical Metrics
- [ ] **Training completion**: 0 OOMs, reproducible with fixed seed
- [ ] **Verbosity control**: ≥3× token separation between styles
- [ ] **IaC validation**: ≥85% toolchain pass rate
- [ ] **Latency**: p50 < N seconds (define N after baseline measurement)

### Operational Metrics
- [ ] **Serving stability**: 99.9% uptime, automated restart on failure
- [ ] **Rollback capability**: Previous adapter archived, quick rollback procedure
- [ ] **Monitoring**: QPS/latency/error dashboards
- [ ] **Cost tracking**: $/hour measured vs budgeted

---

## 9. Archival & Reproducibility

### Version Control
```bash
# Record exact versions
conda list > training_environment_$(date +%Y%m%d).txt
nvidia-smi > gpu_info_$(date +%Y%m%d).txt
cat /proc/version > system_info_$(date +%Y%m%d).txt
```

### S3 Backups
```bash
# Archive training artifacts
tar -czf qwen_complete_training_$(date +%Y%m%d).tar.gz \
  qwen_verbosity_lora/ qwen_complete_lora/ *.txt *.log
aws s3 cp qwen_complete_training_$(date +%Y%m%d).tar.gz s3://asoba-llm-cache/training/

# Archive adapters
aws s3 cp qwen_complete_lora/ s3://asoba-llm-cache/adapters/qwen_complete_$(date +%Y%m%d)/ --recursive
```

### Training Log
Document in `TRAINING_LOG.md`:
- Base model exact ID + revision
- All package versions
- CUDA + driver versions, GPU model, VRAM
- Random seed, training args (full dump)
- Dataset checksums (SHA256)
- Eval suite results and pass rates
- Serving config details

---

## 10. Next Steps (Concrete)

1. **Confirm base model**: Verify exact Qwen model ID and availability
2. **Prepare training data**: Convert current dataset to JSONL with style tokens
3. **Execute Stage 1**: Run QLoRA verbosity training (2-3 hours)
4. **Validate verbosity**: Test style token control before proceeding
5. **Execute Stage 2**: Continue training on IaC corpus (4-6 hours)
6. **Deploy with vLLM**: Set up production serving
7. **Run validation suite**: IaC automated checks and performance benchmarks
8. **Archive and document**: Complete reproducibility documentation

---

**Key Changes from Original Plan**:
- ✅ QLoRA training instead of impossible "AWQ + LoRA training"
- ✅ vLLM serving instead of custom servers
- ✅ Concrete IaC validation metrics instead of speed guesses
- ✅ Style tokens for deterministic verbosity control
- ✅ Proper adapter composition strategy
- ✅ Measurable success criteria
- ✅ Complete reproducibility documentation