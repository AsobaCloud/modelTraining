# Mistral-7B Golden Config Documentation

## PLAN Phase - Validated Configuration Specification ✅

This document specifies the golden-path configuration for Mistral-7B verbosity training, adapted from the proven Qwen approach. The configuration leverages the existing training data and methodology while optimizing for the Mistral architecture and A10G hardware constraints.

## Environment Specifications

### Server Configuration
- **Instance**: AWS g5.2xlarge 
- **Instance IP**: 54.197.142.172
- **SSH Access**: `ssh -i config/mistral-base.pem ubuntu@54.197.142.172`
- **GPU**: NVIDIA A10G (22GB VRAM)
- **Base Model**: `/home/ubuntu/mistral-7b-v0.3`
- **Python**: 3.10+ 
- **CUDA**: 12.6+

### Library Versions (Golden Path Pinned)
```bash
transformers==4.54.0
torch==2.7.1+cu126  
peft==0.16.0
accelerate==1.9.0
bitsandbytes==0.46.1
datasets==3.6.0
```

### Hardware Validation
- **Expected GPU Memory**: ~8-10GB (smaller than Qwen3-14B due to 7B parameter count)
- **VRAM Headroom**: ~12GB available for training on 22GB A10G
- **Target Training Time**: ~30-45 minutes (vs 60 minutes for Qwen3-14B)

## Golden Configuration Parameters

### Environment Setup
```bash
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_USE_DEEPSPEED=0
export ACCELERATE_USE_FSDP=0
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Model Loading (Mistral-Specific)
```python
# BitsAndBytes 4-bit quantization (same as Qwen)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Mistral model loading with forced single GPU
model = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/mistral-7b-v0.3",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},  # Critical: force single GPU (golden path)
    trust_remote_code=True
)
```

### Tokenizer Configuration
```python
# Load Mistral tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ubuntu/mistral-7b-v0.3", 
    trust_remote_code=True
)

# Add style tokens as additional special tokens
special_tokens = {
    "additional_special_tokens": ["<STYLE_TERSE>", "<STYLE_VERBOSE>"]
}
tokenizer.add_special_tokens(special_tokens)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Resize model embeddings for new tokens
model.resize_token_embeddings(len(tokenizer))
```

### QLoRA Configuration (Mistral Architecture)
```python
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,                    # LoRA rank
    lora_alpha=32,          # LoRA alpha  
    lora_dropout=0.05,      # LoRA dropout
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",    # Attention projections
        "gate_proj", "up_proj", "down_proj"        # MLP projections (Mistral-specific)
    ],
    modules_to_save=["embed_tokens", "lm_head"],   # Save for new style tokens
    inference_mode=False,
)
```

### Training Arguments (g5.2xlarge Optimized)
```python
TrainingArguments(
    output_dir="./mistral_verbosity_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,      # Effective batch size: 16
    learning_rate=2e-4,
    num_train_epochs=1,                  # Scale up to 2-3 as needed
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    warmup_ratio=0.03,
    logging_steps=25,
    save_steps=500,
    lr_scheduler_type="cosine",
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,
    packing=False,
    completion_only_loss=True,           # Critical for prompt-completion training
)
```

## Data Format and Processing

### Dataset Configuration
- **Source**: `verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl` (reuse existing data)
- **Format**: Prompt-completion pairs with style token injection
- **Size**: 1600 examples (sufficient for style fine-tuning)

### Style Token Detection Logic
```python
def add_style_token(messages):
    """Add style token based on user prompt keywords"""
    user_content = messages[0]['content']
    
    # Detection logic
    if "briefly" in user_content.lower() or "concise" in user_content.lower():
        style_token = "<STYLE_TERSE>"
    elif "detailed" in user_content.lower() or "well-structured" in user_content.lower():
        style_token = "<STYLE_VERBOSE>"
    else:
        style_token = "<STYLE_TERSE>"  # Default to terse
    
    # Prepend style token
    messages[0]['content'] = f"{style_token}\n{user_content}"
    return messages
```

### Prompt-Completion Format
```python
# Example processed data
{
    "prompt": "<s>[INST] <STYLE_TERSE>\nWhat is rate limiting? [/INST]",
    "completion": "Rate limiting controls request frequency.</s>"
}
```

## Inference Configuration

### Style-Specific Generation Presets
```python
# TERSE preset (concise responses)
TERSE_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_new_tokens": 32,
    "do_sample": True
}

# VERBOSE preset (detailed responses)  
VERBOSE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "min_new_tokens": 64,
    "max_new_tokens": 256,
    "do_sample": True
}
```

### Inference Protocol
```python
# Style-aware generation
def generate_with_style(model, tokenizer, prompt, style):
    style_token = "<STYLE_TERSE>" if style == "terse" else "<STYLE_VERBOSE>"
    styled_prompt = f"{style_token}\n{prompt}"
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": styled_prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).cuda()
    
    # Use appropriate generation preset
    params = TERSE_PARAMS if style == "terse" else VERBOSE_PARAMS
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(inputs, **params)
    
    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()
```

## Deployment Architecture

### Flask Server Configuration
```python
# Production-ready deployment endpoint
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    style = data.get('style', 'terse').lower()
    
    response = generate_with_style(model, tokenizer, prompt, style)
    
    return jsonify({
        "response": response,
        "style": style,
        "tokens": len(tokenizer.encode(response)),
        "latency": "< 30s"  # Target latency requirement
    })
```

### Health Check and Monitoring
```python
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "mistral-7b-v0.3",
        "adapter": "verbosity_lora",
        "gpu_memory": torch.cuda.memory_allocated() / 1024**3  # GB
    })
```

## Key Differences from Qwen Configuration

### Architecture Adaptations
1. **Target Modules**: Mistral uses `gate_proj`, `up_proj`, `down_proj` for MLP (vs Qwen's structure)
2. **Model Size**: 7B parameters vs 14B (faster training, lower memory)
3. **Chat Template**: Mistral uses different format than Qwen's `<|im_start|>` tokens
4. **Memory Requirements**: ~8-10GB vs ~12-14GB for Qwen

### Training Optimizations
1. **Reduced Training Time**: Target 30-45 minutes vs 60 minutes
2. **Lower Memory Pressure**: More headroom on 22GB A10G
3. **Faster Inference**: Smaller model = faster generation
4. **Same Data Reuse**: Leverage existing 1600-example dataset

## Success Criteria

### Training Validation
- [ ] **Model loads in 4-bit**: `model.is_loaded_in_4bit == True`
- [ ] **GPU memory < 12GB**: Sufficient headroom on 22GB A10G
- [ ] **Training completes**: No OOM errors during training
- [ ] **Loss convergence**: Training loss decreases steadily
- [ ] **Adapter saves**: LoRA weights saved successfully

### Style Control Validation  
- [ ] **Token separation**: TERSE < 50 tokens, VERBOSE > 80 tokens
- [ ] **Style consistency**: TERSE responses concise, VERBOSE detailed
- [ ] **Generation speed**: < 30s end-to-end latency
- [ ] **No repetition**: Clean, coherent responses

### Deployment Validation
- [ ] **Server starts**: Flask app loads trained model
- [ ] **Health endpoint**: `/health` returns 200 status
- [ ] **Generate endpoint**: `/generate` produces style-controlled responses
- [ ] **Error handling**: Graceful handling of malformed requests

## Migration Path from Qwen

### Immediate Steps
1. **Copy training data**: Reuse `verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl`
2. **Adapt target modules**: Update LoRA config for Mistral architecture  
3. **Update model path**: Point to `/home/ubuntu/mistral-7b-v0.3`
4. **Adjust memory settings**: Optimize for 22GB vs 24GB VRAM

### Validation Protocol
1. **Run training script**: Execute full training pipeline
2. **Test style control**: Validate TERSE vs VERBOSE outputs
3. **Benchmark performance**: Measure training time and inference speed
4. **Deploy server**: Test Flask endpoints under load

This golden configuration provides a deterministic, optimized path for Mistral-7B verbosity training, leveraging proven techniques from the Qwen implementation while adapting for the different architecture and hardware constraints.