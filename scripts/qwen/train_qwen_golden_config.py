#!/usr/bin/env python3
"""
Golden Config QLoRA Training for Qwen3-14B
Exact pinned configuration for A10G (24GB) - deterministic recipe
"""

import os
import json
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Set environment flags before imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "0"
os.environ["ACCELERATE_USE_FSDP"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig, 
    get_peft_model,
)
from datasets import Dataset

# Configuration
S3_MODEL_PATH = "s3://asoba-llm-cache/models/Qwen/Qwen3-14B/"
MODEL_PATH = "/tmp/Qwen3-14B"
DATASET_PATH = "verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl"
OUTPUT_DIR = "qwen3_14b_verbosity_pc_lora"
MAX_LEN = 1024  # Safe for A10G with proper QLoRA

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StyleDeltaEmbedding(nn.Module):
    """
    Wraps a frozen base embedding and adds a tiny trainable delta
    ONLY at positions where <STYLE_TERSE>/<STYLE_VERBOSE> appear.
    """
    def __init__(self, base_emb: nn.Embedding, terse_id: int, verbose_id: int):
        super().__init__()
        self.base_emb = base_emb  # frozen outside
        self.terse_id = terse_id
        self.verbose_id = verbose_id
        hidden = base_emb.embedding_dim

        # Two trainable rows (<< 1MB even at hidden=5120)
        self.style_delta = nn.Embedding(2, hidden)
        nn.init.zeros_(self.style_delta.weight)

        # Ensure base embedding is frozen
        for p in self.base_emb.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.LongTensor):
        out = self.base_emb(input_ids)
        # masks
        m_terse = (input_ids == self.terse_id).unsqueeze(-1)  # [B,T,1]
        m_verbose = (input_ids == self.verbose_id).unsqueeze(-1)

        if m_terse.any():
            delta = self.style_delta.weight[0].to(device=out.device, dtype=out.dtype)
            out = out + delta * m_terse
        if m_verbose.any():
            delta = self.style_delta.weight[1].to(device=out.device, dtype=out.dtype)
            out = out + delta * m_verbose
        return out

class PromptCompletionCollator:
    """Custom collator with exact masking"""
    
    def __init__(self, tokenizer, max_length=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for ex in examples:
            # Tokenize full sequence
            full_text = ex['prompt'] + ex['completion']
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Get prompt length for masking
            prompt_encoding = self.tokenizer(
                ex['prompt'],
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            prompt_length = prompt_encoding.input_ids.shape[1]
            
            # Create labels with masking
            input_ids = encoding.input_ids[0]
            labels = input_ids.clone()
            labels[:prompt_length] = -100  # Mask prompt tokens
            
            # Clamp sequence length to avoid OOM
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = encoding.attention_mask[0][:self.max_length]
                labels = labels[:self.max_length]
            else:
                attention_mask = encoding.attention_mask[0]
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        # Pad to same length
        max_len = max(len(ids) for ids in batch_input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(batch_input_ids)):
            pad_length = max_len - len(batch_input_ids[i])
            
            padded_input_ids.append(torch.cat([
                batch_input_ids[i],
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ]))
            
            padded_attention_mask.append(torch.cat([
                batch_attention_mask[i],
                torch.zeros(pad_length, dtype=torch.long)
            ]))
            
            padded_labels.append(torch.cat([
                batch_labels[i],
                torch.full((pad_length,), -100, dtype=torch.long)
            ]))
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels)
        }

def download_model_from_s3():
    """Download model from S3 to local cache"""
    import subprocess
    import os
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model already exists at {MODEL_PATH}")
        return
    
    logger.info(f"Downloading model from {S3_MODEL_PATH} to {MODEL_PATH}")
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    cmd = f"aws s3 sync {S3_MODEL_PATH} {MODEL_PATH} --region us-east-1"
    subprocess.run(cmd, shell=True, check=True)
    logger.info("Model download complete")

def setup_model_and_tokenizer():
    """Golden config model setup"""
    logger.info("Setting up model with golden config...")
    
    # BitsAndBytes config - explicit 4-bit
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # A10G supports bf16
    )
    
    # Load model with forced single GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},  # force single GPU, no auto-sharding
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add style tokens and resize
    added = tokenizer.add_tokens(["<STYLE_TERSE>", "<STYLE_VERBOSE>"], special_tokens=False)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Added {added} style tokens, resized embeddings")
        
        # Get token IDs for the wrapper
        terse_id = tokenizer.convert_tokens_to_ids("<STYLE_TERSE>")
        verbose_id = tokenizer.convert_tokens_to_ids("<STYLE_VERBOSE>")
        
        # Locate and replace the embedding module
        inner = getattr(model, "model", None) or getattr(model, "transformer", None)
        assert inner is not None and hasattr(inner, "embed_tokens"), "Cannot locate embed_tokens"
        
        # Replace the embedding with the wrapper
        base_emb = inner.embed_tokens
        wrapped = StyleDeltaEmbedding(base_emb, terse_id, verbose_id)
        inner.embed_tokens = wrapped
        
        logger.info(f"Installed StyleDeltaEmbedding; base frozen: {all(not p.requires_grad for p in base_emb.parameters())}")
    
    # QLoRA config - LoRA + tiny style delta
    peft_cfg = LoraConfig(
        r=16,  # Back to 16 since we're not training full embeddings
        lora_alpha=32,  # Keep 2:1 ratio  
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        modules_to_save=["model.embed_tokens.style_delta"],  # ONLY the 2-row delta
    )
    model = get_peft_model(model, peft_cfg)
    
    # Memory safety
    model.config.use_cache = False
    
    # Sanity check logging
    logger.info(f"is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}")
    try:
        import bitsandbytes as bnbmod
        logger.info(f"bitsandbytes: {getattr(bnbmod, '__version__', 'unknown')}")
    except Exception as e:
        logger.error(f"bitsandbytes import error: {e}")
    
    # Sanity checks from your recommendation
    logger.info(f"is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}")
    style_delta_modules = [n for n,_ in model.named_modules() if "style_delta" in n]
    logger.info(f"style_delta modules found: {style_delta_modules}")
    
    # Confirm only LoRA + delta are trainable
    total, trainable = 0, 0
    hot = []
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad: 
            trainable += p.numel()
            if "style_delta" in n or "lora_" in n:
                hot.append(n)
    logger.info(f"trainable% = {100*trainable/total:.4f}%")
    logger.info(f"trainables (sample): {hot[:10]}")
    
    # One-step grad smoke test (no Trainer) if style delta exists  
    if added > 0 and style_delta_modules:
        logger.info("Running gradient smoke test for style delta...")
        model.train()
        ids = torch.tensor([[terse_id, verbose_id]], device=model.device)
        out = model(input_ids=ids, labels=ids)  # forces a loss
        out.loss.backward()
        try:
            delta_grad_norm = model.model.embed_tokens.style_delta.weight.grad.norm().item()
            logger.info(f"delta grad norm: {delta_grad_norm}")
        except AttributeError as e:
            logger.warning(f"Could not access style_delta grad: {e}")
        model.zero_grad()  # Clean up
    
    logger.info(f"CUDA mem allocated MB: {torch.cuda.memory_allocated()/2**20:.1f}")
    logger.info(f"CUDA mem reserved  MB: {torch.cuda.memory_reserved()/2**20:.1f}")
    logger.info(f"device map: {getattr(model, 'hf_device_map', None)}")
    
    return model, tokenizer

def load_dataset(tokenizer):
    """Load and format dataset for prompt-completion"""
    logger.info(f"Loading dataset from {DATASET_PATH}")
    
    # Load JSONL
    data = []
    with open(DATASET_PATH, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} examples")
    
    def to_prompt_completion(example):
        """Convert to prompt-completion with style tokens"""
        messages = example['messages'].copy()
        user_content = messages[0]['content']
        
        # Add style token
        if "briefly" in user_content.lower():
            style_token = "<STYLE_TERSE>"
        elif "detailed" in user_content.lower():
            style_token = "<STYLE_VERBOSE>"
        else:
            style_token = "<STYLE_TERSE>"
        
        messages[0]['content'] = f"{style_token}\n{user_content}"
        
        # Generate prompt using chat template
        prompt = tokenizer.apply_chat_template(
            [messages[0]], 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Completion with <|im_end|> appended
        completion = messages[1]['content'].rstrip("\n") + "<|im_end|>"
        
        return {"prompt": prompt, "completion": completion}
    
    # Convert all examples
    formatted_data = [to_prompt_completion(item) for item in data]
    dataset = Dataset.from_list(formatted_data)
    
    # Log sample
    sample = formatted_data[0]
    logger.info(f"Sample prompt ending: ...{sample['prompt'][-50:]}")
    logger.info(f"Sample completion: {sample['completion'][:100]}...")
    
    return dataset

def create_trainer(model, tokenizer, train_dataset):
    """Create trainer with A10G-optimized settings"""
    
    # TrainingArguments that fit A10G comfortably
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=42,
        per_device_train_batch_size=1,      # start at 1 for headroom
        gradient_accumulation_steps=16,     # back to 16 with proper QLoRA
        learning_rate=2e-4,
        num_train_epochs=2,
        optim="adamw_torch",  # Standard AdamW (more stable)
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="no",
        save_steps=500,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        dataloader_num_workers=2,           # conservative, avoids RAM thrash
        dataloader_pin_memory=True,
        report_to=[],
        logging_steps=25,
        remove_unused_columns=False,  # Critical for custom data format
    )
    
    # Custom data collator
    data_collator = PromptCompletionCollator(tokenizer, MAX_LEN)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    
    return trainer

def validate_model(model, tokenizer):
    """Test the trained model"""
    logger.info("Validating trained model...")
    
    test_cases = [
        {"role": "user", "content": "<STYLE_TERSE>\nWhat is 2+2?"},
        {"role": "user", "content": "<STYLE_VERBOSE>\nWhat is 2+2?"},
        {"role": "user", "content": "<STYLE_TERSE>\nWhat is machine learning?"},
        {"role": "user", "content": "<STYLE_VERBOSE>\nWhat is machine learning?"}
    ]
    
    model.eval()
    
    for test_case in test_cases:
        # Generate prompt
        prompt = tokenizer.apply_chat_template(
            [test_case], 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).cuda()
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=prompt,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][prompt.shape[-1]:], skip_special_tokens=True)
        token_count = len(tokenizer.encode(response))
        
        logger.info(f"Input: {test_case['content']}")
        logger.info(f"Output ({token_count} tokens): {response}")
        logger.info("---")

def main():
    """Main training pipeline with golden config"""
    logger.info("Starting Qwen QLoRA training - Golden Config")
    
    # Verify CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Download model from S3
    download_model_from_s3()
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer()
    train_dataset = load_dataset(tokenizer)
    trainer = create_trainer(model, tokenizer, train_dataset)
    
    # Log trainable parameters
    model.print_trainable_parameters()
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Validate
    validate_model(trainer.model, tokenizer)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()