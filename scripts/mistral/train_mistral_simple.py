#!/usr/bin/env python3
"""
Simple Mistral Training Script
Clean, focused training script based on proven Qwen approach
Expects pre-processed datasets from S3
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import boto3

# Heavy ML imports will be done lazily in functions that need them

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "mistralai/Mistral-7B-v0.3"
DEFAULT_S3_BUCKET = "asoba-llm-cache"
DEFAULT_DATASET_PREFIX = "datasets/mistral-verbosity"
DEFAULT_MODEL_PREFIX = "models/mistral-verbosity"

class MistralTrainer:
    """Simple Mistral trainer following Qwen pattern"""
    
    def __init__(self, train_dataset_s3: str, val_dataset_s3: str, output_dir: str,
                 model_name: str = DEFAULT_MODEL):
        self.train_dataset_s3 = train_dataset_s3
        self.val_dataset_s3 = val_dataset_s3  
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer: model={model_name}, output={output_dir}")

def download_dataset_from_s3(s3_path: str, local_dir: str) -> str:
    """Download dataset from S3 to local directory"""
    try:
        # Parse S3 path
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        s3_path_parts = s3_path[5:].split("/", 1)  # Remove s3:// prefix
        bucket = s3_path_parts[0]
        key = s3_path_parts[1]
        
        # Local file path
        local_file = Path(local_dir) / Path(key).name
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Download from S3
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket, key, str(local_file))
        
        logger.info(f"Downloaded {s3_path} to {local_file}")
        return str(local_file)
        
    except Exception as e:
        logger.error(f"Failed to download {s3_path}: {e}")
        raise

def get_training_config(model_name: str, max_seq_length: int = 1024, 
                       batch_size: int = 4) -> Dict[str, Any]:
    """Get simple, proven training configuration"""
    
    # Based on working Qwen configuration - safe, proven values
    config = {
        "output_dir": "./mistral_simple_output",
        "per_device_train_batch_size": min(batch_size, 4),  # Safe for A10G
        "per_device_eval_batch_size": min(batch_size, 4),
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "max_steps": -1,
        "learning_rate": 2e-4,  # Proven LoRA learning rate
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps", 
        "save_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
        "optim": "paged_adamw_8bit",
        "fp16": True,
        "report_to": "none",
        "run_name": f"mistral_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    logger.info("Using proven training configuration")
    return config

def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True) -> Tuple[Any, Any]:
    """Setup model and tokenizer with 4-bit quantization (Qwen pattern)"""
    
    # Import heavy ML libraries only when needed
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    
    # 4-bit quantization config (proven with Qwen)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        logger.info("Using 4-bit quantization")
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_4bit else torch.float16,
        trust_remote_code=True
    )
    
    # Setup LoRA (proven config from Qwen)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Proven rank
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    logger.info("Model and tokenizer setup complete")
    return model, tokenizer

def upload_model_to_s3(model_dir: str, s3_bucket: str, s3_prefix: str) -> str:
    """Upload trained model to S3"""
    try:
        model_path = Path(model_dir)
        if not model_path.exists():
            raise ValueError(f"Model directory does not exist: {model_dir}")
        
        s3_client = boto3.client('s3')
        
        # Upload all model files
        uploaded_files = []
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                s3_client.upload_file(str(file_path), s3_bucket, s3_key)
                uploaded_files.append(s3_key)
        
        s3_path = f"s3://{s3_bucket}/{s3_prefix}/"
        logger.info(f"Uploaded {len(uploaded_files)} model files to {s3_path}")
        return s3_path
        
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise

def preprocess_dataset(examples, tokenizer, max_length: int = 1024):
    """Preprocess dataset for training"""
    texts = []
    
    for text in examples["text"]:
        # Simple preprocessing - just tokenize the text
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
    )
    
    # Set labels (for causal LM, labels = input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Simple Mistral training")
    parser.add_argument("--train-dataset", required=True,
                       help="S3 path to training dataset")
    parser.add_argument("--val-dataset", required=True,
                       help="S3 path to validation dataset")
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                       help="Model name/path")
    parser.add_argument("--output-dir", default="./mistral_simple_output",
                       help="Output directory")
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET,
                       help="S3 bucket for model upload")
    parser.add_argument("--s3-prefix", default=DEFAULT_MODEL_PREFIX,
                       help="S3 prefix for model upload")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--local-only", action="store_true",
                       help="Skip S3 upload (local training only)")
    
    args = parser.parse_args()
    
    logger.info("Starting simple Mistral training")
    logger.info(f"Configuration: model={args.model_name}, batch_size={args.batch_size}")
    
    try:
        # Create trainer instance
        trainer = MistralTrainer(
            train_dataset_s3=args.train_dataset,
            val_dataset_s3=args.val_dataset,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
        
        # Download datasets
        logger.info("Downloading datasets from S3...")
        train_file = download_dataset_from_s3(args.train_dataset, args.output_dir)
        val_file = download_dataset_from_s3(args.val_dataset, args.output_dir)
        
        # Load datasets
        from datasets import load_dataset
        logger.info("Loading datasets...")
        train_dataset = load_dataset('json', data_files=train_file, split='train')
        val_dataset = load_dataset('json', data_files=val_file, split='train')
        
        logger.info(f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation examples")
        
        # Setup model and tokenizer
        logger.info("Setting up model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(args.model_name)
        
        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        train_dataset = train_dataset.map(
            lambda x: preprocess_dataset(x, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            lambda x: preprocess_dataset(x, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        # Training configuration
        training_config = get_training_config(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size
        )
        training_config["output_dir"] = args.output_dir
        
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        training_args = TrainingArguments(**training_config)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Create trainer
        trainer_obj = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer_obj.train()
        
        # Save the model
        logger.info("Saving model...")
        trainer_obj.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Upload to S3 if not local-only
        if not args.local_only:
            logger.info("Uploading model to S3...")
            s3_path = upload_model_to_s3(
                model_dir=args.output_dir,
                s3_bucket=args.s3_bucket,
                s3_prefix=args.s3_prefix
            )
            logger.info(f"Model uploaded to: {s3_path}")
        
        logger.info("✅ Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())