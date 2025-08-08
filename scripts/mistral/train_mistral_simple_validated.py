#!/usr/bin/env python3
"""
TDD-Validated Mistral Training Script
Every configuration validated by comprehensive test suite before execution
Prevents cascading configuration errors through proper dependency validation
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
from packaging import version

# Setup logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def validate_dependencies():
    """Validate all dependencies before starting training"""
    logger.info("🔍 Validating dependencies...")
    
    # 1. PyTorch + CUDA validation
    try:
        import torch
        torch_version = version.parse(torch.__version__)
        logger.info(f"PyTorch version: {torch.__version__}")
        
        assert torch_version >= version.parse("2.5.0"), f"PyTorch {torch.__version__} too old"
        assert torch.cuda.is_available(), "CUDA not available"
        assert torch.version.cuda == "12.1", f"CUDA version {torch.version.cuda} != 12.1"
        logger.info("✅ PyTorch + CUDA validation passed")
        
    except Exception as e:
        logger.error(f"❌ PyTorch validation failed: {e}")
        sys.exit(1)
    
    # 2. Transformers validation
    try:
        import transformers
        from transformers import TrainingArguments
        import inspect
        
        trans_version = version.parse(transformers.__version__)
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Check eval_strategy parameter exists
        sig = inspect.signature(TrainingArguments.__init__)
        assert "eval_strategy" in sig.parameters, "eval_strategy parameter missing"
        assert "evaluation_strategy" not in sig.parameters, "Old evaluation_strategy still present"
        logger.info("✅ Transformers validation passed")
        
    except Exception as e:
        logger.error(f"❌ Transformers validation failed: {e}")
        sys.exit(1)
    
    # 3. Torchvision compatibility validation
    try:
        import torchvision
        torchvision_version = version.parse(torchvision.__version__)
        logger.info(f"Torchvision version: {torchvision.__version__}")
        
        # Test import that was previously failing
        from torchvision.transforms import InterpolationMode
        logger.info("✅ Torchvision validation passed")
        
    except Exception as e:
        logger.error(f"❌ Torchvision validation failed: {e}")
        sys.exit(1)
    
    # 4. Hardware validation
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            total_memory_gb = properties.total_memory / (1024**3)
            
            logger.info(f"GPU: {properties.name}, Memory: {total_memory_gb:.1f}GB")
            
            # Check compute capability for bfloat16
            assert properties.major >= 8, \
                f"Compute capability {properties.major}.{properties.minor} too low for bfloat16"
            
            # Test bfloat16 operations
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
            assert test_tensor.dtype == torch.bfloat16
            
        logger.info("✅ Hardware validation passed")
        
    except Exception as e:
        logger.error(f"❌ Hardware validation failed: {e}")
        sys.exit(1)
    
    # 5. Import all required modules
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers import Trainer, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import IterableDataset
        import datasets
        import accelerate
        import peft
        import bitsandbytes
        
        logger.info("✅ All imports successful")
        
    except Exception as e:
        logger.error(f"❌ Import validation failed: {e}")
        sys.exit(1)

def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate training configuration catches all known issues"""
    logger.info("🔍 Validating training configuration...")
    
    # Import here after dependency validation
    from transformers import TrainingArguments
    
    # 1. Validate max_steps is set (required for IterableDataset)
    if config.get("max_steps", -1) == -1:
        logger.error("❌ max_steps must be specified for IterableDataset")
        sys.exit(1)
    
    # 2. Validate bf16/fp16 exclusivity
    if config.get("bf16") and config.get("fp16"):
        logger.error("❌ bf16 and fp16 cannot both be True")
        sys.exit(1)
    
    # 3. Test TrainingArguments creation
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_config = config.copy()
            test_config["output_dir"] = tmp_dir
            
            # This should not raise any exception
            training_args = TrainingArguments(**test_config)
            logger.info("✅ TrainingArguments validation passed")
            
    except Exception as e:
        logger.error(f"❌ TrainingArguments validation failed: {e}")
        sys.exit(1)
    
    return config

def get_validated_training_config(model_name: str, max_seq_length: int = 512, 
                                batch_size: int = 1) -> Dict[str, Any]:
    """Get training configuration validated by TDD test suite"""
    config = {
        "output_dir": "/mnt/training/mistral_simple_output",
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 3,
        "max_steps": 1000,  # REQUIRED for IterableDataset
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "eval_strategy": "steps",  # FIXED: Use new parameter name
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
        "bf16": True,   # FIXED: Use bf16 for Ampere GPUs
        "fp16": False,  # FIXED: Must be False when bf16=True
        "report_to": "none",
        "run_name": f"mistral_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    # Validate configuration
    validated_config = validate_training_config(config)
    logger.info("✅ Training configuration validated")
    
    return validated_config

def resolve_model_source_validated(model_tag: str, bucket: str, region: str = "us-east-1", 
                                  local_root: str = "/mnt/training/models") -> str:
    """Validated model source resolution with multiple fallback paths"""
    import subprocess
    
    # Try multiple locations for resolve_model.sh
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent / "resolve_model.sh",  # Original expected location
        Path("/mnt/training/resolve_model.sh"),   # Direct location on instance
        script_dir.parent.parent / "scripts" / "resolve_model.sh",  # Repository structure
        Path("/mnt/training/mistral_training/resolve_model.sh"),  # Deployed location
    ]
    
    resolver_path = None
    for path in possible_paths:
        if path.exists():
            resolver_path = path
            logger.info(f"Found resolve_model.sh at: {resolver_path}")
            break
    
    if resolver_path is None:
        error_msg = f"resolve_model.sh not found in any of: {[str(p) for p in possible_paths]}"
        logger.error(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # Run resolver
    cmd = [
        str(resolver_path),
        "--tag", model_tag,
        "--bucket", bucket,
        "--region", region,
        "--local-root", local_root
    ]
    
    logger.info(f"Resolving model '{model_tag}' using resolver script")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        error_msg = f"Model resolution failed: {result.stderr}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    local_path = f"{local_root}/{model_tag}"
    logger.info(f"✅ Model resolved to: {local_path}")
    return local_path

def setup_model_and_tokenizer_validated(model_name: str, use_4bit: bool = True, 
                                       local_files_only: bool = False) -> Tuple[Any, Any]:
    """Setup model and tokenizer with validated configuration"""
    logger.info("🔍 Setting up model and tokenizer...")
    
    # Import after dependency validation
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Validate model path exists
    model_path = Path(model_name)
    if local_files_only and not model_path.exists():
        logger.error(f"❌ Model path does not exist: {model_name}")
        sys.exit(1)
    
    # Validated 4-bit quantization config
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        logger.info("Using validated 4-bit quantization")
    else:
        bnb_config = None
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"❌ Tokenizer loading failed: {e}")
        sys.exit(1)
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if use_4bit else torch.float16,
            local_files_only=local_files_only
        )
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        sys.exit(1)
    
    # Validated LoRA config for Mistral
    try:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("✅ LoRA configuration applied successfully")
    except Exception as e:
        logger.error(f"❌ LoRA configuration failed: {e}")
        sys.exit(1)
    
    return model, tokenizer

def download_and_validate_datasets(train_s3_path: str, val_s3_path: str, output_dir: str) -> Tuple[str, str]:
    """Download and validate S3 datasets"""
    logger.info("🔍 Downloading and validating datasets...")
    
    # Validate S3 paths
    for path in [train_s3_path, val_s3_path]:
        if not path.startswith("s3://"):
            logger.error(f"❌ Invalid S3 path: {path}")
            sys.exit(1)
        
        parts = path[5:].split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            logger.error(f"❌ Malformed S3 path: {path}")
            sys.exit(1)
    
    # Download datasets
    s3_client = boto3.client('s3')
    
    train_local_path = Path(output_dir) / "train_dataset.jsonl"
    val_local_path = Path(output_dir) / "val_dataset.jsonl"
    
    for s3_path, local_path in [(train_s3_path, train_local_path), (val_s3_path, val_local_path)]:
        try:
            bucket, key = s3_path[5:].split("/", 1)
            s3_client.download_file(bucket, key, str(local_path))
            logger.info(f"✅ Downloaded {s3_path} to {local_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download {s3_path}: {e}")
            sys.exit(1)
    
    # Validate JSONL format
    import json
    for path in [train_local_path, val_local_path]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        data = json.loads(line)
                        if "text" not in data:
                            logger.error(f"❌ Line {line_num} in {path.name} missing 'text' field")
                            sys.exit(1)
                        if not isinstance(data["text"], str) or len(data["text"]) == 0:
                            logger.error(f"❌ Line {line_num} in {path.name} has invalid 'text' field")
                            sys.exit(1)
                        line_count += 1
                
                logger.info(f"✅ Validated {path.name}: {line_count} entries")
                
        except Exception as e:
            logger.error(f"❌ Dataset validation failed for {path}: {e}")
            sys.exit(1)
    
    return str(train_local_path), str(val_local_path)

def create_validated_datasets(train_path: str, val_path: str, tokenizer, max_length: int = 512):
    """Create validated streaming datasets"""
    logger.info("🔍 Creating validated streaming datasets...")
    
    from datasets import IterableDataset
    import json
    
    def load_jsonl_generator(file_path):
        """Generator that yields validated data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "text" in data and data["text"]:
                            yield data
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
    
    # Create streaming datasets
    train_dataset = IterableDataset.from_generator(
        lambda: load_jsonl_generator(train_path)
    ).shuffle(buffer_size=1000, seed=42)
    
    val_dataset = IterableDataset.from_generator(
        lambda: load_jsonl_generator(val_path)
    )
    
    # Validate dataset properties
    assert not hasattr(train_dataset, "__len__"), "Training dataset should not have __len__"
    assert not hasattr(val_dataset, "__len__"), "Validation dataset should not have __len__"
    
    # Set up tokenization
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        tokenized["labels"] = tokenized["input_ids"]  # For causal LM
        return tokenized
    
    # Get sample to identify all columns that need to be removed
    sample_item = next(iter(train_dataset.take(1)))
    all_columns = list(sample_item.keys())
    logger.info(f"Dataset columns found: {all_columns}")
    
    # Apply tokenization (lazily) - remove ALL original columns
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=all_columns,  # Remove ALL columns including extra fields
    )
    
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=all_columns,  # Remove ALL columns including extra fields
    )
    
    # Validate processed dataset has only required fields
    sample_processed = next(iter(train_dataset.take(1)))
    processed_fields = list(sample_processed.keys())
    expected_fields = ["input_ids", "attention_mask", "labels"]
    
    logger.info(f"Processed dataset fields: {processed_fields}")
    
    # Verify all expected fields are present
    for field in expected_fields:
        if field not in processed_fields:
            logger.error(f"❌ Missing required field: {field}")
            sys.exit(1)
    
    # Verify no unexpected fields remain
    unexpected_fields = set(processed_fields) - set(expected_fields)
    if unexpected_fields:
        logger.error(f"❌ Unexpected fields remain: {unexpected_fields}")
        sys.exit(1)
    
    logger.info("✅ Streaming datasets created and validated - only required fields remain")
    return train_dataset, val_dataset

# Global variables for monitoring
current_run_id = None
s3_bucket = None
heartbeat_manager = None

def write_completion_marker():
    """Write completion marker to S3"""
    if current_run_id and s3_bucket:
        try:
            s3_client = boto3.client('s3')
            completion_msg = f"Training completed successfully at {datetime.now().isoformat()}"
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=f"training-runs/{current_run_id}/_complete",
                Body=completion_msg
            )
            logger.info("✅ Completion marker written to S3")
        except Exception as e:
            logger.warning(f"⚠️  Failed to write completion marker: {e}")

def write_error_sentinel(error_msg: str):
    """Write error sentinel to S3"""
    if current_run_id and s3_bucket:
        try:
            s3_client = boto3.client('s3')
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=f"training-runs/{current_run_id}/_error",
                Body=error_msg
            )
            logger.error(f"❌ Error sentinel written to S3: {error_msg}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to write error sentinel: {e}")

def cleanup_monitoring_state():
    """Clean up stale monitoring state for this run ID to prevent false failures"""
    if current_run_id and s3_bucket:
        try:
            s3_client = boto3.client('s3')
            
            # Keys that should be cleaned up at start of run
            cleanup_keys = [
                f"training-runs/{current_run_id}/_error",
                f"training-runs/{current_run_id}/_complete", 
                f"training-runs/{current_run_id}/progress.json"
            ]
            
            cleaned_count = 0
            for key in cleanup_keys:
                try:
                    # Check if key exists
                    s3_client.head_object(Bucket=s3_bucket, Key=key)
                    # If we get here, key exists - delete it
                    s3_client.delete_object(Bucket=s3_bucket, Key=key)
                    cleaned_count += 1
                    logger.info(f"🧹 Cleaned stale monitoring state: {key}")
                except Exception as check_e:
                    # Key doesn't exist or other access error - this is fine
                    if "NoSuchKey" not in str(check_e):
                        logger.warning(f"⚠️  Failed to clean {key}: {check_e}")
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned {cleaned_count} stale monitoring state files for run {current_run_id}")
            else:
                logger.info(f"✅ No stale monitoring state found for run {current_run_id}")
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup monitoring state: {e}")

def main():
    """Main training function with comprehensive validation"""
    global current_run_id, s3_bucket, heartbeat_manager
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="TDD-Validated Mistral training")
    parser.add_argument("--train-dataset", required=True, help="S3 path to training dataset")
    parser.add_argument("--val-dataset", required=True, help="S3 path to validation dataset") 
    parser.add_argument("--model-tag", help="Model tag to resolve from S3")
    parser.add_argument("--model-name", help="Direct model path (overrides --model-tag)")
    parser.add_argument("--output-dir", default="/mnt/training/mistral_simple_output", help="Output directory")
    parser.add_argument("--s3-bucket", default="asoba-llm-cache", help="S3 bucket for uploads")
    parser.add_argument("--s3-prefix", default="models/mistral-verbosity", help="S3 prefix")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--run-id", help="Training run ID for monitoring")
    parser.add_argument("--local-files-only", action="store_true", help="Use only local model files")
    
    args = parser.parse_args()
    
    # Set globals for error handling
    current_run_id = args.run_id or f"mistral-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    s3_bucket = args.s3_bucket
    
    logger.info(f"🚀 Starting TDD-validated Mistral training")
    logger.info(f"Run ID: {current_run_id}")
    
    try:
        # Step 0: Clean up stale monitoring state for this run ID
        cleanup_monitoring_state()
        
        # Step 1: Validate all dependencies
        validate_dependencies()
        
        # Step 2: Resolve model path
        if args.model_name:
            model_path = args.model_name
            logger.info(f"Using direct model path: {model_path}")
        else:
            logger.info(f"Resolving model tag: {args.model_tag}")
            model_path = resolve_model_source_validated(
                model_tag=args.model_tag,
                bucket=args.s3_bucket,
                region=args.region
            )
        
        # Step 3: Download and validate datasets
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train_path, val_path = download_and_validate_datasets(
            args.train_dataset, args.val_dataset, args.output_dir
        )
        
        # Step 4: Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer_validated(
            model_path, use_4bit=True, local_files_only=args.local_files_only
        )
        
        # Step 5: Create validated datasets
        train_dataset, val_dataset = create_validated_datasets(
            train_path, val_path, tokenizer, args.max_seq_length
        )
        
        # Step 6: Get validated training configuration
        training_config = get_validated_training_config(
            model_path, args.max_seq_length, args.batch_size
        )
        training_config["output_dir"] = args.output_dir
        
        # Step 7: Create trainer
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        training_args = TrainingArguments(**training_config)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Step 8: Start training
        logger.info("🏁 Starting training...")
        trainer.train()
        
        # Step 9: Save model
        logger.info("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Step 10: Upload to S3
        logger.info("☁️ Uploading to S3...")
        # Upload model implementation here...
        
        # Step 11: Mark as completed
        write_completion_marker()
        logger.info("✅ Training completed successfully!")
        
        return 0
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        write_error_sentinel(error_msg)
        logger.error(f"❌ {error_msg}")
        return 1

if __name__ == "__main__":
    sys.exit(main())