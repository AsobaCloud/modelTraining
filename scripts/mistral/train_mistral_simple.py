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
import atexit
import threading
import time

# Setup logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Auto-deploy dependencies if missing
def ensure_dependencies():
    import subprocess
    
    script_dir = Path(__file__).parent
    shared_dir = script_dir.parent / "shared"
    repo_root = script_dir.parent.parent
    requirements_file = repo_root / "requirements.txt"
    
    # Skip auto-install to prevent stalls
    if requirements_file.exists():
        logger.info("Skipping auto-install of requirements (already installed)")
    
    # Check and copy heartbeat_manager.py
    if not (script_dir / "heartbeat_manager.py").exists():
        if (shared_dir / "heartbeat_manager.py").exists():
            import shutil
            shutil.copy2(shared_dir / "heartbeat_manager.py", script_dir / "heartbeat_manager.py")
            logger.info("Auto-deployed heartbeat_manager.py")
    
    # Check and copy heartbeat_callback.py  
    heartbeat_callback_path = script_dir / "heartbeat_callback.py"
    if not heartbeat_callback_path.exists():
        # Create minimal heartbeat_callback.py if missing
        heartbeat_callback_path.write_text('''#!/usr/bin/env python3
"""HeartbeatCallback - Auto-generated minimal version"""
import json
import boto3
from datetime import datetime
from transformers import TrainerCallback
class HeartbeatCallback(TrainerCallback):
    def __init__(self, heartbeat_manager, update_every_steps=20):
        self.hb = heartbeat_manager
        self.update_every_steps = update_every_steps
        self._last_step = -1
        self.s3_client = boto3.client('s3')
    def on_log(self, args, state, control, logs, **kwargs):
        step = int(state.global_step or 0)
        if step - self._last_step >= self.update_every_steps:
            self._last_step = step
            loss = logs.get('loss', 0)
            self.hb.update_phase("training", "active", f"step={step} loss={loss:.4f}")
            # Write separate progress.json for dual heartbeat
            self._write_progress(step, loss, logs)
    def _write_progress(self, step, loss, logs):
        try:
            progress_key = self.hb.s3_key.replace('metadata.json', 'progress.json')
            progress_data = {
                "step": step,
                "loss": loss,
                "logs": logs,
                "timestamp": datetime.now().isoformat()
            }
            self.s3_client.put_object(
                Bucket=self.hb.s3_bucket,
                Key=progress_key,
                Body=json.dumps(progress_data),
                ContentType='application/json'
            )
        except Exception as e:
            pass  # Don't fail training on progress write failure
''')
        logger.info("Auto-generated heartbeat_callback.py")

ensure_dependencies()

# Import monitoring components (tolerate absence)
try:
    from heartbeat_manager import HeartbeatManager
    from heartbeat_callback import HeartbeatCallback
except Exception:
    HeartbeatManager = None
    HeartbeatCallback = None

# Heavy ML imports will be done lazily in functions that need them

# Configuration
DEFAULT_MODEL = "/mnt/training/models/mistral-7b-v0.3"
DEFAULT_S3_BUCKET = "asoba-llm-cache"
DEFAULT_DATASET_PREFIX = "datasets/mistral-verbosity"
DEFAULT_MODEL_PREFIX = "models/mistral-verbosity"

# Global variables for error handling and monitoring
current_run_id = None
s3_bucket = None
monitoring_prefix = "training-runs"
heartbeat_manager = None
log_streamer = None

class LogStreamer:
    """Background thread that streams training log to S3"""
    def __init__(self, log_file_path, s3_bucket, s3_key, interval_seconds=30):
        self.log_file_path = log_file_path
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.interval = interval_seconds
        self.running = False
        self.thread = None
        self.s3_client = boto3.client('s3')
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            
    def _stream_loop(self):
        while self.running:
            try:
                self._upload_log_tail()
            except Exception as e:
                logger.debug(f"Log streaming error: {e}")
            time.sleep(self.interval)
            
    def _upload_log_tail(self):
        """Upload last 1000 lines of log to S3"""
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r') as f:
                    lines = f.readlines()
                    tail_lines = lines[-1000:]  # Last 1000 lines
                    
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=self.s3_key,
                    Body=''.join(tail_lines)
                )
        except Exception:
            pass  # Silent failure - don't interrupt training

def write_error_sentinel(error_msg: str):
    """Write error sentinel to S3 for production monitoring with full context"""
    if current_run_id and s3_bucket:
        try:
            import traceback
            import socket
            
            # Gather full context
            error_data = {
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
                "run_id": current_run_id,
                "instance_id": socket.gethostname(),
                "context": {
                    "model_name": getattr(main, 'model_name', 'unknown'),
                    "train_dataset": getattr(main, 'train_dataset_s3', 'unknown'),
                    "val_dataset": getattr(main, 'val_dataset_s3', 'unknown')
                }
            }
            
            s3_client = boto3.client('s3')
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=f"{monitoring_prefix}/{current_run_id}/_error",
                Body=json.dumps(error_data, indent=2)
            )
            logger.info(f"Error sentinel written to S3: {error_msg}")
        except Exception as e:
            logger.warning(f"Failed to write error sentinel: {e}")

def write_completion_marker():
    """Write completion marker to S3 for production monitoring"""
    if current_run_id and s3_bucket:
        try:
            s3_client = boto3.client('s3')
            completion_msg = f"Training completed successfully at {datetime.now().isoformat()}"
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=f"{monitoring_prefix}/{current_run_id}/_complete",
                Body=completion_msg
            )
            logger.info("Completion marker written to S3")
        except Exception as e:
            logger.warning(f"Failed to write completion marker: {e}")

def resolve_model_source(model_tag: str, bucket: str = DEFAULT_S3_BUCKET, region: str = "us-east-1", local_root: str = "/mnt/training/models") -> str:
    """Programmatically resolve model source using resolve_model.sh"""
    import subprocess
    
    # Find resolve_model.sh script - try multiple locations
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent / "resolve_model.sh",  # Original expected location
        Path("/mnt/training/resolve_model.sh"),   # Direct location on instance
        script_dir.parent.parent / "scripts" / "resolve_model.sh",  # Repository structure
    ]
    
    resolver_path = None
    for path in possible_paths:
        if path.exists():
            resolver_path = path
            break
    
    if resolver_path is None:
        raise FileNotFoundError(f"resolve_model.sh not found in any of: {[str(p) for p in possible_paths]}")
    
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
        # If the script failed, read its full output from the temporary log file on the local instance
        try:
            with open("/tmp/resolve_model_output.log", "r") as f:
                remote_log_content = f.read()
            
            error_msg = f"Model resolution failed: {result.stderr.strip()}. Remote script output:\n{remote_log_content}"
        except FileNotFoundError:
            error_msg = f"Model resolution failed: {result.stderr.strip()}. Remote log file not found."
        except Exception as e:
            error_msg = f"Model resolution failed: {result.stderr.strip()}. Failed to read remote log: {e}"
        
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Return local path
    local_path = f"{local_root}/{model_tag}"
    logger.info(f"Model resolved to: {local_path}")
    return local_path

def handle_error(error_msg: str):
    """Handle errors with monitoring integration"""
    write_error_sentinel(error_msg)
    logger.error(error_msg)
    sys.exit(1)

# Don't register completion marker automatically - only on success

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

def get_training_config(model_name: str, max_seq_length: int = 512, 
                       batch_size: int = 1) -> Dict[str, Any]:
    """Get simple, proven training configuration"""
    
    # Based on working Qwen configuration - safe, proven values
    config = {
        "output_dir": "/mnt/training/mistral_simple_output",
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 16,  # Compensate with more accumulation
        "num_train_epochs": 3,
        "max_steps": -1,
        "learning_rate": 2e-4,  # Proven LoRA learning rate
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps", 
        "save_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "dataloader_num_workers": 0,  # Single worker for reliability
        "remove_unused_columns": False,  # Critical for IterableDataset with lazy tokenization
        "optim": "paged_adamw_8bit",
        # Ampere (A10): prefer bf16 to match model dtype
        "bf16": True,
        "fp16": False,
        "report_to": "none",
        "run_name": f"mistral_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    logger.info("Using proven training configuration")
    return config

def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True, local_files_only: bool = False) -> Tuple[Any, Any]:
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_4bit else torch.float16,
        local_files_only=local_files_only
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
    """Preprocess dataset for training - HF best practices for memory efficiency"""
    # Direct tokenization of examples["text"] - no intermediate Python lists
    tokenized = tokenizer(
        examples["text"],  # Pass directly to tokenizer (not via texts=[])
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
    )
    
    # Labels as view of input_ids, not copy (HF best practice for causal LM)
    tokenized["labels"] = tokenized["input_ids"]  # View, not .copy()
    
    return tokenized

def memory_safe_map(dataset, func, **kwargs):
    """
    Memory-safe wrapper around Dataset.map that:
    • writes each batch to disk immediately
    • keeps no partial state in memory
    """
    return dataset.map(
        func,
        batched=True,
        batch_size=512,              # small enough to stay cache-friendly
        writer_batch_size=512,       # flush each batch to Arrow immediately
        keep_in_memory=False,        # critical: forces disk-backed processing
        load_from_cache_file=False,  # always rebuild when code changes
        num_proc=1,                  # single process to avoid subprocess crashes
        **kwargs
    )

def main():
    """Main training function"""
    global current_run_id, s3_bucket, heartbeat_manager
    
    parser = argparse.ArgumentParser(description="Simple Mistral training")
    parser.add_argument("--train-dataset", required=True,
                       help="S3 path to training dataset")
    parser.add_argument("--val-dataset", required=True,
                       help="S3 path to validation dataset")
    parser.add_argument("--model-tag", default="mistral-7b-v0.3",
                       help="Model tag to resolve from S3 (e.g., mistral-7b-v0.3)")
    parser.add_argument("--model-name", 
                       help="Direct model path (overrides --model-tag)")
    parser.add_argument("--output-dir", default="/mnt/training/mistral_simple_output",
                       help="Output directory")
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET,
                       help="S3 bucket for model upload")
    parser.add_argument("--s3-prefix", default=DEFAULT_MODEL_PREFIX,
                       help="S3 prefix for model upload")
    parser.add_argument("--region", default="us-east-1",
                       help="AWS region")
    parser.add_argument("--max-seq-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--local-only", action="store_true",
                       help="Skip S3 upload (local training only)")
    parser.add_argument("--run-id", help="Training run ID for monitoring")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume training from checkpoint if available")
    parser.add_argument("--local-files-only", action="store_true", default=False,
                       help="Use only locally cached model files (no downloads)")
    
    args = parser.parse_args()
    
    # Set globals for error handling  
    current_run_id = args.run_id or f"mistral-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    s3_bucket = args.s3_bucket
    
    # Resolve model path programmatically
    if args.model_name:
        # Direct path provided, use as-is
        model_path = args.model_name
        logger.info(f"Using direct model path: {model_path}")
    else:
        # Resolve model from tag
        if heartbeat_manager:
            heartbeat_manager.update_phase("initialization", "model_resolution", f"Resolving model '{args.model_tag}'")
        try:
            model_path = resolve_model_source(
                model_tag=args.model_tag,
                bucket=args.s3_bucket,
                region=args.region,
                local_root="/mnt/training/models"
            )
        except Exception as e:
            handle_error(f"Model resolution failed: {e}")
    
    # Use run-specific output directory to prevent collisions
    if args.output_dir == "/mnt/training/mistral_simple_output":  # default
        args.output_dir = f"/mnt/training/outputs/{current_run_id}"
    
    # Initialize heartbeat monitoring
    if HeartbeatManager and s3_bucket and current_run_id:
        s3_key = f"{monitoring_prefix}/{current_run_id}/metadata.json"
        heartbeat_manager = HeartbeatManager(s3_bucket, s3_key, interval_seconds=60)  # 1 minute intervals
        heartbeat_manager.start()
        heartbeat_manager.update_phase("initialization", "starting", "Setting up training environment")
        
        # Start log streaming to S3
        log_file_path = "training_final.log"  # Match the actual log file name
        log_s3_key = f"{monitoring_prefix}/{current_run_id}/logs/training_log_latest.txt"
        log_streamer = LogStreamer(log_file_path, s3_bucket, log_s3_key, interval_seconds=30)
        log_streamer.start()
        
        logger.info(f"Monitoring started: s3://{s3_bucket}/{s3_key}")
    else:
        logger.info("Running without S3 monitoring - HeartbeatManager unavailable")
    
    logger.info("Starting simple Mistral training")
    logger.info(f"Configuration: model={model_path}, batch_size={args.batch_size}")
    logger.info(f"Run ID: {current_run_id}")
    
    try:
        # Create trainer instance
        trainer = MistralTrainer(
            train_dataset_s3=args.train_dataset,
            val_dataset_s3=args.val_dataset,
            output_dir=args.output_dir,
            model_name=model_path
        )
        
        # Check for pre-tokenized data first
        import datasets
        TOKENIZED_DIR = os.getenv("TOKENIZED_DIR")
        
        if TOKENIZED_DIR:
            # Option A: Use pre-tokenized Arrow data
            if heartbeat_manager:
                heartbeat_manager.update_phase("data_prep", "loading", "Loading pre-tokenized Arrow data")
            logger.info(f"Loading pre-tokenized data from: {TOKENIZED_DIR}")
            
            ds = datasets.load_from_disk(TOKENIZED_DIR)
            train_dataset = ds["train"]
            val_dataset = ds.get("validation")
            
            # Set format after loading
            train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            if val_dataset:
                val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            
            # Load tokenizer for model setup (but data is already tokenized)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # No data collator needed - data is already packed and labeled
            data_collator = None
            
            logger.info(f"Pre-tokenized data loaded: {len(train_dataset):,} train examples")
            if val_dataset:
                logger.info(f"Pre-tokenized validation: {len(val_dataset):,} examples")
            
        else:
            # Option B: Original streaming approach with reduced buffer
            # Download datasets
            if heartbeat_manager:
                heartbeat_manager.update_phase("data_prep", "downloading", "Downloading datasets from S3")
            logger.info("Downloading datasets from S3...")
            train_file = download_dataset_from_s3(args.train_dataset, args.output_dir)
            val_file = download_dataset_from_s3(args.val_dataset, args.output_dir)
            
            # Load datasets with STREAMING (no RAM materialization)
            if heartbeat_manager:
                heartbeat_manager.update_phase("data_prep", "loading", "Loading datasets with streaming")
            from datasets import load_dataset
            logger.info("Loading datasets with streaming (reduced buffer size)...")
            raw_train = load_dataset('json', data_files=train_file, split='train', streaming=True)
            raw_val = load_dataset('json', data_files=val_file, split='train', streaming=True)
            
            # Setup tokenizer
            if heartbeat_manager:
                heartbeat_manager.update_phase("data_prep", "preprocessing", "Setting up lazy tokenization")
            logger.info("Setting up tokenizer for lazy tokenization...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Lazy tokenization function
            def lazy_tokenize(example):
                ids = tokenizer(
                    example["text"],
                    truncation=True,
                    max_length=args.max_seq_length,
                    padding=False
                )["input_ids"]
                return {"input_ids": ids, "labels": ids}
            
            # Apply lazy tokenization with SMALL shuffle buffer
            logger.info("Setting up lazy tokenization with small shuffle buffer...")
            train_dataset = raw_train.map(lazy_tokenize)
            train_dataset = train_dataset.shuffle(buffer_size=1000)  # Reduced from 10,000
            
            val_dataset = raw_val.map(lazy_tokenize)
            
            # Standard data collator for streaming
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM
                pad_to_multiple_of=8
            )
            
            # Free raw references
            del train_file, val_file, raw_train, raw_val
            import gc; gc.collect()
            logger.info("Released raw dataset references - using reduced memory footprint")
        
        # NOW load the full model (after tokenization is complete)
        if heartbeat_manager:
            heartbeat_manager.update_phase("model_setup", "loading", "Setting up model for training")
        logger.info("Setting up model for training...")
        model, _ = setup_model_and_tokenizer(model_path, local_files_only=args.local_files_only)
        
        # Training configuration
        training_config = get_training_config(
            model_name=model_path,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size
        )
        training_config["output_dir"] = args.output_dir
        
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        training_args = TrainingArguments(**training_config)
        
        # Data collator (may be None if using pre-tokenized data)
        if 'data_collator' not in locals():
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM, not masked LM
                pad_to_multiple_of=8
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
        
        # Add HeartbeatCallback for real progress tracking
        if heartbeat_manager and HeartbeatCallback:
            trainer_obj.add_callback(HeartbeatCallback(heartbeat_manager, update_every_steps=20))
            logger.info("HeartbeatCallback added for real-time progress tracking")
        else:
            logger.info("Training without HeartbeatCallback - progress tracking via logs only")
        
        # Train the model
        if heartbeat_manager:
            heartbeat_manager.update_phase("training", "active", f"Training {args.model_name}")
            heartbeat_manager.set_status("training")
        
        if args.resume:
            logger.info("Starting training with resume from checkpoint...")
            trainer_obj.train(resume_from_checkpoint=True)
        else:
            logger.info("Starting fresh training...")
            trainer_obj.train()
        
        # Save the model
        if heartbeat_manager:
            heartbeat_manager.update_phase("training", "saving", "Saving trained model")
        logger.info("Saving model...")
        trainer_obj.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Upload to S3 if not local-only
        if not args.local_only:
            if heartbeat_manager:
                heartbeat_manager.update_phase("upload", "s3_upload", "Uploading model to S3")
            logger.info("Uploading model to S3...")
            s3_path = upload_model_to_s3(
                model_dir=args.output_dir,
                s3_bucket=args.s3_bucket,
                s3_prefix=args.s3_prefix
            )
            logger.info(f"Model uploaded to: {s3_path}")
        
        # Success marker only after successful save/upload
        try:
            write_completion_marker()
        except Exception as _e:
            logger.warning(f"Failed to write completion marker: {_e}")
        
        # Mark as completed
        if heartbeat_manager:
            heartbeat_manager.set_status("completed")
            heartbeat_manager.update_phase("complete", "success", "Training completed successfully")
            heartbeat_manager.stop()
        if log_streamer:
            log_streamer.stop()
        
        logger.info("✅ Training completed successfully!")
        return 0
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        if heartbeat_manager:
            heartbeat_manager.set_status("failed")
            heartbeat_manager.update_phase("error", "exception", error_msg)
            heartbeat_manager.stop()
        if log_streamer:
            log_streamer.stop()
        handle_error(error_msg)
        return 1

if __name__ == "__main__":
    sys.exit(main())