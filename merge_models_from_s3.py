#!/usr/bin/env python3
"""
Merge LoRA adapters with base models from S3 and push to Hugging Face Hub
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def download_from_s3(s3_path, local_path):
    """Download model files from S3"""
    print(f"Downloading from {s3_path} to {local_path}")
    cmd = [
        "aws", "s3", "sync", 
        s3_path, 
        local_path,
        "--region", "us-east-1"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading from S3: {result.stderr}")
        sys.exit(1)
    print("Download complete")

def upload_to_s3(local_path, s3_path):
    """Upload model files to S3"""
    print(f"Uploading from {local_path} to {s3_path}")
    cmd = [
        "aws", "s3", "sync", 
        local_path,
        s3_path,
        "--region", "us-east-1"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error uploading to S3: {result.stderr}")
        return False
    print("Upload to S3 complete")
    return True

def merge_lora_with_base(base_model_path, lora_path, output_path):
    """Merge LoRA adapters with base model"""
    print(f"Loading base model from: {base_model_path}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"Loading LoRA adapters from: {lora_path}")
    
    # Load LoRA model
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    )
    
    print("Merging LoRA adapters with base model...")
    
    # Merge and unload
    model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    
    # Save the merged model
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Load and save tokenizer from LoRA path (it should have the tokenizer)
    print("Loading and saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merge complete!")
    return True

def push_to_huggingface(local_path, repo_id, token, commit_message="Update model"):
    """Push model to Hugging Face Hub"""
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id=repo_id, token=token, exist_ok=True)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False
    
    # Upload the folder
    try:
        print(f"Uploading {local_path} to {repo_id}")
        print("This may take a while for large models...")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message
        )
        print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        return False

def create_model_card(local_path, model_name, model_type, dataset_info):
    """Create a model card for the model"""
    if "qwen" in model_type.lower():
        base_info = "Qwen3-14B"
        hardware = "~16GB VRAM minimum, 24GB+ recommended"
    else:
        base_info = "Mistral-7B-v0.3"
        hardware = "~8GB VRAM minimum, 16GB+ recommended"
        
    model_card = f"""---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- llm
- fine-tuned
- {model_type}
datasets:
- custom
---

# {model_name}

## Model Description

This is a fine-tuned {base_info} model specialized for {"code generation and DevOps/Infrastructure as Code tasks" if "code" in model_name.lower() else "policy analysis and document understanding"}.

This is the **full merged model** - the LoRA adapters have been merged with the base model, so you can use it directly without needing the base model separately.

## Training Details

- **Base Model**: {base_info}
- **Fine-tuning Method**: QLoRA (4-bit quantization during training)
- **Training Data**: {dataset_info}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("{model_name}", trust_remote_code=True)

# Generate text
input_text = "Your prompt here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Hardware Requirements

{hardware}

## License

Apache 2.0
"""
    
    card_path = Path(local_path) / "README.md"
    with open(card_path, "w") as f:
        f.write(model_card)
    print(f"Created model card at {card_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA models and push to Hugging Face")
    parser.add_argument("--token", help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--model", choices=["qwen", "mistral", "both"], default="qwen",
                        help="Which model to process")
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Hugging Face token required. Set HF_TOKEN env var or use --token")
        sys.exit(1)
    
    models = []
    
    if args.model in ["qwen", "both"]:
        models.append({
            "name": "qwen",
            "base_model_s3": "s3://asoba-llm-cache/models/Qwen/Qwen3-14B/",
            "lora_s3_path": "s3://asoba-llm-cache/trained-models/qwen3-14b-iac-verbosity-qlora/",
            "merged_s3_path": "s3://asoba-llm-cache/trained-models/qwen3-14b-iac-verbosity-merged/",
            "repo_id": "asoba/AsobaCode-v0.1",
            "model_name": "asoba/AsobaCode-v0.1",
            "model_type": "qwen",
            "dataset_info": "Custom dataset of Infrastructure as Code, DevOps documentation, and technical verbosity examples"
        })
    
    if args.model in ["mistral", "both"]:
        models.append({
            "name": "mistral",
            "base_model_s3": "s3://asoba-llm-cache/models/mistralai/Mistral-7B-v0.3/",
            "lora_s3_path": "s3://asoba-llm-cache/trained-models/mistral-policy-qlora/",
            "merged_s3_path": "s3://asoba-llm-cache/trained-models/mistral-policy-merged/",
            "repo_id": "asoba/PolicyAnalyst-v0.1",
            "model_name": "asoba/PolicyAnalyst-v0.1",
            "model_type": "mistral",
            "dataset_info": "90,000+ policy documents including State Department cables and policy analysis texts"
        })
    
    for model_config in models:
        print(f"\n{'='*60}")
        print(f"Processing {model_config['model_name']}")
        print(f"{'='*60}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            base_path = tmpdir / "base_model"
            lora_path = tmpdir / "lora_model"
            merged_path = tmpdir / "merged_model"
            
            # Download base model from S3
            print("\n1. Downloading base model from S3...")
            download_from_s3(model_config["base_model_s3"], str(base_path))
            
            # Download LoRA model from S3
            print("\n2. Downloading LoRA model from S3...")
            download_from_s3(model_config["lora_s3_path"], str(lora_path))
            
            # Merge LoRA with base model
            print("\n3. Merging LoRA adapters with base model...")
            print("Note: This requires loading the full base model and may take significant time and memory")
            success = merge_lora_with_base(
                str(base_path),
                str(lora_path),
                str(merged_path)
            )
            
            if not success:
                print(f"Failed to merge {model_config['model_name']}")
                continue
            
            # Create model card
            create_model_card(
                str(merged_path),
                model_config["model_name"],
                model_config["model_type"],
                model_config["dataset_info"]
            )
            
            # Upload merged model to S3
            print(f"\n4. Uploading merged model to S3...")
            success = upload_to_s3(str(merged_path), model_config["merged_s3_path"])
            if success:
                print(f"✅ Merged model saved to {model_config['merged_s3_path']}")
            
            # Push to Hugging Face
            print(f"\n5. Pushing to Hugging Face...")
            success = push_to_huggingface(
                str(merged_path),
                model_config["repo_id"],
                token,
                f"Upload {model_config['model_name']} full merged model"
            )
            
            if success:
                print(f"✅ Successfully pushed {model_config['model_name']} to HuggingFace")
            else:
                print(f"❌ Failed to push {model_config['model_name']} to HuggingFace")

if __name__ == "__main__":
    main()