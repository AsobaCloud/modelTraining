#!/usr/bin/env python3
"""
Push trained models from S3 to Hugging Face Hub
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import argparse
import json

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

def push_to_huggingface(local_path, repo_id, token, commit_message="Update model"):
    """Push model to Hugging Face Hub"""
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
    model_card = f"""---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- llm
- fine-tuned
- {"qwen" if "qwen" in model_type.lower() else "mistral"}
datasets:
- custom
---

# {model_name}

## Model Description

This is a fine-tuned {"Qwen3-14B" if "qwen" in model_type.lower() else "Mistral-7B-v0.3"} model specialized for {"code generation and DevOps/Infrastructure as Code tasks" if "code" in model_name.lower() else "policy analysis and document understanding"}.

## Training Details

- **Base Model**: {"Qwen/Qwen3-14B" if "qwen" in model_type.lower() else "mistralai/Mistral-7B-v0.3"}
- **Fine-tuning Method**: QLoRA (4-bit quantization)
- **Training Data**: {dataset_info}

## Usage

This model can be loaded using the Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```

## Limitations

This is a fine-tuned model and may exhibit biases or limitations from both the base model and the training data.

## License

Apache 2.0
"""
    
    card_path = Path(local_path) / "README.md"
    with open(card_path, "w") as f:
        f.write(model_card)
    print(f"Created model card at {card_path}")

def main():
    parser = argparse.ArgumentParser(description="Push models to Hugging Face")
    parser.add_argument("--token", help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--model", choices=["qwen", "mistral", "both"], default="both",
                        help="Which model to push")
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Hugging Face token required. Set HF_TOKEN env var or use --token")
        sys.exit(1)
    
    models = []
    
    if args.model in ["qwen", "both"]:
        models.append({
            "s3_path": "s3://asoba-llm-cache/trained-models/qwen3-14b-iac-verbosity-qlora/",
            "repo_id": "asoba/AsobaCode-v0.1",
            "model_name": "AsobaCode-v0.1",
            "model_type": "qwen",
            "dataset_info": "Custom dataset of Infrastructure as Code, DevOps documentation, and technical verbosity examples"
        })
    
    if args.model in ["mistral", "both"]:
        models.append({
            "s3_path": "s3://asoba-llm-cache/trained-models/mistral-policy-qlora/",
            "repo_id": "asoba/PolicyAnalyst-v0.1",
            "model_name": "PolicyAnalyst-v0.1",
            "model_type": "mistral",
            "dataset_info": "90,000+ policy documents including State Department cables and policy analysis texts"
        })
    
    for model_config in models:
        print(f"\n{'='*60}")
        print(f"Processing {model_config['model_name']}")
        print(f"{'='*60}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / model_config["model_name"]
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download from S3
            download_from_s3(model_config["s3_path"], str(local_path))
            
            # Create model card
            create_model_card(
                str(local_path),
                model_config["repo_id"],
                model_config["model_type"],
                model_config["dataset_info"]
            )
            
            # Push to Hugging Face
            success = push_to_huggingface(
                str(local_path),
                model_config["repo_id"],
                token,
                f"Upload {model_config['model_name']} fine-tuned model"
            )
            
            if success:
                print(f"✅ Successfully pushed {model_config['model_name']}")
            else:
                print(f"❌ Failed to push {model_config['model_name']}")

if __name__ == "__main__":
    main()