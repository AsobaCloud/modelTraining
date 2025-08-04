#!/usr/bin/env python3
"""
Quick Shell Script Corpus Builder - Simplified Version
Processes key shell scripts for immediate training corpus
"""

import os
import json
import hashlib
from pathlib import Path

def extract_key_scripts():
    """Extract key shell scripts for training corpus"""
    
    # Key script files identified from our exploration
    key_scripts = [
        "/home/shingai/api/deploy/deploy-core-api.sh",
        "/home/shingai/api/deploy/scripts/setup-monitoring.sh", 
        "/home/shingai/api/RAG/ETL/scripts/orchestrator.sh",
        "/home/shingai/sort/ona-front-end/express-chatbot/setup-monitoring.sh",
        "/home/shingai/sort/ona-front-end/express-chatbot/deploy-all-endpoints.sh",
        "/home/shingai/api/RAG/ETL/scripts/deploy.sh",
        "/home/shingai/api/core/weather/deploy.sh",
        "/home/shingai/api/deploy/scripts/create-ecr-repos.sh",
        "/home/shingai/sort/deployments/monitoring/setup_monitoring.sh"
    ]
    
    training_examples = []
    
    for script_path in key_scripts:
        if os.path.exists(script_path):
            print(f"Processing: {script_path}")
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip very large files
                if len(content) > 20000:
                    continue
                
                # Generate training examples
                examples = create_training_examples(script_path, content)
                training_examples.extend(examples)
                
            except Exception as e:
                print(f"Error processing {script_path}: {e}")
    
    return training_examples

def create_training_examples(file_path, content):
    """Create training examples from shell script"""
    examples = []
    
    # Determine script category and purpose
    path_lower = file_path.lower()
    
    if 'deploy' in path_lower:
        category = 'deployment'
        base_prompt = "Write a shell script for deploying infrastructure and services"
    elif 'monitor' in path_lower:
        category = 'monitoring'
        base_prompt = "Create a shell script for setting up monitoring and alerts"
    elif 'orchestrat' in path_lower:
        category = 'orchestration'
        base_prompt = "Write a shell script that orchestrates multiple deployment steps"
    elif 'setup' in path_lower:
        category = 'infrastructure'
        base_prompt = "Create a shell script for infrastructure setup"
    else:
        category = 'aws_operations'
        base_prompt = "Write a shell script for AWS resource management"
    
    # Clean content
    clean_content = clean_script_content(content)
    
    # Main example
    examples.append({
        "prompt": f"{base_prompt} using AWS CLI and CloudFormation.",
        "completion": f"```bash\n{clean_content}\n```",
        "metadata": {
            "source": file_path,
            "category": category,
            "type": "complete_script"
        }
    })
    
    # Extract AWS CLI examples
    aws_examples = extract_aws_cli_examples(content, category)
    examples.extend(aws_examples)
    
    return examples

def clean_script_content(content):
    """Clean script content and add placeholders"""
    import re
    
    # Replace account IDs
    content = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', content)
    
    # Replace email addresses  
    content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', content)
    
    # Replace specific URLs (but keep AWS service URLs)
    content = re.sub(r'https?://(?!.*\.amazonaws\.com)[^\s"\']+', '<URL>', content)
    
    # Replace instance IDs
    content = re.sub(r'\bi-[0-9a-f]{8,17}\b', '<INSTANCE_ID>', content)
    
    return content.strip()

def extract_aws_cli_examples(content, category):
    """Extract AWS CLI command examples"""
    examples = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('aws ') and len(line) > 10 and len(line) < 200:
            # Generate prompt for this AWS command
            prompt = generate_aws_prompt(line)
            if prompt:
                examples.append({
                    "prompt": prompt,
                    "completion": f"```bash\n{line}\n```",
                    "metadata": {
                        "category": category,
                        "type": "aws_command"
                    }
                })
    
    return examples

def generate_aws_prompt(aws_cmd):
    """Generate appropriate prompt for AWS CLI command"""
    parts = aws_cmd.split()
    if len(parts) < 2:
        return None
    
    service = parts[1]
    operation = parts[2] if len(parts) > 2 else ""
    
    prompts = {
        ('cloudformation', 'deploy'): "Deploy CloudFormation stack using AWS CLI",
        ('cloudformation', 'describe-stacks'): "Get CloudFormation stack information using AWS CLI",
        ('sns', 'create-topic'): "Create SNS topic using AWS CLI",
        ('lambda', 'create-function'): "Create Lambda function using AWS CLI",
        ('lambda', 'update-function-code'): "Update Lambda function code using AWS CLI",
        ('iam', 'create-role'): "Create IAM role using AWS CLI",
        ('ec2', 'describe-instances'): "List EC2 instances using AWS CLI",
        ('apigateway', 'create-deployment'): "Deploy API Gateway using AWS CLI",
        ('cloudwatch', 'put-metric-alarm'): "Create CloudWatch alarm using AWS CLI"
    }
    
    return prompts.get((service, operation), f"Use AWS CLI for {service} {operation}")

def main():
    print("Starting Quick Shell Corpus Builder...")
    
    # Extract training examples
    training_examples = extract_key_scripts()
    
    print(f"Generated {len(training_examples)} training examples")
    
    # Save to JSONL
    output_dir = Path("shell_corpus")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "shell_training_corpus.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Saved corpus to {output_file}")
    
    # Generate simple statistics
    categories = {}
    for example in training_examples:
        cat = example['metadata']['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCorpus Statistics:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} examples")
    
    print("\nShell corpus building completed!")

if __name__ == "__main__":
    main()