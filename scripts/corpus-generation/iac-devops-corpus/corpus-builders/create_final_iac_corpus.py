#!/usr/bin/env python3
"""
Create Final Comprehensive IaC Training Corpus
Combining all authentic examples from real-world sources
"""

import json

def main():
    """Combine all authentic IaC examples into final corpus"""
    
    # Read all component corpora
    print("Reading component corpora...")
    
    # Shell/Bash scripts
    shell_examples = []
    with open('final_comprehensive_shell_corpus.jsonl', 'r') as f:
        for line in f:
            shell_examples.append(json.loads(line))
    
    # Terraform configurations
    terraform_examples = []
    with open('terraform_real_corpus/terraform_real_corpus.jsonl', 'r') as f:
        for line in f:
            terraform_examples.append(json.loads(line))
    
    # AWS CLI examples
    aws_cli_examples = []
    with open('aws_cli_real_corpus/aws_cli_real_corpus.jsonl', 'r') as f:
        for line in f:
            aws_cli_examples.append(json.loads(line))
    
    # CDK examples (now authentic!)
    cdk_examples = []
    with open('cdk_real_corpus/cdk_real_corpus.jsonl', 'r') as f:
        for line in f:
            cdk_examples.append(json.loads(line))
    
    # Combine all examples
    final_corpus = shell_examples + terraform_examples + aws_cli_examples + cdk_examples
    
    print(f"Final IaC Training Corpus Assembly:")
    print(f"Shell/Bash: {len(shell_examples)} examples")
    print(f"Terraform: {len(terraform_examples)} examples")
    print(f"AWS CLI: {len(aws_cli_examples)} examples")
    print(f"CDK: {len(cdk_examples)} examples")
    print(f"Total: {len(final_corpus)} examples")
    
    # Save final corpus
    with open('final_iac_training_corpus.jsonl', 'w') as f:
        for example in final_corpus:
            f.write(json.dumps(example, ensure_ascii=False) + '\\n')
    
    # Generate comprehensive statistics
    languages = {}
    sources = {}
    categories = {}
    
    for example in final_corpus:
        metadata = example.get('metadata', {})
        source = metadata.get('source', 'unknown')
        category = metadata.get('category', 'unknown')
        
        # Detect language/type
        completion = example['completion']
        if '```bash' in completion or '```shell' in completion:
            language = 'shell_script'
        elif '```hcl' in completion:
            language = 'terraform'
        elif '```typescript' in completion:
            language = 'cdk_typescript'
        elif '```python' in completion:
            language = 'cdk_python'
        elif '```javascript' in completion:
            language = 'cdk_javascript'
        else:
            language = 'aws_cli'
        
        languages[language] = languages.get(language, 0) + 1
        sources[source] = sources.get(source, 0) + 1
        categories[category] = categories.get(category, 0) + 1
    
    print(f"\\nFinal IaC Training Corpus Statistics:")
    print(f"Total examples: {len(final_corpus)}")
    print(f"Languages/Types: {dict(sorted(languages.items()))}")
    print(f"Sources: {dict(sorted(sources.items()))}")
    print(f"Top categories: {dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10])}")
    
    # Calculate final distribution percentages
    total = len(final_corpus)
    shell_pct = len(shell_examples) / total * 100
    terraform_pct = len(terraform_examples) / total * 100
    aws_cli_pct = len(aws_cli_examples) / total * 100
    cdk_pct = len(cdk_examples) / total * 100
    
    print(f"\\nFinal Distribution:")
    print(f"Shell/Bash: {shell_pct:.1f}% ({len(shell_examples)}/{total})")
    print(f"Terraform: {terraform_pct:.1f}% ({len(terraform_examples)}/{total})")
    print(f"AWS CLI: {aws_cli_pct:.1f}% ({len(aws_cli_examples)}/{total})")
    print(f"CDK: {cdk_pct:.1f}% ({len(cdk_examples)}/{total})")
    
    # Verify authenticity
    all_authentic = all(example.get('metadata', {}).get('authentic', False) for example in final_corpus)
    print(f"\\nAll examples authentic: {all_authentic}")
    
    print(f"\\nSaved final corpus to: final_iac_training_corpus.jsonl")
    print("Ready for Mistral 7B IaC model training!")

if __name__ == "__main__":
    main()