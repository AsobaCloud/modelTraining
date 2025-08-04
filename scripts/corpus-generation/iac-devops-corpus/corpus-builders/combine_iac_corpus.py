#!/usr/bin/env python3
"""
Combine Shell and Terraform corpora into comprehensive IaC training dataset
"""

import json

def main():
    # Read shell script corpus
    print('Reading shell script corpus...')
    shell_examples = []
    with open('final_comprehensive_shell_corpus.jsonl', 'r') as f:
        for line in f:
            shell_examples.append(json.loads(line))

    # Read Terraform corpus
    print('Reading Terraform corpus...')
    terraform_examples = []
    with open('terraform_real_corpus/terraform_real_corpus.jsonl', 'r') as f:
        for line in f:
            terraform_examples.append(json.loads(line))

    # Combine
    combined_iac = shell_examples + terraform_examples
    print(f'Combined IaC corpus: {len(shell_examples)} shell + {len(terraform_examples)} terraform = {len(combined_iac)} total')

    # Save combined corpus
    with open('combined_iac_corpus.jsonl', 'w') as f:
        for example in combined_iac:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Generate statistics
    categories = {}
    sources = {}
    languages = {}

    for example in combined_iac:
        metadata = example.get('metadata', {})
        source = metadata.get('source', 'unknown')
        category = metadata.get('category', 'unknown')
        
        # Detect language from completion
        completion = example['completion']
        if '```bash' in completion or '```shell' in completion:
            language = 'shell_script'
        elif '```hcl' in completion:
            language = 'terraform'
        else:
            language = 'unknown'
        
        categories[category] = categories.get(category, 0) + 1
        sources[source] = sources.get(source, 0) + 1
        languages[language] = languages.get(language, 0) + 1

    print(f'Combined IaC Training Corpus Statistics:')
    print(f'Total examples: {len(combined_iac)}')
    print(f'Languages: {dict(sorted(languages.items()))}')
    print(f'Sources: {dict(sorted(sources.items()))}')
    print(f'Top categories: {dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10])}')

    # Calculate percentages according to target plan
    shell_pct = len(shell_examples) / len(combined_iac) * 100
    terraform_pct = len(terraform_examples) / len(combined_iac) * 100

    print(f'Language distribution:')
    print(f'Shell/Bash: {len(shell_examples)} examples ({shell_pct:.1f}%)')
    print(f'Terraform: {len(terraform_examples)} examples ({terraform_pct:.1f}%)')
    print(f'All authentic: {all(example.get("metadata", {}).get("authentic", False) for example in combined_iac)}')
    print('Saved to: combined_iac_corpus.jsonl')

if __name__ == "__main__":
    main()