#!/usr/bin/env python3
"""
Enhanced IaC Corpus Integration Script
Following CLAUDE.md principle: Combine all collected real-world data into comprehensive training corpus

Integrates:
- Existing: Shell, Terraform, AWS CLI, CDK examples  
- New: Docker, Helm, CI/CD, Jupyter, Mermaid examples
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jsonl(file_path: Path) -> list:
    """Load examples from JSONL file"""
    examples = []
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    examples.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line in {file_path}: {e}")
    return examples

def save_jsonl(examples: list, file_path: Path) -> None:
    """Save examples to JSONL file"""
    file_path.parent.mkdir(exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    logger.info(f"Saved {len(examples)} examples to {file_path}")

def main():
    """Integrate enhanced IaC corpus"""
    logger.info("Starting enhanced IaC corpus integration")
    
    # Base paths
    base_dir = Path("/home/shingai/sort/deployments")
    collectors_dir = base_dir / "data" / "collectors"
    comprehensive_dir = base_dir / "comprehensive_iac_corpus"
    
    # Existing corpus files
    existing_files = {
        "shell": comprehensive_dir / "shell_examples.jsonl",
        "terraform": comprehensive_dir / "terraform_examples.jsonl", 
        "aws_cli": comprehensive_dir / "aws_cli_examples.jsonl",
        "cdk": comprehensive_dir / "cdk_examples.jsonl"
    }
    
    # New corpus files
    new_files = {
        "docker": collectors_dir / "docker_real_corpus" / "docker_deployment_corpus.jsonl",
        "helm": collectors_dir / "helm_real_corpus" / "helm_deployment_corpus.jsonl",
        "cicd": collectors_dir / "cicd_real_corpus" / "cicd_pipeline_corpus.jsonl",
        "jupyter": collectors_dir / "jupyter_real_corpus" / "jupyter_deployment_corpus.jsonl", 
        "mermaid": collectors_dir / "mermaid_real_corpus" / "mermaid_architecture_corpus.jsonl"
    }
    
    # Load all examples
    all_examples = []
    category_counts = {}
    
    # Load existing examples
    for category, file_path in existing_files.items():
        examples = load_jsonl(file_path)
        all_examples.extend(examples)
        category_counts[category] = len(examples)
        logger.info(f"Loaded {len(examples)} {category} examples")
    
    # Load new examples  
    for category, file_path in new_files.items():
        examples = load_jsonl(file_path)
        all_examples.extend(examples)
        category_counts[category] = len(examples)
        logger.info(f"Loaded {len(examples)} {category} examples")
    
    # Copy new files to comprehensive corpus directory
    for category, source_path in new_files.items():
        if source_path.exists():
            dest_path = comprehensive_dir / f"{category}_examples.jsonl"
            examples = load_jsonl(source_path)
            save_jsonl(examples, dest_path)
    
    # Create final comprehensive corpus
    final_corpus_path = comprehensive_dir / "final_enhanced_iac_corpus.jsonl"
    save_jsonl(all_examples, final_corpus_path)
    
    # Generate summary
    total_examples = len(all_examples)
    logger.info(f"Enhanced IaC Corpus Integration Complete!")
    logger.info(f"Total examples: {total_examples}")
    logger.info("Category breakdown:")
    for category, count in sorted(category_counts.items()):
        percentage = (count / total_examples * 100) if total_examples > 0 else 0
        logger.info(f"  {category}: {count} examples ({percentage:.1f}%)")
    
    # Generate corpus summary
    summary = {
        "total_examples": total_examples,
        "categories": category_counts,
        "files": {
            "comprehensive_corpus": str(final_corpus_path),
            "individual_files": {
                **{k: str(v) for k, v in existing_files.items()},
                **{k: str(comprehensive_dir / f"{k}_examples.jsonl") for k in new_files.keys()}
            }
        },
        "enhancement_summary": {
            "original_categories": len(existing_files),
            "new_categories": len(new_files), 
            "total_categories": len(category_counts),
            "new_examples_added": sum(category_counts[k] for k in new_files.keys())
        }
    }
    
    summary_path = comprehensive_dir / "enhanced_corpus_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Corpus summary saved to {summary_path}")
    
    print(f"\n🎯 Enhanced IaC Training Corpus Ready!")
    print(f"📊 Total Examples: {total_examples}")
    print(f"🔧 Categories: {', '.join(sorted(category_counts.keys()))}")
    print(f"📁 Location: {final_corpus_path}")

if __name__ == "__main__":
    main()