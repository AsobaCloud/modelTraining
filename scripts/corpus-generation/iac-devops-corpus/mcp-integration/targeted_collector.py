#!/usr/bin/env python3
"""
Targeted High-Value Repository Collector
Focus on the most valuable repositories that timed out
"""

import os
import json
import time
import requests
import subprocess
import tempfile
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TargetedCollector:
    """Collect from high-value repositories that timed out"""
    
    def __init__(self, output_dir: str = "targeted_iac_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        
        # High-value repositories that timed out or weren't processed
        self.target_repos = [
            'aws/aws-cdk',  # Had 2,233 examples
            'hashicorp/terraform-provider-aws',  # Likely hundreds of Terraform examples
            'awsdocs/aws-doc-sdk-examples',  # Official AWS examples
            'aws/serverless-application-model',  # SAM templates
            'toniblyx/my-arsenal-of-aws-security-tools',  # Security tools
            'Netflix/metaflow',  # Netflix infrastructure
        ]

    def collect_from_high_value_repos(self) -> Dict[str, List[Dict]]:
        """Collect from high-value repositories only"""
        logger.info("Starting targeted collection from high-value repositories")
        
        results = {
            'cdk': [],
            'terraform': [],
            'shell': [],
            'aws_cli': []
        }
        
        for repo in self.target_repos:
            logger.info(f"Processing high-value repository: {repo}")
            repo_results = self._collect_from_repository(repo)
            
            for iac_type, examples in repo_results.items():
                results[iac_type].extend(examples)
                logger.info(f"  {iac_type}: {len(examples)} examples")
        
        # Deduplicate each type
        for iac_type in results:
            results[iac_type] = self._deduplicate_examples(results[iac_type])
        
        total_examples = sum(len(examples) for examples in results.values())
        logger.info(f"Total high-value examples collected: {total_examples}")
        
        return results

    def _collect_from_repository(self, repo: str) -> Dict[str, List[Dict]]:
        """Collect all IaC types from a single repository"""
        results = {
            'cdk': [],
            'terraform': [],
            'shell': [],
            'aws_cli': []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                clone_path = os.path.join(temp_dir, repo.replace('/', '_'))
                
                # Clone repository with limited depth
                result = subprocess.run([
                    'git', 'clone', '--depth', '1', '--filter=blob:limit=100k',
                    f'https://github.com/{repo}.git', clone_path
                ], capture_output=True, text=True, timeout=300)  # 5 min timeout
                
                if result.returncode != 0:
                    logger.warning(f"Failed to clone {repo}: {result.stderr}")
                    return results
                
                # Find all relevant files
                file_collections = self._find_iac_files(clone_path)
                
                # Process each file type with limits
                for file_path in file_collections['cdk'][:50]:  # Limit to avoid timeout
                    example = self._process_cdk_file(file_path, repo)
                    if example:
                        results['cdk'].append(example)
                
                for file_path in file_collections['terraform'][:50]:
                    example = self._process_terraform_file(file_path, repo)
                    if example:
                        results['terraform'].append(example)
                
                for file_path in file_collections['shell'][:20]:
                    examples = self._process_shell_file(file_path, repo)
                    results['shell'].extend(examples)
                    
                    # Also extract AWS CLI from shell files
                    cli_examples = self._extract_aws_cli_from_shell(file_path, repo)
                    results['aws_cli'].extend(cli_examples)
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout cloning {repo}")
            except Exception as e:
                logger.warning(f"Error processing {repo}: {e}")
        
        return results

    def _find_iac_files(self, repo_path: str) -> Dict[str, List[str]]:
        """Find all IaC-related files in repository"""
        files = {
            'cdk': [],
            'terraform': [],
            'shell': []
        }
        
        try:
            # Find CDK files (focus on examples directories)
            for ext in ['ts', 'js', 'py']:
                result = subprocess.run([
                    'find', repo_path, '-type', 'f', '-name', f'*.{ext}',
                    '-path', '*/example*',
                    '-not', '-path', '*/node_modules/*',
                    '-not', '-path', '*/venv/*'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    for file_path in result.stdout.strip().split('\n'):
                        if file_path and self._file_contains_cdk(file_path):
                            files['cdk'].append(file_path)
            
            # If no examples found, search more broadly
            if not files['cdk']:
                for ext in ['ts', 'js', 'py']:
                    result = subprocess.run([
                        'find', repo_path, '-type', 'f', '-name', f'*.{ext}',
                        '-not', '-path', '*/node_modules/*',
                        '-not', '-path', '*/venv/*'
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        for file_path in result.stdout.strip().split('\n')[:100]:  # Limit search
                            if file_path and self._file_contains_cdk(file_path):
                                files['cdk'].append(file_path)
            
            # Find Terraform files
            result = subprocess.run([
                'find', repo_path, '-type', 'f',
                '(', '-name', '*.tf', '-o', '-name', '*.hcl', ')',
                '-not', '-path', '*/\.terraform/*'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                files['terraform'] = [f for f in result.stdout.strip().split('\n') if f][:100]
            
            # Find shell scripts
            result = subprocess.run([
                'find', repo_path, '-type', 'f',
                '(', '-name', '*.sh', '-o', '-name', '*.bash', ')',
                '-not', '-path', '*/\.git/*'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                files['shell'] = [f for f in result.stdout.strip().split('\n') if f][:50]
            
        except Exception as e:
            logger.debug(f"Error finding files: {e}")
        
        return files

    def _file_contains_cdk(self, file_path: str) -> bool:
        """Check if file contains CDK code"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()[:5000]  # Only read first 5KB
            
            cdk_indicators = [
                'aws-cdk-lib', 'aws_cdk', '@aws-cdk', 'constructs',
                'from aws_cdk import', 'import * as cdk', 'extends Stack'
            ]
            
            return any(indicator in content for indicator in cdk_indicators)
        except:
            return False

    def _process_cdk_file(self, file_path: str, repo: str) -> Optional[Dict]:
        """Process a CDK file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not self._is_quality_content(content, min_lines=10, max_lines=200):
                return None
            
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            self.collected_hashes.add(content_hash)
            
            clean_content = self._clean_content(content)
            language = self._detect_language(file_path)
            
            return {
                "prompt": f"Create an AWS CDK {language} infrastructure application",
                "completion": f"```{language}\n{clean_content}\n```",
                "metadata": {
                    "source": "github_repository",
                    "repository": repo,
                    "file_path": file_path,
                    "language": language,
                    "category": "cdk",
                    "authentic": True
                }
            }
        except:
            return None

    def _process_terraform_file(self, file_path: str, repo: str) -> Optional[Dict]:
        """Process a Terraform file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not self._is_quality_content(content, min_lines=5, max_lines=300):
                return None
            
            if not any(keyword in content.lower() for keyword in ['resource ', 'provider ', 'module ', 'variable ', 'data ']):
                return None
            
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            self.collected_hashes.add(content_hash)
            
            clean_content = self._clean_content(content)
            
            return {
                "prompt": "Create a Terraform configuration for infrastructure deployment",
                "completion": f"```hcl\n{clean_content}\n```",
                "metadata": {
                    "source": "github_repository",
                    "repository": repo,
                    "file_path": file_path,
                    "category": "terraform",
                    "authentic": True
                }
            }
        except:
            return None

    def _process_shell_file(self, file_path: str, repo: str) -> List[Dict]:
        """Process a shell script file"""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not self._is_quality_content(content, min_lines=5, max_lines=500):
                return examples
            
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return examples
            self.collected_hashes.add(content_hash)
            
            clean_content = self._clean_content(content)
            
            example = {
                "prompt": "Create a shell script for infrastructure automation",
                "completion": f"```bash\n{clean_content}\n```",
                "metadata": {
                    "source": "github_repository",
                    "repository": repo,
                    "file_path": file_path,
                    "category": "shell_script",
                    "authentic": True
                }
            }
            
            examples.append(example)
        except:
            pass
        
        return examples

    def _extract_aws_cli_from_shell(self, file_path: str, repo: str) -> List[Dict]:
        """Extract AWS CLI commands from shell script"""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if 'aws ' not in content.lower():
                return examples
            
            # Find AWS CLI command blocks
            aws_blocks = self._find_aws_cli_blocks(content)
            
            for block in aws_blocks:
                if len(block.split('\n')) >= 3:
                    content_hash = hashlib.md5(block.encode()).hexdigest()
                    if content_hash in self.collected_hashes:
                        continue
                    self.collected_hashes.add(content_hash)
                    
                    clean_block = self._clean_content(block)
                    
                    example = {
                        "prompt": "Write AWS CLI commands for infrastructure management",
                        "completion": f"```bash\n{clean_block}\n```",
                        "metadata": {
                            "source": "github_repository",
                            "repository": repo,
                            "file_path": file_path,
                            "category": "aws_cli",
                            "authentic": True
                        }
                    }
                    
                    examples.append(example)
        except:
            pass
        
        return examples

    def _find_aws_cli_blocks(self, content: str) -> List[str]:
        """Find AWS CLI command blocks in content"""
        blocks = []
        lines = content.split('\n')
        
        current_block = []
        in_aws_block = False
        
        for i, line in enumerate(lines):
            if 'aws ' in line.lower() and not line.strip().startswith('#'):
                if not in_aws_block:
                    start_idx = max(0, i - 1)
                    current_block = lines[start_idx:i]
                    in_aws_block = True
                current_block.append(line)
            elif in_aws_block:
                if (line.strip().startswith('#') or 
                    line.strip().startswith('echo') or
                    line.strip().startswith('if') or
                    line.strip().endswith('\\\\') or
                    '=' in line or
                    line.strip() == ''):
                    current_block.append(line)
                else:
                    if len(current_block) >= 3:
                        blocks.append('\n'.join(current_block))
                    current_block = []
                    in_aws_block = False
        
        if in_aws_block and len(current_block) >= 3:
            blocks.append('\n'.join(current_block))
        
        return blocks

    def _is_quality_content(self, content: str, min_lines: int = 5, max_lines: int = 500) -> bool:
        """Check if content meets quality standards"""
        lines = content.split('\n')
        line_count = len(lines)
        
        return min_lines <= line_count <= max_lines and 100 <= len(content) <= 20000

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension"""
        if file_path.endswith('.ts'):
            return 'typescript'
        elif file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.js'):
            return 'javascript'
        else:
            return 'typescript'

    def _clean_content(self, content: str) -> str:
        """Clean content while preserving authenticity"""
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            if len(line) < 300:
                line = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', line)
                line = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', line)
                line = re.sub(r'\b[a-f0-9]{32,64}\b', '<TOKEN>', line)
                clean_lines.append(line.rstrip())
        
        return '\n'.join(clean_lines).strip()

    def _deduplicate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicate examples"""
        seen = set()
        unique = []
        
        for example in examples:
            content_hash = hashlib.md5(example['completion'].encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(example)
        
        return unique

    def save_results(self, results: Dict[str, List[Dict]]) -> None:
        """Save all results to separate files"""
        for iac_type, examples in results.items():
            if examples:
                output_file = self.output_dir / f"{iac_type}_examples.jsonl"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for example in examples:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
                logger.info(f"Saved {len(examples)} {iac_type} examples to {output_file}")
                
                # Validate
                try:
                    count = 0
                    with open(output_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                json.loads(line)
                                count += 1
                    logger.info(f"Validated {count} parseable examples")
                except Exception as e:
                    logger.error(f"VALIDATION FAILED: {e}")


def main():
    """Main targeted collection pipeline"""
    logger.info("Starting Targeted High-Value Repository Collection")
    
    collector = TargetedCollector()
    
    try:
        results = collector.collect_from_high_value_repos()
        collector.save_results(results)
        
        # Generate statistics
        total_examples = sum(len(examples) for examples in results.values())
        
        logger.info(f"\nTargeted High-Value Collection Complete:")
        logger.info(f"CDK examples: {len(results['cdk'])}")
        logger.info(f"Terraform examples: {len(results['terraform'])}")
        logger.info(f"Shell scripts: {len(results['shell'])}")
        logger.info(f"AWS CLI examples: {len(results['aws_cli'])}")
        logger.info(f"Total examples: {total_examples}")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()