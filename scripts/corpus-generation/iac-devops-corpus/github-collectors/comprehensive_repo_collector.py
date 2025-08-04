#!/usr/bin/env python3
"""
Comprehensive IaC Collector - Extract ALL IaC examples from user-specified repositories
- CDK examples (TypeScript, Python, etc.)
- Terraform configurations (.tf files)
- Shell scripts with AWS CLI commands
- Any other IaC patterns

Following CLAUDE.md: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING
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

class ComprehensiveIaCCollector:
    """Collects ALL IaC examples from real repositories"""
    
    def __init__(self, output_dir: str = "comprehensive_iac_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        
        # User-specified repositories
        self.target_repos = [
            'aws-samples/aws-cdk-examples',
            'ksmin23/my-aws-cdk-examples',
            'swoodford/aws',
            'thomvaill/tads-boilerplate',
            'nguyenanhung/infra-caddy-guy',
            'cloudposse/geodesic',
            'devopsgroup-io/catapult',
            # Additional high-quality repositories - batch 1
            'bregman-arie/devops-exercises',
            'ByteByteGoHq/system-design-101',
            'localstack/localstack',
            'serverless/serverless',
            'open-guides/og-aws',
            'pulumi/pulumi',
            'getsops/sops',
            'kubernetes-sigs/kubespray',
            'aws/aws-cli',
            'caprover/caprover',
            # Additional high-quality repositories - batch 2
            'GoogleCloudPlatform/terraformer',
            'donnemartin/awesome-aws',
            'aws/aws-cdk',
            'hashicorp/terraform-provider-aws',
            'awsdocs/aws-doc-sdk-examples',
            'farhanashrafdev/90DaysOfCyberSecurity',
            'google/go-cloud',
            'aws/serverless-application-model',
            'boto/boto3',
            'toniblyx/my-arsenal-of-aws-security-tools',
            'Netflix/metaflow'
        ]

    def collect_all_iac_examples(self) -> Dict[str, List[Dict]]:
        """Collect ALL IaC examples from repositories"""
        logger.info("Starting comprehensive IaC collection from user-specified repositories")
        logger.info("Looking for: CDK, Terraform, Shell scripts, AWS CLI")
        
        results = {
            'cdk': [],
            'terraform': [],
            'shell': [],
            'aws_cli': []
        }
        
        for repo in self.target_repos:
            logger.info(f"Processing repository: {repo}")
            repo_results = self._collect_from_repository(repo)
            
            for iac_type, examples in repo_results.items():
                results[iac_type].extend(examples)
                logger.info(f"  {iac_type}: {len(examples)} examples")
        
        # Deduplicate each type
        for iac_type in results:
            results[iac_type] = self._deduplicate_examples(results[iac_type])
        
        total_examples = sum(len(examples) for examples in results.values())
        logger.info(f"Total IaC examples collected: {total_examples}")
        
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
                
                # Clone repository with optimizations
                result = subprocess.run([
                    'git', 'clone', '--depth', '1', '--filter=blob:limit=50k',
                    f'https://github.com/{repo}.git', clone_path
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    logger.warning(f"Failed to clone {repo}")
                    return results
                
                # Find all relevant files
                file_collections = self._find_iac_files(clone_path)
                
                # Process each file type with limits to prevent timeout
                for file_path in file_collections['cdk'][:100]:  # Limit to prevent timeout
                    example = self._process_cdk_file(file_path, repo)
                    if example:
                        results['cdk'].append(example)
                
                for file_path in file_collections['terraform'][:100]:  # Limit to prevent timeout
                    example = self._process_terraform_file(file_path, repo)
                    if example:
                        results['terraform'].append(example)
                
                for file_path in file_collections['shell'][:50]:  # Limit to prevent timeout
                    examples = self._process_shell_file(file_path, repo)
                    results['shell'].extend(examples)
                    
                    # Also extract AWS CLI from shell files
                    cli_examples = self._extract_aws_cli_from_shell(file_path, repo)
                    results['aws_cli'].extend(cli_examples)
                
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
            # Find CDK files (TypeScript, Python, etc. with CDK imports)
            for ext in ['ts', 'js', 'py', 'java', 'cs']:
                result = subprocess.run([
                    'find', repo_path, '-type', 'f', '-name', f'*.{ext}',
                    '-not', '-path', '*/node_modules/*',
                    '-not', '-path', '*/venv/*',
                    '-not', '-path', '*/.git/*'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    for file_path in result.stdout.strip().split('\n'):
                        if file_path and self._file_contains_cdk(file_path):
                            files['cdk'].append(file_path)
            
            # Find Terraform files
            result = subprocess.run([
                'find', repo_path, '-type', 'f',
                '(', '-name', '*.tf', '-o', '-name', '*.hcl', ')',
                '-not', '-path', '*/.git/*'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                files['terraform'] = [f for f in result.stdout.strip().split('\n') if f]
            
            # Find shell scripts
            result = subprocess.run([
                'find', repo_path, '-type', 'f',
                '(', '-name', '*.sh', '-o', '-name', '*.bash', ')',
                '-not', '-path', '*/.git/*'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                files['shell'] = [f for f in result.stdout.strip().split('\n') if f]
            
        except Exception as e:
            logger.debug(f"Error finding files: {e}")
        
        return files

    def _file_contains_cdk(self, file_path: str) -> bool:
        """Check if file contains CDK code"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            cdk_indicators = [
                'aws-cdk-lib', 'aws_cdk', '@aws-cdk', 'aws-cdk-core',
                'from constructs import', 'from aws_cdk import',
                'import * as cdk', 'extends Stack', 'extends Construct'
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
            
            # Check for duplicates
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            self.collected_hashes.add(content_hash)
            
            # Clean content
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
            
        except Exception as e:
            logger.debug(f"Error processing CDK file {file_path}: {e}")
            return None

    def _process_terraform_file(self, file_path: str, repo: str) -> Optional[Dict]:
        """Process a Terraform file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not self._is_quality_content(content, min_lines=5, max_lines=300):
                return None
            
            # Must contain Terraform syntax
            if not any(keyword in content.lower() for keyword in ['resource ', 'provider ', 'module ', 'variable ', 'data ']):
                return None
            
            # Check for duplicates
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            self.collected_hashes.add(content_hash)
            
            # Clean content
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
            
        except Exception as e:
            logger.debug(f"Error processing Terraform file {file_path}: {e}")
            return None

    def _process_shell_file(self, file_path: str, repo: str) -> List[Dict]:
        """Process a shell script file"""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not self._is_quality_content(content, min_lines=5, max_lines=500):
                return examples
            
            # Check for duplicates
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return examples
            self.collected_hashes.add(content_hash)
            
            # Clean content
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
            
        except Exception as e:
            logger.debug(f"Error processing shell file {file_path}: {e}")
        
        return examples

    def _extract_aws_cli_from_shell(self, file_path: str, repo: str) -> List[Dict]:
        """Extract AWS CLI commands from shell script"""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Find AWS CLI command blocks
            aws_blocks = self._find_aws_cli_blocks(content)
            
            for block in aws_blocks:
                if len(block.split('\n')) >= 3:  # Minimum size
                    # Check for duplicates
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
            
        except Exception as e:
            logger.debug(f"Error extracting AWS CLI from {file_path}: {e}")
        
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
                    # Start new block with some context
                    start_idx = max(0, i - 1)
                    current_block = lines[start_idx:i]
                    in_aws_block = True
                current_block.append(line)
            elif in_aws_block:
                # Continue block if related
                if (line.strip().startswith('#') or 
                    line.strip().startswith('echo') or
                    line.strip().startswith('if') or
                    line.strip().startswith('fi') or
                    line.strip().startswith('done') or
                    line.strip().endswith('\\\\') or
                    '=' in line or
                    line.strip() == ''):
                    current_block.append(line)
                else:
                    # End block
                    if len(current_block) >= 3:
                        blocks.append('\n'.join(current_block))
                    current_block = []
                    in_aws_block = False
        
        # Handle final block
        if in_aws_block and len(current_block) >= 3:
            blocks.append('\n'.join(current_block))
        
        return blocks

    def _is_quality_content(self, content: str, min_lines: int = 5, max_lines: int = 500) -> bool:
        """Check if content meets quality standards"""
        lines = content.split('\n')
        line_count = len(lines)
        
        if line_count < min_lines or line_count > max_lines:
            return False
        
        if len(content) < 100 or len(content) > 20000:
            return False
        
        return True

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension"""
        if file_path.endswith('.ts'):
            return 'typescript'
        elif file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.js'):
            return 'javascript'
        elif file_path.endswith('.java'):
            return 'java'
        elif file_path.endswith('.cs'):
            return 'csharp'
        else:
            return 'typescript'

    def _clean_content(self, content: str) -> str:
        """Clean content while preserving authenticity"""
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            if len(line) < 300:  # Skip very long lines
                # Basic sanitization
                line = re.sub(r'\\b\\d{12}\\b', '<ACCOUNT_ID>', line)
                line = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}', '<EMAIL>', line)
                line = re.sub(r'\\b[a-f0-9]{32,64}\\b', '<TOKEN>', line)
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
                
                # Validate file can be parsed
                try:
                    count = 0
                    with open(output_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                json.loads(line)
                                count += 1
                    logger.info(f"Validated {count} parseable examples in {output_file}")
                except Exception as e:
                    logger.error(f"VALIDATION FAILED for {output_file}: {e}")


def main():
    """Main collection pipeline"""
    logger.info("Starting Comprehensive IaC Collection")
    logger.info("Collecting ALL IaC types from user-specified repositories")
    
    collector = ComprehensiveIaCCollector()
    
    try:
        # Collect all IaC examples
        results = collector.collect_all_iac_examples()
        
        # Save results
        collector.save_results(results)
        
        # Generate statistics
        total_examples = sum(len(examples) for examples in results.values())
        
        logger.info(f"\\nComprehensive IaC Collection Complete:")
        logger.info(f"CDK examples: {len(results['cdk'])}")
        logger.info(f"Terraform examples: {len(results['terraform'])}")
        logger.info(f"Shell scripts: {len(results['shell'])}")
        logger.info(f"AWS CLI examples: {len(results['aws_cli'])}")
        logger.info(f"Total examples: {total_examples}")
        
        # Repository breakdown
        repos = {}
        for iac_type, examples in results.items():
            for example in examples:
                repo = example['metadata']['repository']
                repos[repo] = repos.get(repo, 0) + 1
        
        logger.info(f"\\nRepository breakdown:")
        for repo, count in sorted(repos.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {repo}: {count} examples")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()