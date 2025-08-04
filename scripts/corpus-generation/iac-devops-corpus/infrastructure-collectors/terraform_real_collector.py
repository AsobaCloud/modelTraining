#!/usr/bin/env python3
"""
Real-World Terraform Configuration Collector
Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING

Collects exclusively from authentic sources:
- Production Terraform configurations
- Official provider examples
- Open source infrastructure projects
- Stack Overflow snippets
- Real DevOps workflows
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
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TerraformRealCollector:
    """Collects Terraform configurations exclusively from real-world sources"""
    
    def __init__(self, output_dir: str = "terraform_real_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        
        # Target infrastructure repositories with quality Terraform configs
        self.target_repos = [
            # Terraform official examples
            'hashicorp/terraform-provider-aws',
            'hashicorp/terraform-provider-azurerm',
            'hashicorp/terraform-provider-google',
            'hashicorp/terraform-guides',
            
            # Infrastructure as Code projects
            'terraform-aws-modules/terraform-aws-vpc',
            'terraform-aws-modules/terraform-aws-eks',
            'terraform-aws-modules/terraform-aws-rds',
            'terraform-aws-modules/terraform-aws-s3-bucket',
            'terraform-aws-modules/terraform-aws-security-group',
            'terraform-aws-modules/terraform-aws-alb',
            
            # Real-world infrastructure
            'gruntwork-io/terragrunt-infrastructure-live-example',
            'gruntwork-io/infrastructure-modules-example',
            'cloudposse/terraform-aws-components',
            'cloudposse/terraform-aws-cloudfront-cdn',
            'cloudposse/terraform-aws-eks-cluster',
            
            # Kubernetes + Terraform
            'terraform-providers/terraform-provider-kubernetes',
            'hashicorp/terraform-provider-helm',
            
            # Multi-cloud examples
            'hashicorp/learn-terraform-provision-eks-cluster',
            'hashicorp/terraform-provider-docker',
            
            # DevOps workflows
            'antonbabenko/terraform-best-practices',
            'bridgecrewio/checkov',
        ]
        
        # Local repositories to explore
        self.local_sources = [
            '/home/shingai/api',
            '/home/shingai/sort/deployments',
            '/home/shingai/sort/ona-front-end'
        ]

    def collect_real_terraform_configs(self, target_count: int = 100) -> List[Dict]:
        """Collect real-world Terraform configurations"""
        logger.info(f"Starting real-world Terraform collection targeting {target_count} configurations")
        
        all_examples = []
        
        # Strategy 1: Local repository mining
        local_examples = self._collect_from_local_repos()
        all_examples.extend(local_examples)
        logger.info(f"Collected {len(local_examples)} from local repositories")
        
        # Strategy 2: Clone Terraform-focused repositories
        repo_examples = self._collect_from_terraform_repos()
        all_examples.extend(repo_examples)
        logger.info(f"Collected {len(repo_examples)} from Terraform repositories")
        
        # Strategy 3: GitHub raw file collection
        github_examples = self._collect_terraform_raw_files()
        all_examples.extend(github_examples)
        logger.info(f"Collected {len(github_examples)} from GitHub raw files")
        
        # Strategy 4: Stack Overflow Terraform snippets
        stackoverflow_examples = self._collect_stackoverflow_terraform()
        all_examples.extend(stackoverflow_examples)
        logger.info(f"Collected {len(stackoverflow_examples)} from Stack Overflow")
        
        # Deduplicate and filter quality
        unique_examples = self._deduplicate_examples(all_examples)
        quality_examples = self._filter_terraform_quality(unique_examples)
        
        logger.info(f"Final real-world Terraform collection: {len(quality_examples)} configurations")
        return quality_examples[:target_count]

    def _collect_from_local_repos(self) -> List[Dict]:
        """Collect Terraform configs from local repositories"""
        examples = []
        
        for repo_path in self.local_sources:
            if os.path.exists(repo_path):
                logger.info(f"Scanning local repository: {repo_path}")
                repo_examples = self._extract_terraform_from_path(repo_path)
                examples.extend(repo_examples)
        
        return examples

    def _extract_terraform_from_path(self, path: str) -> List[Dict]:
        """Extract Terraform configurations from a given path"""
        examples = []
        
        try:
            # Find Terraform files
            result = subprocess.run([
                'find', path, '-type', 'f',
                '(', '-name', '*.tf', '-o', '-name', '*.hcl', ')',
                '-not', '-path', '*/node_modules/*',
                '-not', '-path', '*/venv/*',
                '-not', '-path', '*/.git/*',
                '-not', '-path', '*/build/*',
                '-not', '-path', '*/dist/*',
                '-not', '-path', '*/.terraform/*'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                terraform_files = result.stdout.strip().split('\n')
                
                for tf_file in terraform_files:
                    if tf_file:  # Skip empty lines
                        example = self._process_terraform_file(tf_file)
                        if example:
                            examples.append(example)
            
        except Exception as e:
            logger.debug(f"Error extracting Terraform from {path}: {e}")
        
        return examples

    def _process_terraform_file(self, file_path: str) -> Optional[Dict]:
        """Process a Terraform configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Quality checks for Terraform configs
            if not self._is_quality_terraform_config(content, file_path):
                return None
            
            # Check for duplicates
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            
            self.collected_hashes.add(content_hash)
            
            # Generate training example
            prompt = self._generate_terraform_prompt(content, file_path)
            clean_content = self._clean_terraform_content(content)
            
            return {
                "prompt": prompt,
                "completion": f"```hcl\n{clean_content}\n```",
                "metadata": {
                    "source": "real_local_repository",
                    "file_path": file_path,
                    "category": self._categorize_terraform_config(content, file_path),
                    "authentic": True
                }
            }
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return None

    def _is_quality_terraform_config(self, content: str, file_path: str) -> bool:
        """Check if Terraform config is real and high quality"""
        lines = content.split('\n')
        line_count = len(lines)
        
        # Size checks
        if line_count < 5 or line_count > 500:
            return False
        
        if len(content) < 100 or len(content) > 50000:
            return False
        
        # Must contain Terraform syntax
        terraform_indicators = [
            'resource ', 'provider ', 'variable ', 'output ', 'module ',
            'data ', 'locals ', 'terraform {', 'required_providers'
        ]
        
        content_lower = content.lower()
        has_terraform_syntax = any(indicator in content_lower for indicator in terraform_indicators)
        
        if not has_terraform_syntax:
            return False
        
        # Check for infrastructure relevance
        infra_indicators = [
            'aws_', 'azurerm_', 'google_', 'kubernetes_', 'helm_',
            'docker_', 'vault_', 'consul_', 'nomad_', 'tls_',
            'random_', 'null_', 'local_', 'archive_', 'template_',
            'vpc', 'subnet', 'instance', 'bucket', 'security_group',
            'load_balancer', 'database', 'cluster', 'network'
        ]
        
        matches = sum(1 for indicator in infra_indicators if indicator in content_lower)
        
        # Avoid obviously generated or example configs
        bad_patterns = [
            'this file was automatically generated',
            'do not edit this file',
            'generated by terraform',
            'example configuration',
            'sample config',
            'template file',
            'placeholder values'
        ]
        
        for pattern in bad_patterns:
            if pattern in content_lower:
                return False
        
        return matches >= 2  # At least 2 infrastructure indicators

    def _collect_from_terraform_repos(self) -> List[Dict]:
        """Clone and extract Terraform configs from repositories"""
        examples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo in self.target_repos[:8]:  # Limit to avoid timeout
                try:
                    logger.info(f"Cloning {repo}...")
                    clone_path = os.path.join(temp_dir, repo.replace('/', '_'))
                    
                    # Clone with depth 1 for speed
                    result = subprocess.run([
                        'git', 'clone', '--depth', '1', '--filter=blob:limit=100k',
                        f'https://github.com/{repo}.git', clone_path
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        repo_examples = self._extract_terraform_from_path(clone_path)
                        examples.extend(repo_examples[:5])  # Limit per repo
                        logger.info(f"Extracted {len(repo_examples)} configs from {repo}")
                    
                except Exception as e:
                    logger.debug(f"Failed to process {repo}: {e}")
        
        return examples

    def _collect_terraform_raw_files(self) -> List[Dict]:
        """Collect Terraform configs from GitHub using raw file URLs"""
        examples = []
        
        # Well-known Terraform paths in infrastructure projects
        known_terraform_paths = [
            # AWS provider examples
            ('hashicorp/terraform-provider-aws', 'examples/ec2-instance/main.tf'),
            ('hashicorp/terraform-provider-aws', 'examples/s3-bucket/main.tf'),
            ('hashicorp/terraform-provider-aws', 'examples/vpc/main.tf'),
            
            # Terraform modules
            ('terraform-aws-modules/terraform-aws-vpc', 'main.tf'),
            ('terraform-aws-modules/terraform-aws-vpc', 'variables.tf'),
            ('terraform-aws-modules/terraform-aws-eks', 'main.tf'),
            
            # Real infrastructure examples
            ('gruntwork-io/terragrunt-infrastructure-live-example', 'non-prod/us-east-1/qa/mysql/terragrunt.hcl'),
            ('cloudposse/terraform-aws-components', 'modules/eks/main.tf'),
            
            # Kubernetes
            ('terraform-providers/terraform-provider-kubernetes', 'examples/nginx/main.tf'),
            ('hashicorp/terraform-provider-helm', 'examples/nginx/main.tf'),
        ]
        
        for repo, tf_path in known_terraform_paths:
            try:
                example = self._fetch_terraform_raw_file(repo, tf_path)
                if example:
                    examples.append(example)
                    
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.debug(f"Failed to fetch {repo}/{tf_path}: {e}")
        
        return examples

    def _fetch_terraform_raw_file(self, repo: str, tf_path: str) -> Optional[Dict]:
        """Fetch a Terraform config from GitHub raw URL"""
        try:
            # Try main branch, then master
            for branch in ['main', 'master']:
                url = f"https://raw.githubusercontent.com/{repo}/{branch}/{tf_path}"
                
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.text
                    
                    # Quality checks
                    if not self._is_quality_terraform_config(content, tf_path):
                        return None
                    
                    # Check for duplicates
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    if content_hash in self.collected_hashes:
                        return None
                    
                    self.collected_hashes.add(content_hash)
                    
                    # Generate training example
                    prompt = self._generate_terraform_prompt(content, tf_path)
                    clean_content = self._clean_terraform_content(content)
                    
                    return {
                        "prompt": prompt,
                        "completion": f"```hcl\n{clean_content}\n```",
                        "metadata": {
                            "source": "github_raw",
                            "repository": repo,
                            "file_path": tf_path,
                            "category": self._categorize_terraform_config(content, tf_path),
                            "authentic": True
                        }
                    }
                    
        except Exception as e:
            logger.debug(f"Error fetching {repo}/{tf_path}: {e}")
        
        return None

    def _collect_stackoverflow_terraform(self) -> List[Dict]:
        """Collect Terraform snippets from Stack Overflow"""
        examples = []
        
        # Search for high-quality Terraform questions
        search_queries = [
            'terraform aws vpc subnet',
            'terraform kubernetes deployment',
            'terraform azure resource group',
            'terraform gcp compute instance',
            'terraform docker container',
            'terraform helm chart deployment',
            'terraform s3 bucket policy',
            'terraform security group rules',
            'terraform load balancer configuration',
            'terraform database rds configuration'
        ]
        
        for query in search_queries[:5]:  # Limit queries
            try:
                query_examples = self._search_stackoverflow_terraform(query)
                examples.extend(query_examples)
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"Stack Overflow search failed for '{query}': {e}")
        
        return examples

    def _search_stackoverflow_terraform(self, query: str) -> List[Dict]:
        """Search Stack Overflow for Terraform examples"""
        examples = []
        
        try:
            # Use Stack Exchange API
            api_url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'votes',
                'q': query,
                'accepted': 'True',
                'site': 'stackoverflow',
                'pagesize': 3,
                'filter': 'withbody'
            }
            
            response = requests.get(api_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('items', []):
                    # Extract Terraform code blocks from answers
                    if 'body' in item:
                        code_examples = self._extract_terraform_from_stackoverflow(item['body'])
                        for code in code_examples:
                            example = self._create_stackoverflow_terraform_example(code, item, query)
                            if example:
                                examples.append(example)
                                
        except Exception as e:
            logger.debug(f"Stack Overflow API error: {e}")
        
        return examples

    def _extract_terraform_from_stackoverflow(self, html_body: str) -> List[str]:
        """Extract Terraform code blocks from Stack Overflow HTML"""
        code_blocks = []
        
        # Simple regex to extract code blocks (between <pre><code> tags)
        code_pattern = r'<pre[^>]*><code[^>]*>(.*?)</code></pre>'
        matches = re.findall(code_pattern, html_body, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            # Clean HTML entities and tags
            clean_code = re.sub(r'<[^>]+>', '', match)
            clean_code = clean_code.replace('&lt;', '<').replace('&gt;', '>')
            clean_code = clean_code.replace('&amp;', '&').replace('&quot;', '"')
            
            # Check if it looks like Terraform config
            if self._looks_like_terraform(clean_code):
                code_blocks.append(clean_code.strip())
        
        return code_blocks

    def _looks_like_terraform(self, code: str) -> bool:
        """Check if code looks like Terraform configuration"""
        lines = code.split('\n')
        if len(lines) < 3:  # Too short
            return False
        
        # Terraform indicators
        terraform_indicators = [
            'resource ' in code,
            'provider ' in code,
            'variable ' in code,
            'output ' in code,
            'module ' in code,
            'data ' in code,
            any(line.strip().startswith('aws_') for line in lines),
            any(line.strip().startswith('azurerm_') for line in lines),
            any(line.strip().startswith('google_') for line in lines),
            'terraform {' in code,
            '.tf' in code.lower() or 'hcl' in code.lower(),
        ]
        
        return sum(terraform_indicators) >= 2

    def _create_stackoverflow_terraform_example(self, code: str, item: Dict, query: str) -> Optional[Dict]:
        """Create training example from Stack Overflow Terraform code"""
        try:
            # Quality checks
            if not self._is_quality_terraform_config(code, "stackoverflow"):
                return None
            
            # Check for duplicates
            content_hash = hashlib.md5(code.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            
            self.collected_hashes.add(content_hash)
            
            # Generate prompt based on question context
            title = item.get('title', '')
            prompt = self._generate_stackoverflow_terraform_prompt(title, query)
            
            clean_content = self._clean_terraform_content(code)
            
            return {
                "prompt": prompt,
                "completion": f"```hcl\n{clean_content}\n```",
                "metadata": {
                    "source": "stackoverflow",
                    "question_id": item.get('question_id', ''),
                    "title": title,
                    "score": item.get('score', 0),
                    "category": self._categorize_terraform_config(code, "stackoverflow"),
                    "authentic": True
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating Stack Overflow Terraform example: {e}")
            return None

    def _generate_terraform_prompt(self, content: str, file_path: str) -> str:
        """Generate contextual prompt from Terraform analysis"""
        content_lower = content.lower()
        filename = os.path.basename(file_path).lower()
        
        # Analyze providers
        if 'aws_' in content_lower:
            provider_context = "AWS"
        elif 'azurerm_' in content_lower:
            provider_context = "Azure"
        elif 'google_' in content_lower:
            provider_context = "Google Cloud"
        elif 'kubernetes_' in content_lower:
            provider_context = "Kubernetes"
        elif 'helm_' in content_lower:
            provider_context = "Helm"
        elif 'docker_' in content_lower:
            provider_context = "Docker"
        else:
            provider_context = "infrastructure"
        
        # Analyze resources
        if 'aws_vpc' in content_lower or 'vpc' in filename:
            return f"Create a Terraform configuration for {provider_context} VPC networking"
        elif 'aws_instance' in content_lower or 'ec2' in content_lower:
            return f"Write a Terraform configuration for {provider_context} compute instances"
        elif 'aws_s3_bucket' in content_lower or 's3' in content_lower:
            return f"Create a Terraform configuration for {provider_context} storage bucket"
        elif 'aws_rds' in content_lower or 'database' in content_lower:
            return f"Write a Terraform configuration for {provider_context} database"
        elif 'aws_eks' in content_lower or 'kubernetes' in content_lower:
            return f"Create a Terraform configuration for {provider_context} Kubernetes cluster"
        elif 'aws_lb' in content_lower or 'load_balancer' in content_lower:
            return f"Write a Terraform configuration for {provider_context} load balancer"
        elif 'security_group' in content_lower:
            return f"Create a Terraform configuration for {provider_context} security groups"
        elif 'module ' in content_lower:
            return f"Write a Terraform module for {provider_context} infrastructure"
        elif 'variable ' in content_lower and 'output ' in content_lower:
            return f"Create a Terraform configuration with variables and outputs for {provider_context}"
        else:
            return f"Write a Terraform configuration for {provider_context} infrastructure deployment"

    def _generate_stackoverflow_terraform_prompt(self, title: str, query: str) -> str:
        """Generate prompt from Stack Overflow question title"""
        title_lower = title.lower()
        
        if 'vpc' in title_lower:
            return "Create a Terraform configuration for VPC networking"
        elif 's3' in title_lower or 'bucket' in title_lower:
            return "Write a Terraform configuration for S3 bucket setup"
        elif 'kubernetes' in title_lower or 'k8s' in title_lower:
            return "Create a Terraform configuration for Kubernetes deployment"
        elif 'rds' in title_lower or 'database' in title_lower:
            return "Write a Terraform configuration for database infrastructure"
        elif 'security' in title_lower:
            return "Create a Terraform configuration for security groups"
        elif 'load balancer' in title_lower or 'alb' in title_lower:
            return "Write a Terraform configuration for load balancer setup"
        elif 'module' in title_lower:
            return "Create a reusable Terraform module"
        else:
            return "Write a Terraform configuration for infrastructure deployment"

    def _categorize_terraform_config(self, content: str, file_path: str) -> str:
        """Categorize Terraform config based on content and context"""
        content_lower = content.lower()
        filename = os.path.basename(file_path).lower()
        
        if 'aws_vpc' in content_lower or 'vpc' in filename:
            return 'networking'
        elif 'aws_instance' in content_lower or 'ec2' in content_lower:
            return 'compute'
        elif 'aws_s3' in content_lower or 's3' in content_lower:
            return 'storage'
        elif 'aws_rds' in content_lower or 'database' in content_lower:
            return 'database'
        elif 'aws_eks' in content_lower or 'kubernetes' in content_lower:
            return 'orchestration'
        elif 'security_group' in content_lower:
            return 'security'
        elif 'aws_lb' in content_lower or 'load_balancer' in content_lower:
            return 'load_balancing'
        elif 'module ' in content_lower:
            return 'module'
        elif 'variable ' in filename or 'variables.tf' in filename:
            return 'variables'
        elif 'output' in filename or 'outputs.tf' in filename:
            return 'outputs'
        elif 'provider' in content_lower:
            return 'provider'
        else:
            return 'infrastructure'

    def _clean_terraform_content(self, content: str) -> str:
        """Clean Terraform content while preserving authenticity"""
        # Basic cleaning - preserve real-world nature
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            # Remove extremely long lines
            if len(line) < 300:
                clean_lines.append(line.rstrip())
        
        content = '\n'.join(clean_lines)
        
        # Only sanitize obvious sensitive data
        content = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', content)
        content = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', content)
        content = re.sub(r'\b[a-f0-9]{32,64}\b', '<TOKEN>', content)
        content = re.sub(r'(password|secret|key)\s*=\s*"[^"]+"', r'\1 = "<REDACTED>"', content, flags=re.IGNORECASE)
        
        return content.strip()

    def _deduplicate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicate examples"""
        seen_hashes = set()
        unique_examples = []
        
        for example in examples:
            content = example['completion']
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_examples.append(example)
        
        return unique_examples

    def _filter_terraform_quality(self, examples: List[Dict]) -> List[Dict]:
        """Filter for highest quality Terraform examples"""
        quality_examples = []
        
        for example in examples:
            content = example['completion']
            
            # Extract Terraform content from markdown
            tf_content = content.replace('```hcl\n', '').replace('\n```', '')
            
            # Quality checks
            if self._is_quality_terraform_config(tf_content, "filter"):
                quality_examples.append(example)
        
        return quality_examples

    def save_corpus(self, examples: List[Dict], filename: str = "terraform_real_corpus.jsonl") -> None:
        """Save real-world Terraform corpus"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} real-world Terraform examples to {output_file}")


def main():
    """Main Terraform real-world collection pipeline"""
    logger.info("Starting Real-World Terraform Configuration Collection")
    logger.info("Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA")
    
    collector = TerraformRealCollector()
    
    try:
        # Collect real-world Terraform configs
        examples = collector.collect_real_terraform_configs(target_count=100)
        
        # Save corpus
        collector.save_corpus(examples)
        
        # Generate statistics
        sources = {}
        categories = {}
        providers = {}
        
        for example in examples:
            metadata = example.get('metadata', {})
            source = metadata.get('source', 'unknown')
            category = metadata.get('category', 'unknown')
            
            # Detect provider from content
            content = example['completion'].lower()
            if 'aws_' in content:
                provider = 'aws'
            elif 'azurerm_' in content:
                provider = 'azure'
            elif 'google_' in content:
                provider = 'gcp'
            elif 'kubernetes_' in content:
                provider = 'kubernetes'
            else:
                provider = 'multi_cloud'
            
            sources[source] = sources.get(source, 0) + 1
            categories[category] = categories.get(category, 0) + 1
            providers[provider] = providers.get(provider, 0) + 1
        
        logger.info(f"Real-World Terraform Collection Statistics:")
        logger.info(f"Total authentic examples: {len(examples)}")
        logger.info(f"Sources: {dict(sorted(sources.items()))}")
        logger.info(f"Categories: {dict(sorted(categories.items()))}")
        logger.info(f"Providers: {dict(sorted(providers.items()))}")
        
        logger.info("Real-world Terraform collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Real-world Terraform collection failed: {e}")
        raise


if __name__ == "__main__":
    main()