#!/usr/bin/env python3
"""
GitHub Search Collector for Shell Scripts
Aggressive collection strategy to reach 400 shell script target

Uses multiple search strategies and rate limit management
"""

import os
import json
import time
import requests
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubSearchCollector:
    """Aggressive GitHub search for infrastructure shell scripts"""
    
    def __init__(self, output_dir: str = "github_search_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.base_url = "https://api.github.com/search/code"
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Rate limiting
        self.requests_made = 0
        self.last_request_time = 0
        self.rate_limit_delay = 2  # seconds between requests
        
        # Search strategies for infrastructure scripts
        self.search_queries = [
            # AWS Infrastructure
            'filename:deploy.sh aws cloudformation',
            'filename:setup.sh aws s3',
            'filename:install.sh aws ec2',
            'filename:build.sh aws ecr',
            'filename:deploy.sh terraform aws',
            'aws cli script infrastructure',
            'cloudformation deploy script',
            'aws lambda deploy shell',
            'ec2 instance launch script',
            's3 backup automation script',
            
            # Kubernetes & Container Infrastructure
            'filename:deploy.sh kubernetes',
            'filename:setup.sh kubectl',
            'filename:install.sh docker',
            'kubernetes cluster setup script',
            'helm chart deploy script',
            'dockerfile build script',
            'container deployment automation',
            'k8s manifest deploy',
            'istio service mesh setup',
            'ingress controller setup',
            
            # CI/CD & DevOps
            'filename:deploy.sh jenkins',
            'filename:build.sh gitlab',
            'github actions deploy script',
            'ci cd pipeline shell',
            'deployment automation script',
            'infrastructure pipeline script',
            'automated testing deploy',
            'release deployment script',
            
            # Monitoring & Operations
            'filename:monitor.sh prometheus',
            'filename:setup.sh grafana',
            'monitoring stack deploy',
            'alerting setup script',
            'logging infrastructure script',
            'metrics collection script',
            'observability stack setup',
            
            # Infrastructure as Code
            'infrastructure provision script',
            'terraform wrapper script',
            'pulumi deployment script',
            'ansible playbook script',
            'infrastructure automation',
            'cloud resource provision',
            'multi cloud deploy script',
            
            # Specific Cloud Providers
            'gcp deployment script',
            'azure infrastructure script',
            'digitalocean setup script',
            'aws cdk deploy script',
            'serverless deploy script',
            
            # Database & Storage
            'database setup script',
            'backup automation script',
            'storage provisioning script',
            'database migration script',
            'redis cluster setup',
            'postgresql setup script',
            
            # Security & Compliance
            'security scan script',
            'vulnerability check script',
            'compliance audit script',
            'ssl certificate script',
            'secrets management script',
            
            # Networking
            'network setup script',
            'vpn configuration script',
            'load balancer setup',
            'dns configuration script',
            'firewall rules script'
        ]

    def collect_github_examples(self, target_count: int = 310) -> List[Dict]:
        """Aggressive collection to reach target count"""
        logger.info(f"Starting GitHub search collection targeting {target_count} examples")
        
        collected_count = 0
        query_index = 0
        
        while collected_count < target_count and query_index < len(self.search_queries):
            query = self.search_queries[query_index]
            logger.info(f"Searching query {query_index + 1}/{len(self.search_queries)}: {query}")
            
            try:
                query_examples = self._search_and_collect(query, max_per_query=20)
                new_examples = self._deduplicate_and_add(query_examples)
                collected_count += new_examples
                
                logger.info(f"Query yielded {new_examples} new examples. Total: {len(self.examples)}")
                
                # Rate limiting
                self._respect_rate_limits()
                
            except Exception as e:
                logger.warning(f"Query failed: {query[:50]}... Error: {e}")
                time.sleep(5)  # Longer delay on error
            
            query_index += 1
            
            # Checkpoint every 50 examples
            if len(self.examples) % 50 == 0 and len(self.examples) > 0:
                self._save_checkpoint()
        
        logger.info(f"GitHub search completed. Collected {len(self.examples)} total examples")
        return self.examples

    def _search_and_collect(self, query: str, max_per_query: int = 20) -> List[Dict]:
        """Search GitHub and collect examples from a single query"""
        examples = []
        
        params = {
            'q': f'{query} language:Shell size:>100 size:<10000',
            'sort': 'stars',
            'order': 'desc',
            'per_page': min(max_per_query, 100)
        }
        
        try:
            response = self._make_api_request(self.base_url, params)
            
            if response and response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                logger.debug(f"Found {len(items)} files for query: {query[:30]}...")
                
                for item in items:
                    try:
                        example = self._process_github_item(item, query)
                        if example:
                            examples.append(example)
                    except Exception as e:
                        logger.debug(f"Failed to process item: {e}")
                        
            elif response and response.status_code == 403:
                logger.warning("Hit GitHub rate limit. Waiting...")
                time.sleep(60)  # Wait 1 minute on rate limit
                
        except Exception as e:
            logger.debug(f"Search request failed: {e}")
        
        return examples

    def _make_api_request(self, url: str, params: Dict) -> Optional[requests.Response]:
        """Make API request with rate limiting"""
        # Respect rate limits
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        try:
            response = requests.get(url, params=params, timeout=15)
            self.requests_made += 1
            self.last_request_time = time.time()
            
            # Check if we're approaching rate limits
            remaining = response.headers.get('X-RateLimit-Remaining')
            if remaining and int(remaining) < 10:
                logger.warning(f"Rate limit low: {remaining} remaining")
                time.sleep(10)
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request failed: {e}")
            return None

    def _process_github_item(self, item: Dict, query: str) -> Optional[Dict]:
        """Process a GitHub search result item"""
        try:
            # Get file metadata
            file_path = item.get('path', '')
            repository = item.get('repository', {})
            repo_name = repository.get('full_name', 'unknown')
            download_url = item.get('download_url')
            
            if not download_url:
                return None
            
            # Skip certain patterns
            skip_patterns = [
                'node_modules/', 'vendor/', '.git/', 'test/', 'tests/',
                'examples/', 'sample/', 'demo/', 'tmp/', 'temp/'
            ]
            
            if any(pattern in file_path.lower() for pattern in skip_patterns):
                return None
            
            # Get file content
            content_response = self._make_api_request(download_url, {})
            if not content_response or content_response.status_code != 200:
                return None
            
            content = content_response.text
            
            # Quality checks
            if not self._is_quality_script(content, file_path):
                return None
            
            # Generate training example
            example = self._create_training_example(content, file_path, repo_name, query)
            return example
            
        except Exception as e:
            logger.debug(f"Failed to process GitHub item: {e}")
            return None

    def _is_quality_script(self, content: str, file_path: str) -> bool:
        """Check if script meets quality criteria"""
        lines = content.split('\n')
        line_count = len(lines)
        
        # Size checks
        if line_count < 10 or line_count > 500:
            return False
        
        if len(content) < 200 or len(content) > 20000:
            return False
        
        # Must have shebang or be .sh file
        has_shebang = content.startswith('#!')
        is_sh_file = file_path.endswith('.sh')
        
        if not (has_shebang or is_sh_file):
            return False
        
        # Infrastructure relevance check
        infra_keywords = [
            'aws', 'docker', 'kubernetes', 'terraform', 'ansible',
            'deploy', 'setup', 'install', 'build', 'provision',
            'cloudformation', 'helm', 'kubectl', 'infrastructure',
            'monitoring', 'grafana', 'prometheus', 'jenkins',
            'gitlab', 'github', 'ci', 'cd', 'pipeline'
        ]
        
        content_lower = content.lower()
        keyword_count = sum(1 for keyword in infra_keywords if keyword in content_lower)
        
        if keyword_count < 2:  # Must have at least 2 infrastructure keywords
            return False
        
        # Avoid generated scripts
        generated_patterns = [
            'auto-generated', 'generated by', 'do not edit',
            'autogenerated', 'this file was automatically',
            '# WARNING: do not modify'
        ]
        
        if any(pattern in content_lower for pattern in generated_patterns):
            return False
        
        return True

    def _create_training_example(self, content: str, file_path: str, repo_name: str, query: str) -> Dict:
        """Create training example from script"""
        # Clean content
        clean_content = self._clean_script_content(content)
        
        # Generate prompt based on script analysis
        prompt = self._generate_contextual_prompt(clean_content, file_path, query)
        
        # Determine category
        category = self._categorize_script(clean_content, file_path, query)
        
        return {
            "prompt": prompt,
            "completion": f"```bash\n{clean_content}\n```",
            "metadata": {
                "source": "github_search",
                "repository": repo_name,
                "file_path": file_path,
                "query": query,
                "category": category
            }
        }

    def _generate_contextual_prompt(self, content: str, file_path: str, query: str) -> str:
        """Generate contextual prompt based on script content"""
        content_lower = content.lower()
        filename = os.path.basename(file_path).lower()
        
        # Analyze script purpose
        if 'aws' in content_lower and 'deploy' in content_lower:
            return "Write a shell script to deploy AWS infrastructure using CloudFormation and AWS CLI"
        elif 'kubernetes' in content_lower and 'deploy' in content_lower:
            return "Create a shell script to deploy applications to Kubernetes cluster"
        elif 'docker' in content_lower and 'build' in content_lower:
            return "Write a shell script to build and deploy Docker containers"
        elif 'monitoring' in content_lower or 'prometheus' in content_lower:
            return "Create a shell script to setup monitoring infrastructure"
        elif 'terraform' in content_lower:
            return "Write a shell script to manage Terraform infrastructure deployment"
        elif 'backup' in content_lower:
            return "Create a shell script for automated infrastructure backup"
        elif 'setup' in filename or 'install' in filename:
            return "Write a shell script to setup and configure infrastructure components"
        elif 'ci' in content_lower or 'pipeline' in content_lower:
            return "Create a shell script for CI/CD pipeline automation"
        elif 'security' in content_lower:
            return "Write a shell script for security scanning and compliance"
        elif 'database' in content_lower:
            return "Create a shell script for database setup and management"
        
        # Fallback based on query
        if 'aws' in query:
            return "Write a shell script for AWS cloud infrastructure automation"
        elif 'kubernetes' in query:
            return "Create a shell script for Kubernetes cluster management"
        elif 'docker' in query:
            return "Write a shell script for Docker container management"
        elif 'deploy' in query:
            return "Create a shell script for infrastructure deployment automation"
        
        return "Write a shell script for infrastructure automation and management"

    def _categorize_script(self, content: str, file_path: str, query: str) -> str:
        """Categorize script based on content analysis"""
        content_lower = content.lower()
        
        if 'deploy' in content_lower or 'deployment' in content_lower:
            return 'deployment'
        elif 'aws' in content_lower and any(svc in content_lower for svc in ['ec2', 's3', 'cloudformation', 'lambda']):
            return 'aws_operations'
        elif 'monitor' in content_lower or 'prometheus' in content_lower or 'grafana' in content_lower:
            return 'monitoring'
        elif 'kubernetes' in content_lower or 'kubectl' in content_lower:
            return 'orchestration'
        elif 'docker' in content_lower or 'container' in content_lower:
            return 'containerization'
        elif 'terraform' in content_lower or 'infrastructure' in content_lower:
            return 'infrastructure'
        elif 'backup' in content_lower or 'restore' in content_lower:
            return 'backup'
        elif 'security' in content_lower or 'audit' in content_lower:
            return 'security'
        elif 'database' in content_lower or 'db' in content_lower:
            return 'database'
        elif 'network' in content_lower or 'vpn' in content_lower:
            return 'networking'
        else:
            return 'general'

    def _clean_script_content(self, content: str) -> str:
        """Clean and sanitize script content"""
        import re
        
        # Remove excessive whitespace
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            # Remove lines that are too long (likely not real commands)
            if len(line) < 300:
                clean_lines.append(line.rstrip())
        
        content = '\n'.join(clean_lines)
        
        # Replace sensitive patterns with placeholders
        content = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', content)
        content = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', content)
        content = re.sub(r'https?://(?!.*\.(amazonaws\.com|k8s\.io|docker\.(com|io)|github\.com))[^\s"\']+', '<URL>', content)
        content = re.sub(r'\bi-[0-9a-f]{8,17}\b', '<INSTANCE_ID>', content)
        content = re.sub(r'\bami-[0-9a-f]{8,17}\b', '<AMI_ID>', content)
        content = re.sub(r'\bsg-[0-9a-f]{8,17}\b', '<SECURITY_GROUP_ID>', content)
        content = re.sub(r'\bsubnet-[0-9a-f]{8,17}\b', '<SUBNET_ID>', content)
        content = re.sub(r'\bvpc-[0-9a-f]{8,17}\b', '<VPC_ID>', content)
        
        return content.strip()

    def _deduplicate_and_add(self, new_examples: List[Dict]) -> int:
        """Add new examples, avoiding duplicates"""
        added_count = 0
        
        for example in new_examples:
            # Create hash from content
            content = example['completion']
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in self.collected_hashes:
                self.collected_hashes.add(content_hash)
                self.examples.append(example)
                added_count += 1
        
        return added_count

    def _respect_rate_limits(self):
        """Implement rate limiting strategy"""
        # Random delay to avoid patterns
        delay = random.uniform(1, 3)
        time.sleep(delay)
        
        # Longer delay every 30 requests
        if self.requests_made % 30 == 0:
            logger.info("Taking rate limit break...")
            time.sleep(10)

    def _save_checkpoint(self):
        """Save progress checkpoint"""
        checkpoint_file = self.output_dir / f"checkpoint_{len(self.examples)}.jsonl"
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            for example in self.examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Checkpoint saved: {len(self.examples)} examples")

    def save_corpus(self, filename: str = "github_search_corpus.jsonl") -> None:
        """Save final corpus"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in self.examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(self.examples)} GitHub search examples to {output_file}")


def main():
    """Main GitHub search collection pipeline"""
    logger.info("Starting GitHub Search Collection for Shell Scripts")
    
    collector = GitHubSearchCollector()
    
    try:
        # Target: 310 examples to reach 400 total
        examples = collector.collect_github_examples(target_count=310)
        
        # Save corpus
        collector.save_corpus()
        
        # Generate statistics
        categories = {}
        sources = {}
        
        for example in examples:
            metadata = example.get('metadata', {})
            category = metadata.get('category', 'unknown')
            repo = metadata.get('repository', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
            sources[repo] = sources.get(repo, 0) + 1
        
        logger.info(f"GitHub Search Collection Statistics:")
        logger.info(f"Total examples: {len(examples)}")
        logger.info(f"Categories: {dict(sorted(categories.items()))}")
        logger.info(f"Top repositories: {dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        logger.info("GitHub search collection completed successfully!")
        
    except Exception as e:
        logger.error(f"GitHub search collection failed: {e}")
        raise


if __name__ == "__main__":
    main()