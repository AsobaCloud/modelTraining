#!/usr/bin/env python3
"""
Diverse Real-World Shell Script Collector
Collects authentic shell scripts from multiple real sources:
- Public GitHub repositories (top starred infrastructure projects)
- Open source projects (Docker, Kubernetes, etc.)
- Stack Overflow answers (code snippets)

Follows CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA
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
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiverseRealCollector:
    """Collects shell scripts from diverse real-world sources"""
    
    def __init__(self, output_dir: str = "diverse_real_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        
        # Target infrastructure repositories (high quality, real-world scripts)
        self.target_repos = [
            # Container/Docker
            'docker/docker',
            'docker/compose',
            'moby/moby',
            
            # Kubernetes ecosystem
            'kubernetes/kubernetes',
            'kubernetes/minikube',
            'helm/helm',
            'istio/istio',
            
            # Infrastructure as Code
            'hashicorp/terraform',
            'hashicorp/vault',
            'hashicorp/consul',
            'ansible/ansible',
            'pulumi/pulumi',
            
            # CI/CD
            'jenkins-x/jx',
            'argoproj/argo-cd',
            'tektoncd/pipeline',
            'fluxcd/flux2',
            
            # Monitoring
            'prometheus/prometheus',
            'grafana/grafana',
            'elastic/elasticsearch',
            'jaegertracing/jaeger',
            
            # Cloud providers
            'aws/aws-cli',
            'Azure/azure-cli',
            'GoogleCloudPlatform/cloud-sdk-docker',
            
            # Development tools
            'git/git',
            'nginx/nginx',
            'apache/httpd',
            'nodejs/node',
        ]

    def collect_diverse_real_scripts(self, target_count: int = 200, stackoverflow_only: bool = False) -> List[Dict]:
        """Collect diverse real-world shell scripts"""
        logger.info(f"Starting diverse real-world collection targeting {target_count} scripts")
        
        all_examples = []
        
        if not stackoverflow_only:
            # Strategy 1: Clone popular infrastructure repositories
            repo_examples = self._collect_from_repositories()
            all_examples.extend(repo_examples)
            logger.info(f"Collected {len(repo_examples)} from repositories")
            
            # Strategy 2: GitHub raw file collection (without API limits)
            github_examples = self._collect_github_raw_files()
            all_examples.extend(github_examples)
            logger.info(f"Collected {len(github_examples)} from GitHub raw files")
        
        # Strategy 3: Stack Overflow shell script snippets (expanded)
        stackoverflow_examples = self._collect_stackoverflow_snippets()
        all_examples.extend(stackoverflow_examples)
        logger.info(f"Collected {len(stackoverflow_examples)} from Stack Overflow")
        
        # Deduplicate and filter quality
        unique_examples = self._deduplicate_examples(all_examples)
        quality_examples = self._filter_real_quality(unique_examples)
        
        logger.info(f"Final diverse real-world collection: {len(quality_examples)} scripts")
        return quality_examples[:target_count]

    def _collect_from_repositories(self) -> List[Dict]:
        """Clone and extract scripts from real repositories"""
        examples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo in self.target_repos[:10]:  # Limit to avoid timeout
                try:
                    logger.info(f"Cloning {repo}...")
                    clone_path = os.path.join(temp_dir, repo.replace('/', '_'))
                    
                    # Clone with depth 1 for speed
                    result = subprocess.run([
                        'git', 'clone', '--depth', '1', '--filter=blob:limit=100k',
                        f'https://github.com/{repo}.git', clone_path
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        repo_scripts = self._extract_repo_scripts(clone_path, repo)
                        examples.extend(repo_scripts)
                        logger.info(f"Extracted {len(repo_scripts)} scripts from {repo}")
                    
                except Exception as e:
                    logger.debug(f"Failed to process {repo}: {e}")
        
        return examples

    def _extract_repo_scripts(self, repo_path: str, repo_name: str) -> List[Dict]:
        """Extract shell scripts from a cloned repository"""
        examples = []
        
        try:
            # Find shell scripts, prioritizing infrastructure-related ones
            find_patterns = [
                '*.sh',
                'scripts/*.sh',
                'hack/*.sh',
                'build/*.sh',
                'deploy/*.sh',
                'install/*.sh',
                'setup/*.sh'
            ]
            
            script_files = []
            for pattern in find_patterns:
                result = subprocess.run([
                    'find', repo_path, '-name', pattern, '-type', 'f',
                    '-not', '-path', '*/.*',  # Exclude hidden dirs
                    '-not', '-path', '*/vendor/*',
                    '-not', '-path', '*/node_modules/*'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    script_files.extend(result.stdout.strip().split('\n'))
            
            # Remove duplicates and empty entries
            script_files = list(set(f for f in script_files if f))
            
            # Process each script file
            for script_file in script_files[:5]:  # Limit per repo
                example = self._process_repo_script_file(script_file, repo_name)
                if example:
                    examples.append(example)
                    
        except Exception as e:
            logger.debug(f"Error extracting scripts from {repo_path}: {e}")
        
        return examples

    def _process_repo_script_file(self, file_path: str, repo_name: str) -> Optional[Dict]:
        """Process a script file from a repository"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Quality checks
            if not self._is_quality_infrastructure_script(content, file_path):
                return None
            
            # Check for duplicates
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            
            self.collected_hashes.add(content_hash)
            
            # Generate training example
            prompt = self._generate_prompt_from_script(content, file_path, repo_name)
            clean_content = self._clean_real_script_content(content)
            
            return {
                "prompt": prompt,
                "completion": f"```bash\n{clean_content}\n```",
                "metadata": {
                    "source": "github_repository",
                    "repository": repo_name,
                    "file_path": os.path.basename(file_path),
                    "category": self._categorize_script(content, file_path),
                    "authentic": True
                }
            }
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return None

    def _collect_github_raw_files(self) -> List[Dict]:
        """Collect shell scripts from GitHub using raw file URLs"""
        examples = []
        
        # Well-known shell script paths in infrastructure projects
        known_script_paths = [
            # Docker
            ('docker/docker', 'hack/make.sh'),
            ('docker/docker', 'scripts/build/binary'),
            ('docker/compose', 'script/build/linux'),
            
            # Kubernetes
            ('kubernetes/kubernetes', 'hack/local-up-cluster.sh'),
            ('kubernetes/kubernetes', 'cluster/gce/util.sh'),
            ('kubernetes/minikube', 'scripts/release.sh'),
            
            # Terraform
            ('hashicorp/terraform', 'scripts/build.sh'),
            ('hashicorp/vault', 'scripts/build.sh'),
            
            # Monitoring
            ('prometheus/prometheus', 'scripts/build_promu.sh'),
            ('grafana/grafana', 'scripts/build.sh'),
            
            # CI/CD
            ('argoproj/argo-cd', 'hack/install.sh'),
            ('tektoncd/pipeline', 'hack/release.sh'),
        ]
        
        for repo, script_path in known_script_paths:
            try:
                example = self._fetch_github_raw_script(repo, script_path)
                if example:
                    examples.append(example)
                    
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.debug(f"Failed to fetch {repo}/{script_path}: {e}")
        
        return examples

    def _fetch_github_raw_script(self, repo: str, script_path: str) -> Optional[Dict]:
        """Fetch a script from GitHub raw URL"""
        try:
            raw_url = f"https://raw.githubusercontent.com/{repo}/main/{script_path}"
            
            # Try main branch, then master
            for branch in ['main', 'master']:
                url = f"https://raw.githubusercontent.com/{repo}/{branch}/{script_path}"
                
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.text
                    
                    # Quality checks
                    if not self._is_quality_infrastructure_script(content, script_path):
                        return None
                    
                    # Check for duplicates
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    if content_hash in self.collected_hashes:
                        return None
                    
                    self.collected_hashes.add(content_hash)
                    
                    # Generate training example
                    prompt = self._generate_prompt_from_script(content, script_path, repo)
                    clean_content = self._clean_real_script_content(content)
                    
                    return {
                        "prompt": prompt,
                        "completion": f"```bash\n{clean_content}\n```",
                        "metadata": {
                            "source": "github_raw",
                            "repository": repo,
                            "file_path": script_path,
                            "category": self._categorize_script(content, script_path),
                            "authentic": True
                        }
                    }
                    
        except Exception as e:
            logger.debug(f"Error fetching {repo}/{script_path}: {e}")
        
        return None

    def _collect_stackoverflow_snippets(self) -> List[Dict]:
        """Collect shell script snippets from Stack Overflow"""
        examples = []
        
        # Extended search for high-quality shell script questions
        search_queries = [
            'bash script deployment aws',
            'shell script docker kubernetes',
            'bash automation infrastructure',
            'shell script ci cd pipeline',
            'bash monitoring logging script',
            'shell script backup automation',
            'bash script git hooks',
            'shell script system administration',
            'bash script nginx apache setup',
            'shell script database backup mysql',
            'bash script cron automation',
            'shell script ssh remote management',
            'bash script systemd service',
            'shell script terraform automation',
            'bash script ansible playbook',
            'shell script prometheus monitoring'
        ]
        
        for query in search_queries:  # Use all queries for more examples
            try:
                query_examples = self._search_stackoverflow(query)
                examples.extend(query_examples)
                time.sleep(1.5)  # Reduced rate limiting for more throughput
                
            except Exception as e:
                logger.debug(f"Stack Overflow search failed for '{query}': {e}")
        
        return examples

    def _search_stackoverflow(self, query: str) -> List[Dict]:
        """Search Stack Overflow for shell script examples"""
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
                'pagesize': 5,
                'filter': 'withbody'
            }
            
            response = requests.get(api_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('items', []):
                    # Extract code blocks from answers
                    if 'body' in item:
                        code_examples = self._extract_code_from_stackoverflow(item['body'])
                        for code in code_examples:
                            example = self._create_stackoverflow_example(code, item, query)
                            if example:
                                examples.append(example)
                                
        except Exception as e:
            logger.debug(f"Stack Overflow API error: {e}")
        
        return examples

    def _extract_code_from_stackoverflow(self, html_body: str) -> List[str]:
        """Extract shell code blocks from Stack Overflow HTML"""
        code_blocks = []
        
        # Simple regex to extract code blocks (between <pre><code> tags)
        code_pattern = r'<pre[^>]*><code[^>]*>(.*?)</code></pre>'
        matches = re.findall(code_pattern, html_body, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            # Clean HTML entities and tags
            clean_code = re.sub(r'<[^>]+>', '', match)
            clean_code = clean_code.replace('&lt;', '<').replace('&gt;', '>')
            clean_code = clean_code.replace('&amp;', '&').replace('&quot;', '"')
            
            # Check if it looks like shell script
            if self._looks_like_shell_script(clean_code):
                code_blocks.append(clean_code.strip())
        
        return code_blocks

    def _looks_like_shell_script(self, code: str) -> bool:
        """Check if code looks like a shell script"""
        lines = code.split('\n')
        if len(lines) < 3:  # Too short
            return False
        
        # Shell script indicators
        shell_indicators = [
            code.startswith('#!/bin/bash'),
            code.startswith('#!/bin/sh'),
            'bash' in code.lower(),
            any(line.strip().startswith('$') for line in lines),
            any(cmd in code for cmd in ['aws', 'docker', 'kubectl', 'git', 'ssh', 'rsync']),
            any(pattern in code for pattern in ['if [', 'for ', 'while ', 'function ', 'echo ']),
        ]
        
        return sum(shell_indicators) >= 2

    def _create_stackoverflow_example(self, code: str, item: Dict, query: str) -> Optional[Dict]:
        """Create training example from Stack Overflow code"""
        try:
            # Quality checks
            if not self._is_quality_infrastructure_script(code, "stackoverflow"):
                return None
            
            # Check for duplicates
            content_hash = hashlib.md5(code.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            
            self.collected_hashes.add(content_hash)
            
            # Generate prompt based on question context
            title = item.get('title', '')
            prompt = self._generate_stackoverflow_prompt(title, query)
            
            clean_content = self._clean_real_script_content(code)
            
            return {
                "prompt": prompt,
                "completion": f"```bash\n{clean_content}\n```",
                "metadata": {
                    "source": "stackoverflow",
                    "question_id": item.get('question_id', ''),
                    "title": title,
                    "score": item.get('score', 0),
                    "category": self._categorize_script(code, "stackoverflow"),
                    "authentic": True
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating Stack Overflow example: {e}")
            return None

    def _generate_stackoverflow_prompt(self, title: str, query: str) -> str:
        """Generate prompt from Stack Overflow question title"""
        title_lower = title.lower()
        
        if 'deploy' in title_lower:
            return "Write a shell script for deployment automation"
        elif 'docker' in title_lower:
            return "Create a shell script for Docker container management"
        elif 'kubernetes' in title_lower or 'k8s' in title_lower:
            return "Write a shell script for Kubernetes operations"
        elif 'aws' in title_lower:
            return "Create a shell script for AWS infrastructure management"
        elif 'backup' in title_lower:
            return "Write a shell script for backup automation"
        elif 'monitoring' in title_lower:
            return "Create a shell script for monitoring setup"
        elif 'ci' in title_lower or 'cd' in title_lower:
            return "Write a shell script for CI/CD pipeline"
        elif 'git' in title_lower:
            return "Create a shell script for Git workflow automation"
        else:
            return "Write a shell script for infrastructure automation"

    def _is_quality_infrastructure_script(self, content: str, file_path: str) -> bool:
        """Check if script is high quality and infrastructure-related"""
        if not content or len(content) < 50:
            return False
        
        lines = content.split('\n')
        line_count = len(lines)
        
        # Size checks
        if line_count < 5 or line_count > 500:
            return False
        
        # Infrastructure relevance check
        infra_keywords = [
            'docker', 'kubernetes', 'kubectl', 'helm', 'terraform', 'ansible',
            'aws', 'gcp', 'azure', 'deploy', 'build', 'install', 'setup',
            'monitoring', 'prometheus', 'grafana', 'jenkins', 'git', 'ci', 'cd',
            'systemctl', 'service', 'nginx', 'apache', 'backup', 'restore'
        ]
        
        content_lower = content.lower()
        keyword_matches = sum(1 for keyword in infra_keywords if keyword in content_lower)
        
        if keyword_matches < 2:
            return False
        
        # Avoid obviously generated or template scripts
        bad_patterns = [
            'this file was automatically generated',
            'do not edit this file',
            'generated by',
            'auto-generated',
            'template file'
        ]
        
        for pattern in bad_patterns:
            if pattern in content_lower:
                return False
        
        return True

    def _generate_prompt_from_script(self, content: str, file_path: str, repo_name: str) -> str:
        """Generate contextual prompt from script analysis"""
        content_lower = content.lower()
        filename = os.path.basename(file_path).lower()
        
        # Context from repository
        repo_context = ""
        if 'docker' in repo_name:
            repo_context = "Docker container"
        elif 'kubernetes' in repo_name:
            repo_context = "Kubernetes cluster"
        elif 'terraform' in repo_name:
            repo_context = "Terraform infrastructure"
        elif 'prometheus' in repo_name or 'grafana' in repo_name:
            repo_context = "monitoring"
        elif 'jenkins' in repo_name or 'argo' in repo_name:
            repo_context = "CI/CD"
        
        # Context from filename
        if 'build' in filename:
            return f"Write a shell script to build {repo_context} components"
        elif 'deploy' in filename:
            return f"Create a shell script to deploy {repo_context} infrastructure"
        elif 'install' in filename or 'setup' in filename:
            return f"Write a shell script to install and setup {repo_context}"
        elif 'release' in filename:
            return f"Create a shell script for {repo_context} release automation"
        elif 'test' in filename:
            return f"Write a shell script for {repo_context} testing automation"
        
        # Context from content
        if 'docker build' in content_lower:
            return "Create a shell script for Docker image building and deployment"
        elif 'kubectl' in content_lower:
            return "Write a shell script for Kubernetes resource management"
        elif 'terraform' in content_lower:
            return "Create a shell script for Terraform infrastructure deployment"
        elif 'aws' in content_lower:
            return "Write a shell script for AWS infrastructure automation"
        
        return f"Create a shell script for {repo_context} automation"

    def _categorize_script(self, content: str, file_path: str) -> str:
        """Categorize script based on content and context"""
        content_lower = content.lower()
        filename = os.path.basename(file_path).lower()
        
        if 'deploy' in filename or 'deploy' in content_lower:
            return 'deployment'
        elif 'build' in filename or 'build' in content_lower:
            return 'build'
        elif 'test' in filename or 'test' in content_lower:
            return 'testing'
        elif 'docker' in content_lower:
            return 'containerization'
        elif 'kubernetes' in content_lower or 'kubectl' in content_lower:
            return 'orchestration'
        elif 'aws' in content_lower:
            return 'aws_operations'
        elif 'monitor' in content_lower or 'prometheus' in content_lower:
            return 'monitoring'
        elif 'backup' in content_lower:
            return 'backup'
        elif 'install' in filename or 'setup' in filename:
            return 'infrastructure'
        else:
            return 'automation'

    def _clean_real_script_content(self, content: str) -> str:
        """Clean real script content while preserving authenticity"""
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

    def _filter_real_quality(self, examples: List[Dict]) -> List[Dict]:
        """Filter for highest quality real examples"""
        quality_examples = []
        
        for example in examples:
            content = example['completion']
            
            # Extract script content from markdown
            script_content = content.replace('```bash\n', '').replace('\n```', '')
            
            # Quality checks
            if self._is_quality_infrastructure_script(script_content, "filter"):
                quality_examples.append(example)
        
        return quality_examples

    def save_corpus(self, examples: List[Dict], filename: str = "diverse_real_corpus.jsonl") -> None:
        """Save diverse real-world corpus"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} diverse real-world examples to {output_file}")


def main():
    """Main diverse real-world collection pipeline"""
    import sys
    
    # Check for Stack Overflow only mode
    stackoverflow_only = len(sys.argv) > 1 and sys.argv[1] == "--stackoverflow-only"
    
    if stackoverflow_only:
        logger.info("Starting Stack Overflow-Only Shell Script Collection")
        logger.info("Source: Stack Overflow only")
    else:
        logger.info("Starting Diverse Real-World Shell Script Collection")
        logger.info("Sources: GitHub repos, open source projects, Stack Overflow")
    
    collector = DiverseRealCollector()
    
    try:
        # Collect diverse real-world scripts
        examples = collector.collect_diverse_real_scripts(target_count=50 if stackoverflow_only else 200, 
                                                         stackoverflow_only=stackoverflow_only)
        
        # Save corpus
        filename = "stackoverflow_only_corpus.jsonl" if stackoverflow_only else "diverse_real_corpus.jsonl"
        collector.save_corpus(examples, filename)
        
        # Generate statistics
        sources = {}
        categories = {}
        repos = {}
        
        for example in examples:
            metadata = example.get('metadata', {})
            source = metadata.get('source', 'unknown')
            category = metadata.get('category', 'unknown')
            repo = metadata.get('repository', 'unknown')
            
            sources[source] = sources.get(source, 0) + 1
            categories[category] = categories.get(category, 0) + 1
            repos[repo] = repos.get(repo, 0) + 1
        
        logger.info(f"Collection Statistics:")
        logger.info(f"Total examples: {len(examples)}")
        logger.info(f"Sources: {dict(sorted(sources.items()))}")
        logger.info(f"Categories: {dict(sorted(categories.items()))}")
        logger.info(f"Top repositories: {dict(sorted(repos.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        logger.info("Collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()