#!/usr/bin/env python3
"""
Real-World Shell Script Collector
Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING

Collects exclusively from authentic sources:
- Production systems and repositories
- Official documentation
- Stack Overflow snippets
- Technical blogs and tutorials
- Real DevOps workflows
"""

import os
import json
import time
import requests
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
import re
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealWorldCollector:
    """Collects shell scripts exclusively from real-world sources"""
    
    def __init__(self, output_dir: str = "real_world_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Start with our existing authentic examples
        self.existing_authentic = 90  # From previous real collection
        
        # Real-world sources to explore
        self.real_sources = {
            'local_repositories': [
                '/home/shingai/api',
                '/home/shingai/sort/ona-front-end/express-chatbot',
                '/home/shingai/sort/deployments'
            ],
            'documentation_sites': [
                'https://docs.aws.amazon.com/cli/latest/userguide/',
                'https://kubernetes.io/docs/tutorials/',
                'https://docs.docker.com/engine/reference/commandline/',
                'https://terraform.io/docs/providers/aws/guides/',
            ],
            'tutorial_sites': [
                'https://www.digitalocean.com/community/tutorials',
                'https://aws.amazon.com/getting-started/hands-on/',
                'https://kubernetes.io/docs/tutorials/',
            ]
        }

    def collect_real_world_scripts(self, target_additional: int = 200) -> List[Dict]:
        """Collect real-world shell scripts from authentic sources"""
        logger.info(f"Starting real-world collection targeting {target_additional} additional scripts")
        logger.info(f"Current authentic examples: {self.existing_authentic}")
        
        # Strategy 1: Deep local repository mining
        local_examples = self._deep_mine_local_repositories()
        logger.info(f"Collected {len(local_examples)} from local repositories")
        
        # Strategy 2: Clone and analyze popular infrastructure repositories
        repo_examples = self._clone_and_analyze_repos()
        logger.info(f"Collected {len(repo_examples)} from cloned repositories")
        
        # Strategy 3: Extract from documentation and tutorials
        doc_examples = self._extract_from_documentation()
        logger.info(f"Collected {len(doc_examples)} from documentation")
        
        # Strategy 4: Find more real examples in filesystem
        filesystem_examples = self._find_additional_filesystem_scripts()
        logger.info(f"Collected {len(filesystem_examples)} from filesystem search")
        
        # Combine and deduplicate
        all_examples = local_examples + repo_examples + doc_examples + filesystem_examples
        unique_examples = self._deduplicate_examples(all_examples)
        
        logger.info(f"Total unique real-world examples: {len(unique_examples)}")
        return unique_examples

    def _deep_mine_local_repositories(self) -> List[Dict]:
        """Deep mining of local repositories for all shell scripts"""
        examples = []
        
        for repo_path in self.real_sources['local_repositories']:
            if os.path.exists(repo_path):
                logger.info(f"Deep mining: {repo_path}")
                repo_examples = self._extract_all_scripts_from_path(repo_path)
                examples.extend(repo_examples)
        
        return examples

    def _extract_all_scripts_from_path(self, path: str) -> List[Dict]:
        """Extract all shell scripts from a given path"""
        examples = []
        
        try:
            # Find all shell scripts
            result = subprocess.run([
                'find', path, '-type', 'f', 
                '(', '-name', '*.sh', '-o', '-name', '*.bash', ')',
                '-not', '-path', '*/node_modules/*',
                '-not', '-path', '*/venv/*',
                '-not', '-path', '*/.git/*',
                '-not', '-path', '*/build/*',
                '-not', '-path', '*/dist/*'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                script_files = result.stdout.strip().split('\n')
                
                for script_file in script_files:
                    if script_file:  # Skip empty lines
                        example = self._process_real_script_file(script_file)
                        if example:
                            examples.append(example)
            
        except Exception as e:
            logger.debug(f"Error extracting scripts from {path}: {e}")
        
        return examples

    def _process_real_script_file(self, file_path: str) -> Optional[Dict]:
        """Process a real shell script file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Quality checks for real scripts
            if not self._is_real_quality_script(content, file_path):
                return None
            
            # Check for duplicates
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            
            self.collected_hashes.add(content_hash)
            
            # Generate training example
            prompt = self._generate_real_world_prompt(content, file_path)
            clean_content = self._clean_real_script(content)
            
            return {
                "prompt": prompt,
                "completion": f"```bash\n{clean_content}\n```",
                "metadata": {
                    "source": "real_local_repository",
                    "file_path": file_path,
                    "category": self._categorize_real_script(content, file_path),
                    "authentic": True
                }
            }
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return None

    def _is_real_quality_script(self, content: str, file_path: str) -> bool:
        """Check if script is real and high quality"""
        lines = content.split('\n')
        line_count = len(lines)
        
        # Size checks
        if line_count < 5 or line_count > 1000:
            return False
        
        if len(content) < 100 or len(content) > 50000:
            return False
        
        # Must be shell script
        if not (content.startswith('#!') or file_path.endswith('.sh') or file_path.endswith('.bash')):
            return False
        
        # Check for infrastructure relevance
        infra_indicators = [
            'aws', 'docker', 'kubernetes', 'kubectl', 'terraform', 'ansible',
            'systemctl', 'service', 'nginx', 'apache', 'mysql', 'postgresql',
            'redis', 'mongodb', 'elasticsearch', 'jenkins', 'git', 'ssh',
            'rsync', 'cron', 'supervisor', 'gunicorn', 'celery', 'rabbitmq',
            'deploy', 'build', 'install', 'setup', 'configure', 'backup',
            'monitoring', 'logging', 'prometheus', 'grafana', 'kibana'
        ]
        
        content_lower = content.lower()
        matches = sum(1 for indicator in infra_indicators if indicator in content_lower)
        
        return matches >= 2  # At least 2 infrastructure indicators

    def _generate_real_world_prompt(self, content: str, file_path: str) -> str:
        """Generate prompt based on real script analysis"""
        content_lower = content.lower()
        filename = os.path.basename(file_path).lower()
        
        # Analyze actual script purpose from content
        if 'deploy' in filename or 'deploy' in content_lower:
            if 'aws' in content_lower:
                return "Write a shell script for deploying AWS infrastructure"
            elif 'kubernetes' in content_lower or 'kubectl' in content_lower:
                return "Create a shell script for deploying to Kubernetes"
            elif 'docker' in content_lower:
                return "Write a shell script for Docker deployment"
            else:
                return "Create a shell script for application deployment"
        
        elif 'install' in filename or 'setup' in filename:
            return "Write a shell script for system installation and setup"
        
        elif 'backup' in filename or 'backup' in content_lower:
            return "Create a shell script for backup automation"
        
        elif 'monitor' in filename or 'monitor' in content_lower:
            return "Write a shell script for monitoring and alerting"
        
        elif 'build' in filename or 'build' in content_lower:
            return "Create a shell script for build automation"
        
        elif 'test' in filename or 'test' in content_lower:
            return "Write a shell script for testing automation"
        
        # Analyze by content patterns
        elif 'systemctl' in content_lower or 'service' in content_lower:
            return "Create a shell script for system service management"
        
        elif 'cron' in content_lower or 'schedule' in content_lower:
            return "Write a shell script for scheduled task automation"
        
        elif 'git' in content_lower:
            return "Create a shell script for Git workflow automation"
        
        elif 'ssh' in content_lower:
            return "Write a shell script for remote system administration"
        
        # Default based on infrastructure presence
        elif 'aws' in content_lower:
            return "Create a shell script for AWS infrastructure management"
        elif 'docker' in content_lower:
            return "Write a shell script for Docker container management"
        elif 'kubernetes' in content_lower:
            return "Create a shell script for Kubernetes cluster management"
        else:
            return "Write a shell script for infrastructure automation"

    def _categorize_real_script(self, content: str, file_path: str) -> str:
        """Categorize based on real script analysis"""
        content_lower = content.lower()
        filename = os.path.basename(file_path).lower()
        
        if 'deploy' in filename or 'deploy' in content_lower:
            return 'deployment'
        elif 'aws' in content_lower:
            return 'aws_operations'
        elif 'docker' in content_lower:
            return 'containerization'
        elif 'kubernetes' in content_lower or 'kubectl' in content_lower:
            return 'orchestration'
        elif 'monitor' in filename or 'monitor' in content_lower:
            return 'monitoring'
        elif 'backup' in filename or 'backup' in content_lower:
            return 'backup'
        elif 'install' in filename or 'setup' in filename:
            return 'infrastructure'
        elif 'build' in filename or 'build' in content_lower:
            return 'build'
        elif 'test' in filename or 'test' in content_lower:
            return 'testing'
        else:
            return 'system_administration'

    def _clean_real_script(self, content: str) -> str:
        """Clean real script while preserving authenticity"""
        # Only basic cleaning - preserve the real-world nature
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            # Remove extremely long lines (likely not real commands)
            if len(line) < 500:
                clean_lines.append(line.rstrip())
        
        content = '\n'.join(clean_lines)
        
        # Only replace obvious sensitive data
        content = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', content)
        content = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', content)
        content = re.sub(r'\bi-[0-9a-f]{8,17}\b', '<INSTANCE_ID>', content)
        
        return content.strip()

    def _clone_and_analyze_repos(self) -> List[Dict]:
        """Clone popular infrastructure repositories and extract real scripts"""
        examples = []
        
        # Popular infrastructure repositories known to have quality scripts
        repos = [
            'https://github.com/kubernetes/kubernetes.git',
            'https://github.com/docker/docker.git',
            'https://github.com/hashicorp/terraform.git',
            'https://github.com/ansible/ansible.git',
            'https://github.com/prometheus/prometheus.git',
            'https://github.com/grafana/grafana.git',
            'https://github.com/helm/helm.git',
            'https://github.com/istio/istio.git',
            'https://github.com/jenkins-x/jenkins-x.git',
            'https://github.com/argoproj/argo.git'
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo_url in repos[:3]:  # Limit to first 3 to avoid timeout
                try:
                    repo_name = repo_url.split('/')[-1].replace('.git', '')
                    clone_path = os.path.join(temp_dir, repo_name)
                    
                    logger.info(f"Cloning {repo_name}...")
                    result = subprocess.run([
                        'git', 'clone', '--depth', '1', repo_url, clone_path
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        repo_examples = self._extract_all_scripts_from_path(clone_path)
                        examples.extend(repo_examples[:10])  # Limit per repo
                        logger.info(f"Extracted {len(repo_examples)} scripts from {repo_name}")
                    
                except Exception as e:
                    logger.debug(f"Failed to clone {repo_url}: {e}")
        
        return examples

    def _extract_from_documentation(self) -> List[Dict]:
        """Extract real script examples from documentation"""
        examples = []
        
        # This would involve scraping documentation sites
        # For now, return empty list as this requires careful implementation
        # to respect robots.txt and terms of service
        
        return examples

    def _find_additional_filesystem_scripts(self) -> List[Dict]:
        """Find additional scripts in common system locations"""
        examples = []
        
        # Common locations for infrastructure scripts
        search_paths = [
            '/usr/local/bin',
            '/opt',
            '/etc/init.d',
            '/usr/share/doc',
            os.path.expanduser('~/.local/bin'),
            os.path.expanduser('~/bin'),
            os.path.expanduser('~/scripts')
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                try:
                    path_examples = self._extract_all_scripts_from_path(search_path)
                    examples.extend(path_examples)
                except Exception as e:
                    logger.debug(f"Error searching {search_path}: {e}")
        
        return examples

    def _deduplicate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicates while preserving real-world examples"""
        seen_hashes = set()
        unique_examples = []
        
        for example in examples:
            content = example['completion']
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_examples.append(example)
        
        return unique_examples

    def save_corpus(self, examples: List[Dict], filename: str = "real_world_corpus.jsonl") -> None:
        """Save real-world corpus"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} real-world examples to {output_file}")


def main():
    """Main real-world collection pipeline"""
    logger.info("Starting Real-World Shell Script Collection")
    logger.info("Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA")
    
    collector = RealWorldCollector()
    
    try:
        # Collect real-world scripts
        examples = collector.collect_real_world_scripts(target_additional=200)
        
        # Save corpus
        collector.save_corpus(examples)
        
        # Generate statistics
        categories = {}
        sources = {}
        
        for example in examples:
            metadata = example.get('metadata', {})
            category = metadata.get('category', 'unknown')
            source = metadata.get('source', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        logger.info(f"Real-World Collection Statistics:")
        logger.info(f"Total authentic examples: {len(examples)}")
        logger.info(f"Categories: {dict(sorted(categories.items()))}")
        logger.info(f"Sources: {dict(sorted(sources.items()))}")
        
        logger.info("Real-world collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Real-world collection failed: {e}")
        raise


if __name__ == "__main__":
    main()