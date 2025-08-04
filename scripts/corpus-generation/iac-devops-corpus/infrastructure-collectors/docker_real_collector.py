#!/usr/bin/env python3
"""
Real-World Docker Deployment Examples Collector
Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING

Collects exclusively from authentic sources:
- Production Docker configurations and workflows
- Official Docker Hub deployment patterns
- Open source container orchestration projects
- Real Kubernetes manifests and Docker Compose files
- Enterprise containerization examples
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

class DockerRealCollector:
    """Collects Docker deployment examples exclusively from real-world sources"""
    
    def __init__(self, output_dir: str = "docker_real_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        
        # Target repositories with quality Docker deployment examples
        self.target_repos = [
            # Core Docker and container projects
            'docker/compose',
            'docker/awesome-compose',
            'moby/moby',
            'containerd/containerd',
            'docker/docker-bench-security',
            'docker/buildx',
            'GoogleContainerTools/distroless',
            'hadolint/hadolint',
            
            # Kubernetes and orchestration
            'kubernetes/kubernetes',
            'kubernetes/examples',
            'argoproj/argo-workflows',
            'helm/charts',
            'istio/istio',
            'envoyproxy/envoy',
            
            # Enterprise production examples
            'Netflix/metaflow',
            'airbnb/knowledge-repo',
            'uber/cadence',
            'spotify/docker-maven-plugin',
            'pinterest/ktranslate',
            
            # ML and data science platforms
            'kubeflow/kubeflow',
            'ray-project/ray',
            'mlflow/mlflow',
            'bentoml/BentoML',
            'seldon-core/seldon-core',
            
            # CI/CD and DevOps
            'drone/drone',
            'tektoncd/pipeline',
            'jenkins-x/jx',
            'spinnaker/spinnaker',
        ]
        
        # Local repositories to explore
        self.local_sources = [
            '/home/shingai/api',
            '/home/shingai/sort/deployments',
            '/home/shingai/sort/ona-front-end'
        ]

    def collect_real_docker_examples(self, target_count: int = 500) -> List[Dict]:
        """Collect real-world Docker deployment examples"""
        logger.info(f"Starting real-world Docker collection targeting {target_count} examples")
        
        all_examples = []
        
        # Strategy 1: Local repository mining for Docker usage
        local_examples = self._collect_from_local_repos()
        all_examples.extend(local_examples)
        logger.info(f"Collected {len(local_examples)} from local repositories")
        
        # Strategy 2: Clone Docker and container repositories
        repo_examples = self._collect_from_docker_repos()
        all_examples.extend(repo_examples)
        logger.info(f"Collected {len(repo_examples)} from Docker repositories")
        
        # Strategy 3: Official Docker examples and documentation
        official_examples = self._collect_from_official_sources()
        all_examples.extend(official_examples)
        logger.info(f"Collected {len(official_examples)} from official sources")
        
        # Deduplicate and clean
        deduplicated = self._deduplicate_examples(all_examples)
        logger.info(f"Deduplicated to {len(deduplicated)} unique examples")
        
        # Filter for quality and return top examples
        quality_examples = [ex for ex in deduplicated if self._is_quality_example(ex)]
        
        # Sort by complexity and authenticity, take best examples
        sorted_examples = sorted(quality_examples, 
                               key=lambda x: (len(x['completion']), x['metadata'].get('stars', 0)), 
                               reverse=True)
        
        final_examples = sorted_examples[:target_count]
        logger.info(f"Final corpus: {len(final_examples)} high-quality Docker examples")
        
        return final_examples

    def _collect_from_local_repos(self) -> List[Dict]:
        """Extract Docker examples from local repositories"""
        examples = []
        
        for source_path in self.local_sources:
            if not os.path.exists(source_path):
                continue
                
            logger.info(f"Mining Docker examples from {source_path}")
            path_examples = self._extract_from_path(source_path)
            examples.extend(path_examples)
            
        return examples

    def _extract_from_path(self, path: str) -> List[Dict]:
        """Extract Docker configurations from a given path"""
        examples = []
        path_obj = Path(path)
        
        # Find Docker-related files
        docker_patterns = [
            "**/Dockerfile*",
            "**/docker-compose*.yml",
            "**/docker-compose*.yaml", 
            "**/.dockerignore",
            "**/kubernetes/*.yml",
            "**/kubernetes/*.yaml",
            "**/k8s/*.yml",
            "**/k8s/*.yaml",
            "**/helm/**/*.yml",
            "**/helm/**/*.yaml"
        ]
        
        for pattern in docker_patterns:
            for file_path in path_obj.glob(pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if self._is_quality_docker_content(content):
                            example = self._create_docker_example(content, str(file_path))
                            if example:
                                examples.append(example)
                                
                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")
                        continue
                        
        return examples

    def _collect_from_docker_repos(self) -> List[Dict]:
        """Clone and extract from Docker repositories"""
        examples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo in self.target_repos[:15]:  # Limit to avoid timeout
                try:
                    logger.info(f"Cloning {repo}")
                    clone_path = Path(temp_dir) / repo.replace('/', '_')
                    
                    # Clone with depth limit and file size filter
                    result = subprocess.run([
                        'git', 'clone', '--depth', '1', '--filter=blob:limit=50k',
                        f'https://github.com/{repo}.git', str(clone_path)
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        repo_examples = self._extract_from_path(str(clone_path))
                        # Add repository metadata
                        for example in repo_examples:
                            example['metadata']['source_repo'] = repo
                        examples.extend(repo_examples)
                        
                    time.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Error cloning {repo}: {e}")
                    continue
                    
        return examples

    def _collect_from_official_sources(self) -> List[Dict]:
        """Collect from official Docker documentation and examples"""
        examples = []
        
        # Official Docker example repositories
        official_sources = [
            ('docker/awesome-compose', 'master'),
            ('docker/labs', 'master'),
            ('kubernetes/examples', 'master')
        ]
        
        for repo, branch in official_sources:
            try:
                logger.info(f"Fetching official examples from {repo}")
                # Get repository file listing
                api_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
                response = requests.get(api_url, timeout=10)
                
                if response.status_code == 200:
                    tree = response.json()
                    
                    # Find Docker files
                    docker_files = [
                        item for item in tree.get('tree', [])
                        if item['type'] == 'blob' and 
                        (item['path'].endswith('Dockerfile') or 
                         'docker-compose' in item['path'] or
                         item['path'].endswith('.yml') or
                         item['path'].endswith('.yaml'))
                    ]
                    
                    for file_info in docker_files[:20]:  # Limit files per repo
                        try:
                            # Fetch file content
                            file_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_info['path']}"
                            file_response = requests.get(file_url, timeout=10)
                            
                            if file_response.status_code == 200:
                                content = file_response.text
                                if self._is_quality_docker_content(content):
                                    example = self._create_docker_example(
                                        content, 
                                        f"{repo}/{file_info['path']}",
                                        repo_name=repo
                                    )
                                    if example:
                                        examples.append(example)
                                        
                            time.sleep(1)  # Rate limiting
                            
                        except Exception as e:
                            logger.debug(f"Error fetching {file_info['path']}: {e}")
                            continue
                            
                time.sleep(2)  # Rate limiting between repos
                
            except Exception as e:
                logger.debug(f"Error accessing {repo}: {e}")
                continue
                
        return examples

    def _is_quality_docker_content(self, content: str) -> bool:
        """Check if Docker content meets quality standards"""
        # Size checks
        if len(content) < 30 or len(content) > 8000:
            return False
            
        content_upper = content.upper()
        
        # Must contain Docker indicators
        dockerfile_indicators = ['FROM', 'RUN', 'COPY', 'ADD', 'WORKDIR', 'EXPOSE', 'CMD', 'ENTRYPOINT']
        compose_indicators = ['VERSION:', 'SERVICES:', 'IMAGE:', 'BUILD:', 'PORTS:', 'VOLUMES:']
        k8s_indicators = ['APIVERSION:', 'KIND:', 'METADATA:', 'SPEC:']
        
        has_docker = any(indicator in content_upper for indicator in dockerfile_indicators)
        has_compose = any(indicator in content_upper for indicator in compose_indicators)
        has_k8s = any(indicator in content_upper for indicator in k8s_indicators)
        
        if not (has_docker or has_compose or has_k8s):
            return False
            
        # Avoid template/example placeholders
        bad_patterns = [
            'your-image', 'example.com', 'placeholder', 'template',
            'changeme', 'your-app', 'your-service', 'your-domain'
        ]
        content_lower = content.lower()
        if any(pattern in content_lower for pattern in bad_patterns):
            return False
            
        # Must have some complexity (multiple commands/services)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 5:
            return False
            
        return True

    def _create_docker_example(self, content: str, file_path: str, repo_name: str = None) -> Optional[Dict]:
        """Create a training example from Docker content"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            self.collected_hashes.add(content_hash)
            
            # Clean and prepare content
            cleaned_content = self._clean_docker_content(content)
            
            # Determine file type and category
            category = self._categorize_docker_content(content, file_path)
            language = self._determine_language(file_path)
            
            # Generate appropriate prompt
            prompt = self._generate_docker_prompt(cleaned_content, category, language)
            
            # Format completion with proper syntax highlighting
            completion = f"```{language}\n{cleaned_content}\n```"
            
            return {
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "source": "real_production_repository" if repo_name else "real_local_repository",
                    "file_path": file_path,
                    "category": category,
                    "authentic": True,
                    "language": language,
                    "source_repo": repo_name or "local",
                    "deployment_type": self._determine_deployment_type(content, file_path)
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating example from {file_path}: {e}")
            return None

    def _clean_docker_content(self, content: str) -> str:
        """Clean Docker content while preserving authenticity"""
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            # Remove excessively long lines
            if len(line) > 500:
                continue
                
            # Basic sanitization of sensitive data
            line = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', line)
            line = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', line)
            line = re.sub(r'(password|secret|token|key)[\s]*[:=][\s]*["\'][^"\']+["\']', 
                         r'\1: "<REDACTED>"', line, flags=re.IGNORECASE)
            
            clean_lines.append(line.rstrip())
            
        return '\n'.join(clean_lines).strip()

    def _categorize_docker_content(self, content: str, file_path: str) -> str:
        """Categorize Docker content by type"""
        file_path_lower = file_path.lower()
        content_upper = content.upper()
        
        if 'dockerfile' in file_path_lower:
            if 'FROM' in content_upper and 'RUN' in content_upper:
                return 'dockerfile_production'
            return 'dockerfile_basic'
        elif 'docker-compose' in file_path_lower:
            return 'docker_compose_deployment'
        elif any(k8s in file_path_lower for k8s in ['kubernetes', 'k8s']):
            if 'DEPLOYMENT' in content_upper:
                return 'kubernetes_deployment'
            elif 'SERVICE' in content_upper:
                return 'kubernetes_service'
            return 'kubernetes_manifest'
        elif 'helm' in file_path_lower:
            return 'helm_chart'
        else:
            return 'container_config'

    def _determine_language(self, file_path: str) -> str:
        """Determine syntax highlighting language"""
        file_path_lower = file_path.lower()
        
        if 'dockerfile' in file_path_lower:
            return 'dockerfile'
        elif file_path_lower.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif file_path_lower.endswith('.json'):
            return 'json'
        else:
            return 'dockerfile'

    def _determine_deployment_type(self, content: str, file_path: str) -> str:
        """Determine the deployment context"""
        content_upper = content.upper()
        file_path_lower = file_path.lower()
        
        if any(k8s in content_upper for k8s in ['KUBERNETES', 'KUBECTL', 'APIVERSION']):
            return 'kubernetes'
        elif 'docker-compose' in file_path_lower:
            return 'docker_compose'
        elif any(cloud in content_upper for cloud in ['AWS', 'GCP', 'AZURE']):
            return 'cloud'
        else:
            return 'docker'

    def _generate_docker_prompt(self, content: str, category: str, language: str) -> str:
        """Generate appropriate prompt for Docker content"""
        content_upper = content.upper()
        
        if category == 'dockerfile_production':
            if 'MULTI-STAGE' in content or 'AS BUILDER' in content_upper:
                return "Create a production-ready multi-stage Dockerfile with build optimization"
            elif 'ALPINE' in content_upper:
                return "Create a secure Dockerfile using Alpine Linux base image"
            else:
                return "Create a production-ready Dockerfile for application deployment"
                
        elif category == 'docker_compose_deployment':
            services = content_upper.count('IMAGE:') + content_upper.count('BUILD:')
            if services > 1:
                return "Create a Docker Compose configuration for multi-service application deployment"
            else:
                return "Create a Docker Compose configuration for application deployment"
                
        elif category.startswith('kubernetes'):
            if 'DEPLOYMENT' in content_upper:
                return "Create a Kubernetes Deployment manifest for container orchestration"
            elif 'SERVICE' in content_upper:
                return "Create a Kubernetes Service manifest for container networking"
            else:
                return "Create a Kubernetes manifest for container deployment"
                
        elif category == 'helm_chart':
            return "Create a Helm chart template for Kubernetes application deployment"
            
        else:
            return "Create a container configuration for application deployment"

    def _deduplicate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicate examples based on content similarity"""
        seen_hashes = set()
        unique_examples = []
        
        for example in examples:
            # Create hash from completion content
            content_hash = hashlib.md5(example['completion'].encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_examples.append(example)
                
        return unique_examples

    def _is_quality_example(self, example: Dict) -> bool:
        """Final quality check for examples"""
        completion = example['completion']
        
        # Must have reasonable length
        if len(completion) < 100 or len(completion) > 10000:
            return False
            
        # Must contain actual configuration content
        if completion.count('\n') < 3:
            return False
            
        return True

    def save_corpus(self, examples: List[Dict], filename: str = "docker_deployment_corpus.jsonl") -> None:
        """Save examples to JSONL corpus file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
                
        logger.info(f"Saved {len(examples)} Docker examples to {output_path}")

def main():
    """Collect Docker deployment examples for Mistral 7B training"""
    collector = DockerRealCollector()
    
    # Collect examples
    examples = collector.collect_real_docker_examples(target_count=500)
    
    # Save to corpus
    collector.save_corpus(examples)
    
    # Print summary
    categories = {}
    for example in examples:
        cat = example['metadata']['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nDocker Corpus Summary:")
    print(f"Total examples: {len(examples)}")
    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()