#!/usr/bin/env python3
"""
Real-World Jupyter Deployment Examples Collector
Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING

Collects exclusively from authentic sources:
- Production JupyterHub deployments and configurations
- Real MLOps platform deployments with Jupyter
- Kubeflow and data science platform configurations
- Enterprise data science infrastructure examples
- Production-ready notebook server deployments
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

class JupyterRealCollector:
    """Collects Jupyter deployment examples exclusively from real-world sources"""
    
    def __init__(self, output_dir: str = "jupyter_real_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        
        # Target repositories with quality Jupyter deployment examples
        self.target_repos = [
            # Official JupyterHub deployments
            'jupyterhub/zero-to-jupyterhub-k8s',
            'jupyterhub/jupyterhub-deploy-docker',
            'jupyterhub/the-littlest-jupyterhub',
            'jupyterhub/dockerspawner',
            'jupyterhub/kubespawner',
            
            # ML platform deployments
            'kubeflow/kubeflow',
            'kubeflow/examples',
            'kubeflow/manifests',
            'ray-project/ray',
            'mlflow/mlflow',
            'bentoml/BentoML',
            'seldon-core/seldon-core',
            
            # Enterprise ML platforms
            'Netflix/metaflow',
            'airbnb/knowledge-repo',
            'uber/ludwig',
            'microsoft/kubemlops',
            'deployKF/deployKF',
            'feast-dev/feast',
            
            # Data science infrastructure
            'dask/dask-kubernetes',
            'pangeo-data/pangeo-docker-images',
            'jupyter/docker-stacks',
            'jupyter/repo2docker',
            'yarnpkg/berry',
            
            # Cloud provider examples
            'aws/amazon-sagemaker-examples',
            'GoogleCloudPlatform/ai-platform-samples',
            'Azure/MachineLearningNotebooks',
            
            # Academic and research platforms
            'berkeley-dsep-infra/datahub',
            'data-8/materials-fa20',
            '2i2c-org/infrastructure',
        ]
        
        # Local repositories to explore
        self.local_sources = [
            '/home/shingai/api',
            '/home/shingai/sort/deployments',
            '/home/shingai/sort/ona-front-end'
        ]

    def collect_real_jupyter_examples(self, target_count: int = 200) -> List[Dict]:
        """Collect real-world Jupyter deployment examples"""
        logger.info(f"Starting real-world Jupyter collection targeting {target_count} examples")
        
        all_examples = []
        
        # Strategy 1: Local repository mining for Jupyter configs
        local_examples = self._collect_from_local_repos()
        all_examples.extend(local_examples)
        logger.info(f"Collected {len(local_examples)} from local repositories")
        
        # Strategy 2: Clone Jupyter and ML platform repositories
        repo_examples = self._collect_from_jupyter_repos()
        all_examples.extend(repo_examples)
        logger.info(f"Collected {len(repo_examples)} from Jupyter repositories")
        
        # Strategy 3: Official JupyterHub and Kubeflow examples
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
        logger.info(f"Final corpus: {len(final_examples)} high-quality Jupyter examples")
        
        return final_examples

    def _collect_from_local_repos(self) -> List[Dict]:
        """Extract Jupyter deployment examples from local repositories"""
        examples = []
        
        for source_path in self.local_sources:
            if not os.path.exists(source_path):
                continue
                
            logger.info(f"Mining Jupyter examples from {source_path}")
            path_examples = self._extract_from_path(source_path)
            examples.extend(path_examples)
            
        return examples

    def _extract_from_path(self, path: str) -> List[Dict]:
        """Extract Jupyter configurations from a given path"""
        examples = []
        path_obj = Path(path)
        
        # Find Jupyter-related files
        jupyter_patterns = [
            # JupyterHub configurations
            "**/jupyterhub_config.py",
            "**/jupyterhub_config.yaml",
            "**/jupyterhub*.yml",
            "**/jupyterhub*.yaml",
            
            # Kubernetes deployments for Jupyter
            "**/jupyter*.yml",
            "**/jupyter*.yaml",
            "**/notebook*.yml",
            "**/notebook*.yaml",
            
            # Helm charts for JupyterHub
            "**/hub/templates/*.yml",
            "**/hub/templates/*.yaml",
            "**/charts/jupyterhub/*.yml",
            "**/charts/jupyterhub/*.yaml",
            
            # Docker configurations for Jupyter
            "**/jupyter/Dockerfile*",
            "**/notebook/Dockerfile*",
            "**/lab/Dockerfile*",
            
            # MLOps and data science platform configs
            "**/kubeflow/*.yml",
            "**/kubeflow/*.yaml",
            "**/mlflow/*.yml",
            "**/mlflow/*.yaml",
            "**/sagemaker/*.yml",
            "**/sagemaker/*.yaml",
            
            # Data science infrastructure
            "**/dask*.yml",
            "**/dask*.yaml",
            "**/ray*.yml",
            "**/ray*.yaml",
        ]
        
        for pattern in jupyter_patterns:
            for file_path in path_obj.glob(pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if self._is_quality_jupyter_content(content):
                            example = self._create_jupyter_example(content, str(file_path))
                            if example:
                                examples.append(example)
                                
                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")
                        continue
                        
        return examples

    def _collect_from_jupyter_repos(self) -> List[Dict]:
        """Clone and extract from Jupyter and ML platform repositories"""
        examples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo in self.target_repos[:12]:  # Limit to avoid timeout
                try:
                    logger.info(f"Cloning {repo}")
                    clone_path = Path(temp_dir) / repo.replace('/', '_')
                    
                    # Clone with depth limit and file size filter
                    result = subprocess.run([
                        'git', 'clone', '--depth', '1', '--filter=blob:limit=100k',
                        f'https://github.com/{repo}.git', str(clone_path)
                    ], capture_output=True, text=True, timeout=180)
                    
                    if result.returncode == 0:
                        repo_examples = self._extract_from_path(str(clone_path))
                        # Add repository metadata
                        for example in repo_examples:
                            example['metadata']['source_repo'] = repo
                        examples.extend(repo_examples)
                        
                    time.sleep(3)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Error cloning {repo}: {e}")
                    continue
                    
        return examples

    def _collect_from_official_sources(self) -> List[Dict]:
        """Collect from official Jupyter and ML platform documentation"""
        examples = []
        
        # Official Jupyter and ML platform repositories
        official_sources = [
            ('jupyterhub/zero-to-jupyterhub-k8s', 'main'),
            ('kubeflow/manifests', 'master'),
            ('jupyter/docker-stacks', 'main'),
            ('jupyterhub/the-littlest-jupyterhub', 'main')
        ]
        
        for repo, branch in official_sources:
            try:
                logger.info(f"Fetching official examples from {repo}")
                # Get repository file listing
                api_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
                response = requests.get(api_url, timeout=15)
                
                if response.status_code == 200:
                    tree = response.json()
                    
                    # Find Jupyter deployment files
                    jupyter_files = [
                        item for item in tree.get('tree', [])
                        if item['type'] == 'blob' and (
                            'jupyter' in item['path'].lower() or
                            'notebook' in item['path'].lower() or
                            'hub' in item['path'].lower() or
                            'kubeflow' in item['path'].lower() or
                            item['path'].endswith('.yml') or
                            item['path'].endswith('.yaml') or
                            'config.py' in item['path']
                        )
                    ]
                    
                    for file_info in jupyter_files[:25]:  # Limit files per repo
                        try:
                            # Fetch file content
                            file_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_info['path']}"
                            file_response = requests.get(file_url, timeout=15)
                            
                            if file_response.status_code == 200:
                                content = file_response.text
                                if self._is_quality_jupyter_content(content):
                                    example = self._create_jupyter_example(
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
                            
                time.sleep(3)  # Rate limiting between repos
                
            except Exception as e:
                logger.debug(f"Error accessing {repo}: {e}")
                continue
                
        return examples

    def _is_quality_jupyter_content(self, content: str) -> bool:
        """Check if Jupyter content meets quality standards"""
        # Size checks
        if len(content) < 50 or len(content) > 15000:
            return False
            
        content_lower = content.lower()
        content_upper = content.upper()
        
        # Must contain Jupyter or ML platform indicators
        jupyter_indicators = [
            'jupyter', 'jupyterhub', 'notebook', 'spawner',
            'kubeflow', 'mlflow', 'sagemaker', 'ray',
            'authenticator', 'hub.config', 'notebook.config'
        ]
        
        k8s_indicators = [
            'apiversion:', 'kind:', 'metadata:', 'spec:',
            'deployment', 'service', 'configmap', 'secret'
        ]
        
        config_indicators = [
            'c.jupyterhub', 'c.spawner', 'c.authenticator',
            'image:', 'port:', 'env:', 'mount:'
        ]
        
        has_jupyter = any(indicator in content_lower for indicator in jupyter_indicators)
        has_k8s = any(indicator in content_lower for indicator in k8s_indicators)
        has_config = any(indicator in content_lower for indicator in config_indicators)
        
        if not (has_jupyter or (has_k8s and ('jupyter' in content_lower or 'notebook' in content_lower))):
            return False
            
        # Avoid template/example placeholders
        bad_patterns = [
            'your-domain', 'example.com', 'changeme', 'your-image',
            'your-hub', 'your-notebook', 'placeholder', 'template'
        ]
        if any(pattern in content_lower for pattern in bad_patterns):
            return False
            
        # Must have some complexity (multiple configuration options)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 8:
            return False
            
        return True

    def _create_jupyter_example(self, content: str, file_path: str, repo_name: str = None) -> Optional[Dict]:
        """Create a training example from Jupyter content"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            self.collected_hashes.add(content_hash)
            
            # Clean and prepare content
            cleaned_content = self._clean_jupyter_content(content)
            
            # Determine file type and category
            category = self._categorize_jupyter_content(content, file_path)
            language = self._determine_language(file_path)
            
            # Generate appropriate prompt
            prompt = self._generate_jupyter_prompt(cleaned_content, category, language)
            
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
                    "deployment_platform": self._determine_deployment_platform(content, file_path),
                    "features": self._extract_features(content)
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating example from {file_path}: {e}")
            return None

    def _clean_jupyter_content(self, content: str) -> str:
        """Clean Jupyter content while preserving authenticity"""
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            # Remove excessively long lines
            if len(line) > 800:
                continue
                
            # Basic sanitization of sensitive data
            line = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', line)
            line = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', line)
            line = re.sub(r'(password|secret|token|key)[\s]*[:=][\s]*["\'][^"\']+["\']', 
                         r'\1: "<REDACTED>"', line, flags=re.IGNORECASE)
            line = re.sub(r'(domain|host)[\s]*[:=][\s]*["\'][^"\']+["\']', 
                         r'\1: "<DOMAIN>"', line, flags=re.IGNORECASE)
            
            clean_lines.append(line.rstrip())
            
        return '\n'.join(clean_lines).strip()

    def _categorize_jupyter_content(self, content: str, file_path: str) -> str:
        """Categorize Jupyter content by type"""
        file_path_lower = file_path.lower()
        content_lower = content.lower()
        
        if 'jupyterhub_config' in file_path_lower or 'c.jupyterhub' in content_lower:
            return 'jupyter_hub_config'
        elif 'spawner' in content_lower or 'kubespawner' in content_lower:
            return 'jupyter_spawner_config'
        elif any(k8s in file_path_lower for k8s in ['kubernetes', 'k8s']) or 'apiversion:' in content_lower:
            if 'deployment' in content_lower:
                return 'jupyter_k8s_deployment'
            elif 'service' in content_lower:
                return 'jupyter_k8s_service'
            return 'jupyter_k8s_manifest'
        elif 'dockerfile' in file_path_lower:
            return 'jupyter_docker_deploy'
        elif 'helm' in file_path_lower or 'chart' in file_path_lower:
            return 'jupyter_helm_chart'
        elif 'kubeflow' in content_lower:
            return 'kubeflow_jupyter_config'
        elif 'mlflow' in content_lower:
            return 'mlflow_jupyter_config'
        elif 'sagemaker' in content_lower:
            return 'jupyter_cloud_deploy'
        else:
            return 'jupyter_deployment_config'

    def _determine_language(self, file_path: str) -> str:
        """Determine syntax highlighting language"""
        file_path_lower = file_path.lower()
        
        if file_path_lower.endswith('.py'):
            return 'python'
        elif file_path_lower.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif file_path_lower.endswith('.json'):
            return 'json'
        elif 'dockerfile' in file_path_lower:
            return 'dockerfile'
        else:
            return 'yaml'

    def _determine_deployment_platform(self, content: str, file_path: str) -> str:
        """Determine the deployment platform"""
        content_lower = content.lower()
        
        if any(k8s in content_lower for k8s in ['kubernetes', 'kubectl', 'apiversion']):
            return 'kubernetes'
        elif 'docker' in content_lower or 'dockerfile' in file_path.lower():
            return 'docker'
        elif any(cloud in content_lower for cloud in ['aws', 'sagemaker', 'gcp', 'azure']):
            return 'cloud'
        elif 'helm' in content_lower:
            return 'helm'
        else:
            return 'standalone'

    def _extract_features(self, content: str) -> List[str]:
        """Extract key features from the deployment configuration"""
        features = []
        content_lower = content.lower()
        
        # Authentication features
        if any(auth in content_lower for auth in ['oauth', 'authenticator', 'ldap', 'github']):
            features.append('authentication')
        
        # Storage features  
        if any(storage in content_lower for storage in ['persistent', 'volume', 'storage', 'pvc']):
            features.append('persistent_storage')
            
        # Scaling features
        if any(scale in content_lower for scale in ['replica', 'autoscal', 'hpa']):
            features.append('scaling')
            
        # Security features
        if any(sec in content_lower for sec in ['tls', 'ssl', 'rbac', 'security']):
            features.append('security')
            
        # Multi-user features
        if any(multi in content_lower for multi in ['spawner', 'multi', 'user']):
            features.append('multi_user')
            
        return features

    def _generate_jupyter_prompt(self, content: str, category: str, language: str) -> str:
        """Generate appropriate prompt for Jupyter content"""
        content_lower = content.lower()
        
        if category == 'jupyter_hub_config':
            if 'authenticator' in content_lower:
                return "Configure JupyterHub with authentication and user management"
            else:
                return "Configure JupyterHub for multi-user notebook server deployment"
                
        elif category == 'jupyter_k8s_deployment':
            if 'persistent' in content_lower:
                return "Deploy JupyterHub on Kubernetes with persistent storage and scaling"
            else:
                return "Deploy JupyterHub on Kubernetes for container orchestration"
                
        elif category == 'jupyter_docker_deploy':
            return "Create Docker configuration for Jupyter notebook server deployment"
            
        elif category == 'kubeflow_jupyter_config':
            return "Configure Jupyter notebooks within Kubeflow ML platform"
            
        elif category == 'jupyter_cloud_deploy':
            return "Deploy Jupyter notebook server on cloud platform with MLOps integration"
            
        elif category == 'jupyter_helm_chart':
            return "Create Helm chart for JupyterHub deployment on Kubernetes"
            
        else:
            return "Configure Jupyter deployment for data science and machine learning workflows"

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
        if len(completion) < 150 or len(completion) > 15000:
            return False
            
        # Must contain actual configuration content
        if completion.count('\n') < 5:
            return False
            
        # Must be deployment-related (not just notebook content)
        deployment_keywords = [
            'config', 'deploy', 'service', 'image', 'port',
            'mount', 'volume', 'env', 'secret', 'auth'
        ]
        if not any(keyword in completion.lower() for keyword in deployment_keywords):
            return False
            
        return True

    def save_corpus(self, examples: List[Dict], filename: str = "jupyter_deployment_corpus.jsonl") -> None:
        """Save examples to JSONL corpus file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
                
        logger.info(f"Saved {len(examples)} Jupyter examples to {output_path}")

def main():
    """Collect Jupyter deployment examples for Mistral 7B training"""
    collector = JupyterRealCollector()
    
    # Collect examples
    examples = collector.collect_real_jupyter_examples(target_count=200)
    
    # Save to corpus
    collector.save_corpus(examples)
    
    # Print summary
    categories = {}
    platforms = {}
    for example in examples:
        cat = example['metadata']['category']
        platform = example['metadata']['deployment_platform']
        categories[cat] = categories.get(cat, 0) + 1
        platforms[platform] = platforms.get(platform, 0) + 1
    
    print(f"\nJupyter Corpus Summary:")
    print(f"Total examples: {len(examples)}")
    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print("Platforms:")
    for platform, count in sorted(platforms.items()):
        print(f"  {platform}: {count}")

if __name__ == "__main__":
    main()