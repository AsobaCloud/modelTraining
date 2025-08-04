#!/usr/bin/env python3
"""
Real-World Helm Chart Examples Collector
Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING

Collects exclusively from authentic sources:
- Production Helm charts from major repositories
- Enterprise Kubernetes application packaging patterns
- Official Helm chart repositories
- Real-world deployment configurations
- Production values and template examples
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

class HelmRealCollector:
    """Collects Helm chart examples exclusively from real-world sources"""
    
    def __init__(self, output_dir: str = "helm_real_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        
        # Target repositories with quality Helm chart examples
        self.target_repos = [
            # Major Helm chart repositories
            'bitnami/charts',
            'prometheus-community/helm-charts',
            'grafana/helm-charts',
            'elastic/helm-charts',
            'jetstack/cert-manager',
            'kubernetes/ingress-nginx',
            'hashicorp/vault-helm',
            'consul-helm',
            
            # Enterprise and production examples
            'argoproj/argo-helm',
            'jaegertracing/helm-charts',
            'open-telemetry/opentelemetry-helm-charts',
            'istio/istio',
            'linkerd/linkerd2',
            'envoyproxy/gateway',
            
            # Data and ML platforms
            'jupyterhub/helm-chart',
            'kubeflow/manifests',
            'ray-project/kuberay-helm',
            'mlflow/mlflow',
            
            # Databases and storage
            'mongodb/helm-charts',
            'redis/helm-charts',
            'confluentinc/cp-helm-charts',
            'cockroachdb/helm-charts',
            
            # Security and monitoring
            'falcosecurity/charts',
            'aquasecurity/trivy-helm-chart',
            'sigstore/helm-charts',
        ]
        
        # Local repositories to explore
        self.local_sources = [
            '/home/shingai/api',
            '/home/shingai/sort/deployments',
            '/home/shingai/sort/ona-front-end'
        ]

    def collect_real_helm_examples(self, target_count: int = 300) -> List[Dict]:
        """Collect real-world Helm chart examples"""
        logger.info(f"Starting real-world Helm collection targeting {target_count} examples")
        
        all_examples = []
        
        # Strategy 1: Local repository mining for Helm charts
        local_examples = self._collect_from_local_repos()
        all_examples.extend(local_examples)
        logger.info(f"Collected {len(local_examples)} from local repositories")
        
        # Strategy 2: Clone Helm chart repositories
        repo_examples = self._collect_from_helm_repos()
        all_examples.extend(repo_examples)
        logger.info(f"Collected {len(repo_examples)} from Helm repositories")
        
        # Strategy 3: Official Helm Hub and artifact repositories
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
        logger.info(f"Final corpus: {len(final_examples)} high-quality Helm examples")
        
        return final_examples

    def _collect_from_local_repos(self) -> List[Dict]:
        """Extract Helm chart examples from local repositories"""
        examples = []
        
        for source_path in self.local_sources:
            if not os.path.exists(source_path):
                continue
                
            logger.info(f"Mining Helm examples from {source_path}")
            path_examples = self._extract_from_path(source_path)
            examples.extend(path_examples)
            
        return examples

    def _extract_from_path(self, path: str) -> List[Dict]:
        """Extract Helm configurations from a given path"""
        examples = []
        path_obj = Path(path)
        
        # Find Helm-related files
        helm_patterns = [
            # Chart definitions
            "**/Chart.yaml",
            "**/Chart.yml", 
            
            # Values configurations
            "**/values.yaml",
            "**/values.yml",
            "**/values-*.yaml",
            "**/values-*.yml",
            
            # Templates
            "**/templates/*.yaml",
            "**/templates/*.yml",
            
            # Helm chart directories
            "**/charts/*/Chart.yaml",
            "**/helm/*/Chart.yaml",
            
            # Requirements and dependencies
            "**/requirements.yaml",
            "**/requirements.yml",
        ]
        
        for pattern in helm_patterns:
            for file_path in path_obj.glob(pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if self._is_quality_helm_content(content):
                            example = self._create_helm_example(content, str(file_path))
                            if example:
                                examples.append(example)
                                
                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")
                        continue
                        
        return examples

    def _collect_from_helm_repos(self) -> List[Dict]:
        """Clone and extract from Helm chart repositories"""
        examples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo in self.target_repos[:10]:  # Limit to avoid timeout
                try:
                    logger.info(f"Cloning {repo}")
                    clone_path = Path(temp_dir) / repo.replace('/', '_')
                    
                    # Clone with depth limit and file size filter
                    result = subprocess.run([
                        'git', 'clone', '--depth', '1', '--filter=blob:limit=200k',
                        f'https://github.com/{repo}.git', str(clone_path)
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        repo_examples = self._extract_from_path(str(clone_path))
                        # Add repository metadata
                        for example in repo_examples:
                            example['metadata']['source_repo'] = repo
                        examples.extend(repo_examples)
                        
                    time.sleep(4)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Error cloning {repo}: {e}")
                    continue
                    
        return examples

    def _collect_from_official_sources(self) -> List[Dict]:
        """Collect from official Helm chart repositories"""
        examples = []
        
        # Official Helm chart repositories
        official_sources = [
            ('bitnami/charts', 'main'),
            ('prometheus-community/helm-charts', 'main'),
            ('grafana/helm-charts', 'main'),
            ('elastic/helm-charts', 'main')
        ]
        
        for repo, branch in official_sources:
            try:
                logger.info(f"Fetching official charts from {repo}")
                # Get repository file listing
                api_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
                response = requests.get(api_url, timeout=20)
                
                if response.status_code == 200:
                    tree = response.json()
                    
                    # Find Helm chart files
                    helm_files = [
                        item for item in tree.get('tree', [])
                        if item['type'] == 'blob' and (
                            item['path'].endswith('Chart.yaml') or
                            item['path'].endswith('values.yaml') or
                            '/templates/' in item['path'] and item['path'].endswith('.yaml')
                        )
                    ]
                    
                    for file_info in helm_files[:30]:  # Limit files per repo
                        try:
                            # Fetch file content
                            file_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_info['path']}"
                            file_response = requests.get(file_url, timeout=15)
                            
                            if file_response.status_code == 200:
                                content = file_response.text
                                if self._is_quality_helm_content(content):
                                    example = self._create_helm_example(
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
                            
                time.sleep(4)  # Rate limiting between repos
                
            except Exception as e:
                logger.debug(f"Error accessing {repo}: {e}")
                continue
                
        return examples

    def _is_quality_helm_content(self, content: str) -> bool:
        """Check if Helm content meets quality standards"""
        # Size checks
        if len(content) < 30 or len(content) > 20000:
            return False
            
        content_lower = content.lower()
        
        # Must contain Helm indicators
        helm_indicators = [
            'apiversion:', 'name:', 'version:', 'description:',
            'chart:', 'values:', 'template', 'helm',
            'kind:', 'metadata:', 'spec:',
            'replicacount:', 'image:', 'service:', 'ingress:',
            'resources:', 'deployment:', 'configmap:'
        ]
        
        if not any(indicator in content_lower for indicator in helm_indicators):
            return False
            
        # Avoid template/example placeholders (but allow chart-example.local as it's common in Helm)
        bad_patterns = [
            'your-app', 'your-chart', 'changeme',
            'placeholder', 'template-name', 'your-domain', 
            'your-image', 'your-service', 'replace-me'
        ]
        # Check for bad patterns but allow chart-example.local
        has_bad_patterns = any(pattern in content_lower for pattern in bad_patterns)
        if has_bad_patterns and 'chart-example.local' not in content_lower:
            return False
            
        # Must have some structure (multiple lines with content)
        meaningful_lines = [line.strip() for line in content.split('\n') 
                          if line.strip() and not line.strip().startswith('#')]
        if len(meaningful_lines) < 4:
            return False
            
        return True

    def _create_helm_example(self, content: str, file_path: str, repo_name: str = None) -> Optional[Dict]:
        """Create a training example from Helm content"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            self.collected_hashes.add(content_hash)
            
            # Clean and prepare content
            cleaned_content = self._clean_helm_content(content)
            
            # Determine file type and category
            category = self._categorize_helm_content(content, file_path)
            language = self._determine_language(file_path)
            
            # Generate appropriate prompt
            prompt = self._generate_helm_prompt(cleaned_content, category, language)
            
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
                    "helm_type": self._determine_helm_type(content, file_path),
                    "deployment_scope": self._determine_deployment_scope(content)
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating example from {file_path}: {e}")
            return None

    def _clean_helm_content(self, content: str) -> str:
        """Clean Helm content while preserving authenticity"""
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            # Remove excessively long lines
            if len(line) > 1000:
                continue
                
            # Basic sanitization of sensitive data
            line = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', line)
            line = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', line)
            line = re.sub(r'(password|secret|token|key)[\s]*:[\s]*["\'][^"\']+["\']', 
                         r'\1: "<REDACTED>"', line, flags=re.IGNORECASE)
            line = re.sub(r'(host|domain)[\s]*:[\s]*["\'][^"\']+["\']', 
                         r'\1: "<DOMAIN>"', line, flags=re.IGNORECASE)
            
            clean_lines.append(line.rstrip())
            
        return '\n'.join(clean_lines).strip()

    def _categorize_helm_content(self, content: str, file_path: str) -> str:
        """Categorize Helm content by type"""
        file_path_lower = file_path.lower()
        content_lower = content.lower()
        
        if 'chart.yaml' in file_path_lower or 'chart.yml' in file_path_lower:
            return 'helm_chart_definition'
        elif 'values.yaml' in file_path_lower or 'values.yml' in file_path_lower:
            return 'helm_values_config'
        elif '/templates/' in file_path_lower:
            if 'deployment' in file_path_lower:
                return 'helm_deployment_template'
            elif 'service' in file_path_lower:
                return 'helm_service_template'
            elif 'configmap' in file_path_lower:
                return 'helm_configmap_template'
            elif 'ingress' in file_path_lower:
                return 'helm_ingress_template'
            else:
                return 'helm_template'
        elif 'requirements' in file_path_lower:
            return 'helm_dependencies'
        else:
            return 'helm_configuration'

    def _determine_language(self, file_path: str) -> str:
        """Determine syntax highlighting language"""
        file_path_lower = file_path.lower()
        
        if file_path_lower.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif file_path_lower.endswith('.json'):
            return 'json'
        else:
            return 'yaml'

    def _determine_helm_type(self, content: str, file_path: str) -> str:
        """Determine the Helm component type"""
        content_lower = content.lower()
        file_path_lower = file_path.lower()
        
        if 'chart.yaml' in file_path_lower:
            return 'chart_metadata'
        elif 'values' in file_path_lower:
            return 'configuration_values'
        elif '/templates/' in file_path_lower:
            return 'kubernetes_template'
        else:
            return 'helm_config'

    def _determine_deployment_scope(self, content: str) -> str:
        """Determine the deployment scope/complexity"""
        content_lower = content.lower()
        
        # Count configuration complexity indicators
        complexity_indicators = [
            'replicas', 'image', 'service', 'ingress', 'volume',
            'configmap', 'secret', 'deployment', 'statefulset'
        ]
        
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in content_lower)
        
        if complexity_score >= 6:
            return 'enterprise'
        elif complexity_score >= 3:
            return 'production'
        else:
            return 'basic'

    def _generate_helm_prompt(self, content: str, category: str, language: str) -> str:
        """Generate appropriate prompt for Helm content"""
        content_lower = content.lower()
        
        if category == 'helm_chart_definition':
            if 'dependencies' in content_lower:
                return "Create a Helm chart definition with dependencies for complex application deployment"
            else:
                return "Create a Helm chart definition for Kubernetes application packaging"
                
        elif category == 'helm_values_config':
            if 'ingress' in content_lower and 'service' in content_lower:
                return "Create Helm values configuration for application with ingress and service exposure"
            elif 'replicas' in content_lower:
                return "Create Helm values configuration for scalable application deployment"
            else:
                return "Create Helm values configuration for application customization"
                
        elif category.startswith('helm_') and 'template' in category:
            if 'deployment' in category:
                return "Create Helm template for Kubernetes Deployment with configurable parameters"
            elif 'service' in category:
                return "Create Helm template for Kubernetes Service with port configuration"
            elif 'ingress' in category:
                return "Create Helm template for Kubernetes Ingress with routing rules"
            else:
                return "Create Helm template for Kubernetes resource deployment"
                
        elif category == 'helm_dependencies':
            return "Define Helm chart dependencies for complex application architecture"
            
        else:
            return "Create Helm configuration for Kubernetes application deployment"

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
        if len(completion) < 200 or len(completion) > 25000:
            return False
            
        # Must contain actual configuration content
        if completion.count('\n') < 5:
            return False
            
        # Must be Helm-related
        if 'helm' not in completion.lower() and 'chart' not in completion.lower() and 'values' not in completion.lower():
            # Check for Kubernetes templates (also valid for Helm)
            k8s_indicators = ['apiversion', 'kind', 'metadata', 'spec']
            if not any(indicator in completion.lower() for indicator in k8s_indicators):
                return False
            
        return True

    def save_corpus(self, examples: List[Dict], filename: str = "helm_deployment_corpus.jsonl") -> None:
        """Save examples to JSONL corpus file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
                
        logger.info(f"Saved {len(examples)} Helm examples to {output_path}")

def main():
    """Collect Helm chart examples for Mistral 7B training"""
    collector = HelmRealCollector()
    
    # Collect examples
    examples = collector.collect_real_helm_examples(target_count=300)
    
    # Save to corpus
    collector.save_corpus(examples)
    
    # Print summary
    categories = {}
    helm_types = {}
    for example in examples:
        cat = example['metadata']['category']
        helm_type = example['metadata']['helm_type']
        categories[cat] = categories.get(cat, 0) + 1
        helm_types[helm_type] = helm_types.get(helm_type, 0) + 1
    
    print(f"\nHelm Corpus Summary:")
    print(f"Total examples: {len(examples)}")
    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print("Helm Types:")
    for helm_type, count in sorted(helm_types.items()):
        print(f"  {helm_type}: {count}")

if __name__ == "__main__":
    main()