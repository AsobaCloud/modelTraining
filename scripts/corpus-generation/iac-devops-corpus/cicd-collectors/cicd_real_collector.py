#!/usr/bin/env python3
"""
Real-World CI/CD Pipeline Examples Collector
Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING

Collects exclusively from authentic sources:
- Production CI/CD workflows from major repositories
- GitHub Actions, GitLab CI, Jenkins pipeline examples
- Real deployment automation scripts
- Enterprise DevOps pipeline configurations
- Production-tested CI/CD patterns
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

class CICDRealCollector:
    """Collects CI/CD pipeline examples exclusively from real-world sources"""
    
    def __init__(self, output_dir: str = "cicd_real_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        
        # Target repositories with quality CI/CD pipeline examples
        self.target_repos = [
            # Official workflow repositories
            'actions/starter-workflows',
            'docker/build-push-action',
            'actions/checkout',
            'actions/setup-node',
            'actions/setup-python',
            
            # Major open source projects with excellent CI/CD
            'kubernetes/kubernetes',
            'docker/compose',
            'hashicorp/terraform',
            'prometheus/prometheus',
            'grafana/grafana',
            'envoyproxy/envoy',
            'istio/istio',
            
            # Enterprise and production examples
            'Netflix/metaflow',
            'airbnb/knowledge-repo',
            'uber/cadence',
            'microsoft/vscode',
            'facebook/react',
            'google/go-github',
            
            # DevOps and infrastructure projects
            'argoproj/argo-workflows',
            'tektoncd/pipeline',
            'jenkins-x/jx',
            'spinnaker/spinnaker',
            'drone/drone',
            
            # Container and orchestration
            'helm/helm',
            'containerd/containerd',
            'kubernetes-sigs/kind',
            'kubernetes/minikube',
            
            # Security and compliance
            'aquasecurity/trivy',
            'falcosecurity/falco',
            'open-policy-agent/opa',
        ]
        
        # Local repositories to explore
        self.local_sources = [
            '/home/shingai/api',
            '/home/shingai/sort/deployments',
            '/home/shingai/sort/ona-front-end'
        ]

    def collect_real_cicd_examples(self, target_count: int = 400) -> List[Dict]:
        """Collect real-world CI/CD pipeline examples"""
        logger.info(f"Starting real-world CI/CD collection targeting {target_count} examples")
        
        all_examples = []
        
        # Strategy 1: Local repository mining for CI/CD configs
        local_examples = self._collect_from_local_repos()
        all_examples.extend(local_examples)
        logger.info(f"Collected {len(local_examples)} from local repositories")
        
        # Strategy 2: Clone CI/CD repositories
        repo_examples = self._collect_from_cicd_repos()
        all_examples.extend(repo_examples)
        logger.info(f"Collected {len(repo_examples)} from CI/CD repositories")
        
        # Strategy 3: Official workflow repositories
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
        logger.info(f"Final corpus: {len(final_examples)} high-quality CI/CD examples")
        
        return final_examples

    def _collect_from_local_repos(self) -> List[Dict]:
        """Extract CI/CD pipeline examples from local repositories"""
        examples = []
        
        for source_path in self.local_sources:
            if not os.path.exists(source_path):
                continue
                
            logger.info(f"Mining CI/CD examples from {source_path}")
            path_examples = self._extract_from_path(source_path)
            examples.extend(path_examples)
            
        return examples

    def _extract_from_path(self, path: str) -> List[Dict]:
        """Extract CI/CD configurations from a given path"""
        examples = []
        path_obj = Path(path)
        
        # Find CI/CD-related files
        cicd_patterns = [
            # GitHub Actions
            "**/.github/workflows/*.yml",
            "**/.github/workflows/*.yaml",
            
            # GitLab CI
            "**/.gitlab-ci.yml",
            "**/.gitlab-ci.yaml",
            
            # Jenkins
            "**/Jenkinsfile*",
            "**/pipeline.groovy",
            
            # Azure Pipelines
            "**/azure-pipelines.yml",
            "**/azure-pipelines.yaml",
            "**/.azuredevops/*.yml",
            
            # CircleCI
            "**/.circleci/config.yml",
            
            # Travis CI
            "**/.travis.yml",
            
            # Drone CI
            "**/.drone.yml",
            
            # Tekton
            "**/tekton/*.yml",
            "**/tekton/*.yaml",
            
            # Argo Workflows
            "**/argo-workflows/*.yml",
            "**/argo-workflows/*.yaml",
        ]
        
        for pattern in cicd_patterns:
            for file_path in path_obj.glob(pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if self._is_quality_cicd_content(content):
                            example = self._create_cicd_example(content, str(file_path))
                            if example:
                                examples.append(example)
                                
                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")
                        continue
                        
        return examples

    def _collect_from_cicd_repos(self) -> List[Dict]:
        """Clone and extract from CI/CD repositories"""
        examples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo in self.target_repos[:12]:  # Limit to avoid timeout
                try:
                    logger.info(f"Cloning {repo}")
                    clone_path = Path(temp_dir) / repo.replace('/', '_')
                    
                    # Clone with depth limit and file size filter
                    result = subprocess.run([
                        'git', 'clone', '--depth', '1', '--filter=blob:limit=150k',
                        f'https://github.com/{repo}.git', str(clone_path)
                    ], capture_output=True, text=True, timeout=240)
                    
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
        """Collect from official CI/CD workflow repositories"""
        examples = []
        
        # Official CI/CD repositories
        official_sources = [
            ('actions/starter-workflows', 'main'),
            ('docker/build-push-action', 'master'),
            ('kubernetes/kubernetes', 'master'),
            ('hashicorp/terraform', 'main')
        ]
        
        for repo, branch in official_sources:
            try:
                logger.info(f"Fetching official workflows from {repo}")
                # Get repository file listing
                api_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
                response = requests.get(api_url, timeout=20)
                
                if response.status_code == 200:
                    tree = response.json()
                    
                    # Find CI/CD workflow files
                    cicd_files = [
                        item for item in tree.get('tree', [])
                        if item['type'] == 'blob' and (
                            '/.github/workflows/' in item['path'] or
                            item['path'].endswith('.gitlab-ci.yml') or
                            'Jenkinsfile' in item['path'] or
                            item['path'].endswith('.yml') or
                            item['path'].endswith('.yaml')
                        )
                    ]
                    
                    for file_info in cicd_files[:25]:  # Limit files per repo
                        try:
                            # Fetch file content
                            file_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_info['path']}"
                            file_response = requests.get(file_url, timeout=15)
                            
                            if file_response.status_code == 200:
                                content = file_response.text
                                if self._is_quality_cicd_content(content):
                                    example = self._create_cicd_example(
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

    def _is_quality_cicd_content(self, content: str) -> bool:
        """Check if CI/CD content meets quality standards"""
        # Size checks
        if len(content) < 50 or len(content) > 25000:
            return False
            
        content_lower = content.lower()
        
        # Must contain CI/CD indicators
        cicd_indicators = [
            # GitHub Actions
            'on:', 'jobs:', 'steps:', 'runs-on:', 'uses:', 'with:', 'env:',
            # GitLab CI
            'stages:', 'stage:', 'script:', 'before_script:', 'after_script:',
            # Jenkins
            'pipeline', 'agent', 'stage(', 'steps {', 'sh ',
            # General CI/CD
            'build', 'test', 'deploy', 'docker', 'kubectl', 'npm', 'yarn'
        ]
        
        if not any(indicator in content_lower for indicator in cicd_indicators):
            return False
            
        # Avoid template/example placeholders
        bad_patterns = [
            'your-app', 'your-repo', 'example.com', 'changeme',
            'placeholder', 'template-name', 'your-domain',
            'your-image', 'your-service', 'replace this',
            'add your', 'replace with your'
        ]
        if any(pattern in content_lower for pattern in bad_patterns):
            return False
            
        # Must have some structure (multiple lines with content)
        meaningful_lines = [line.strip() for line in content.split('\n') 
                          if line.strip() and not line.strip().startswith('#')]
        if len(meaningful_lines) < 6:
            return False
            
        return True

    def _create_cicd_example(self, content: str, file_path: str, repo_name: str = None) -> Optional[Dict]:
        """Create a training example from CI/CD content"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            self.collected_hashes.add(content_hash)
            
            # Clean and prepare content
            cleaned_content = self._clean_cicd_content(content)
            
            # Determine file type and category
            category = self._categorize_cicd_content(content, file_path)
            language = self._determine_language(file_path)
            
            # Generate appropriate prompt
            prompt = self._generate_cicd_prompt(cleaned_content, category, language)
            
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
                    "cicd_platform": self._determine_cicd_platform(content, file_path),
                    "deployment_patterns": self._extract_deployment_patterns(content),
                    "pipeline_complexity": self._determine_pipeline_complexity(content)
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating example from {file_path}: {e}")
            return None

    def _clean_cicd_content(self, content: str) -> str:
        """Clean CI/CD content while preserving authenticity"""
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            # Remove excessively long lines
            if len(line) > 1200:
                continue
                
            # Basic sanitization of sensitive data
            line = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', line)
            line = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', line)
            line = re.sub(r'(password|secret|token|key)[\s]*:[\s]*["\'][^"\']+["\']', 
                         r'\1: "<REDACTED>"', line, flags=re.IGNORECASE)
            line = re.sub(r'(host|domain|registry)[\s]*:[\s]*["\'][^"\']+["\']', 
                         r'\1: "<DOMAIN>"', line, flags=re.IGNORECASE)
            
            clean_lines.append(line.rstrip())
            
        return '\n'.join(clean_lines).strip()

    def _categorize_cicd_content(self, content: str, file_path: str) -> str:
        """Categorize CI/CD content by platform and type"""
        file_path_lower = file_path.lower()
        content_lower = content.lower()
        
        if '/.github/workflows/' in file_path_lower:
            return 'github_actions_workflow'
        elif '.gitlab-ci' in file_path_lower:
            return 'gitlab_ci_pipeline'
        elif 'jenkinsfile' in file_path_lower or 'pipeline' in content_lower:
            return 'jenkins_pipeline'
        elif 'azure-pipelines' in file_path_lower:
            return 'azure_pipelines'
        elif '.circleci' in file_path_lower:
            return 'circleci_config'
        elif '.drone' in file_path_lower:
            return 'drone_ci_pipeline'
        elif 'tekton' in file_path_lower:
            return 'tekton_pipeline'
        elif 'argo' in file_path_lower:
            return 'argo_workflow'
        else:
            return 'cicd_configuration'

    def _determine_language(self, file_path: str) -> str:
        """Determine syntax highlighting language"""
        file_path_lower = file_path.lower()
        
        if file_path_lower.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif file_path_lower.endswith('.json'):
            return 'json'
        elif 'jenkinsfile' in file_path_lower or file_path_lower.endswith('.groovy'):
            return 'groovy'
        else:
            return 'yaml'

    def _determine_cicd_platform(self, content: str, file_path: str) -> str:
        """Determine the CI/CD platform"""
        file_path_lower = file_path.lower()
        content_lower = content.lower()
        
        if '/.github/workflows/' in file_path_lower or 'runs-on:' in content_lower:
            return 'github_actions'
        elif '.gitlab-ci' in file_path_lower or 'gitlab' in content_lower:
            return 'gitlab_ci'
        elif 'jenkinsfile' in file_path_lower or 'pipeline {' in content_lower:
            return 'jenkins'
        elif 'azure-pipelines' in file_path_lower:
            return 'azure_devops'
        elif '.circleci' in file_path_lower:
            return 'circleci'
        elif 'tekton' in content_lower:
            return 'tekton'
        else:
            return 'generic_cicd'

    def _extract_deployment_patterns(self, content: str) -> List[str]:
        """Extract deployment patterns from CI/CD content"""
        patterns = []
        content_lower = content.lower()
        
        # Container patterns
        if any(docker in content_lower for docker in ['docker build', 'docker push', 'docker run']):
            patterns.append('docker')
        
        # Kubernetes patterns
        if any(k8s in content_lower for k8s in ['kubectl', 'kubernetes', 'k8s', 'helm']):
            patterns.append('kubernetes')
            
        # Cloud patterns
        if any(aws in content_lower for aws in ['aws', 'terraform', 'cloudformation']):
            patterns.append('cloud_infrastructure')
            
        # Package managers
        if any(pkg in content_lower for pkg in ['npm', 'yarn', 'pip', 'mvn', 'gradle']):
            patterns.append('package_management')
            
        # Testing patterns
        if any(test in content_lower for test in ['test', 'jest', 'pytest', 'junit']):
            patterns.append('testing')
            
        return patterns

    def _determine_pipeline_complexity(self, content: str) -> str:
        """Determine the complexity level of the CI/CD pipeline"""
        content_lower = content.lower()
        
        # Count complexity indicators
        complexity_indicators = [
            'matrix:', 'strategy:', 'parallel:', 'needs:', 'depends_on:',
            'environment:', 'secrets:', 'variables:', 'cache:',
            'services:', 'artifacts:', 'deploy:', 'stage:'
        ]
        
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in content_lower)
        
        # Count job/stage count
        job_count = content_lower.count('job:') + content_lower.count('stage:') + content_lower.count('- name:')
        
        if complexity_score >= 5 or job_count >= 8:
            return 'enterprise'
        elif complexity_score >= 3 or job_count >= 4:
            return 'production'
        else:
            return 'basic'

    def _generate_cicd_prompt(self, content: str, category: str, language: str) -> str:
        """Generate appropriate prompt for CI/CD content"""
        content_lower = content.lower()
        
        if category == 'github_actions_workflow':
            if 'docker' in content_lower and 'deploy' in content_lower:
                return "Create a GitHub Actions workflow for Docker-based application deployment"
            elif 'test' in content_lower and 'build' in content_lower:
                return "Create a GitHub Actions workflow for testing and building applications"
            else:
                return "Create a GitHub Actions workflow for CI/CD automation"
                
        elif category == 'gitlab_ci_pipeline':
            if 'kubernetes' in content_lower or 'kubectl' in content_lower:
                return "Create a GitLab CI pipeline for Kubernetes deployment"
            else:
                return "Create a GitLab CI pipeline for continuous integration and deployment"
                
        elif category == 'jenkins_pipeline':
            return "Create a Jenkins pipeline for automated build and deployment"
            
        elif category == 'tekton_pipeline':
            return "Create a Tekton pipeline for Kubernetes-native CI/CD"
            
        elif category == 'argo_workflow':
            return "Create an Argo Workflow for container-native workflow orchestration"
            
        else:
            return "Create a CI/CD pipeline configuration for automated deployment"

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
        if len(completion) < 200 or len(completion) > 30000:
            return False
            
        # Must contain actual pipeline content
        if completion.count('\n') < 8:
            return False
            
        # Must be CI/CD-related
        cicd_keywords = [
            'build', 'test', 'deploy', 'pipeline', 'workflow',
            'job', 'stage', 'step', 'run', 'script'
        ]
        if not any(keyword in completion.lower() for keyword in cicd_keywords):
            return False
            
        return True

    def save_corpus(self, examples: List[Dict], filename: str = "cicd_pipeline_corpus.jsonl") -> None:
        """Save examples to JSONL corpus file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
                
        logger.info(f"Saved {len(examples)} CI/CD examples to {output_path}")

def main():
    """Collect CI/CD pipeline examples for Mistral 7B training"""
    collector = CICDRealCollector()
    
    # Collect examples
    examples = collector.collect_real_cicd_examples(target_count=400)
    
    # Save to corpus
    collector.save_corpus(examples)
    
    # Print summary
    categories = {}
    platforms = {}
    complexity = {}
    for example in examples:
        cat = example['metadata']['category']
        platform = example['metadata']['cicd_platform']
        comp = example['metadata']['pipeline_complexity']
        categories[cat] = categories.get(cat, 0) + 1
        platforms[platform] = platforms.get(platform, 0) + 1
        complexity[comp] = complexity.get(comp, 0) + 1
    
    print(f"\nCI/CD Corpus Summary:")
    print(f"Total examples: {len(examples)}")
    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print("Platforms:")
    for platform, count in sorted(platforms.items()):
        print(f"  {platform}: {count}")
    print("Complexity:")
    for comp, count in sorted(complexity.items()):
        print(f"  {comp}: {count}")

if __name__ == "__main__":
    main()