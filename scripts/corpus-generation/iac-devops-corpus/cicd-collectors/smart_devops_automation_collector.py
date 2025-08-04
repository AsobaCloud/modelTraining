#!/usr/bin/env python3
"""
Smart DevOps Automation Collector - Content-First, Star-Based Quality

Searches for high-quality DevOps automation patterns across popular repositories.
Focuses on CI/CD, Docker, Kubernetes, Terraform, and automation scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import base class
sys.path.append(str(Path(__file__).parent))
from smart_collector_base import SmartCollectorBase

# Import validation pipeline
sys.path.append(str(Path(__file__).parent.parent / 'validation'))
from universal_validation_pipeline import UniversalValidationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_devops_automation")

class SmartDevOpsAutomationCollector(SmartCollectorBase):
    """Smart collector for DevOps automation examples using content-first approach"""
    
    def __init__(self, github_token: Optional[str] = None):
        super().__init__("devops_automation", github_token)
        self.validation_pipeline = UniversalValidationPipeline()
        
        # Content-based search configuration
        self.search_config = {
            'ci_cd_pipelines': {
                'search_terms': [
                    'github-actions',
                    'gitlab-ci',
                    'jenkins',
                    'pipeline',
                    'workflow'
                ],
                'content_patterns': [
                    'workflow', 'pipeline', 'ci', 'cd', 'build',
                    'deploy', 'test', 'action', 'job', 'step'
                ],
                'min_stars': 500,
                'max_repos': 30,
                'target_examples': 120
            },
            'containerization': {
                'search_terms': [
                    'Dockerfile',
                    'docker-compose',
                    'container',
                    'kubernetes',
                    'helm'
                ],
                'content_patterns': [
                    'FROM', 'RUN', 'COPY', 'WORKDIR', 'EXPOSE',
                    'CMD', 'ENTRYPOINT', 'docker', 'container'
                ],
                'min_stars': 300,
                'max_repos': 25,
                'target_examples': 100
            },
            'infrastructure_as_code': {
                'search_terms': [
                    'terraform',
                    'cloudformation',
                    'ansible',
                    'infrastructure',
                    'provisioning'
                ],
                'content_patterns': [
                    'resource', 'provider', 'variable', 'output',
                    'module', 'terraform', 'ansible', 'aws'
                ],
                'min_stars': 400,
                'max_repos': 20,
                'target_examples': 80
            },
            'monitoring_alerting': {
                'search_terms': [
                    'prometheus',
                    'grafana',
                    'monitoring',
                    'alerting',
                    'metrics'
                ],
                'content_patterns': [
                    'prometheus', 'grafana', 'alert', 'metric',
                    'dashboard', 'monitor', 'query', 'rule'
                ],
                'min_stars': 200,
                'max_repos': 25,
                'target_examples': 60
            },
            'automation_scripts': {
                'search_terms': [
                    'automation',
                    'script',
                    'bash',
                    'deploy',
                    'setup'
                ],
                'content_patterns': [
                    'bash', 'script', 'deploy', 'setup', 'install',
                    'configure', 'automation', 'provision'
                ],
                'min_stars': 150,
                'max_repos': 30,
                'target_examples': 40
            }
        }
    
    def collect_domain_examples(self, target_count: int = 400) -> List[Dict]:
        """Collect DevOps automation examples using smart content-first approach"""
        logger.info(f"Starting smart DevOps automation collection (target: {target_count})")
        
        all_examples = []
        
        for category, config in self.search_config.items():
            logger.info(f"Collecting {category} examples (target: {config['target_examples']})")
            
            # Search for high-quality repositories
            repositories = self.search_repositories_by_content(
                search_terms=config['search_terms'],
                min_stars=config['min_stars'],
                max_repos=config['max_repos']
            )
            
            category_examples = []
            
            # Extract examples from each repository
            for repo in repositories:
                if len(category_examples) >= config['target_examples']:
                    break
                
                try:
                    repo_examples = self.extract_content_by_patterns(
                        repo=repo,
                        content_patterns=config['content_patterns'],
                        file_extensions=['.yml', '.yaml', '.json', '.sh', '.tf', '.py', '.md'],
                        max_files=15
                    )
                    
                    # Process and validate examples
                    for example in repo_examples:
                        if len(category_examples) >= config['target_examples']:
                            break
                        
                        processed_example = self._create_devops_example(
                            example, category
                        )
                        
                        if processed_example and self._validate_devops_example(processed_example):
                            category_examples.append(processed_example)
                
                except Exception as e:
                    logger.warning(f"Error processing repo {repo.full_name}: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(category_examples)} {category} examples")
            all_examples.extend(category_examples)
        
        logger.info(f"Total collected: {len(all_examples)} DevOps automation examples")
        return all_examples[:target_count]
    
    def _create_example_from_content(self, 
                                   content: str, 
                                   file_path: str, 
                                   repo_name: str,
                                   star_count: int) -> Optional[Dict]:
        """Create example from content (required by base class)"""
        return {
            'content': content,
            'file_path': file_path,
            'repo_name': repo_name,
            'star_count': star_count,
            'quality_score': self._assess_content_quality(content, star_count)
        }
    
    def _create_devops_example(self, raw_example: Dict, category: str) -> Optional[Dict]:
        """Create training example for DevOps automation"""
        content = raw_example.get('content', '')
        if len(content.strip()) < 30:
            return None
        
        # Generate category-specific prompt
        prompt = self._generate_devops_prompt(content, category)
        
        # Assess DevOps complexity
        complexity = self._assess_devops_complexity(content, category)
        
        # Extract DevOps features
        features = self._extract_devops_features(content)
        
        return {
            'domain': 'devops_automation',
            'category': category,
            'prompt': prompt,
            'completion': content,
            'metadata': {
                'source': f"github.com/{raw_example.get('repo_name', 'unknown')}",
                'file_path': raw_example.get('file_path', ''),
                'language': self._detect_devops_language(raw_example.get('file_path', '')),
                'complexity': complexity,
                'star_count': raw_example.get('star_count', 0),
                'quality_score': raw_example.get('quality_score', 0.5),
                'features': features,
                'authority_level': 'high' if raw_example.get('star_count', 0) > 1000 else 'medium',
                'authentic': True
            }
        }
    
    def _generate_devops_prompt(self, content: str, category: str) -> str:
        """Generate category-specific DevOps prompt"""
        prompts = {
            'ci_cd_pipelines': 'Create CI/CD pipeline for automated build, test, and deployment',
            'containerization': 'Create containerization setup with Docker and orchestration',
            'infrastructure_as_code': 'Create infrastructure as code for cloud resource provisioning',
            'monitoring_alerting': 'Create monitoring and alerting configuration for system observability',
            'automation_scripts': 'Create automation scripts for deployment and system management'
        }
        
        base_prompt = prompts.get(category, 'Create DevOps automation solution')
        
        # Add specific details based on content patterns
        if 'github' in content.lower() and 'workflow' in content.lower():
            base_prompt += ' using GitHub Actions'
        elif 'FROM ' in content and 'RUN ' in content:
            base_prompt += ' with Dockerfile configuration'
        elif 'terraform' in content.lower():
            base_prompt += ' using Terraform infrastructure as code'
        elif 'prometheus' in content.lower() or 'grafana' in content.lower():
            base_prompt += ' with Prometheus/Grafana monitoring stack'
        elif 'bash' in content.lower() or '#!/bin/bash' in content:
            base_prompt += ' with bash automation scripts'
        
        return self._generate_claude_md_prompt(content, 'devops_automation', base_prompt)
    
    def _assess_devops_complexity(self, content: str, category: str) -> str:
        """Assess DevOps automation complexity"""
        line_count = len(content.split('\n'))
        
        # Category-specific complexity indicators
        complexity_indicators = {
            'ci_cd_pipelines': ['matrix', 'strategy', 'environment', 'secrets', 'artifact'],
            'containerization': ['multi-stage', 'compose', 'network', 'volume', 'secret'],
            'infrastructure_as_code': ['module', 'data', 'locals', 'dynamic', 'for_each'],
            'monitoring_alerting': ['rule', 'query', 'dashboard', 'alert', 'threshold'],
            'automation_scripts': ['function', 'loop', 'condition', 'variable', 'error']
        }
        
        indicators = complexity_indicators.get(category, [])
        indicator_count = sum(1 for indicator in indicators if indicator.lower() in content.lower())
        
        # DevOps-specific complexity scoring
        devops_patterns = ['deploy', 'build', 'test', 'monitor', 'scale']
        devops_count = sum(1 for pattern in devops_patterns if pattern.lower() in content.lower())
        
        total_complexity = indicator_count + devops_count
        
        if line_count > 150 or total_complexity > 4:
            return 'complex'
        elif line_count > 75 or total_complexity > 2:
            return 'medium'
        else:
            return 'simple'
    
    def _extract_devops_features(self, content: str) -> Dict:
        """Extract DevOps-specific features"""
        features = {
            'has_ci_cd': any(ci in content.lower() for ci in ['workflow', 'pipeline', 'build', 'deploy']),
            'has_docker': 'docker' in content.lower() or 'FROM ' in content,
            'has_kubernetes': any(k8s in content.lower() for k8s in ['kubernetes', 'kubectl', 'helm']),
            'has_terraform': 'terraform' in content.lower() or 'resource "' in content,
            'has_ansible': 'ansible' in content.lower() or 'playbook' in content.lower(),
            'has_monitoring': any(mon in content.lower() for mon in ['prometheus', 'grafana', 'alert']),
            'has_secrets': any(sec in content.lower() for sec in ['secret', 'password', 'token', 'key']),
            'has_environment_vars': any(env in content for env in ['$ENV', '${', 'ENV_VAR']),
            'has_scripts': any(script in content for script in ['#!/bin/', 'bash', 'sh']),
            'has_automation': any(auto in content.lower() for auto in ['automation', 'deploy', 'provision']),
            'file_count': 1,
            'line_count': len(content.split('\n'))
        }
        
        return features
    
    def _detect_devops_language(self, file_path: str) -> str:
        """Detect DevOps file type from extension"""
        if file_path.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif file_path.endswith('.json'):
            return 'json'
        elif file_path.endswith('.tf'):
            return 'terraform'
        elif file_path.endswith('.sh'):
            return 'bash'
        elif file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.md'):
            return 'markdown'
        else:
            return 'unknown'
    
    def _validate_devops_example(self, example: Dict) -> bool:
        """Validate DevOps example with enhanced criteria"""
        try:
            # Use universal validation pipeline
            validation_result = self.validation_pipeline.validate_example(example)
            overall_quality = validation_result.get('overall_quality', 0)
            
            # DevOps-specific validation bonuses
            features = example.get('metadata', {}).get('features', {})
            
            devops_bonus = 0
            if features.get('has_ci_cd'):
                devops_bonus += 0.1
            if features.get('has_docker') or features.get('has_kubernetes'):
                devops_bonus += 0.05
            if features.get('has_terraform') or features.get('has_ansible'):
                devops_bonus += 0.05
            if features.get('has_monitoring'):
                devops_bonus += 0.05
            
            final_score = overall_quality + devops_bonus
            
            # Accept DevOps examples with reasonable quality
            return final_score >= 0.65
            
        except Exception as e:
            logger.debug(f"DevOps validation failed: {str(e)}")
            return False
    
    def save_examples(self, examples: List[Dict], output_path: str):
        """Save examples to JSONL format"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} DevOps examples to {output_path}")

def main():
    """Main collection process"""
    collector = SmartDevOpsAutomationCollector()
    
    # Collect examples
    examples = collector.collect_domain_examples(target_count=400)
    
    # Save to file
    output_path = "/home/shingai/sort/deployments/data/corpus/smart_devops_automation_corpus.jsonl"
    collector.save_examples(examples, output_path)
    
    # Print summary
    categories = {}
    for example in examples:
        category = example.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    print(f"Smart DevOps Automation Collection Complete:")
    for category, count in categories.items():
        print(f"  {category}: {count} examples")
    print(f"Total: {len(examples)} examples")

if __name__ == "__main__":
    main()