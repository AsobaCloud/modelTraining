#!/usr/bin/env python3
"""
Smart System Architecture Collector - Content-First, Star-Based Quality

Searches for high-quality system architecture patterns across popular repositories.
Focuses on microservices, scalability, distributed systems, and architecture documentation.
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
logger = logging.getLogger("smart_system_architecture")

class SmartSystemArchitectureCollector(SmartCollectorBase):
    """Smart collector for system architecture examples using content-first approach"""
    
    def __init__(self, github_token: Optional[str] = None):
        super().__init__("system_architecture", github_token)
        self.validation_pipeline = UniversalValidationPipeline()
        
        # Content-based search configuration
        self.search_config = {
            'microservices_patterns': {
                'search_terms': [
                    'microservices',
                    'microservice',
                    'service-mesh',
                    'distributed',
                    'architecture'
                ],
                'content_patterns': [
                    'microservice', 'service', 'distributed', 'mesh',
                    'gateway', 'load balancer', 'circuit breaker'
                ],
                'min_stars': 500,
                'max_repos': 25,
                'target_examples': 100
            },
            'scalability_patterns': {
                'search_terms': [
                    'scalability',
                    'horizontal',
                    'vertical',
                    'load-balancing',
                    'caching'
                ],
                'content_patterns': [
                    'scale', 'load', 'cache', 'redis', 'memcached',
                    'horizontal', 'vertical', 'sharding', 'partition'
                ],
                'min_stars': 300,
                'max_repos': 30,
                'target_examples': 80
            },
            'distributed_systems': {
                'search_terms': [
                    'distributed',
                    'consensus',
                    'raft',
                    'eventual',
                    'consistency'
                ],
                'content_patterns': [
                    'distributed', 'consensus', 'raft', 'paxos',
                    'eventual', 'consistency', 'partition', 'tolerance'
                ],
                'min_stars': 400,
                'max_repos': 20,
                'target_examples': 60
            },
            'cloud_architecture': {
                'search_terms': [
                    'cloud',
                    'serverless',
                    'lambda',
                    'container',
                    'kubernetes'
                ],
                'content_patterns': [
                    'cloud', 'aws', 'azure', 'gcp', 'serverless',
                    'lambda', 'container', 'kubernetes', 'docker'
                ],
                'min_stars': 600,
                'max_repos': 25,
                'target_examples': 80
            },
            'api_design': {
                'search_terms': [
                    'api-design',
                    'rest',
                    'graphql',
                    'grpc',
                    'swagger'
                ],
                'content_patterns': [
                    'api', 'rest', 'graphql', 'grpc', 'swagger',
                    'openapi', 'endpoint', 'schema', 'versioning'
                ],
                'min_stars': 200,
                'max_repos': 35,
                'target_examples': 80
            }
        }
    
    def collect_domain_examples(self, target_count: int = 400) -> List[Dict]:
        """Collect system architecture examples using smart content-first approach"""
        logger.info(f"Starting smart system architecture collection (target: {target_count})")
        
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
                        file_extensions=['.py', '.js', '.go', '.java', '.md', '.yml', '.yaml', '.json'],
                        max_files=15
                    )
                    
                    # Process and validate examples
                    for example in repo_examples:
                        if len(category_examples) >= config['target_examples']:
                            break
                        
                        processed_example = self._create_architecture_example(
                            example, category
                        )
                        
                        if processed_example and self._validate_architecture_example(processed_example):
                            category_examples.append(processed_example)
                
                except Exception as e:
                    logger.warning(f"Error processing repo {repo.full_name}: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(category_examples)} {category} examples")
            all_examples.extend(category_examples)
        
        logger.info(f"Total collected: {len(all_examples)} system architecture examples")
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
    
    def _create_architecture_example(self, raw_example: Dict, category: str) -> Optional[Dict]:
        """Create training example for system architecture"""
        content = raw_example.get('content', '')
        if len(content.strip()) < 40:
            return None
        
        # Generate category-specific prompt
        prompt = self._generate_architecture_prompt(content, category)
        
        # Assess architecture complexity
        complexity = self._assess_architecture_complexity(content, category)
        
        # Extract architecture features
        features = self._extract_architecture_features(content)
        
        return {
            'domain': 'system_architecture',
            'category': category,
            'prompt': prompt,
            'completion': content,
            'metadata': {
                'source': f"github.com/{raw_example.get('repo_name', 'unknown')}",
                'file_path': raw_example.get('file_path', ''),
                'language': self._detect_architecture_language(raw_example.get('file_path', '')),
                'complexity': complexity,
                'star_count': raw_example.get('star_count', 0),
                'quality_score': raw_example.get('quality_score', 0.5),
                'features': features,
                'authority_level': 'high' if raw_example.get('star_count', 0) > 2000 else 'medium',
                'authentic': True
            }
        }
    
    def _generate_architecture_prompt(self, content: str, category: str) -> str:
        """Generate category-specific architecture prompt"""
        prompts = {
            'microservices_patterns': 'Design microservices architecture with service communication patterns',
            'scalability_patterns': 'Design scalable system architecture with load balancing and caching',
            'distributed_systems': 'Design distributed system with consensus and consistency patterns',
            'cloud_architecture': 'Design cloud-native architecture with serverless and container patterns',
            'api_design': 'Design RESTful API architecture with proper versioning and documentation'
        }
        
        base_prompt = prompts.get(category, 'Design system architecture solution')
        
        # Add specific details based on content patterns
        if 'microservice' in content.lower():
            base_prompt += ' using microservices design patterns'
        elif 'kubernetes' in content.lower() or 'docker' in content.lower():
            base_prompt += ' with containerized deployment'
        elif 'aws' in content.lower() or 'cloud' in content.lower():
            base_prompt += ' for cloud infrastructure'
        elif 'api' in content.lower() and 'rest' in content.lower():
            base_prompt += ' with RESTful API design'
        elif 'distributed' in content.lower():
            base_prompt += ' for distributed system requirements'
        
        return self._generate_claude_md_prompt(content, 'system_architecture', base_prompt)
    
    def _assess_architecture_complexity(self, content: str, category: str) -> str:
        """Assess system architecture complexity"""
        line_count = len(content.split('\n'))
        
        # Category-specific complexity indicators
        complexity_indicators = {
            'microservices_patterns': ['gateway', 'circuit breaker', 'service mesh', 'discovery', 'tracing'],
            'scalability_patterns': ['load balancer', 'caching', 'sharding', 'partition', 'replica'],
            'distributed_systems': ['consensus', 'raft', 'paxos', 'eventual consistency', 'CAP'],
            'cloud_architecture': ['serverless', 'lambda', 'container', 'orchestration', 'auto-scaling'],
            'api_design': ['versioning', 'authentication', 'rate limiting', 'pagination', 'swagger']
        }
        
        indicators = complexity_indicators.get(category, [])
        indicator_count = sum(1 for indicator in indicators if indicator.lower() in content.lower())
        
        # Architecture-specific complexity scoring
        arch_patterns = ['pattern', 'design', 'architecture', 'system', 'service']
        arch_count = sum(1 for pattern in arch_patterns if pattern.lower() in content.lower())
        
        total_complexity = indicator_count + (arch_count // 2)
        
        if line_count > 200 or total_complexity > 5:
            return 'complex'
        elif line_count > 100 or total_complexity > 2:
            return 'medium'
        else:
            return 'simple'
    
    def _extract_architecture_features(self, content: str) -> Dict:
        """Extract architecture-specific features"""
        features = {
            'has_microservices': 'microservice' in content.lower(),
            'has_api_design': 'api' in content.lower() or 'rest' in content.lower(),
            'has_load_balancing': 'load' in content.lower() and 'balanc' in content.lower(),
            'has_caching': any(cache in content.lower() for cache in ['cache', 'redis', 'memcached']),
            'has_database_design': any(db in content.lower() for db in ['database', 'sql', 'nosql', 'mongodb']),
            'has_security': any(sec in content.lower() for sec in ['auth', 'security', 'encryption', 'token']),
            'has_monitoring': any(mon in content.lower() for mon in ['monitor', 'metric', 'log', 'trace']),
            'has_scaling': any(scale in content.lower() for scale in ['scale', 'horizontal', 'vertical']),
            'has_cloud_patterns': any(cloud in content.lower() for cloud in ['aws', 'azure', 'gcp', 'cloud']),
            'has_containers': any(cont in content.lower() for cont in ['docker', 'container', 'kubernetes']),
            'has_diagrams': any(diag in content for diag in ['```mermaid', '![', 'diagram', 'architecture']),
            'line_count': len(content.split('\n'))
        }
        
        return features
    
    def _detect_architecture_language(self, file_path: str) -> str:
        """Detect architecture file type from extension"""
        if file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.js'):
            return 'javascript'
        elif file_path.endswith('.go'):
            return 'go'
        elif file_path.endswith('.java'):
            return 'java'
        elif file_path.endswith('.md'):
            return 'markdown'
        elif file_path.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif file_path.endswith('.json'):
            return 'json'
        else:
            return 'unknown'
    
    def _validate_architecture_example(self, example: Dict) -> bool:
        """Validate architecture example with enhanced criteria"""
        try:
            # Use universal validation pipeline
            validation_result = self.validation_pipeline.validate_example(example)
            overall_quality = validation_result.get('overall_quality', 0)
            
            # Architecture-specific validation bonuses
            features = example.get('metadata', {}).get('features', {})
            
            arch_bonus = 0
            if features.get('has_microservices'):
                arch_bonus += 0.1
            if features.get('has_api_design'):
                arch_bonus += 0.05
            if features.get('has_load_balancing') or features.get('has_scaling'):
                arch_bonus += 0.05
            if features.get('has_cloud_patterns'):
                arch_bonus += 0.05
            if features.get('has_security'):
                arch_bonus += 0.05
            
            final_score = overall_quality + arch_bonus
            
            # Accept architecture examples with good quality
            return final_score >= 0.7
            
        except Exception as e:
            logger.debug(f"Architecture validation failed: {str(e)}")
            return False
    
    def save_examples(self, examples: List[Dict], output_path: str):
        """Save examples to JSONL format"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} architecture examples to {output_path}")

def main():
    """Main collection process"""
    collector = SmartSystemArchitectureCollector()
    
    # Collect examples
    examples = collector.collect_domain_examples(target_count=400)
    
    # Save to file
    output_path = "/home/shingai/sort/deployments/data/corpus/smart_system_architecture_corpus.jsonl"
    collector.save_examples(examples, output_path)
    
    # Print summary
    categories = {}
    for example in examples:
        category = example.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    print(f"Smart System Architecture Collection Complete:")
    for category, count in categories.items():
        print(f"  {category}: {count} examples")
    print(f"Total: {len(examples)} examples")

if __name__ == "__main__":
    main()