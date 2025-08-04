#!/usr/bin/env python3
"""
Smart Technical Documentation Collector - Content-First, Star-Based Quality

Searches for high-quality technical documentation patterns across popular repositories.
Focuses on API docs, tutorials, README files, and documentation generation.
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
logger = logging.getLogger("smart_technical_documentation")

class SmartTechnicalDocumentationCollector(SmartCollectorBase):
    """Smart collector for technical documentation examples using content-first approach"""
    
    def __init__(self, github_token: Optional[str] = None):
        super().__init__("technical_documentation", github_token)
        self.validation_pipeline = UniversalValidationPipeline()
        
        # Content-based search configuration
        self.search_config = {
            'api_documentation': {
                'search_terms': [
                    'OpenAPI',
                    'Swagger',
                    'documentation',
                    'API',
                    'docs'
                ],
                'content_patterns': [
                    'openapi', 'swagger', 'api', 'endpoint', 'docs',
                    'documentation', 'schema', 'response', 'request'
                ],
                'min_stars': 500,
                'max_repos': 30,
                'target_examples': 120
            },
            'tutorials_guides': {
                'search_terms': [
                    'tutorial',
                    'guide',
                    'howto',
                    'example',
                    'demo'
                ],
                'content_patterns': [
                    'tutorial', 'guide', 'step', 'example', 'demo',
                    'how to', 'getting started', 'quickstart', 'introduction'
                ],
                'min_stars': 300,
                'max_repos': 25,
                'target_examples': 80
            },
            'readme_documentation': {
                'search_terms': [
                    'README',
                    'installation',
                    'usage',
                    'getting-started',
                    'quickstart'
                ],
                'content_patterns': [
                    'install', 'usage', 'example', 'demo', 'getting started',
                    'quickstart', 'setup', 'configuration', 'readme'
                ],
                'min_stars': 1000,
                'max_repos': 20,
                'target_examples': 60
            },
            'code_comments': {
                'search_terms': [
                    'docstring',
                    'comments',
                    'documentation',
                    'sphinx',
                    'pydoc'
                ],
                'content_patterns': [
                    '"""', "'''", 'docstring', 'param', 'return',
                    'raises', 'example', 'note', 'warning'
                ],
                'min_stars': 200,
                'max_repos': 35,
                'target_examples': 40
            }
        }
    
    def collect_domain_examples(self, target_count: int = 300) -> List[Dict]:
        """Collect technical documentation examples using smart content-first approach"""
        logger.info(f"Starting smart technical documentation collection (target: {target_count})")
        
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
                        file_extensions=['.md', '.rst', '.txt', '.py', '.yml', '.yaml'],
                        max_files=15
                    )
                    
                    # Process and validate examples
                    for example in repo_examples:
                        if len(category_examples) >= config['target_examples']:
                            break
                        
                        processed_example = self._create_documentation_example(
                            example, category
                        )
                        
                        if processed_example and self._validate_documentation_example(processed_example):
                            category_examples.append(processed_example)
                
                except Exception as e:
                    logger.warning(f"Error processing repo {repo.full_name}: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(category_examples)} {category} examples")
            all_examples.extend(category_examples)
        
        logger.info(f"Total collected: {len(all_examples)} technical documentation examples")
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
    
    def _create_documentation_example(self, raw_example: Dict, category: str) -> Optional[Dict]:
        """Create training example for technical documentation"""
        content = raw_example.get('content', '')
        if len(content.strip()) < 50:
            return None
        
        # Generate category-specific prompt
        prompt = self._generate_documentation_prompt(content, category)
        
        # Assess documentation complexity
        complexity = self._assess_documentation_complexity(content, category)
        
        # Extract documentation features
        features = self._extract_documentation_features(content)
        
        return {
            'domain': 'technical_documentation',
            'category': category,
            'prompt': prompt,
            'completion': content,
            'metadata': {
                'source': f"github.com/{raw_example.get('repo_name', 'unknown')}",
                'file_path': raw_example.get('file_path', ''),
                'language': self._detect_file_type(raw_example.get('file_path', '')),
                'complexity': complexity,
                'star_count': raw_example.get('star_count', 0),
                'quality_score': raw_example.get('quality_score', 0.5),
                'features': features,
                'authority_level': 'high' if raw_example.get('star_count', 0) > 2000 else 'medium',
                'authentic': True
            }
        }
    
    def _generate_documentation_prompt(self, content: str, category: str) -> str:
        """Generate category-specific documentation prompt"""
        prompts = {
            'api_documentation': 'Create comprehensive API documentation with examples and schemas',
            'tutorials_guides': 'Create step-by-step tutorial with clear explanations and examples',
            'readme_documentation': 'Create project README with installation, usage, and examples',
            'code_comments': 'Create detailed code documentation with docstrings and comments'
        }
        
        base_prompt = prompts.get(category, 'Create technical documentation')
        
        # Add specific details based on content patterns
        if 'openapi' in content.lower() or 'swagger' in content.lower():
            base_prompt += ' using OpenAPI/Swagger specification'
        elif 'installation' in content.lower():
            base_prompt += ' with installation and setup instructions'
        elif 'example' in content.lower():
            base_prompt += ' with practical examples and code samples'
        elif 'api' in content.lower():
            base_prompt += ' for API endpoints and usage'
        
        return self._generate_claude_md_prompt(content, 'technical_documentation', base_prompt)
    
    def _assess_documentation_complexity(self, content: str, category: str) -> str:
        """Assess documentation complexity"""
        line_count = len(content.split('\n'))
        
        # Category-specific complexity indicators
        complexity_indicators = {
            'api_documentation': ['openapi', 'swagger', 'schema', 'endpoint', 'response'],
            'tutorials_guides': ['step', 'tutorial', 'example', 'guide', 'walkthrough'],
            'readme_documentation': ['installation', 'usage', 'example', 'configuration', 'setup'],
            'code_comments': ['param', 'return', 'raises', 'example', 'docstring']
        }
        
        indicators = complexity_indicators.get(category, [])
        indicator_count = sum(1 for indicator in indicators if indicator.lower() in content.lower())
        
        # Documentation-specific complexity scoring
        doc_patterns = ['example', 'code', 'snippet', 'tutorial', 'guide']
        doc_count = sum(1 for pattern in doc_patterns if pattern.lower() in content.lower())
        
        total_complexity = indicator_count + doc_count
        
        if line_count > 200 or total_complexity > 5:
            return 'complex'
        elif line_count > 100 or total_complexity > 2:
            return 'medium'
        else:
            return 'simple'
    
    def _extract_documentation_features(self, content: str) -> Dict:
        """Extract documentation-specific features"""
        features = {
            'has_code_examples': any(marker in content for marker in ['```', '`', 'python', 'javascript']),
            'has_installation_guide': 'install' in content.lower(),
            'has_usage_examples': 'usage' in content.lower() or 'example' in content.lower(),
            'has_api_documentation': 'api' in content.lower() or 'endpoint' in content.lower(),
            'has_configuration': 'config' in content.lower() or 'setting' in content.lower(),
            'has_troubleshooting': 'troubleshoot' in content.lower() or 'error' in content.lower(),
            'has_links': 'http' in content or '[' in content,
            'has_images': '![' in content or '.png' in content or '.jpg' in content,
            'has_tables': '|' in content and '-' in content,
            'has_headers': '#' in content,
            'line_count': len(content.split('\n')),
            'word_count': len(content.split())
        }
        
        return features
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension"""
        if file_path.endswith('.md'):
            return 'markdown'
        elif file_path.endswith('.rst'):
            return 'restructuredtext'
        elif file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif file_path.endswith('.txt'):
            return 'text'
        else:
            return 'unknown'
    
    def _validate_documentation_example(self, example: Dict) -> bool:
        """Validate documentation example with enhanced criteria"""
        try:
            # Use universal validation pipeline
            validation_result = self.validation_pipeline.validate_example(example)
            overall_quality = validation_result.get('overall_quality', 0)
            
            # Documentation-specific validation bonuses
            features = example.get('metadata', {}).get('features', {})
            
            doc_bonus = 0
            if features.get('has_code_examples'):
                doc_bonus += 0.1
            if features.get('has_usage_examples'):
                doc_bonus += 0.05
            if features.get('has_installation_guide'):
                doc_bonus += 0.05
            if features.get('has_api_documentation'):
                doc_bonus += 0.05
            
            final_score = overall_quality + doc_bonus
            
            # Accept documentation examples with good quality
            return final_score >= 0.65
            
        except Exception as e:
            logger.debug(f"Documentation validation failed: {str(e)}")
            return False
    
    def save_examples(self, examples: List[Dict], output_path: str):
        """Save examples to JSONL format"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} documentation examples to {output_path}")

def main():
    """Main collection process"""
    collector = SmartTechnicalDocumentationCollector()
    
    # Collect examples
    examples = collector.collect_domain_examples(target_count=300)
    
    # Save to file
    output_path = "/home/shingai/sort/deployments/data/corpus/smart_technical_documentation_corpus.jsonl"
    collector.save_examples(examples, output_path)
    
    # Print summary
    categories = {}
    for example in examples:
        category = example.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    print(f"Smart Technical Documentation Collection Complete:")
    for category, count in categories.items():
        print(f"  {category}: {count} examples")
    print(f"Total: {len(examples)} examples")

if __name__ == "__main__":
    main()