#!/usr/bin/env python3
"""
Smart Code Generation Collector - Content-First, Star-Based Quality

Searches for high-quality code generation patterns across popular repositories.
Focuses on API development, database integration, testing, and framework patterns.
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
logger = logging.getLogger("smart_code_generation")

class SmartCodeGenerationCollector(SmartCollectorBase):
    """Smart collector for code generation examples using content-first approach"""
    
    def __init__(self, github_token: Optional[str] = None):
        super().__init__("code_generation", github_token)
        self.validation_pipeline = UniversalValidationPipeline()
        
        # Content-based search configuration
        self.search_config = {
            'api_development': {
                'search_terms': [
                    'FastAPI',
                    'Flask',
                    'Django',
                    'REST',
                    'GraphQL'
                ],
                'content_patterns': [
                    '@app.route', '@app.get', '@app.post', 'def ', 'api',
                    'endpoint', 'router', 'view', 'request', 'response'
                ],
                'min_stars': 1000,
                'max_repos': 30,
                'target_examples': 150
            },
            'database_integration': {
                'search_terms': [
                    'SQLAlchemy',
                    'Django',
                    'pymongo',
                    'PostgreSQL',
                    'database'
                ],
                'content_patterns': [
                    'model', 'db.', 'session', 'query', 'filter',
                    'join', 'relationship', 'table', 'column', 'migrate'
                ],
                'min_stars': 800,
                'max_repos': 25,
                'target_examples': 120
            },
            'testing_patterns': {
                'search_terms': [
                    'pytest',
                    'unittest',
                    'testing',
                    'mock',
                    'fixture'
                ],
                'content_patterns': [
                    'test_', 'def test', 'assert', 'mock', 'fixture',
                    'pytest', 'unittest', '@patch', 'setUp', 'tearDown'
                ],
                'min_stars': 500,
                'max_repos': 35,
                'target_examples': 100
            },
            'async_patterns': {
                'search_terms': [
                    'asyncio',
                    'aiohttp',
                    'async',
                    'await',
                    'coroutine'
                ],
                'content_patterns': [
                    'async def', 'await ', 'asyncio', 'aiohttp',
                    'coroutine', 'gather', 'create_task', 'loop'
                ],
                'min_stars': 600,
                'max_repos': 20,
                'target_examples': 80
            },
            'error_handling': {
                'search_terms': [
                    'exception',
                    'try',
                    'logging',
                    'error',
                    'validation'
                ],
                'content_patterns': [
                    'try:', 'except', 'raise', 'Exception', 'Error',
                    'logging', 'logger', 'validate', 'finally:'
                ],
                'min_stars': 400,
                'max_repos': 30,
                'target_examples': 70
            },
            'data_structures': {
                'search_terms': [
                    'dataclass',
                    'pydantic',
                    'BaseModel',
                    'Enum',
                    'class'
                ],
                'content_patterns': [
                    'class ', '@dataclass', 'BaseModel', 'Enum',
                    '__init__', 'property', 'method', 'validate'
                ],
                'min_stars': 300,
                'max_repos': 25,
                'target_examples': 60
            }
        }
    
    def collect_domain_examples(self, target_count: int = 500) -> List[Dict]:
        """Collect code generation examples using smart content-first approach"""
        logger.info(f"Starting smart code generation collection (target: {target_count})")
        
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
                        file_extensions=['.py'],
                        max_files=15
                    )
                    
                    # Process and validate examples
                    for example in repo_examples:
                        if len(category_examples) >= config['target_examples']:
                            break
                        
                        processed_example = self._create_code_generation_example(
                            example, category
                        )
                        
                        if processed_example and self._validate_code_example(processed_example):
                            category_examples.append(processed_example)
                
                except Exception as e:
                    logger.warning(f"Error processing repo {repo.full_name}: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(category_examples)} {category} examples")
            all_examples.extend(category_examples)
        
        logger.info(f"Total collected: {len(all_examples)} code generation examples")
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
    
    def _create_code_generation_example(self, raw_example: Dict, category: str) -> Optional[Dict]:
        """Create training example for code generation"""
        content = raw_example.get('content', '')
        if len(content.strip()) < 50:
            return None
        
        # Generate category-specific prompt
        prompt = self._generate_code_prompt(content, category)
        
        # Assess code complexity
        complexity = self._assess_code_complexity(content, category)
        
        # Extract code features
        features = self._extract_code_features(content)
        
        return {
            'domain': 'code_generation',
            'category': category,
            'prompt': prompt,
            'completion': content,
            'metadata': {
                'source': f"github.com/{raw_example.get('repo_name', 'unknown')}",
                'file_path': raw_example.get('file_path', ''),
                'language': 'python',
                'complexity': complexity,
                'star_count': raw_example.get('star_count', 0),
                'quality_score': raw_example.get('quality_score', 0.5),
                'features': features,
                'authority_level': 'high' if raw_example.get('star_count', 0) > 2000 else 'medium',
                'authentic': True
            }
        }
    
    def _generate_code_prompt(self, content: str, category: str) -> str:
        """Generate category-specific code prompt"""
        prompts = {
            'api_development': 'Create REST API endpoint with proper request/response handling',
            'database_integration': 'Create database model with ORM relationships and queries',
            'testing_patterns': 'Create comprehensive test suite with fixtures and mocks',
            'async_patterns': 'Create asynchronous code with proper concurrency handling',
            'error_handling': 'Create robust error handling with logging and validation',
            'data_structures': 'Create data classes with validation and business logic'
        }
        
        base_prompt = prompts.get(category, 'Create Python code with best practices')
        
        # Add specific details based on content patterns
        if '@app.route' in content or '@app.get' in content:
            base_prompt += ' using Flask/FastAPI decorators'
        elif 'class ' in content and 'Model' in content:
            base_prompt += ' with SQLAlchemy/Django ORM patterns'
        elif 'async def' in content:
            base_prompt += ' with asyncio coroutines'
        elif 'test_' in content and 'assert' in content:
            base_prompt += ' with pytest testing framework'
        
        return self._generate_claude_md_prompt(content, 'code_generation', base_prompt)
    
    def _assess_code_complexity(self, content: str, category: str) -> str:
        """Assess code complexity"""
        line_count = len(content.split('\n'))
        
        # Category-specific complexity indicators
        complexity_indicators = {
            'api_development': ['@app.', 'router', 'middleware', 'auth', 'validation'],
            'database_integration': ['relationship', 'join', 'query', 'migrate', 'foreign'],
            'testing_patterns': ['fixture', 'mock', 'patch', 'parametrize', 'setUp'],
            'async_patterns': ['await', 'gather', 'create_task', 'semaphore', 'lock'],
            'error_handling': ['raise', 'custom', 'logging', 'traceback', 'context'],
            'data_structures': ['@property', '@classmethod', '__', 'inherit', 'abstract']
        }
        
        indicators = complexity_indicators.get(category, [])
        indicator_count = sum(1 for indicator in indicators if indicator.lower() in content.lower())
        
        # Function and class counting
        function_count = len([line for line in content.split('\n') if 'def ' in line])
        class_count = len([line for line in content.split('\n') if 'class ' in line])
        
        total_complexity = indicator_count + (function_count // 2) + class_count
        
        if line_count > 150 or total_complexity > 6:
            return 'complex'
        elif line_count > 75 or total_complexity > 3:
            return 'medium'
        else:
            return 'simple'
    
    def _extract_code_features(self, content: str) -> Dict:
        """Extract code-specific features"""
        features = {
            'has_classes': 'class ' in content,
            'has_functions': 'def ' in content,
            'has_async': 'async def' in content or 'await ' in content,
            'has_decorators': '@' in content,
            'has_imports': 'import ' in content or 'from ' in content,
            'has_error_handling': 'try:' in content and 'except' in content,
            'has_documentation': '"""' in content or "'''" in content,
            'has_type_hints': '->' in content or ': ' in content,
            'has_testing': any(test in content.lower() for test in ['test_', 'assert', 'mock']),
            'has_logging': 'log' in content.lower(),
            'function_count': len([line for line in content.split('\n') if 'def ' in line]),
            'class_count': len([line for line in content.split('\n') if 'class ' in line]),
            'line_count': len(content.split('\n'))
        }
        
        return features
    
    def _validate_code_example(self, example: Dict) -> bool:
        """Validate code example quality"""
        try:
            # Use universal validation pipeline
            validation_result = self.validation_pipeline.validate_example(example)
            overall_quality = validation_result.get('overall_quality', 0)
            
            # Code-specific validation bonuses
            features = example.get('metadata', {}).get('features', {})
            
            code_bonus = 0
            if features.get('has_functions') or features.get('has_classes'):
                code_bonus += 0.1
            if features.get('has_error_handling'):
                code_bonus += 0.05
            if features.get('has_documentation'):
                code_bonus += 0.05
            if features.get('has_type_hints'):
                code_bonus += 0.05
            
            final_score = overall_quality + code_bonus
            
            # Accept code examples with good quality
            return final_score >= 0.7
            
        except Exception as e:
            logger.debug(f"Code validation failed: {str(e)}")
            return False
    
    def save_examples(self, examples: List[Dict], output_path: str):
        """Save examples to JSONL format"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} code examples to {output_path}")

def main():
    """Main collection process"""
    collector = SmartCodeGenerationCollector()
    
    # Collect examples
    examples = collector.collect_domain_examples(target_count=500)
    
    # Save to file
    output_path = "/home/shingai/sort/deployments/data/corpus/smart_code_generation_corpus.jsonl"
    collector.save_examples(examples, output_path)
    
    # Print summary
    categories = {}
    for example in examples:
        category = example.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    print(f"Smart Code Generation Collection Complete:")
    for category, count in categories.items():
        print(f"  {category}: {count} examples")
    print(f"Total: {len(examples)} examples")

if __name__ == "__main__":
    main()