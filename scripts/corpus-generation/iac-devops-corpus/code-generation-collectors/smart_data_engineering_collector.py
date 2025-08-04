#!/usr/bin/env python3
"""
Smart Data Engineering Collector - Content-First, Star-Based Quality

Searches for data engineering patterns by content, not folder structure.
Targets high-star repositories for quality training examples.
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
logger = logging.getLogger("smart_data_engineering")

class SmartDataEngineeringCollector(SmartCollectorBase):
    """Smart collector for data engineering examples using content-first approach"""
    
    def __init__(self, github_token: Optional[str] = None):
        super().__init__("data_engineering", github_token)
        self.validation_pipeline = UniversalValidationPipeline()
        
        # Content-based search configuration
        self.search_config = {
            'airflow_etl': {
                'search_terms': [
                    'airflow',
                    'DAG',
                    'PythonOperator',
                    'BashOperator',
                    'ETL'
                ],
                'content_patterns': [
                    'DAG(', 'PythonOperator', 'BashOperator', 'SqlOperator',
                    'airflow', 'task_id', 'dag_id', 'schedule_interval'
                ],
                'min_stars': 300,
                'max_repos': 30,
                'target_examples': 80
            },
            'data_processing': {
                'search_terms': [
                    'pandas',
                    'spark',
                    'pyspark',
                    'pipeline',
                    'ETL'
                ],
                'content_patterns': [
                    'pandas', 'DataFrame', 'spark', 'pyspark', 'sql',
                    'transform', 'extract', 'load', 'pipeline', 'etl'
                ],
                'min_stars': 500,
                'max_repos': 25,
                'target_examples': 60
            },
            'sql_analytics': {
                'search_terms': [
                    'SQL',
                    'SELECT',
                    'dbt',
                    'analytics',
                    'query'
                ],
                'content_patterns': [
                    'select', 'from', 'where', 'join', 'group by',
                    'order by', 'having', 'with', 'cte', 'case when'
                ],
                'min_stars': 200,
                'max_repos': 35,
                'target_examples': 60
            },
            'data_modeling': {
                'search_terms': [
                    'dbt',
                    'warehouse',
                    'modeling',
                    'dimensional',
                    'schema'
                ],
                'content_patterns': [
                    'dbt', 'model', 'fact_', 'dim_', 'warehouse',
                    'star schema', 'snowflake', 'kimball', 'dimensional'
                ],
                'min_stars': 150,
                'max_repos': 20,
                'target_examples': 40
            }
        }
    
    def collect_domain_examples(self, target_count: int = 250) -> List[Dict]:
        """Collect data engineering examples using smart content-first approach"""
        logger.info(f"Starting smart data engineering collection (target: {target_count})")
        
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
                        file_extensions=['.py', '.sql', '.yml', '.yaml'],
                        max_files=15
                    )
                    
                    # Process and validate examples
                    for example in repo_examples:
                        if len(category_examples) >= config['target_examples']:
                            break
                        
                        processed_example = self._create_data_engineering_example(
                            example, category
                        )
                        
                        if processed_example and self._validate_example(processed_example):
                            category_examples.append(processed_example)
                
                except Exception as e:
                    logger.warning(f"Error processing repo {repo.full_name}: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(category_examples)} {category} examples")
            all_examples.extend(category_examples)
        
        logger.info(f"Total collected: {len(all_examples)} data engineering examples")
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
    
    def _create_data_engineering_example(self, raw_example: Dict, category: str) -> Optional[Dict]:
        """Create training example for data engineering"""
        content = raw_example.get('content', '')
        if len(content.strip()) < 50:
            return None
        
        # Generate category-specific prompt
        prompt = self._generate_category_prompt(content, category)
        
        # Assess complexity
        complexity = self._assess_data_complexity(content, category)
        
        return {
            'domain': 'data_engineering',
            'category': category,
            'prompt': prompt,
            'completion': content,
            'metadata': {
                'source': f"github.com/{raw_example.get('repo_name', 'unknown')}",
                'file_path': raw_example.get('file_path', ''),
                'language': self._detect_language(raw_example.get('file_path', '')),
                'complexity': complexity,
                'star_count': raw_example.get('star_count', 0),
                'quality_score': raw_example.get('quality_score', 0.5),
                'authority_level': 'high' if raw_example.get('star_count', 0) > 1000 else 'medium',
                'authentic': True
            }
        }
    
    def _generate_category_prompt(self, content: str, category: str) -> str:
        """Generate category-specific prompt"""
        prompts = {
            'airflow_etl': 'Create Airflow DAG for data pipeline automation',
            'data_processing': 'Create data processing pipeline for transformation and analysis',
            'sql_analytics': 'Create SQL query for data analytics and reporting',
            'data_modeling': 'Create data model for warehouse dimensional design'
        }
        
        base_prompt = prompts.get(category, 'Create data engineering solution')
        
        # Add specific details based on content
        if 'pandas' in content.lower():
            base_prompt += ' using pandas for data manipulation'
        elif 'spark' in content.lower():
            base_prompt += ' using Spark for big data processing'
        elif 'select' in content.lower():
            base_prompt += ' with SQL analytics queries'
        
        return self._generate_claude_md_prompt(content, 'data_engineering', base_prompt)
    
    def _assess_data_complexity(self, content: str, category: str) -> str:
        """Assess data engineering complexity"""
        line_count = len(content.split('\n'))
        
        # Category-specific complexity indicators
        complexity_indicators = {
            'airflow_etl': ['XCom', 'Variable', 'sensor', 'branch', 'trigger_rule'],
            'data_processing': ['join', 'groupby', 'transform', 'pipeline', 'parallel'],
            'sql_analytics': ['window', 'cte', 'recursive', 'pivot', 'partition'],
            'data_modeling': ['fact', 'dimension', 'scd', 'star', 'snowflake']
        }
        
        indicators = complexity_indicators.get(category, [])
        indicator_count = sum(1 for indicator in indicators if indicator.lower() in content.lower())
        
        if line_count > 200 or indicator_count > 3:
            return 'complex'
        elif line_count > 100 or indicator_count > 1:
            return 'medium'
        else:
            return 'simple'
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        if file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.sql'):
            return 'sql'
        elif file_path.endswith(('.yml', '.yaml')):
            return 'yaml'
        else:
            return 'unknown'
    
    def _validate_example(self, example: Dict) -> bool:
        """Validate example quality"""
        try:
            # Use universal validation pipeline
            validation_result = self.validation_pipeline.validate_example(example)
            overall_quality = validation_result.get('overall_quality', 0)
            
            # Accept examples with decent quality (lower threshold for data engineering)
            return overall_quality >= 0.6
            
        except Exception as e:
            logger.debug(f"Validation failed: {str(e)}")
            return False
    
    def save_examples(self, examples: List[Dict], output_path: str):
        """Save examples to JSONL format"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")

def main():
    """Main collection process"""
    collector = SmartDataEngineeringCollector()
    
    # Collect examples
    examples = collector.collect_domain_examples(target_count=250)
    
    # Save to file
    output_path = "/home/shingai/sort/deployments/data/corpus/smart_data_engineering_corpus.jsonl"
    collector.save_examples(examples, output_path)
    
    # Print summary
    categories = {}
    for example in examples:
        category = example.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    print(f"Smart Data Engineering Collection Complete:")
    for category, count in categories.items():
        print(f"  {category}: {count} examples")
    print(f"Total: {len(examples)} examples")

if __name__ == "__main__":
    main()