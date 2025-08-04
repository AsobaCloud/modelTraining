#!/usr/bin/env python3
"""
Data Engineering Collector for Qwen Multi-Domain Training

Collects high-quality data engineering examples from ETL/ELT pipelines,
data modeling patterns, analytics queries, and stream processing configurations.
Target: 500 examples across ETL pipelines, data modeling, and analytics/reporting.
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import yaml
from github import Github

# Import our validation pipeline
import sys
sys.path.append(str(Path(__file__).parent.parent / 'validation'))
from universal_validation_pipeline import UniversalValidationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_engineering_collector")

class AirflowDAGExtractor:
    """Extract Airflow DAG examples from repositories"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.github_client = Github(self.github_token) if self.github_token else None
        
        # Airflow-related patterns
        self.airflow_patterns = {
            'dags': ['dags/*.py', 'airflow/dags/*.py', 'src/dags/*.py'],
            'plugins': ['plugins/*.py', 'airflow/plugins/*.py'],
            'operators': ['operators/*.py', 'airflow/operators/*.py'],
            'hooks': ['hooks/*.py', 'airflow/hooks/*.py']
        }
    
    def extract_airflow_examples(self, repo_name: str) -> List[Dict]:
        """Extract Airflow DAG and plugin examples from repository"""
        if not self.github_client:
            logger.warning("No GitHub client available for Airflow extraction")
            return []
        
        try:
            repo = self.github_client.get_repo(repo_name)
            examples = []
            
            for component_type, patterns in self.airflow_patterns.items():
                component_examples = self._extract_component_type(
                    repo, component_type, patterns
                )
                examples.extend(component_examples)
            
            logger.info(f"Extracted {len(examples)} Airflow examples from {repo_name}")
            return examples
        
        except Exception as e:
            logger.error(f"Error extracting Airflow from {repo_name}: {str(e)}")
            return []
    
    def _extract_component_type(self, repo, component_type: str, patterns: List[str]) -> List[Dict]:
        """Extract specific type of Airflow component"""
        examples = []
        
        for pattern in patterns:
            try:
                files = self._find_python_files_by_pattern(repo, pattern)
                
                for file_path in files[:10]:  # Limit files per pattern
                    content = self._get_file_content(repo, file_path)
                    if content and self._is_airflow_content(content, component_type):
                        example = self._create_airflow_example(
                            content, file_path, component_type, repo.full_name
                        )
                        if example:
                            examples.append(example)
            
            except Exception as e:
                logger.debug(f"Could not process pattern {pattern}: {str(e)}")
                continue
        
        return examples
    
    def _find_python_files_by_pattern(self, repo, pattern: str) -> List[str]:
        """Find Python files matching pattern"""
        try:
            if '/' in pattern:
                directory = pattern.split('/')[0]
                contents = repo.get_contents(directory)
            else:
                contents = repo.get_contents("")
            
            python_files = []
            for content in contents:
                if content.type == 'file' and content.name.endswith('.py'):
                    python_files.append(content.path)
            
            return python_files
        except:
            return []
    
    def _get_file_content(self, repo, file_path: str) -> Optional[str]:
        """Get file content from repository"""
        try:
            file_content = repo.get_contents(file_path)
            if file_content.encoding == 'base64':
                import base64
                return base64.b64decode(file_content.content).decode('utf-8')
            else:
                return file_content.content
        except:
            return None
    
    def _is_airflow_content(self, content: str, component_type: str) -> bool:
        """Check if content is valid Airflow component"""
        airflow_indicators = {
            'dags': ['DAG', 'dag =', '@dag', 'PythonOperator', 'BashOperator'],
            'plugins': ['AirflowPlugin', 'BaseOperator', 'BaseHook'],
            'operators': ['BaseOperator', 'execute(', 'class.*Operator'],
            'hooks': ['BaseHook', 'get_conn', 'class.*Hook']
        }
        
        indicators = airflow_indicators.get(component_type, ['airflow'])
        
        # Check for Airflow imports
        has_airflow_import = 'from airflow' in content or 'import airflow' in content
        
        # Check for component-specific patterns
        has_component_patterns = any(
            re.search(indicator, content, re.IGNORECASE) for indicator in indicators
        )
        
        return has_airflow_import and has_component_patterns
    
    def _create_airflow_example(self, content: str, file_path: str, 
                               component_type: str, repo_name: str) -> Optional[Dict]:
        """Create training example from Airflow component"""
        if len(content.strip()) < 200:  # Minimum content length
            return None
        
        # Extract key information
        dag_info = self._extract_dag_info(content) if component_type == 'dags' else {}
        
        # Generate descriptive prompt
        prompt = self._generate_airflow_prompt(dag_info, component_type, file_path)
        
        # Assess complexity
        complexity = self._assess_airflow_complexity(content, dag_info)
        
        return {
            'content': content,
            'dag_info': dag_info,
            'prompt': prompt,
            'component_type': component_type,
            'file_path': file_path,
            'repo_name': repo_name,
            'complexity': complexity
        }
    
    def _extract_dag_info(self, content: str) -> Dict:
        """Extract DAG metadata from content"""
        dag_info = {}
        
        # Extract DAG ID
        dag_id_match = re.search(r'dag_id\s*=\s*["\']([^"\']+)["\']', content)
        if dag_id_match:
            dag_info['dag_id'] = dag_id_match.group(1)
        
        # Extract schedule interval
        schedule_match = re.search(r'schedule_interval\s*=\s*["\']([^"\']+)["\']', content)
        if schedule_match:
            dag_info['schedule_interval'] = schedule_match.group(1)
        
        # Extract operators used
        operator_patterns = [
            r'(\w*Operator)\s*\(',
            r'from airflow\.operators\.(\w+)',
            r'import.*(\w+Operator)'
        ]
        
        operators = set()
        for pattern in operator_patterns:
            matches = re.findall(pattern, content)
            operators.update(matches)
        
        dag_info['operators'] = list(operators)
        
        # Count tasks
        task_count = len(re.findall(r'\w+\s*=\s*\w*Operator\s*\(', content))
        dag_info['task_count'] = task_count
        
        return dag_info
    
    def _generate_airflow_prompt(self, dag_info: Dict, component_type: str, file_path: str) -> str:
        """Generate descriptive prompt for Airflow example"""
        if component_type == 'dags':
            dag_id = dag_info.get('dag_id', 'data processing')
            operators = dag_info.get('operators', [])
            task_count = dag_info.get('task_count', 0)
            
            if operators:
                operator_text = f"using {', '.join(operators[:3])}"
            else:
                operator_text = "with multiple tasks"
            
            return f"Create Airflow DAG for {dag_id} {operator_text} ({task_count} tasks)"
        
        elif component_type == 'operators':
            return "Create custom Airflow operator for data processing tasks"
        elif component_type == 'hooks':
            return "Create custom Airflow hook for external system integration"
        else:
            return f"Create Airflow {component_type} for data pipeline automation"
    
    def _assess_airflow_complexity(self, content: str, dag_info: Dict) -> str:
        """Assess Airflow component complexity"""
        # Count complexity indicators
        line_count = len(content.split('\n'))
        task_count = dag_info.get('task_count', 0)
        operator_count = len(dag_info.get('operators', []))
        
        # Check for advanced patterns
        advanced_patterns = [
            'XCom', 'Variable', 'Connection', 'template', 'jinja',
            'branch', 'sensor', 'trigger_rule', 'depends_on_past'
        ]
        
        advanced_count = sum(1 for pattern in advanced_patterns if pattern in content)
        
        if line_count > 200 or task_count > 10 or advanced_count > 3:
            return 'complex'
        elif line_count > 100 or task_count > 5 or advanced_count > 1:
            return 'medium'
        else:
            return 'simple'

class DBTProjectExtractor:
    """Extract dbt (data build tool) project examples"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.github_client = Github(self.github_token) if self.github_token else None
        
        # dbt-related patterns
        self.dbt_patterns = {
            'models': ['models/*.sql', 'dbt/models/*.sql'],
            'macros': ['macros/*.sql', 'dbt/macros/*.sql'],
            'tests': ['tests/*.sql', 'dbt/tests/*.sql'],
            'snapshots': ['snapshots/*.sql', 'dbt/snapshots/*.sql'],
            'config': ['dbt_project.yml', 'profiles.yml']
        }
    
    def extract_dbt_examples(self, repo_name: str) -> List[Dict]:
        """Extract dbt project examples from repository"""
        if not self.github_client:
            return []
        
        try:
            repo = self.github_client.get_repo(repo_name)
            examples = []
            
            # Check if it's a dbt project
            if not self._is_dbt_project(repo):
                return []
            
            for component_type, patterns in self.dbt_patterns.items():
                component_examples = self._extract_dbt_component_type(
                    repo, component_type, patterns
                )
                examples.extend(component_examples)
            
            logger.info(f"Extracted {len(examples)} dbt examples from {repo_name}")
            return examples
        
        except Exception as e:
            logger.error(f"Error extracting dbt from {repo_name}: {str(e)}")
            return []
    
    def _is_dbt_project(self, repo) -> bool:
        """Check if repository is a dbt project"""
        try:
            # Look for dbt_project.yml
            repo.get_contents("dbt_project.yml")
            return True
        except:
            try:
                # Look for dbt directory structure
                contents = repo.get_contents("")
                has_models = any(c.name == 'models' and c.type == 'dir' for c in contents)
                has_dbt_files = any('.sql' in c.name for c in contents if c.type == 'file')
                return has_models or has_dbt_files
            except:
                return False
    
    def _extract_dbt_component_type(self, repo, component_type: str, patterns: List[str]) -> List[Dict]:
        """Extract specific type of dbt component"""
        examples = []
        
        for pattern in patterns:
            try:
                files = self._find_files_by_pattern(repo, pattern)
                
                for file_path in files[:10]:  # Limit files per pattern
                    content = self._get_file_content(repo, file_path)
                    if content and len(content.strip()) > 50:
                        example = self._create_dbt_example(
                            content, file_path, component_type, repo.full_name
                        )
                        if example:
                            examples.append(example)
            
            except Exception as e:
                logger.debug(f"Could not process dbt pattern {pattern}: {str(e)}")
                continue
        
        return examples
    
    def _find_files_by_pattern(self, repo, pattern: str) -> List[str]:
        """Find files matching dbt pattern"""
        try:
            if '/' in pattern:
                directory = pattern.split('/')[0]
                try:
                    contents = repo.get_contents(directory)
                    files = []
                    for content in contents:
                        if content.type == 'file':
                            if pattern.endswith('*.sql') and content.name.endswith('.sql'):
                                files.append(content.path)
                            elif pattern.endswith('*.yml') and content.name.endswith('.yml'):
                                files.append(content.path)
                    return files
                except:
                    return []
            else:
                # Single file
                try:
                    repo.get_contents(pattern)
                    return [pattern]
                except:
                    return []
        except:
            return []
    
    def _get_file_content(self, repo, file_path: str) -> Optional[str]:
        """Get file content from repository"""
        try:
            file_content = repo.get_contents(file_path)
            if file_content.encoding == 'base64':
                import base64
                return base64.b64decode(file_content.content).decode('utf-8')
            else:
                return file_content.content
        except:
            return None
    
    def _create_dbt_example(self, content: str, file_path: str,
                           component_type: str, repo_name: str) -> Optional[Dict]:
        """Create training example from dbt component"""
        # Generate descriptive prompt
        prompt = self._generate_dbt_prompt(content, component_type, file_path)
        
        # Assess complexity
        complexity = self._assess_dbt_complexity(content, component_type)
        
        return {
            'content': content,
            'prompt': prompt,
            'component_type': component_type,
            'file_path': file_path,
            'repo_name': repo_name,
            'complexity': complexity
        }
    
    def _generate_dbt_prompt(self, content: str, component_type: str, file_path: str) -> str:
        """Generate descriptive prompt for dbt example"""
        if component_type == 'models':
            # Extract model name from file path
            model_name = Path(file_path).stem
            return f"Create dbt model '{model_name}' for data transformation"
        
        elif component_type == 'macros':
            # Try to extract macro name
            macro_match = re.search(r'macro\s+(\w+)', content)
            macro_name = macro_match.group(1) if macro_match else 'data processing'
            return f"Create dbt macro for {macro_name} functionality"
        
        elif component_type == 'tests':
            return "Create dbt test for data quality validation"
        
        elif component_type == 'snapshots':
            return "Create dbt snapshot for slowly changing dimensions"
        
        else:
            return f"Create dbt {component_type} configuration"
    
    def _assess_dbt_complexity(self, content: str, component_type: str) -> str:
        """Assess dbt component complexity"""
        line_count = len(content.split('\n'))
        
        # Check for advanced dbt features
        advanced_features = [
            'ref(', 'var(', 'source(', 'macro', 'materialized',
            'pre_hook', 'post_hook', 'incremental', 'snapshot'
        ]
        
        advanced_count = sum(1 for feature in advanced_features if feature in content)
        
        if component_type == 'config':
            return 'medium'  # Config files are typically medium complexity
        elif line_count > 100 or advanced_count > 5:
            return 'complex'
        elif line_count > 50 or advanced_count > 2:
            return 'medium'
        else:
            return 'simple'

class StreamProcessingExtractor:
    """Extract stream processing examples (Kafka, Flink, etc.)"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.github_client = Github(self.github_token) if self.github_token else None
        
        # Stream processing patterns
        self.streaming_patterns = {
            'kafka': ['kafka*.py', '*kafka*.py', 'producers/*.py', 'consumers/*.py'],
            'flink': ['flink*.py', '*flink*.py', 'streaming/*.py'],
            'spark_streaming': ['spark_streaming*.py', 'streaming/*.py'],
            'kinesis': ['kinesis*.py', '*kinesis*.py']
        }
    
    def extract_streaming_examples(self, repo_name: str) -> List[Dict]:
        """Extract stream processing examples from repository"""
        if not self.github_client:
            return []
        
        try:
            repo = self.github_client.get_repo(repo_name)
            examples = []
            
            for platform, patterns in self.streaming_patterns.items():
                platform_examples = self._extract_streaming_platform(
                    repo, platform, patterns
                )
                examples.extend(platform_examples)
            
            logger.info(f"Extracted {len(examples)} streaming examples from {repo_name}")
            return examples
        
        except Exception as e:
            logger.error(f"Error extracting streaming from {repo_name}: {str(e)}")
            return []
    
    def _extract_streaming_platform(self, repo, platform: str, patterns: List[str]) -> List[Dict]:
        """Extract examples for specific streaming platform"""
        examples = []
        
        for pattern in patterns:
            try:
                files = self._find_streaming_files(repo, pattern)
                
                for file_path in files[:5]:  # Limit files per pattern
                    content = self._get_file_content(repo, file_path)
                    if content and self._is_streaming_content(content, platform):
                        example = self._create_streaming_example(
                            content, file_path, platform, repo.full_name
                        )
                        if example:
                            examples.append(example)
            
            except Exception as e:
                logger.debug(f"Could not process streaming pattern {pattern}: {str(e)}")
                continue
        
        return examples
    
    def _find_streaming_files(self, repo, pattern: str) -> List[str]:
        """Find streaming-related files"""
        try:
            contents = repo.get_contents("")
            matching_files = []
            
            for content in contents:
                if content.type == 'file' and content.name.endswith('.py'):
                    if '*' in pattern:
                        pattern_cleaned = pattern.replace('*', '')
                        if pattern_cleaned in content.name:
                            matching_files.append(content.path)
                    elif content.name == pattern:
                        matching_files.append(content.path)
            
            return matching_files
        except:
            return []
    
    def _get_file_content(self, repo, file_path: str) -> Optional[str]:
        """Get file content from repository"""
        try:
            file_content = repo.get_contents(file_path)
            if file_content.encoding == 'base64':
                import base64
                return base64.b64decode(file_content.content).decode('utf-8')
            else:
                return file_content.content
        except:
            return None
    
    def _is_streaming_content(self, content: str, platform: str) -> bool:
        """Check if content is valid streaming code"""
        platform_indicators = {
            'kafka': ['kafka', 'KafkaProducer', 'KafkaConsumer', 'produce', 'consume'],
            'flink': ['flink', 'StreamExecutionEnvironment', 'DataStream'],
            'spark_streaming': ['spark', 'StreamingContext', 'DStream', 'streaming'],
            'kinesis': ['kinesis', 'boto3', 'put_record', 'get_records']
        }
        
        indicators = platform_indicators.get(platform, [platform])
        
        return any(indicator.lower() in content.lower() for indicator in indicators)
    
    def _create_streaming_example(self, content: str, file_path: str,
                                 platform: str, repo_name: str) -> Optional[Dict]:
        """Create training example from streaming code"""
        if len(content.strip()) < 100:
            return None
        
        # Generate descriptive prompt
        prompt = self._generate_streaming_prompt(content, platform, file_path)
        
        # Assess complexity
        complexity = self._assess_streaming_complexity(content)
        
        return {
            'content': content,
            'prompt': prompt,
            'platform': platform,
            'file_path': file_path,
            'repo_name': repo_name,
            'complexity': complexity
        }
    
    def _generate_streaming_prompt(self, content: str, platform: str, file_path: str) -> str:
        """Generate descriptive prompt for streaming example"""
        if 'producer' in file_path.lower():
            return f"Create {platform} producer for real-time data streaming"
        elif 'consumer' in file_path.lower():
            return f"Create {platform} consumer for stream processing"
        elif 'transform' in content.lower() or 'process' in content.lower():
            return f"Create {platform} stream transformation pipeline"
        else:
            return f"Create {platform} streaming application for data processing"
    
    def _assess_streaming_complexity(self, content: str) -> str:
        """Assess streaming code complexity"""
        line_count = len(content.split('\n'))
        
        # Check for advanced streaming patterns
        advanced_patterns = [
            'windowing', 'watermark', 'checkpoint', 'state',
            'join', 'aggregate', 'tumbling', 'sliding'
        ]
        
        advanced_count = sum(1 for pattern in advanced_patterns 
                           if pattern.lower() in content.lower())
        
        if line_count > 200 or advanced_count > 3:
            return 'complex'
        elif line_count > 100 or advanced_count > 1:
            return 'medium'
        else:
            return 'simple'

class DataEngineeringCollector:
    """Main collector for data engineering examples"""
    
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_client = Github(self.github_token) if self.github_token else None
        
        self.airflow_extractor = AirflowDAGExtractor(self.github_token)
        self.dbt_extractor = DBTProjectExtractor(self.github_token)
        self.streaming_extractor = StreamProcessingExtractor(self.github_token)
        self.validation_pipeline = UniversalValidationPipeline()
        
        # Collection targets by category
        self.targets = {
            'etl_pipelines': 200,
            'data_modeling': 150,
            'analytics_reporting': 150
        }
        
        # Source configuration
        self.sources = self._configure_sources()
    
    def _configure_sources(self) -> Dict:
        """Configure data engineering sources for each category"""
        return {
            'etl_pipelines': {
                'airflow_repos': [
                    'apache/airflow',
                    'astronomer/astronomer-cosmos',
                    'apache/airflow-providers-amazon',
                    'datadog/airflow-guides'
                ],
                'general_etl_repos': [
                    'great-expectations/great_expectations',
                    'dagster-io/dagster',
                    'prefecthq/prefect'
                ]
            },
            'data_modeling': {
                'dbt_repos': [
                    'dbt-labs/dbt-core',
                    'dbt-labs/dbt-utils',
                    'fishtown-analytics/dbt-external-tables',
                    'calogica/dbt-expectations'
                ],
                'modeling_repos': [
                    'aws-samples/aws-data-lake-solution',
                    'GoogleCloudPlatform/professional-services'
                ]
            },
            'analytics_reporting': {
                'sql_repos': [
                    'Evidence-dev/evidence',
                    'sqlfluff/sqlfluff',
                    'tobymao/sqlglot'
                ],
                'streaming_repos': [
                    'apache/kafka',
                    'apache/flink',
                    'confluentinc/kafka-streams-examples'
                ]
            }
        }
    
    async def collect_all_categories(self) -> Dict[str, List[Dict]]:
        """Collect examples from all data engineering categories"""
        logger.info("Starting collection for all data engineering categories")
        
        results = {}
        for category in self.targets:
            logger.info(f"Collecting {category} examples (target: {self.targets[category]})")
            examples = await self.collect_category_examples(category)
            results[category] = examples
            
            logger.info(f"Collected {len(examples)} examples for {category}")
        
        return results
    
    async def collect_category_examples(self, category: str) -> List[Dict]:
        """Collect examples for a specific category"""
        config = self.sources[category]
        examples = []
        
        # Process repositories based on category
        if category == 'etl_pipelines':
            # Collect Airflow examples
            for repo_name in config.get('airflow_repos', []):
                logger.info(f"Processing Airflow repository: {repo_name}")
                try:
                    airflow_examples = self.airflow_extractor.extract_airflow_examples(repo_name)
                    for raw_example in airflow_examples:
                        training_example = self._create_etl_training_example(raw_example)
                        if training_example:
                            examples.append(training_example)
                    await asyncio.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error processing Airflow repo {repo_name}: {str(e)}")
                    continue
            
            # Collect general ETL examples
            for repo_name in config.get('general_etl_repos', []):
                logger.info(f"Processing ETL repository: {repo_name}")
                try:
                    etl_examples = await self._collect_general_etl_examples(repo_name)
                    examples.extend(etl_examples)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error processing ETL repo {repo_name}: {str(e)}")
                    continue
        
        elif category == 'data_modeling':
            # Collect dbt examples
            for repo_name in config.get('dbt_repos', []):
                logger.info(f"Processing dbt repository: {repo_name}")
                try:
                    dbt_examples = self.dbt_extractor.extract_dbt_examples(repo_name)
                    for raw_example in dbt_examples:
                        training_example = self._create_modeling_training_example(raw_example)
                        if training_example:
                            examples.append(training_example)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error processing dbt repo {repo_name}: {str(e)}")
                    continue
            
            # Collect general modeling examples
            for repo_name in config.get('modeling_repos', []):
                logger.info(f"Processing modeling repository: {repo_name}")
                try:
                    modeling_examples = await self._collect_modeling_examples(repo_name)
                    examples.extend(modeling_examples)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error processing modeling repo {repo_name}: {str(e)}")
                    continue
        
        elif category == 'analytics_reporting':
            # Collect streaming examples
            for repo_name in config.get('streaming_repos', []):
                logger.info(f"Processing streaming repository: {repo_name}")
                try:
                    streaming_examples = self.streaming_extractor.extract_streaming_examples(repo_name)
                    for raw_example in streaming_examples:
                        training_example = self._create_analytics_training_example(raw_example)
                        if training_example:
                            examples.append(training_example)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error processing streaming repo {repo_name}: {str(e)}")
                    continue
            
            # Collect SQL examples
            for repo_name in config.get('sql_repos', []):
                logger.info(f"Processing SQL repository: {repo_name}")
                try:
                    sql_examples = await self._collect_sql_examples(repo_name)
                    examples.extend(sql_examples)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error processing SQL repo {repo_name}: {str(e)}")
                    continue
        
        # Validate and filter examples
        validated_examples = self._validate_and_filter_examples(examples, category)
        
        # Ensure we meet target count
        if len(validated_examples) < self.targets[category]:
            logger.warning(f"Only collected {len(validated_examples)} examples for {category}, "
                         f"target was {self.targets[category]}")
        
        return validated_examples[:self.targets[category]]
    
    def _create_etl_training_example(self, raw_example: Dict) -> Optional[Dict]:
        """Create training example from ETL/pipeline code"""
        content = raw_example.get('content', '')
        if not content or len(content.strip()) < 100:
            return None
        
        prompt = f"""### Data Engineering Request (CLAUDE.md):
{raw_example.get('prompt', 'Create ETL pipeline for data processing')}

### PLAN Phase:
Analyze data flow requirements and design pipeline architecture.

### CODE Phase:"""
        
        return {
            'domain': 'data_engineering',
            'category': 'etl_pipelines',
            'prompt': prompt,
            'completion': content,
            'metadata': {
                'source': f"github.com/{raw_example.get('repo_name', 'unknown')}",
                'language': 'python',
                'complexity': raw_example.get('complexity', 'medium'),
                'component_type': raw_example.get('component_type', 'pipeline'),
                'file_path': raw_example.get('file_path', ''),
                'authentic': True
            }
        }
    
    def _create_modeling_training_example(self, raw_example: Dict) -> Optional[Dict]:
        """Create training example from data modeling code"""
        content = raw_example.get('content', '')
        if not content or len(content.strip()) < 50:
            return None
        
        prompt = f"""### Data Engineering Request (CLAUDE.md):
{raw_example.get('prompt', 'Create data model for analytics')}

### PLAN Phase:
Analyze data requirements and design model structure.

### CODE Phase:"""
        
        # Determine language based on content
        language = 'sql' if raw_example.get('file_path', '').endswith('.sql') else 'yaml'
        
        return {
            'domain': 'data_engineering',
            'category': 'data_modeling',
            'prompt': prompt,
            'completion': content,
            'metadata': {
                'source': f"github.com/{raw_example.get('repo_name', 'unknown')}",
                'language': language,
                'complexity': raw_example.get('complexity', 'medium'),
                'component_type': raw_example.get('component_type', 'model'),
                'file_path': raw_example.get('file_path', ''),
                'authentic': True
            }
        }
    
    def _create_analytics_training_example(self, raw_example: Dict) -> Optional[Dict]:
        """Create training example from analytics/streaming code"""
        content = raw_example.get('content', '')
        if not content or len(content.strip()) < 100:
            return None
        
        prompt = f"""### Data Engineering Request (CLAUDE.md):
{raw_example.get('prompt', 'Create analytics processing pipeline')}

### PLAN Phase:
Analyze streaming requirements and design processing architecture.

### CODE Phase:"""
        
        return {
            'domain': 'data_engineering',
            'category': 'analytics_reporting',
            'prompt': prompt,
            'completion': content,
            'metadata': {
                'source': f"github.com/{raw_example.get('repo_name', 'unknown')}",
                'language': 'python',
                'complexity': raw_example.get('complexity', 'medium'),
                'platform': raw_example.get('platform', 'streaming'),
                'file_path': raw_example.get('file_path', ''),
                'authentic': True
            }
        }
    
    async def _collect_general_etl_examples(self, repo_name: str) -> List[Dict]:
        """Collect general ETL examples from repository"""
        # Placeholder implementation - would need specific extraction logic
        return []
    
    async def _collect_modeling_examples(self, repo_name: str) -> List[Dict]:
        """Collect data modeling examples from repository"""
        # Placeholder implementation - would need specific extraction logic
        return []
    
    async def _collect_sql_examples(self, repo_name: str) -> List[Dict]:
        """Collect SQL examples from repository"""
        # Placeholder implementation - would need specific extraction logic
        return []
    
    def _validate_and_filter_examples(self, examples: List[Dict], category: str) -> List[Dict]:
        """Validate and filter examples using the universal pipeline"""
        logger.info(f"Validating {len(examples)} examples for {category}")
        
        validated_examples = []
        
        for example in examples:
            try:
                validation_result = self.validation_pipeline.validate_example(example)
                
                if validation_result['passes_quality_threshold']:
                    # Add validation metadata
                    example['validation'] = {
                        'overall_quality': validation_result['overall_quality'],
                        'syntax_valid': validation_result['syntax_valid'],
                        'completeness_score': validation_result['completeness_score'],
                        'authenticity_score': validation_result['authenticity_score']
                    }
                    validated_examples.append(example)
                else:
                    logger.debug(f"Example failed validation with score: {validation_result['overall_quality']:.3f}")
            
            except Exception as e:
                logger.warning(f"Validation failed for example: {str(e)}")
                continue
        
        pass_rate = len(validated_examples) / len(examples) if examples else 0
        logger.info(f"Validation complete. Pass rate: {pass_rate:.2%} ({len(validated_examples)}/{len(examples)})")
        
        return validated_examples
    
    def save_examples(self, examples: Dict[str, List[Dict]], output_path: str):
        """Save collected examples to JSONL format"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        total_examples = sum(len(category_examples) for category_examples in examples.values())
        logger.info(f"Saving {total_examples} examples to {output_path}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for category, category_examples in examples.items():
                for example in category_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Examples saved successfully to {output_path}")

async def main():
    """Main collection process"""
    collector = DataEngineeringCollector()
    
    # Collect all categories
    examples = await collector.collect_all_categories()
    
    # Save to file
    output_path = "/home/shingai/sort/deployments/data/corpus/data_engineering_corpus.jsonl"
    collector.save_examples(examples, output_path)
    
    # Print summary
    for category, category_examples in examples.items():
        print(f"{category}: {len(category_examples)} examples")
    
    total = sum(len(examples) for examples in examples.values())
    print(f"Total collected: {total} examples")

if __name__ == "__main__":
    asyncio.run(main())