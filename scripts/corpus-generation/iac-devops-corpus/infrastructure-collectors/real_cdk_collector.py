#!/usr/bin/env python3
"""
Real CDK Examples Collector - From ACTUAL repositories
Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING
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

class RealCDKCollector:
    """Collects CDK examples from REAL repositories only"""
    
    def __init__(self, output_dir: str = "cdk_real_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        
        # REAL CDK repositories provided by user
        self.target_repos = [
            'aws-samples/aws-cdk-examples',
            'ksmin23/my-aws-cdk-examples',
            'swoodford/aws',
            'thomvaill/tads-boilerplate',
            'nguyenanhung/infra-caddy-guy',
            'cloudposse/geodesic',
            'devopsgroup-io/catapult'
        ]

    def collect_real_cdk_examples(self) -> List[Dict]:
        """Collect real CDK examples from specified repositories"""
        logger.info("Starting REAL CDK collection from user-specified repositories")
        
        all_examples = []
        
        # Strategy 1: Direct file download from aws-samples/aws-cdk-examples
        direct_examples = self._collect_from_aws_cdk_examples()
        all_examples.extend(direct_examples)
        logger.info(f"Collected {len(direct_examples)} from aws-samples/aws-cdk-examples")
        
        # Strategy 2: Clone and extract from other repositories
        for repo in self.target_repos[1:]:  # Skip first one, already handled
            repo_examples = self._collect_from_repository(repo)
            all_examples.extend(repo_examples)
            logger.info(f"Collected {len(repo_examples)} from {repo}")
        
        # Deduplicate
        unique_examples = self._deduplicate_examples(all_examples)
        
        logger.info(f"Total unique real CDK examples: {len(unique_examples)}")
        return unique_examples

    def _collect_from_aws_cdk_examples(self) -> List[Dict]:
        """Collect directly from aws-samples/aws-cdk-examples known paths"""
        examples = []
        
        # Known good CDK example paths
        known_paths = [
            ('typescript/lambda-cron/index.ts', 'typescript'),
            ('typescript/s3-trigger-lambda/index.ts', 'typescript'),
            ('typescript/api-cors-lambda-crud-dynamodb/index.ts', 'typescript'),
            ('python/lambda-cron/app.py', 'python'),
            ('python/s3-trigger-lambda/app.py', 'python'),
            ('python/api-cors-lambda-crud-dynamodb/app.py', 'python'),
            ('typescript/eks/cluster-graviton/index.ts', 'typescript'),
            ('typescript/fargate-alb-service/index.ts', 'typescript'),
            ('python/eks/cluster/app.py', 'python'),
            ('typescript/rds/oracle/oracle.ts', 'typescript'),
        ]
        
        for path, language in known_paths:
            try:
                url = f"https://raw.githubusercontent.com/aws-samples/aws-cdk-examples/master/{path}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    content = response.text
                    example = self._create_example_from_content(content, path, language, 'aws-samples/aws-cdk-examples')
                    if example:
                        examples.append(example)
                        
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"Failed to fetch {path}: {e}")
        
        return examples

    def _collect_from_repository(self, repo: str) -> List[Dict]:
        """Clone and extract CDK examples from a repository"""
        examples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                logger.info(f"Cloning {repo}...")
                clone_path = os.path.join(temp_dir, repo.replace('/', '_'))
                
                # Clone with depth 1
                result = subprocess.run([
                    'git', 'clone', '--depth', '1',
                    f'https://github.com/{repo}.git', clone_path
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    # Find CDK files
                    cdk_files = self._find_cdk_files(clone_path)
                    
                    for file_path in cdk_files[:5]:  # Limit per repo
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            
                            if self._is_cdk_content(content):
                                language = self._detect_language(file_path)
                                example = self._create_example_from_content(
                                    content, file_path, language, repo
                                )
                                if example:
                                    examples.append(example)
                        except:
                            pass
                    
            except Exception as e:
                logger.debug(f"Failed to process {repo}: {e}")
        
        return examples

    def _find_cdk_files(self, path: str) -> List[str]:
        """Find potential CDK files in a directory"""
        cdk_files = []
        
        try:
            # Look for TypeScript/JavaScript CDK files
            result = subprocess.run([
                'find', path, '-type', 'f',
                '(', '-name', '*.ts', '-o', '-name', '*.js', ')',
                '-not', '-path', '*/node_modules/*',
                '-not', '-path', '*/.git/*',
                '-exec', 'grep', '-l', 'aws-cdk', '{}', ';'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                cdk_files.extend(result.stdout.strip().split('\n'))
            
            # Look for Python CDK files
            result = subprocess.run([
                'find', path, '-type', 'f',
                '-name', '*.py',
                '-not', '-path', '*/venv/*',
                '-not', '-path', '*/.git/*',
                '-exec', 'grep', '-l', 'aws_cdk', '{}', ';'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                cdk_files.extend(result.stdout.strip().split('\n'))
            
        except Exception as e:
            logger.debug(f"Error finding CDK files: {e}")
        
        return [f for f in cdk_files if f]

    def _is_cdk_content(self, content: str) -> bool:
        """Check if content is actual CDK code"""
        # Must have CDK imports
        cdk_indicators = [
            'aws-cdk-lib', 'aws_cdk', '@aws-cdk', 'aws-cdk-core',
            'from constructs import', 'from aws_cdk import'
        ]
        
        has_cdk = any(indicator in content for indicator in cdk_indicators)
        
        # Must have actual constructs
        construct_indicators = [
            'new ', 'Stack(', 'Construct(', 'Bucket(', 'Function(',
            'Table(', 'Vpc(', 'Queue(', 'Topic('
        ]
        
        has_constructs = any(indicator in content for indicator in construct_indicators)
        
        # Must be reasonable size
        lines = content.split('\n')
        
        return has_cdk and has_constructs and 10 < len(lines) < 500

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension"""
        if file_path.endswith('.ts'):
            return 'typescript'
        elif file_path.endswith('.js'):
            return 'javascript'
        elif file_path.endswith('.py'):
            return 'python'
        else:
            return 'typescript'

    def _create_example_from_content(self, content: str, file_path: str, language: str, repo: str) -> Optional[Dict]:
        """Create training example from CDK content"""
        try:
            # Clean and validate
            clean_content = self._clean_content(content)
            
            if not clean_content or len(clean_content) < 200:
                return None
            
            # Check duplicate
            content_hash = hashlib.md5(clean_content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            
            self.collected_hashes.add(content_hash)
            
            # Generate prompt
            prompt = self._generate_prompt(clean_content, language)
            
            return {
                "prompt": prompt,
                "completion": f"```{language}\n{clean_content}\n```",
                "metadata": {
                    "source": "github_repository",
                    "repository": repo,
                    "file_path": file_path,
                    "language": language,
                    "category": self._categorize_content(clean_content),
                    "authentic": True
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating example: {e}")
            return None

    def _clean_content(self, content: str) -> str:
        """Clean content while preserving authenticity"""
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            # Skip very long lines
            if len(line) < 200:
                # Sanitize obvious secrets
                line = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', line)
                line = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', line)
                clean_lines.append(line.rstrip())
        
        return '\n'.join(clean_lines).strip()

    def _generate_prompt(self, content: str, language: str) -> str:
        """Generate appropriate prompt based on content"""
        content_lower = content.lower()
        
        if 'lambda' in content_lower or 'function' in content_lower:
            return f"Create an AWS CDK {language} application with Lambda functions"
        elif 's3' in content_lower or 'bucket' in content_lower:
            return f"Write an AWS CDK {language} application for S3 storage"
        elif 'dynamodb' in content_lower or 'table' in content_lower:
            return f"Create an AWS CDK {language} application with DynamoDB"
        elif 'vpc' in content_lower:
            return f"Write an AWS CDK {language} application for VPC networking"
        elif 'fargate' in content_lower or 'ecs' in content_lower:
            return f"Create an AWS CDK {language} application for container orchestration"
        elif 'api' in content_lower:
            return f"Write an AWS CDK {language} application with API Gateway"
        else:
            return f"Create an AWS CDK {language} infrastructure application"

    def _categorize_content(self, content: str) -> str:
        """Categorize based on AWS services used"""
        content_lower = content.lower()
        
        if 'lambda' in content_lower:
            return 'serverless'
        elif 's3' in content_lower:
            return 'storage'
        elif 'dynamodb' in content_lower or 'rds' in content_lower:
            return 'database'
        elif 'vpc' in content_lower:
            return 'networking'
        elif 'fargate' in content_lower or 'ecs' in content_lower:
            return 'containers'
        elif 'api' in content_lower:
            return 'api'
        else:
            return 'infrastructure'

    def _deduplicate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicates"""
        seen = set()
        unique = []
        
        for example in examples:
            content_hash = hashlib.md5(example['completion'].encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(example)
        
        return unique

    def save_corpus(self, examples: List[Dict]) -> None:
        """Save CDK corpus"""
        output_file = self.output_dir / "cdk_real_corpus.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} real CDK examples to {output_file}")


def main():
    """Main collection pipeline"""
    logger.info("Starting REAL CDK Examples Collection")
    logger.info("Using repositories specified by user")
    
    collector = RealCDKCollector()
    
    try:
        # Collect real examples
        examples = collector.collect_real_cdk_examples()
        
        # Save corpus
        collector.save_corpus(examples)
        
        # Statistics
        languages = {}
        categories = {}
        repos = {}
        
        for example in examples:
            metadata = example.get('metadata', {})
            language = metadata.get('language', 'unknown')
            category = metadata.get('category', 'unknown')
            repo = metadata.get('repository', 'unknown')
            
            languages[language] = languages.get(language, 0) + 1
            categories[category] = categories.get(category, 0) + 1
            repos[repo] = repos.get(repo, 0) + 1
        
        logger.info(f"Real CDK Collection Statistics:")
        logger.info(f"Total authentic examples: {len(examples)}")
        logger.info(f"Languages: {dict(sorted(languages.items()))}")
        logger.info(f"Categories: {dict(sorted(categories.items()))}")
        logger.info(f"Repositories: {dict(sorted(repos.items()))}")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()