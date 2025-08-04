#!/usr/bin/env python3
"""
Real-World AWS CLI Examples Collector
Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING

Collects exclusively from authentic sources:
- Production AWS CLI scripts and workflows
- Official AWS documentation examples
- Open source DevOps projects
- Stack Overflow CLI snippets
- Real infrastructure automation scripts
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

class AWSCLIRealCollector:
    """Collects AWS CLI examples exclusively from real-world sources"""
    
    def __init__(self, output_dir: str = "aws_cli_real_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        
        # Target repositories with quality AWS CLI examples
        self.target_repos = [
            # AWS official examples and tools
            'aws/aws-cli',
            'awslabs/aws-shell',
            'awslabs/aws-sam-cli',
            'aws/aws-cdk',
            'aws/copilot-cli',
            
            # AWS infrastructure automation
            'awslabs/ec2-spot-labs',
            'awslabs/ecs-refarch-cloudformation',
            'aws-samples/aws-batch-architecture-for-protein-folding',
            'aws-samples/aws-lambda-developer-guide',
            'aws-samples/amazon-eks-example-for-stateful-java-service',
            
            # DevOps and automation projects
            'ansible/ansible',
            'hashicorp/terraform-provider-aws',
            'gruntwork-io/terragrunt',
            'cloudposse/terraform-aws-components',
            
            # Infrastructure monitoring and management
            'Netflix/security_monkey',
            'prowler-cloud/prowler',
            'bridgecrewio/checkov',
            
            # Kubernetes + AWS
            'kubernetes-sigs/aws-load-balancer-controller',
            'aws/amazon-vpc-cni-k8s',
            'eksctl-io/eksctl',
            
            # Serverless frameworks
            'serverless/serverless',
            'Miserlou/Zappa',
        ]
        
        # Local repositories to explore
        self.local_sources = [
            '/home/shingai/api',
            '/home/shingai/sort/deployments',
            '/home/shingai/sort/ona-front-end'
        ]

    def collect_real_aws_cli_examples(self, target_count: int = 40) -> List[Dict]:
        """Collect real-world AWS CLI examples"""
        logger.info(f"Starting real-world AWS CLI collection targeting {target_count} examples")
        
        all_examples = []
        
        # Strategy 1: Local repository mining for AWS CLI usage
        local_examples = self._collect_from_local_repos()
        all_examples.extend(local_examples)
        logger.info(f"Collected {len(local_examples)} from local repositories")
        
        # Strategy 2: Clone AWS and DevOps repositories
        repo_examples = self._collect_from_aws_repos()
        all_examples.extend(repo_examples)
        logger.info(f"Collected {len(repo_examples)} from AWS repositories")
        
        # Strategy 3: GitHub raw file collection for known AWS CLI scripts
        github_examples = self._collect_aws_cli_raw_files()
        all_examples.extend(github_examples)
        logger.info(f"Collected {len(github_examples)} from GitHub raw files")
        
        # Strategy 4: Stack Overflow AWS CLI snippets
        stackoverflow_examples = self._collect_stackoverflow_aws_cli()
        all_examples.extend(stackoverflow_examples)
        logger.info(f"Collected {len(stackoverflow_examples)} from Stack Overflow")
        
        # Deduplicate and filter quality
        unique_examples = self._deduplicate_examples(all_examples)
        quality_examples = self._filter_aws_cli_quality(unique_examples)
        
        logger.info(f"Final real-world AWS CLI collection: {len(quality_examples)} examples")
        return quality_examples[:target_count]

    def _collect_from_local_repos(self) -> List[Dict]:
        """Collect AWS CLI usage from local repositories"""
        examples = []
        
        for repo_path in self.local_sources:
            if os.path.exists(repo_path):
                logger.info(f"Scanning local repository for AWS CLI: {repo_path}")
                repo_examples = self._extract_aws_cli_from_path(repo_path)
                examples.extend(repo_examples)
        
        return examples

    def _extract_aws_cli_from_path(self, path: str) -> List[Dict]:
        """Extract AWS CLI usage from shell scripts in a given path"""
        examples = []
        
        try:
            # Find shell scripts that likely contain AWS CLI commands
            result = subprocess.run([
                'find', path, '-type', 'f',
                '(', '-name', '*.sh', '-o', '-name', '*.bash', ')',
                '-not', '-path', '*/node_modules/*',
                '-not', '-path', '*/venv/*',
                '-not', '-path', '*/.git/*',
                '-not', '-path', '*/build/*',
                '-not', '-path', '*/dist/*'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                script_files = result.stdout.strip().split('\n')
                
                for script_file in script_files:
                    if script_file:  # Skip empty lines
                        examples.extend(self._extract_aws_cli_from_script(script_file))
            
        except Exception as e:
            logger.debug(f"Error extracting AWS CLI from {path}: {e}")
        
        return examples

    def _extract_aws_cli_from_script(self, file_path: str) -> List[Dict]:
        """Extract AWS CLI commands from a shell script"""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract AWS CLI command blocks
            aws_cli_blocks = self._find_aws_cli_blocks(content, file_path)
            
            for block in aws_cli_blocks:
                example = self._create_aws_cli_example(block, file_path)
                if example:
                    examples.append(example)
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
        
        return examples

    def _find_aws_cli_blocks(self, content: str, file_path: str) -> List[str]:
        """Find blocks of code containing AWS CLI commands"""
        blocks = []
        lines = content.split('\n')
        
        # Find AWS CLI command patterns
        aws_cli_patterns = [
            r'aws\s+\w+\s+\w+',  # Basic aws command pattern
            r'aws\s+ec2\s+',
            r'aws\s+s3\s+',
            r'aws\s+lambda\s+',
            r'aws\s+iam\s+',
            r'aws\s+cloudformation\s+',
            r'aws\s+eks\s+',
            r'aws\s+rds\s+',
            r'aws\s+sts\s+',
            r'aws\s+logs\s+',
            r'aws\s+ecr\s+',
            r'aws\s+elbv2\s+',
            r'aws\s+route53\s+',
            r'aws\s+apigateway\s+',
            r'aws\s+events\s+',
            r'aws\s+sns\s+',
            r'aws\s+sqs\s+',
        ]
        
        current_block = []
        in_aws_block = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if line contains AWS CLI command
            has_aws_cli = any(re.search(pattern, line_lower) for pattern in aws_cli_patterns)
            
            if has_aws_cli:
                if not in_aws_block:
                    # Start new block, include some context before
                    start_idx = max(0, i - 2)
                    current_block = lines[start_idx:i]
                    in_aws_block = True
                current_block.append(line)
            elif in_aws_block:
                # Continue block if line seems related (continuation, variable, etc.)
                if (line_lower.startswith('#') or 
                    line_lower.startswith('echo') or
                    line_lower.startswith('if') or
                    line_lower.startswith('then') or
                    line_lower.startswith('fi') or
                    line_lower.startswith('done') or
                    line_lower.endswith('\\\\') or
                    '=' in line or
                    line.strip() == ''):
                    current_block.append(line)
                else:
                    # End of block
                    if len(current_block) >= 3:  # Minimum block size
                        blocks.append('\\n'.join(current_block))
                    current_block = []
                    in_aws_block = False
        
        # Handle final block
        if in_aws_block and len(current_block) >= 3:
            blocks.append('\\n'.join(current_block))
        
        return blocks

    def _create_aws_cli_example(self, block: str, file_path: str) -> Optional[Dict]:
        """Create training example from AWS CLI block"""
        try:
            # Quality checks
            if not self._is_quality_aws_cli_block(block):
                return None
            
            # Check for duplicates
            content_hash = hashlib.md5(block.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            
            self.collected_hashes.add(content_hash)
            
            # Generate training example
            prompt = self._generate_aws_cli_prompt(block, file_path)
            clean_content = self._clean_aws_cli_content(block)
            
            return {
                "prompt": prompt,
                "completion": f"```bash\\n{clean_content}\\n```",
                "metadata": {
                    "source": "real_local_repository",
                    "file_path": file_path,
                    "category": self._categorize_aws_cli_block(block),
                    "authentic": True
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating AWS CLI example: {e}")
            return None

    def _is_quality_aws_cli_block(self, block: str) -> bool:
        """Check if AWS CLI block is high quality"""
        lines = block.split('\\n')
        line_count = len(lines)
        
        # Size checks
        if line_count < 3 or line_count > 50:
            return False
        
        if len(block) < 50 or len(block) > 5000:
            return False
        
        # Must contain actual AWS CLI commands
        aws_command_count = len([line for line in lines if 'aws ' in line.lower() and not line.strip().startswith('#')])
        if aws_command_count < 1:
            return False
        
        # Avoid obviously generated or example blocks
        bad_patterns = [
            'this is an example',
            'replace with your',
            'example configuration',
            'sample command',
            'template script',
            'placeholder',
            'your-bucket-name',
            'your-instance-id',
            'example-value'
        ]
        
        block_lower = block.lower()
        for pattern in bad_patterns:
            if pattern in block_lower:
                return False
        
        return True

    def _collect_from_aws_repos(self) -> List[Dict]:
        """Clone and extract AWS CLI usage from repositories"""
        examples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo in self.target_repos[:6]:  # Limit to avoid timeout
                try:
                    logger.info(f"Cloning {repo} for AWS CLI examples...")
                    clone_path = os.path.join(temp_dir, repo.replace('/', '_'))
                    
                    # Clone with depth 1 for speed
                    result = subprocess.run([
                        'git', 'clone', '--depth', '1', '--filter=blob:limit=50k',
                        f'https://github.com/{repo}.git', clone_path
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        repo_examples = self._extract_aws_cli_from_path(clone_path)
                        examples.extend(repo_examples[:3])  # Limit per repo
                        logger.info(f"Extracted {len(repo_examples)} AWS CLI examples from {repo}")
                    
                except Exception as e:
                    logger.debug(f"Failed to process {repo}: {e}")
        
        return examples

    def _collect_aws_cli_raw_files(self) -> List[Dict]:
        """Collect AWS CLI scripts from GitHub using raw file URLs"""
        examples = []
        
        # Well-known AWS CLI script paths
        known_aws_cli_paths = [
            # AWS CLI official examples
            ('aws/aws-cli', 'awscli/examples/ec2/run-instances.rst'),
            ('aws/aws-cli', 'awscli/examples/s3/cp.rst'),
            
            # AWS samples
            ('aws-samples/aws-lambda-developer-guide', 'sample-apps/blank/deploy.sh'),
            ('awslabs/ec2-spot-labs', 'ec2-spot-fleet-web-app/deploy.sh'),
            
            # EKS examples
            ('eksctl-io/eksctl', 'examples/01-simple-cluster.yaml'),
            ('aws/amazon-vpc-cni-k8s', 'scripts/install-cni.sh'),
            
            # DevOps automation
            ('gruntwork-io/terragrunt', 'test/fixture-hooks/before-and-after-hooks/terraform-aws-get-caller-identity/terragrunt.hcl'),
        ]
        
        for repo, script_path in known_aws_cli_paths:
            try:
                example = self._fetch_aws_cli_raw_file(repo, script_path)
                if example:
                    examples.append(example)
                    
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.debug(f"Failed to fetch {repo}/{script_path}: {e}")
        
        return examples

    def _fetch_aws_cli_raw_file(self, repo: str, script_path: str) -> Optional[Dict]:
        """Fetch an AWS CLI script from GitHub raw URL"""
        try:
            # Try main branch, then master
            for branch in ['main', 'master']:
                url = f"https://raw.githubusercontent.com/{repo}/{branch}/{script_path}"
                
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.text
                    
                    # Extract AWS CLI blocks from content
                    aws_cli_blocks = self._find_aws_cli_blocks(content, script_path)
                    
                    if aws_cli_blocks:
                        # Use the best block
                        best_block = max(aws_cli_blocks, key=len)
                        
                        if self._is_quality_aws_cli_block(best_block):
                            # Check for duplicates
                            content_hash = hashlib.md5(best_block.encode()).hexdigest()
                            if content_hash not in self.collected_hashes:
                                self.collected_hashes.add(content_hash)
                                
                                # Generate training example
                                prompt = self._generate_aws_cli_prompt(best_block, script_path)
                                clean_content = self._clean_aws_cli_content(best_block)
                                
                                return {
                                    "prompt": prompt,
                                    "completion": f"```bash\\n{clean_content}\\n```",
                                    "metadata": {
                                        "source": "github_raw",
                                        "repository": repo,
                                        "file_path": script_path,
                                        "category": self._categorize_aws_cli_block(best_block),
                                        "authentic": True
                                    }
                                }
                    
        except Exception as e:
            logger.debug(f"Error fetching {repo}/{script_path}: {e}")
        
        return None

    def _collect_stackoverflow_aws_cli(self) -> List[Dict]:
        """Collect AWS CLI snippets from Stack Overflow"""
        examples = []
        
        # Search for high-quality AWS CLI questions
        search_queries = [
            'aws cli ec2 instance launch',
            'aws cli s3 bucket policy',
            'aws cli lambda function deploy',
            'aws cli iam role policy',
            'aws cli cloudformation stack',
            'aws cli eks cluster create',
            'aws cli rds database setup',
            'aws cli ecr docker push',
            'aws cli vpc subnet configuration',
            'aws cli route53 dns record'
        ]
        
        for query in search_queries[:5]:  # Limit queries
            try:
                query_examples = self._search_stackoverflow_aws_cli(query)
                examples.extend(query_examples)
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"Stack Overflow search failed for '{query}': {e}")
        
        return examples

    def _search_stackoverflow_aws_cli(self, query: str) -> List[Dict]:
        """Search Stack Overflow for AWS CLI examples"""
        examples = []
        
        try:
            # Use Stack Exchange API
            api_url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                'order': 'desc',
                'sort': 'votes',
                'q': query,
                'accepted': 'True',
                'site': 'stackoverflow',
                'pagesize': 3,
                'filter': 'withbody'
            }
            
            response = requests.get(api_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('items', []):
                    # Extract AWS CLI code blocks from answers
                    if 'body' in item:
                        code_examples = self._extract_aws_cli_from_stackoverflow(item['body'])
                        for code in code_examples:
                            example = self._create_stackoverflow_aws_cli_example(code, item, query)
                            if example:
                                examples.append(example)
                                
        except Exception as e:
            logger.debug(f"Stack Overflow API error: {e}")
        
        return examples

    def _extract_aws_cli_from_stackoverflow(self, html_body: str) -> List[str]:
        """Extract AWS CLI code blocks from Stack Overflow HTML"""
        code_blocks = []
        
        # Simple regex to extract code blocks (between <pre><code> tags)
        code_pattern = r'<pre[^>]*><code[^>]*>(.*?)</code></pre>'
        matches = re.findall(code_pattern, html_body, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            # Clean HTML entities and tags
            clean_code = re.sub(r'<[^>]+>', '', match)
            clean_code = clean_code.replace('&lt;', '<').replace('&gt;', '>')
            clean_code = clean_code.replace('&amp;', '&').replace('&quot;', '"')
            
            # Check if it contains AWS CLI commands
            if 'aws ' in clean_code.lower() and self._is_quality_aws_cli_block(clean_code):
                code_blocks.append(clean_code.strip())
        
        return code_blocks

    def _create_stackoverflow_aws_cli_example(self, code: str, item: Dict, query: str) -> Optional[Dict]:
        """Create training example from Stack Overflow AWS CLI code"""
        try:
            # Check for duplicates
            content_hash = hashlib.md5(code.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            
            self.collected_hashes.add(content_hash)
            
            # Generate prompt based on question context
            title = item.get('title', '')
            prompt = self._generate_stackoverflow_aws_cli_prompt(title, query)
            
            clean_content = self._clean_aws_cli_content(code)
            
            return {
                "prompt": prompt,
                "completion": f"```bash\\n{clean_content}\\n```",
                "metadata": {
                    "source": "stackoverflow",
                    "question_id": item.get('question_id', ''),
                    "title": title,
                    "score": item.get('score', 0),
                    "category": self._categorize_aws_cli_block(code),
                    "authentic": True
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating Stack Overflow AWS CLI example: {e}")
            return None

    def _generate_aws_cli_prompt(self, block: str, file_path: str) -> str:
        """Generate contextual prompt from AWS CLI analysis"""
        block_lower = block.lower()
        
        # Analyze AWS services used
        if 'aws ec2' in block_lower:
            if 'run-instances' in block_lower:
                return "Write AWS CLI commands to launch EC2 instances"
            elif 'describe-instances' in block_lower:
                return "Create AWS CLI commands to describe EC2 instances"
            else:
                return "Write AWS CLI commands for EC2 management"
        elif 'aws s3' in block_lower:
            if 'cp' in block_lower or 'sync' in block_lower:
                return "Create AWS CLI commands for S3 file operations"
            elif 'mb' in block_lower:
                return "Write AWS CLI commands to create S3 buckets"
            else:
                return "Write AWS CLI commands for S3 management"
        elif 'aws lambda' in block_lower:
            if 'create-function' in block_lower:
                return "Create AWS CLI commands to deploy Lambda functions"
            elif 'invoke' in block_lower:
                return "Write AWS CLI commands to invoke Lambda functions"
            else:
                return "Write AWS CLI commands for Lambda management"
        elif 'aws iam' in block_lower:
            if 'create-role' in block_lower:
                return "Create AWS CLI commands for IAM role management"
            elif 'attach-policy' in block_lower:
                return "Write AWS CLI commands for IAM policy management"
            else:
                return "Write AWS CLI commands for IAM operations"
        elif 'aws eks' in block_lower:
            return "Create AWS CLI commands for EKS cluster management"
        elif 'aws rds' in block_lower:
            return "Write AWS CLI commands for RDS database management"
        elif 'aws cloudformation' in block_lower:
            return "Create AWS CLI commands for CloudFormation stack management"
        elif 'aws ecr' in block_lower:
            return "Write AWS CLI commands for ECR container registry operations"
        elif 'aws sts' in block_lower:
            return "Create AWS CLI commands for STS token operations"
        else:
            return "Write AWS CLI commands for infrastructure management"

    def _generate_stackoverflow_aws_cli_prompt(self, title: str, query: str) -> str:
        """Generate prompt from Stack Overflow question title"""
        title_lower = title.lower()
        
        if 'ec2' in title_lower:
            return "Write AWS CLI commands for EC2 instance management"
        elif 's3' in title_lower:
            return "Create AWS CLI commands for S3 operations"
        elif 'lambda' in title_lower:
            return "Write AWS CLI commands for Lambda function management"
        elif 'iam' in title_lower:
            return "Create AWS CLI commands for IAM operations"
        elif 'eks' in title_lower:
            return "Write AWS CLI commands for EKS cluster operations"
        elif 'rds' in title_lower:
            return "Create AWS CLI commands for RDS database operations"
        elif 'cloudformation' in title_lower:
            return "Write AWS CLI commands for CloudFormation operations"
        else:
            return "Write AWS CLI commands for AWS infrastructure management"

    def _categorize_aws_cli_block(self, block: str) -> str:
        """Categorize AWS CLI block based on services used"""
        block_lower = block.lower()
        
        if 'aws ec2' in block_lower:
            return 'compute'
        elif 'aws s3' in block_lower:
            return 'storage'
        elif 'aws lambda' in block_lower:
            return 'serverless'
        elif 'aws iam' in block_lower:
            return 'security'
        elif 'aws eks' in block_lower or 'aws ecs' in block_lower:
            return 'orchestration'
        elif 'aws rds' in block_lower:
            return 'database'
        elif 'aws cloudformation' in block_lower:
            return 'infrastructure'
        elif 'aws ecr' in block_lower:
            return 'containerization'
        elif 'aws vpc' in block_lower or 'aws route53' in block_lower:
            return 'networking'
        elif 'aws logs' in block_lower or 'aws cloudwatch' in block_lower:
            return 'monitoring'
        elif 'aws sts' in block_lower:
            return 'authentication'
        else:
            return 'aws_operations'

    def _clean_aws_cli_content(self, content: str) -> str:
        """Clean AWS CLI content while preserving authenticity"""
        # Basic cleaning - preserve real-world nature
        lines = content.split('\\n')
        clean_lines = []
        
        for line in lines:
            # Remove extremely long lines
            if len(line) < 300:
                clean_lines.append(line.rstrip())
        
        content = '\\n'.join(clean_lines)
        
        # Only sanitize obvious sensitive data
        content = re.sub(r'\\b\\d{12}\\b', '<ACCOUNT_ID>', content)
        content = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}', '<EMAIL>', content)
        content = re.sub(r'\\b[a-f0-9]{32,64}\\b', '<TOKEN>', content)
        content = re.sub(r'\\bi-[0-9a-f]{8,17}\\b', '<INSTANCE_ID>', content)
        content = re.sub(r'(--secret-access-key|--session-token)\\s+\\S+', r'\\1 <REDACTED>', content)
        
        return content.strip()

    def _deduplicate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicate examples"""
        seen_hashes = set()
        unique_examples = []
        
        for example in examples:
            content = example['completion']
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_examples.append(example)
        
        return unique_examples

    def _filter_aws_cli_quality(self, examples: List[Dict]) -> List[Dict]:
        """Filter for highest quality AWS CLI examples"""
        quality_examples = []
        
        for example in examples:
            content = example['completion']
            
            # Extract CLI content from markdown
            cli_content = content.replace('```bash\\n', '').replace('\\n```', '')
            
            # Quality checks
            if self._is_quality_aws_cli_block(cli_content):
                quality_examples.append(example)
        
        return quality_examples

    def save_corpus(self, examples: List[Dict], filename: str = "aws_cli_real_corpus.jsonl") -> None:
        """Save real-world AWS CLI corpus"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\\n')
        
        logger.info(f"Saved {len(examples)} real-world AWS CLI examples to {output_file}")


def main():
    """Main AWS CLI real-world collection pipeline"""
    logger.info("Starting Real-World AWS CLI Examples Collection")
    logger.info("Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA")
    
    collector = AWSCLIRealCollector()
    
    try:
        # Collect real-world AWS CLI examples
        examples = collector.collect_real_aws_cli_examples(target_count=40)
        
        # Save corpus
        collector.save_corpus(examples)
        
        # Generate statistics
        sources = {}
        categories = {}
        services = {}
        
        for example in examples:
            metadata = example.get('metadata', {})
            source = metadata.get('source', 'unknown')
            category = metadata.get('category', 'unknown')
            
            # Detect AWS service from content
            content = example['completion'].lower()
            if 'aws ec2' in content:
                service = 'ec2'
            elif 'aws s3' in content:
                service = 's3'
            elif 'aws lambda' in content:
                service = 'lambda'
            elif 'aws iam' in content:
                service = 'iam'
            elif 'aws eks' in content:
                service = 'eks'
            elif 'aws rds' in content:
                service = 'rds'
            elif 'aws cloudformation' in content:
                service = 'cloudformation'
            else:
                service = 'multi_service'
            
            sources[source] = sources.get(source, 0) + 1
            categories[category] = categories.get(category, 0) + 1
            services[service] = services.get(service, 0) + 1
        
        logger.info(f"Real-World AWS CLI Collection Statistics:")
        logger.info(f"Total authentic examples: {len(examples)}")
        logger.info(f"Sources: {dict(sorted(sources.items()))}")
        logger.info(f"Categories: {dict(sorted(categories.items()))}")
        logger.info(f"AWS Services: {dict(sorted(services.items()))}")
        
        logger.info("Real-world AWS CLI collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Real-world AWS CLI collection failed: {e}")
        raise


if __name__ == "__main__":
    main()