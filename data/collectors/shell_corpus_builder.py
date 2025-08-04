#!/usr/bin/env python3
"""
Shell Script Corpus Builder for IaC Training Dataset
Collects, analyzes, and formats shell scripts for Mistral-7B fine-tuning

Follows CLAUDE.md principles and implements iac_model_tuning_pipeline.md methodology
"""

import os
import re
import json
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ShellScript:
    """Represents a shell script with metadata"""
    file_path: str
    content: str
    hash: str
    size_bytes: int
    line_count: int
    category: str
    aws_commands: List[str]
    patterns: List[str]
    has_errors: bool
    shellcheck_score: int

class ShellCorpusBuilder:
    """Builds training corpus from shell scripts following IaC methodology"""
    
    def __init__(self, output_dir: str = "shell_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Tracking
        self.processed_files = 0
        self.skipped_files = 0
        self.duplicate_hashes: Set[str] = set()
        self.scripts: List[ShellScript] = []
        
        # Categories for IaC training
        self.categories = {
            'deployment': ['deploy', 'release', 'build', 'push'],
            'monitoring': ['monitor', 'alarm', 'alert', 'cloudwatch'],
            'orchestration': ['orchestrat', 'pipeline', 'workflow'],
            'infrastructure': ['infra', 'provision', 'setup', 'create'],
            'aws_operations': ['aws', 'cloudformation', 'lambda', 's3', 'ec2'],
            'data_processing': ['etl', 'process', 'transform', 'load'],
            'security': ['auth', 'security', 'permission', 'role', 'policy']
        }
        
        # AWS CLI patterns for training focus
        self.aws_patterns = [
            r'aws\s+(\w+)\s+([a-z-]+)',
            r'--region\s+(\S+)',
            r'--query\s+["\']([^"\']+)["\']',
            r'--output\s+(text|json|table)',
            r'--stack-name\s+(\S+)',
            r'--function-name\s+(\S+)'
        ]
        
        # Destructive operations to flag
        self.destructive_patterns = [
            r'rm\s+-rf',
            r'aws\s+\w+\s+delete',
            r'aws\s+\w+\s+terminate',
            r'aws\s+\w+\s+destroy',
            r'shutdown\s+-h',
            r'mkfs',
            r'dd\s+.*=/dev/',
            r':(){ :|:& };:'
        ]

    def scan_repositories(self, repo_paths: List[str]) -> None:
        """Scan multiple repositories for shell scripts"""
        logger.info(f"Scanning {len(repo_paths)} repositories for shell scripts")
        
        for repo_path in repo_paths:
            logger.info(f"Processing repository: {repo_path}")
            self._scan_directory(repo_path)
        
        logger.info(f"Scan complete: {self.processed_files} processed, {self.skipped_files} skipped")

    def _scan_directory(self, directory: str) -> None:
        """Recursively scan directory for shell scripts"""
        repo_path = Path(directory)
        
        if not repo_path.exists():
            logger.warning(f"Repository path does not exist: {directory}")
            return
        
        # Find shell scripts
        shell_files = []
        
        # .sh files
        shell_files.extend(repo_path.rglob("*.sh"))
        
        # Files with shell shebang
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and self._has_shell_shebang(file_path):
                shell_files.append(file_path)
        
        # Filter out unwanted directories
        filtered_files = []
        exclude_dirs = {'node_modules', 'venv', '.git', '__pycache__', 'dist', 'build'}
        
        for file_path in shell_files:
            if not any(part in exclude_dirs for part in file_path.parts):
                filtered_files.append(file_path)
        
        logger.info(f"Found {len(filtered_files)} shell scripts in {directory}")
        
        for file_path in filtered_files:
            self._process_shell_script(file_path)

    def _has_shell_shebang(self, file_path: Path) -> bool:
        """Check if file has shell shebang"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                return first_line.startswith('#!') and ('bash' in first_line or 'sh' in first_line)
        except:
            return False

    def _process_shell_script(self, file_path: Path) -> None:
        """Process individual shell script"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Skip very small or very large files
            if len(content) < 50 or len(content) > 50000:
                self.skipped_files += 1
                return
            
            # Check for duplicates
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.duplicate_hashes:
                self.skipped_files += 1
                return
            
            self.duplicate_hashes.add(content_hash)
            
            # Analyze script
            script = self._analyze_script(file_path, content, content_hash)
            self.scripts.append(script)
            self.processed_files += 1
            
            if self.processed_files % 10 == 0:
                logger.info(f"Processed {self.processed_files} scripts...")
                
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            self.skipped_files += 1

    def _analyze_script(self, file_path: Path, content: str, content_hash: str) -> ShellScript:
        """Analyze shell script content and categorize"""
        lines = content.split('\n')
        line_count = len(lines)
        size_bytes = len(content.encode('utf-8'))
        
        # Categorize script
        category = self._categorize_script(file_path, content)
        
        # Extract AWS commands
        aws_commands = self._extract_aws_commands(content)
        
        # Extract patterns
        patterns = self._extract_patterns(content)
        
        # Check for syntax errors using shellcheck if available
        has_errors, shellcheck_score = self._check_syntax(file_path)
        
        return ShellScript(
            file_path=str(file_path),
            content=content,
            hash=content_hash,
            size_bytes=size_bytes,
            line_count=line_count,
            category=category,
            aws_commands=aws_commands,
            patterns=patterns,
            has_errors=has_errors,
            shellcheck_score=shellcheck_score
        )

    def _categorize_script(self, file_path: Path, content: str) -> str:
        """Categorize script based on path and content"""
        path_str = str(file_path).lower()
        content_lower = content.lower()
        
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in path_str or keyword in content_lower:
                    return category
        
        return 'general'

    def _extract_aws_commands(self, content: str) -> List[str]:
        """Extract AWS CLI commands from script"""
        aws_commands = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('aws ') and not line.startswith('aws #'):
                # Clean up the command
                command = re.sub(r'\s+', ' ', line)
                aws_commands.append(command)
        
        return aws_commands

    def _extract_patterns(self, content: str) -> List[str]:
        """Extract important patterns from script"""
        patterns = []
        
        # AWS CLI patterns
        for pattern in self.aws_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    patterns.append(' '.join(match))
                else:
                    patterns.append(match)
        
        # Destructive operations
        for pattern in self.destructive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                patterns.append(f"DESTRUCTIVE: {pattern}")
        
        return list(set(patterns))

    def _check_syntax(self, file_path: Path) -> Tuple[bool, int]:
        """Check script syntax using shellcheck if available"""
        try:
            result = subprocess.run(
                ['shellcheck', '-f', 'json', str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return False, 100  # No errors, perfect score
            
            # Parse shellcheck output
            try:
                issues = json.loads(result.stdout)
                error_count = len([i for i in issues if i.get('level') == 'error'])
                warning_count = len([i for i in issues if i.get('level') == 'warning'])
                
                # Calculate score (100 - penalties)
                score = max(0, 100 - (error_count * 10) - (warning_count * 2))
                return error_count > 0, score
            except:
                return True, 0
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # shellcheck not available or timeout
            return False, 75  # Assume OK but not verified

    def filter_high_quality_scripts(self, min_score: int = 70) -> List[ShellScript]:
        """Filter scripts for high quality examples"""
        high_quality = []
        
        for script in self.scripts:
            # Quality criteria
            if (script.shellcheck_score >= min_score and
                script.line_count >= 10 and
                script.line_count <= 500 and
                len(script.aws_commands) > 0 and
                not script.has_errors):
                high_quality.append(script)
        
        logger.info(f"Filtered to {len(high_quality)} high-quality scripts from {len(self.scripts)} total")
        return high_quality

    def generate_training_examples(self, scripts: List[ShellScript]) -> List[Dict]:
        """Generate prompt-completion training pairs"""
        training_examples = []
        
        for script in scripts:
            # Generate multiple training examples per script
            examples = self._create_examples_for_script(script)
            training_examples.extend(examples)
        
        logger.info(f"Generated {len(training_examples)} training examples")
        return training_examples

    def _create_examples_for_script(self, script: ShellScript) -> List[Dict]:
        """Create multiple training examples from a single script"""
        examples = []
        
        # 1. Complete script example
        prompt = self._generate_prompt_for_script(script)
        completion = self._format_script_completion(script)
        
        examples.append({
            "prompt": prompt,
            "completion": completion,
            "metadata": {
                "source_file": script.file_path,
                "category": script.category,
                "aws_commands_count": len(script.aws_commands),
                "destructive": any("DESTRUCTIVE" in p for p in script.patterns),
                "quality_score": script.shellcheck_score
            }
        })
        
        # 2. AWS command examples (if present)
        for aws_cmd in script.aws_commands[:3]:  # Limit to 3 per script
            aws_example = self._create_aws_command_example(aws_cmd, script.category)
            if aws_example:
                examples.append(aws_example)
        
        return examples

    def _generate_prompt_for_script(self, script: ShellScript) -> str:
        """Generate appropriate prompt for script based on its purpose"""
        base_prompts = {
            'deployment': "Write a shell script for deploying infrastructure resources",
            'monitoring': "Create a shell script for setting up monitoring and alerts",
            'orchestration': "Write a shell script that orchestrates multiple deployment steps",
            'infrastructure': "Create a shell script for provisioning cloud infrastructure",
            'aws_operations': "Write a shell script for AWS resource management",
            'data_processing': "Create a shell script for data processing pipeline",
            'security': "Write a shell script for security and permissions setup"
        }
        
        base_prompt = base_prompts.get(script.category, "Write a shell script for infrastructure automation")
        
        # Add specific requirements based on script analysis
        requirements = []
        
        if 'cloudformation' in str(script.file_path).lower():
            requirements.append("using CloudFormation")
        if script.aws_commands:
            requirements.append("with AWS CLI commands")
        if any("region" in p for p in script.patterns):
            requirements.append("that accepts region parameter")
        if any("DESTRUCTIVE" in p for p in script.patterns):
            requirements.append("with safety checks for destructive operations")
        
        if requirements:
            base_prompt += " " + ", ".join(requirements)
        
        base_prompt += "."
        
        return base_prompt

    def _format_script_completion(self, script: ShellScript) -> str:
        """Format script as completion with markdown"""
        # Clean up script content
        content = script.content.strip()
        
        # Add placeholders for sensitive data
        content = self._add_placeholders(content)
        
        return f"```bash\n{content}\n```"

    def _add_placeholders(self, content: str) -> str:
        """Replace sensitive or specific data with placeholders"""
        # Account IDs
        content = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', content)
        
        # ARNs
        content = re.sub(r'arn:aws:[^:]+:[^:]*:\d{12}:[^\s"\']+', '<ARN>', content)
        
        # Email addresses
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', content)
        
        # URLs (except AWS service URLs)
        content = re.sub(r'https?://(?!.*\.amazonaws\.com)[^\s"\']+', '<URL>', content)
        
        # S3 bucket names (format: word-word-word)
        content = re.sub(r'\b[a-z0-9]+-[a-z0-9]+-[a-z0-9]+\b', '<BUCKET_NAME>', content)
        
        # Instance IDs
        content = re.sub(r'\bi-[0-9a-f]{8,17}\b', '<INSTANCE_ID>', content)
        
        return content

    def _create_aws_command_example(self, aws_cmd: str, category: str) -> Optional[Dict]:
        """Create training example for specific AWS command"""
        # Skip very long commands
        if len(aws_cmd) > 200:
            return None
        
        # Generate prompt based on command
        prompt = self._generate_aws_command_prompt(aws_cmd)
        
        if not prompt:
            return None
        
        completion = f"```bash\n{aws_cmd}\n```"
        
        return {
            "prompt": prompt,
            "completion": completion,
            "metadata": {
                "type": "aws_command",
                "category": category,
                "command_type": aws_cmd.split()[1] if len(aws_cmd.split()) > 1 else "unknown"
            }
        }

    def _generate_aws_command_prompt(self, aws_cmd: str) -> Optional[str]:
        """Generate prompt for AWS CLI command"""
        parts = aws_cmd.split()
        if len(parts) < 2:
            return None
        
        service = parts[1]
        operation = parts[2] if len(parts) > 2 else ""
        
        command_prompts = {
            ('s3', 'cp'): "Copy a file to S3 bucket using AWS CLI",
            ('s3', 'sync'): "Sync local directory with S3 bucket",
            ('ec2', 'describe-instances'): "List EC2 instances using AWS CLI",
            ('ec2', 'run-instances'): "Launch an EC2 instance using AWS CLI",
            ('cloudformation', 'deploy'): "Deploy CloudFormation stack using AWS CLI",
            ('cloudformation', 'describe-stacks'): "Get CloudFormation stack information",
            ('lambda', 'create-function'): "Create Lambda function using AWS CLI",
            ('lambda', 'update-function-code'): "Update Lambda function code",
            ('iam', 'create-role'): "Create IAM role using AWS CLI",
            ('sns', 'create-topic'): "Create SNS topic using AWS CLI",
            ('cloudwatch', 'put-metric-alarm'): "Create CloudWatch alarm using AWS CLI"
        }
        
        prompt = command_prompts.get((service, operation))
        
        if not prompt:
            # Generic prompt based on service
            service_prompts = {
                's3': "Perform S3 operation using AWS CLI",
                'ec2': "Manage EC2 resources using AWS CLI", 
                'lambda': "Manage Lambda function using AWS CLI",
                'cloudformation': "Manage CloudFormation stack using AWS CLI",
                'iam': "Manage IAM resources using AWS CLI",
                'sns': "Manage SNS resources using AWS CLI",
                'cloudwatch': "Manage CloudWatch resources using AWS CLI"
            }
            prompt = service_prompts.get(service, f"Use AWS CLI for {service} operations")
        
        return prompt

    def save_corpus(self, training_examples: List[Dict], filename: str = "shell_training_corpus.jsonl") -> None:
        """Save training corpus to JSONL format"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(training_examples)} examples to {output_file}")

    def generate_statistics(self) -> Dict:
        """Generate corpus statistics"""
        if not self.scripts:
            return {}
        
        categories = {}
        quality_scores = []
        aws_commands_total = 0
        destructive_count = 0
        
        for script in self.scripts:
            categories[script.category] = categories.get(script.category, 0) + 1
            quality_scores.append(script.shellcheck_score)
            aws_commands_total += len(script.aws_commands)
            
            if any("DESTRUCTIVE" in p for p in script.patterns):
                destructive_count += 1
        
        return {
            "total_scripts": len(self.scripts),
            "categories": categories,
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "total_aws_commands": aws_commands_total,
            "destructive_scripts": destructive_count,
            "average_lines": sum(s.line_count for s in self.scripts) / len(self.scripts)
        }


def main():
    """Main corpus building pipeline"""
    logger.info("Starting Shell Script Corpus Builder")
    
    # Initialize builder
    builder = ShellCorpusBuilder()
    
    # Repository paths to scan
    repo_paths = [
        "/home/shingai/api",
        "/home/shingai/sort/ona-front-end/express-chatbot"
    ]
    
    # Build corpus following CLAUDE.md methodology
    try:
        # 1. EXPLORE: Scan repositories
        builder.scan_repositories(repo_paths)
        
        # 2. PLAN: Filter high quality scripts
        high_quality_scripts = builder.filter_high_quality_scripts(min_score=70)
        
        # 3. CODE: Generate training examples
        training_examples = builder.generate_training_examples(high_quality_scripts)
        
        # 4. COMMIT: Save corpus and statistics
        builder.save_corpus(training_examples)
        
        # Generate statistics
        stats = builder.generate_statistics()
        stats_file = builder.output_dir / "corpus_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Corpus Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Shell corpus building completed successfully!")
        
    except Exception as e:
        logger.error(f"Corpus building failed: {e}")
        raise


if __name__ == "__main__":
    main()