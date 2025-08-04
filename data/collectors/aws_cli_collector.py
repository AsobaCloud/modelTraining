#!/usr/bin/env python3
"""
AWS CLI Official Examples Collector
Extracts shell commands from AWS CLI official documentation and examples

Priority 1 Source: 50-75 high-quality examples expected
"""

import os
import re
import json
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AWSCLICollector:
    """Collects AWS CLI examples from official sources"""
    
    def __init__(self, output_dir: str = "aws_cli_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # AWS CLI GitHub repository
        self.aws_cli_repo = "https://api.github.com/repos/aws/aws-cli"
        self.examples_path = "/awscli/examples"
        
        # Tracking
        self.collected_examples = []
        self.processed_files = 0
        self.skipped_files = 0
        
        # AWS Services to prioritize for IaC focus
        self.iac_services = {
            'cloudformation', 'ec2', 's3', 'iam', 'lambda', 'apigateway',
            'cloudwatch', 'sns', 'sqs', 'rds', 'ecs', 'eks', 'ecr',
            'route53', 'elb', 'autoscaling', 'vpc', 'sts', 'kms'
        }

    def collect_aws_cli_examples(self) -> List[Dict]:
        """Main collection method"""
        logger.info("Starting AWS CLI examples collection from official repository")
        
        # Method 1: Try to use AWS CLI directly to get examples
        local_examples = self._collect_local_aws_examples()
        
        # Method 2: Use known AWS CLI patterns and services
        service_examples = self._generate_service_examples()
        
        # Method 3: Extract from AWS documentation patterns
        doc_examples = self._collect_common_patterns()
        
        all_examples = local_examples + service_examples + doc_examples
        
        # Remove duplicates and filter quality
        unique_examples = self._deduplicate_examples(all_examples)
        quality_examples = self._filter_quality_examples(unique_examples)
        
        logger.info(f"Collected {len(quality_examples)} AWS CLI examples")
        return quality_examples

    def _collect_local_aws_examples(self) -> List[Dict]:
        """Collect examples using local AWS CLI help"""
        examples = []
        
        # Get list of AWS services
        services = self._get_aws_services()
        
        for service in services[:20]:  # Limit to top 20 services for now
            if service in self.iac_services:
                service_examples = self._get_service_examples(service)
                examples.extend(service_examples)
        
        logger.info(f"Collected {len(examples)} examples from local AWS CLI help")
        return examples

    def _get_aws_services(self) -> List[str]:
        """Get list of AWS services from CLI"""
        try:
            result = subprocess.run(['aws', 'help'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse service names from help output
                services = []
                in_services_section = False
                
                for line in result.stdout.split('\n'):
                    if 'AVAILABLE SERVICES' in line:
                        in_services_section = True
                        continue
                    
                    if in_services_section and line.strip():
                        if line.startswith('   '):  # Service line
                            service = line.strip().split()[0]
                            if service and not service.startswith('o'):
                                services.append(service)
                        elif not line.startswith(' '):  # End of services section
                            break
                
                return services[:50]  # Limit to first 50
            
        except Exception as e:
            logger.warning(f"Could not get AWS services list: {e}")
        
        # Fallback to known IaC services
        return list(self.iac_services)

    def _get_service_examples(self, service: str) -> List[Dict]:
        """Get examples for specific AWS service"""
        examples = []
        
        try:
            # Get service help to find subcommands
            result = subprocess.run(['aws', service, 'help'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Extract example commands from help text
                help_text = result.stdout
                command_examples = self._extract_commands_from_help(help_text, service)
                examples.extend(command_examples)
                
        except Exception as e:
            logger.debug(f"Could not get examples for service {service}: {e}")
        
        return examples

    def _extract_commands_from_help(self, help_text: str, service: str) -> List[Dict]:
        """Extract AWS CLI commands from help text"""
        examples = []
        
        # Pattern to find AWS CLI commands in help text
        aws_cmd_pattern = rf'aws {service} [a-z-]+[^\n]*'
        
        commands = re.findall(aws_cmd_pattern, help_text, re.IGNORECASE)
        
        for cmd in commands:
            if len(cmd) > 15 and '--' in cmd:  # Ensure it's a meaningful command
                prompt = self._generate_prompt_for_command(cmd, service)
                if prompt:
                    examples.append({
                        "prompt": prompt,
                        "completion": f"```bash\n{cmd}\n```",
                        "metadata": {
                            "source": "aws_cli_help",
                            "service": service,
                            "type": "official_example"
                        }
                    })
        
        return examples

    def _generate_service_examples(self) -> List[Dict]:
        """Generate examples for common AWS CLI patterns"""
        examples = []
        
        # Common patterns for each service
        service_patterns = {
            'ec2': [
                ('describe-instances', 'List all EC2 instances using AWS CLI'),
                ('run-instances', 'Launch an EC2 instance using AWS CLI'),
                ('terminate-instances', 'Terminate EC2 instances using AWS CLI'),
                ('describe-security-groups', 'List EC2 security groups using AWS CLI'),
                ('create-security-group', 'Create EC2 security group using AWS CLI')
            ],
            'cloudformation': [
                ('deploy', 'Deploy CloudFormation stack using AWS CLI'),
                ('describe-stacks', 'Get CloudFormation stack information using AWS CLI'),
                ('list-stacks', 'List CloudFormation stacks using AWS CLI'),
                ('delete-stack', 'Delete CloudFormation stack using AWS CLI'),
                ('validate-template', 'Validate CloudFormation template using AWS CLI')
            ],
            's3': [
                ('cp', 'Copy files to S3 bucket using AWS CLI'),
                ('sync', 'Sync directory with S3 bucket using AWS CLI'),
                ('ls', 'List S3 bucket contents using AWS CLI'),
                ('mb', 'Create S3 bucket using AWS CLI'),
                ('rb', 'Delete S3 bucket using AWS CLI')
            ],
            'iam': [
                ('create-role', 'Create IAM role using AWS CLI'),
                ('attach-role-policy', 'Attach policy to IAM role using AWS CLI'),
                ('create-user', 'Create IAM user using AWS CLI'),
                ('list-roles', 'List IAM roles using AWS CLI'),
                ('get-role', 'Get IAM role details using AWS CLI')
            ],
            'lambda': [
                ('create-function', 'Create Lambda function using AWS CLI'),
                ('update-function-code', 'Update Lambda function code using AWS CLI'),
                ('invoke', 'Invoke Lambda function using AWS CLI'),
                ('list-functions', 'List Lambda functions using AWS CLI'),
                ('delete-function', 'Delete Lambda function using AWS CLI')
            ],
            'sns': [
                ('create-topic', 'Create SNS topic using AWS CLI'),
                ('publish', 'Publish message to SNS topic using AWS CLI'),
                ('subscribe', 'Subscribe to SNS topic using AWS CLI'),
                ('list-topics', 'List SNS topics using AWS CLI'),
                ('delete-topic', 'Delete SNS topic using AWS CLI')
            ],
            'cloudwatch': [
                ('put-metric-alarm', 'Create CloudWatch alarm using AWS CLI'),
                ('describe-alarms', 'List CloudWatch alarms using AWS CLI'),
                ('put-metric-data', 'Send custom metrics to CloudWatch using AWS CLI'),
                ('get-metric-statistics', 'Get CloudWatch metrics using AWS CLI'),
                ('delete-alarms', 'Delete CloudWatch alarms using AWS CLI')
            ]
        }
        
        for service, patterns in service_patterns.items():
            for operation, prompt in patterns:
                # Generate realistic command example
                command = self._generate_realistic_command(service, operation)
                
                examples.append({
                    "prompt": prompt,
                    "completion": f"```bash\n{command}\n```",
                    "metadata": {
                        "source": "generated_pattern",
                        "service": service,
                        "operation": operation,
                        "type": "common_pattern"
                    }
                })
        
        return examples

    def _generate_realistic_command(self, service: str, operation: str) -> str:
        """Generate realistic AWS CLI command with common parameters"""
        
        base_cmd = f"aws {service} {operation}"
        
        # Common parameters by service and operation
        params = {
            ('ec2', 'describe-instances'): '--query "Reservations[].Instances[].{ID:InstanceId,State:State.Name}" --output table',
            ('ec2', 'run-instances'): '--image-id ami-12345678 --count 1 --instance-type t3.micro --key-name <KEY_NAME> --security-group-ids sg-12345678',
            ('ec2', 'terminate-instances'): '--instance-ids <INSTANCE_ID>',
            ('ec2', 'create-security-group'): '--group-name <SECURITY_GROUP_NAME> --description "Security group for web servers"',
            
            ('cloudformation', 'deploy'): '--template-file template.yaml --stack-name <STACK_NAME> --capabilities CAPABILITY_IAM',
            ('cloudformation', 'describe-stacks'): '--stack-name <STACK_NAME> --query "Stacks[0].StackStatus" --output text',
            ('cloudformation', 'list-stacks'): '--stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE --query "StackSummaries[].{Name:StackName,Status:StackStatus}"',
            ('cloudformation', 'delete-stack'): '--stack-name <STACK_NAME>',
            ('cloudformation', 'validate-template'): '--template-body file://template.yaml',
            
            ('s3', 'cp'): '<FILE_PATH> s3://<BUCKET_NAME>/<KEY>',
            ('s3', 'sync'): '<LOCAL_DIR> s3://<BUCKET_NAME>/ --delete',
            ('s3', 'ls'): 's3://<BUCKET_NAME>/ --recursive',
            ('s3', 'mb'): 's3://<BUCKET_NAME>',
            ('s3', 'rb'): 's3://<BUCKET_NAME> --force',
            
            ('iam', 'create-role'): '--role-name <ROLE_NAME> --assume-role-policy-document file://trust-policy.json',
            ('iam', 'attach-role-policy'): '--role-name <ROLE_NAME> --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess',
            ('iam', 'create-user'): '--user-name <USER_NAME>',
            ('iam', 'list-roles'): '--query "Roles[].{Name:RoleName,Created:CreateDate}" --output table',
            ('iam', 'get-role'): '--role-name <ROLE_NAME>',
            
            ('lambda', 'create-function'): '--function-name <FUNCTION_NAME> --runtime python3.9 --role arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME> --handler lambda_function.lambda_handler --zip-file fileb://function.zip',
            ('lambda', 'update-function-code'): '--function-name <FUNCTION_NAME> --zip-file fileb://function.zip',
            ('lambda', 'invoke'): '--function-name <FUNCTION_NAME> --payload \'{"key": "value"}\' response.json',
            ('lambda', 'list-functions'): '--query "Functions[].{Name:FunctionName,Runtime:Runtime}" --output table',
            ('lambda', 'delete-function'): '--function-name <FUNCTION_NAME>',
            
            ('sns', 'create-topic'): '--name <TOPIC_NAME>',
            ('sns', 'publish'): '--topic-arn <TOPIC_ARN> --message "Hello World"',
            ('sns', 'subscribe'): '--topic-arn <TOPIC_ARN> --protocol email --notification-endpoint <EMAIL>',
            ('sns', 'list-topics'): '--query "Topics[].TopicArn" --output table',
            ('sns', 'delete-topic'): '--topic-arn <TOPIC_ARN>',
            
            ('cloudwatch', 'put-metric-alarm'): '--alarm-name <ALARM_NAME> --alarm-description "CPU utilization" --metric-name CPUUtilization --namespace AWS/EC2 --statistic Average --period 300 --threshold 80 --comparison-operator GreaterThanThreshold --evaluation-periods 2',
            ('cloudwatch', 'describe-alarms'): '--query "MetricAlarms[].{Name:AlarmName,State:StateValue}" --output table',
            ('cloudwatch', 'put-metric-data'): '--namespace "Custom/Application" --metric-data MetricName=PageViews,Value=100,Unit=Count',
            ('cloudwatch', 'get-metric-statistics'): '--namespace AWS/EC2 --metric-name CPUUtilization --start-time 2023-01-01T00:00:00Z --end-time 2023-01-01T23:59:59Z --period 3600 --statistics Average',
            ('cloudwatch', 'delete-alarms'): '--alarm-names <ALARM_NAME>'
        }
        
        param_str = params.get((service, operation), '')
        
        if param_str:
            return f"{base_cmd} {param_str}"
        else:
            return f"{base_cmd} --help"

    def _collect_common_patterns(self) -> List[Dict]:
        """Collect common AWS CLI usage patterns"""
        examples = []
        
        # Common infrastructure patterns
        patterns = [
            {
                "prompt": "Create AWS CLI script to launch EC2 instance with security group",
                "completion": """```bash
#!/bin/bash
# Launch EC2 instance with custom security group

# Create security group
aws ec2 create-security-group \\
    --group-name web-servers \\
    --description "Security group for web servers"

# Add SSH access rule
aws ec2 authorize-security-group-ingress \\
    --group-name web-servers \\
    --protocol tcp \\
    --port 22 \\
    --cidr 0.0.0.0/0

# Add HTTP access rule  
aws ec2 authorize-security-group-ingress \\
    --group-name web-servers \\
    --protocol tcp \\
    --port 80 \\
    --cidr 0.0.0.0/0

# Launch instance
aws ec2 run-instances \\
    --image-id ami-12345678 \\
    --count 1 \\
    --instance-type t3.micro \\
    --key-name <KEY_NAME> \\
    --security-groups web-servers \\
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]'
```""",
                "metadata": {"source": "common_pattern", "type": "multi_step_deployment"}
            },
            {
                "prompt": "Create AWS CLI script to deploy and monitor CloudFormation stack",
                "completion": """```bash
#!/bin/bash
# Deploy CloudFormation stack with monitoring

STACK_NAME="<STACK_NAME>"
TEMPLATE_FILE="infrastructure.yaml"

# Validate template
echo "Validating CloudFormation template..."
aws cloudformation validate-template --template-body file://$TEMPLATE_FILE

# Deploy stack
echo "Deploying CloudFormation stack..."
aws cloudformation deploy \\
    --template-file $TEMPLATE_FILE \\
    --stack-name $STACK_NAME \\
    --capabilities CAPABILITY_IAM \\
    --parameter-overrides Environment=production

# Wait for completion
echo "Waiting for stack deployment to complete..."
aws cloudformation wait stack-deploy-complete --stack-name $STACK_NAME

# Get stack outputs
echo "Stack deployment completed. Outputs:"
aws cloudformation describe-stacks \\
    --stack-name $STACK_NAME \\
    --query "Stacks[0].Outputs[].{Key:OutputKey,Value:OutputValue}" \\
    --output table
```""",
                "metadata": {"source": "common_pattern", "type": "cloudformation_deployment"}
            },
            {
                "prompt": "Create AWS CLI script for S3 backup with lifecycle management",
                "completion": """```bash
#!/bin/bash
# S3 backup script with lifecycle management

BUCKET_NAME="<BUCKET_NAME>"
LOCAL_DIR="<LOCAL_DIR>"

# Create S3 bucket
aws s3 mb s3://$BUCKET_NAME

# Enable versioning
aws s3api put-bucket-versioning \\
    --bucket $BUCKET_NAME \\
    --versioning-configuration Status=Enabled

# Sync local directory to S3
aws s3 sync $LOCAL_DIR s3://$BUCKET_NAME/ \\
    --delete \\
    --exclude "*.tmp" \\
    --exclude ".DS_Store"

# Set lifecycle policy
aws s3api put-bucket-lifecycle-configuration \\
    --bucket $BUCKET_NAME \\
    --lifecycle-configuration file://lifecycle-policy.json

echo "Backup completed to s3://$BUCKET_NAME"
```""",
                "metadata": {"source": "common_pattern", "type": "s3_backup"}
            }
        ]
        
        return patterns

    def _generate_prompt_for_command(self, command: str, service: str) -> Optional[str]:
        """Generate appropriate prompt for AWS CLI command"""
        parts = command.split()
        if len(parts) < 3:
            return None
        
        operation = parts[2]
        
        # Operation-specific prompts
        prompt_templates = {
            'describe': f"List {service} resources using AWS CLI",
            'list': f"List {service} resources using AWS CLI", 
            'create': f"Create {service} resource using AWS CLI",
            'delete': f"Delete {service} resource using AWS CLI",
            'update': f"Update {service} resource using AWS CLI",
            'deploy': f"Deploy {service} using AWS CLI",
            'put': f"Configure {service} using AWS CLI",
            'get': f"Retrieve {service} information using AWS CLI"
        }
        
        for verb, template in prompt_templates.items():
            if operation.startswith(verb):
                return template
        
        return f"Use AWS CLI for {service} {operation}"

    def _deduplicate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicate examples based on command similarity"""
        seen_commands = set()
        unique_examples = []
        
        for example in examples:
            # Extract command from completion
            completion = example['completion']
            command = re.search(r'```bash\n(.*?)\n```', completion, re.DOTALL)
            
            if command:
                cmd_text = command.group(1).strip()
                # Normalize command for deduplication
                normalized = re.sub(r'<[^>]+>', '<PLACEHOLDER>', cmd_text)
                normalized = re.sub(r'\s+', ' ', normalized)
                
                if normalized not in seen_commands:
                    seen_commands.add(normalized)
                    unique_examples.append(example)
        
        return unique_examples

    def _filter_quality_examples(self, examples: List[Dict]) -> List[Dict]:
        """Filter examples for quality and relevance"""
        quality_examples = []
        
        for example in examples:
            completion = example['completion']
            
            # Quality checks
            if len(completion) < 30:  # Too short
                continue
            
            if 'aws help' in completion:  # Help commands not useful for training
                continue
            
            if completion.count('aws ') > 5:  # Too many commands in one example
                continue
            
            quality_examples.append(example)
        
        return quality_examples

    def save_corpus(self, examples: List[Dict], filename: str = "aws_cli_corpus.jsonl") -> None:
        """Save AWS CLI corpus to JSONL format"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} AWS CLI examples to {output_file}")


def main():
    """Main AWS CLI collection pipeline"""
    logger.info("Starting AWS CLI Official Examples Collection")
    
    collector = AWSCLICollector()
    
    try:
        # Collect examples
        examples = collector.collect_aws_cli_examples()
        
        # Save corpus
        collector.save_corpus(examples)
        
        # Generate statistics
        services = {}
        for example in examples:
            service = example['metadata'].get('service', 'unknown')
            services[service] = services.get(service, 0) + 1
        
        logger.info(f"AWS CLI Collection Statistics:")
        logger.info(f"Total examples: {len(examples)}")
        logger.info(f"Services covered: {len(services)}")
        for service, count in sorted(services.items()):
            logger.info(f"  {service}: {count} examples")
        
        logger.info("AWS CLI examples collection completed successfully!")
        
    except Exception as e:
        logger.error(f"AWS CLI collection failed: {e}")
        raise


if __name__ == "__main__":
    main()