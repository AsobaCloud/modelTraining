#!/usr/bin/env python3
"""
MCP Tools for IaC Development with CLAUDE.md Methodology
Provides structured tools for each phase of infrastructure development
"""

import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class WorkflowDepth(Enum):
    DIRECT_CODE = "direct_code"
    PLAN_AND_CODE = "plan_and_code" 
    FULL_CLAUDE_MD = "full_claude_md"


@dataclass
class InfrastructureRequirements:
    """Structured requirements from EXPLORE phase"""
    domain: str  # web_app, data_platform, microservices, etc.
    scale: str   # small, medium, large, enterprise
    constraints: List[str]  # budget, compliance, existing_systems
    security_requirements: List[str]
    operational_requirements: List[str]
    

@dataclass  
class InfrastructureArchitecture:
    """Architecture design from PLAN phase"""
    topology: Dict[str, any]
    security_design: Dict[str, any]
    scalability_patterns: List[str]
    cost_optimization: List[str]
    disaster_recovery: Dict[str, any]
    monitoring_design: Dict[str, any]


def analyze_request_complexity(prompt: str) -> WorkflowDepth:
    """
    Analyze request complexity to determine appropriate workflow depth
    
    Args:
        prompt: User's infrastructure request
        
    Returns:
        WorkflowDepth enum indicating recommended approach
    """
    prompt_lower = prompt.lower()
    
    # Complex request triggers - require full CLAUDE.md methodology
    complex_triggers = [
        "production", "enterprise", "multi-tier", "scalable", 
        "security", "monitoring", "ci/cd", "complete solution",
        "architecture", "design", "plan", "full-stack", "end-to-end",
        "disaster recovery", "high availability", "compliance",
        "multi-region", "fault tolerant", "zero downtime"
    ]
    
    # Medium complexity triggers - plan + code approach
    medium_triggers = [
        "web app", "api", "database", "microservice", "deployment",
        "load balancer", "auto scaling", "backup", "logging", "pipeline"
    ]
    
    # Count complexity indicators
    complex_score = sum(1 for trigger in complex_triggers if trigger in prompt_lower)
    medium_score = sum(1 for trigger in medium_triggers if trigger in prompt_lower)
    
    # Word count and technical depth indicators
    word_count = len(prompt.split())
    has_multiple_services = len(re.findall(r'\b(service|component|tier|layer)\b', prompt_lower)) > 1
    has_specific_requirements = len(re.findall(r'\b(requirement|constraint|must|should|need)\b', prompt_lower)) > 0
    
    # Scoring algorithm
    total_score = complex_score * 3 + medium_score * 1
    if word_count > 50:
        total_score += 1
    if has_multiple_services:
        total_score += 2
    if has_specific_requirements:
        total_score += 1
        
    # Determine workflow depth
    if total_score >= 7 or complex_score >= 2:
        return WorkflowDepth.FULL_CLAUDE_MD
    elif total_score >= 3 or medium_score >= 2:
        return WorkflowDepth.PLAN_AND_CODE
    else:
        return WorkflowDepth.DIRECT_CODE


class IaCMCPTools:
    """MCP Tools for Infrastructure as Code development with CLAUDE.md methodology"""
    
    @staticmethod
    def iac_explore(domain: str, scale: str, constraints: List[str]) -> InfrastructureRequirements:
        """
        EXPLORE phase: Systematically gather infrastructure requirements
        
        Args:
            domain: Infrastructure domain (web_app, data_platform, microservices)
            scale: Scale requirements (small, medium, large, enterprise)
            constraints: List of constraints (budget, compliance, existing_systems)
            
        Returns:
            InfrastructureRequirements object with gathered requirements
        """
        # Domain-specific requirement templates
        domain_requirements = {
            "web_app": {
                "security_requirements": ["TLS termination", "WAF protection", "secure headers"],
                "operational_requirements": ["health checks", "auto scaling", "CDN integration"]
            },
            "data_platform": {
                "security_requirements": ["data encryption", "access logging", "PII protection"],
                "operational_requirements": ["backup automation", "data retention", "monitoring"]
            },
            "microservices": {
                "security_requirements": ["service mesh security", "mTLS", "API gateway"],
                "operational_requirements": ["service discovery", "distributed tracing", "circuit breakers"]
            }
        }
        
        # Scale-specific requirements
        scale_requirements = {
            "small": {
                "operational_requirements": ["basic monitoring", "manual scaling"]
            },
            "medium": {
                "operational_requirements": ["auto scaling", "log aggregation", "alerting"]
            },
            "large": {
                "operational_requirements": ["multi-region", "advanced monitoring", "SRE practices"]
            },
            "enterprise": {
                "operational_requirements": ["compliance reporting", "audit trails", "disaster recovery"]
            }
        }
        
        # Combine requirements based on domain and scale
        base_reqs = domain_requirements.get(domain, {
            "security_requirements": ["basic security"],
            "operational_requirements": ["basic operations"]
        })
        
        scale_reqs = scale_requirements.get(scale, {
            "operational_requirements": ["standard operations"]
        })
        
        # Merge requirements
        security_requirements = base_reqs["security_requirements"]
        operational_requirements = list(set(
            base_reqs["operational_requirements"] + scale_reqs["operational_requirements"]
        ))
        
        return InfrastructureRequirements(
            domain=domain,
            scale=scale,
            constraints=constraints,
            security_requirements=security_requirements,
            operational_requirements=operational_requirements
        )
    
    @staticmethod
    def iac_plan(requirements: InfrastructureRequirements, preferences: List[str]) -> InfrastructureArchitecture:
        """
        PLAN phase: Design infrastructure architecture with trade-offs
        
        Args:
            requirements: Requirements from EXPLORE phase
            preferences: User preferences (cloud provider, tools, patterns)
            
        Returns:
            InfrastructureArchitecture object with detailed design
        """
        # Architecture patterns based on domain and scale
        topology_patterns = {
            "web_app": {
                "small": {"pattern": "single_tier", "components": ["ALB", "EC2", "RDS"]},
                "medium": {"pattern": "three_tier", "components": ["ALB", "ASG", "RDS", "ElastiCache"]},
                "large": {"pattern": "microservices", "components": ["ALB", "ECS", "RDS", "ElastiCache", "EFS"]},
                "enterprise": {"pattern": "multi_region", "components": ["Route53", "ALB", "ECS", "Aurora", "CloudFront"]}
            }
        }
        
        # Security design patterns
        security_patterns = {
            "basic": {"vpc_design": "public_private_subnets", "iam": "least_privilege"},
            "enhanced": {"vpc_design": "private_only", "iam": "role_based", "encryption": "all_data"},
            "enterprise": {"vpc_design": "multi_tier", "iam": "federated", "encryption": "all_data", "compliance": "audit_ready"}
        }
        
        # Get base topology
        domain_patterns = topology_patterns.get(requirements.domain, topology_patterns["web_app"])
        topology = domain_patterns.get(requirements.scale, domain_patterns["medium"])
        
        # Determine security level based on requirements
        if "compliance" in requirements.constraints:
            security_level = "enterprise"
        elif len(requirements.security_requirements) > 3:
            security_level = "enhanced" 
        else:
            security_level = "basic"
            
        security_design = security_patterns[security_level]
        
        # Scalability patterns based on scale
        scalability_patterns = {
            "small": ["manual_scaling"],
            "medium": ["auto_scaling", "load_balancing"],
            "large": ["auto_scaling", "load_balancing", "multi_az"],
            "enterprise": ["auto_scaling", "load_balancing", "multi_az", "multi_region"]
        }
        
        scalability = scalability_patterns.get(requirements.scale, ["auto_scaling"])
        
        # Cost optimization strategies
        cost_optimization = []
        if "budget" in requirements.constraints:
            cost_optimization = ["reserved_instances", "spot_instances", "scheduled_scaling"]
        
        # Disaster recovery design
        disaster_recovery = {
            "backup_strategy": "automated_snapshots",
            "rto_target": "4_hours" if requirements.scale in ["large", "enterprise"] else "24_hours",
            "rpo_target": "1_hour" if requirements.scale in ["large", "enterprise"] else "4_hours"
        }
        
        # Monitoring design
        monitoring_design = {
            "metrics": ["CloudWatch", "custom_metrics"],
            "logging": ["CloudWatch_Logs"],
            "alerting": ["SNS", "email"]
        }
        
        if requirements.scale in ["large", "enterprise"]:
            monitoring_design["metrics"].append("Prometheus")
            monitoring_design["logging"].extend(["ELK_stack", "centralized"])
            monitoring_design["alerting"].extend(["PagerDuty", "Slack"])
        
        return InfrastructureArchitecture(
            topology=topology,
            security_design=security_design,
            scalability_patterns=scalability,
            cost_optimization=cost_optimization,
            disaster_recovery=disaster_recovery,
            monitoring_design=monitoring_design
        )
    
    @staticmethod
    def iac_implement(architecture: InfrastructureArchitecture, format: str = "terraform") -> Dict[str, str]:
        """
        CODE phase: Generate production-ready IaC code
        
        Args:
            architecture: Architecture design from PLAN phase
            format: Desired IaC format (terraform, kubernetes, docker, ansible)
            
        Returns:
            Dictionary with generated code files and configurations
        """
        if format == "terraform":
            return IaCMCPTools._generate_terraform_code(architecture)
        elif format == "kubernetes":
            return IaCMCPTools._generate_kubernetes_manifests(architecture)
        elif format == "docker":
            return IaCMCPTools._generate_docker_configs(architecture)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _generate_terraform_code(architecture: InfrastructureArchitecture) -> Dict[str, str]:
        """Generate Terraform code based on architecture"""
        files = {}
        
        # Main infrastructure
        files["main.tf"] = f"""
# Generated Infrastructure following CLAUDE.md methodology
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
  
  default_tags {{
    tags = {{
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner
    }}
  }}
}}

# VPC and Networking
module "vpc" {{
  source = "./modules/vpc"
  
  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = var.vpc_cidr
  
  security_design = "{architecture.security_design.get('vpc_design', 'public_private_subnets')}"
}}

# Application Infrastructure
module "application" {{
  source = "./modules/application"
  
  project_name = var.project_name
  environment  = var.environment
  vpc_id       = module.vpc.vpc_id
  subnet_ids   = module.vpc.private_subnet_ids
  
  topology = {architecture.topology}
  scalability_patterns = {architecture.scalability_patterns}
}}

# Monitoring and Observability
module "monitoring" {{
  source = "./modules/monitoring"
  
  project_name = var.project_name
  environment  = var.environment
  
  monitoring_design = {architecture.monitoring_design}
}}
"""
        
        # Variables
        files["variables.tf"] = """
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}
"""
        
        # Outputs
        files["outputs.tf"] = """
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "application_endpoints" {
  description = "Application endpoints"
  value       = module.application.endpoints
}

output "monitoring_dashboard_url" {
  description = "Monitoring dashboard URL"
  value       = module.monitoring.dashboard_url
}
"""
        
        return files
    
    @staticmethod
    def _generate_kubernetes_manifests(architecture: InfrastructureArchitecture) -> Dict[str, str]:
        """Generate Kubernetes manifests based on architecture"""
        files = {}
        
        # Namespace
        files["namespace.yaml"] = """
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    name: production
    managed-by: claude-md
"""
        
        # Deployment
        files["deployment.yaml"] = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
  labels:
    app: web-app
    managed-by: claude-md
spec:
  replicas: {3 if architecture.topology.get('pattern') == 'microservices' else 2}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: web-app
        image: nginx:1.21-alpine
        ports:
        - containerPort: 80
          name: http
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
"""
        
        return files
    
    @staticmethod  
    def _generate_docker_configs(architecture: InfrastructureArchitecture) -> Dict[str, str]:
        """Generate Docker configurations based on architecture"""
        files = {}
        
        # Multi-stage Dockerfile
        files["Dockerfile"] = """
# Multi-stage Dockerfile following CLAUDE.md security best practices
FROM node:18-alpine AS builder

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY src/ ./src/

# Build application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Set working directory
WORKDIR /app

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --chown=nextjs:nodejs package*.json ./

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start application
CMD ["node", "dist/server.js"]
"""
        
        # Docker Compose for local development
        files["docker-compose.yml"] = f"""
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    depends_on:
      - db
      - redis
    networks:
      - app-network
    restart: unless-stopped
    
  db:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=appdb
      - POSTGRES_USER=appuser
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - app-network
    restart: unless-stopped
    secrets:
      - db_password
      
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass_file /run/secrets/redis_password
    networks:
      - app-network
    restart: unless-stopped
    secrets:
      - redis_password

networks:
  app-network:
    driver: bridge

volumes:
  postgres_data:

secrets:
  db_password:
    file: ./secrets/db_password.txt
  redis_password:
    file: ./secrets/redis_password.txt
"""
        
        return files
    
    @staticmethod
    def iac_validate(code: str, environment: str) -> Dict[str, any]:
        """
        COMMIT phase: Create tests and deployment procedures
        
        Args:
            code: Generated IaC code to validate
            environment: Target environment (dev, staging, prod)
            
        Returns:
            Dictionary with validation results and deployment procedures
        """
        validation_results = {
            "tests": IaCMCPTools._generate_infrastructure_tests(code),
            "deployment_procedures": IaCMCPTools._generate_deployment_procedures(environment),
            "monitoring_setup": IaCMCPTools._generate_monitoring_setup(),
            "runbooks": IaCMCPTools._generate_operational_runbooks()
        }
        
        return validation_results
    
    @staticmethod
    def _generate_infrastructure_tests(code: str) -> Dict[str, str]:
        """Generate infrastructure testing code"""
        if "terraform" in code.lower() or "resource" in code.lower():
            return {
                "terraform_test.py": """
import pytest
import boto3
import subprocess
import json

class TestInfrastructure:
    def test_terraform_plan_valid(self):
        \"\"\"Test that Terraform plan is valid\"\"\"
        result = subprocess.run(
            ["terraform", "plan", "-out=tfplan"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Terraform plan failed: {result.stderr}"
    
    def test_vpc_configuration(self):
        \"\"\"Test VPC is configured correctly\"\"\"
        ec2 = boto3.client('ec2')
        vpcs = ec2.describe_vpcs(
            Filters=[{'Name': 'tag:Project', 'Values': ['test-project']}]
        )
        assert len(vpcs['Vpcs']) == 1, "Expected exactly one VPC"
        
    def test_security_groups(self):
        \"\"\"Test security groups follow least privilege\"\"\"
        ec2 = boto3.client('ec2')
        sgs = ec2.describe_security_groups()
        
        for sg in sgs['SecurityGroups']:
            for rule in sg.get('IpPermissions', []):
                for ip_range in rule.get('IpRanges', []):
                    assert ip_range.get('CidrIp') != '0.0.0.0/0' or rule.get('FromPort') in [80, 443], \
                        f"Overly permissive security group rule found in {sg['GroupId']}"
"""
            }
        else:
            return {
                "kubernetes_test.py": """
import pytest
import subprocess
import yaml

class TestKubernetesManifests:
    def test_manifests_valid_yaml(self):
        \"\"\"Test all manifests are valid YAML\"\"\"
        manifest_files = ['namespace.yaml', 'deployment.yaml', 'service.yaml']
        
        for file in manifest_files:
            with open(file, 'r') as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {file}: {e}")
    
    def test_security_context(self):
        \"\"\"Test containers run with non-root user\"\"\"
        with open('deployment.yaml', 'r') as f:
            deployment = yaml.safe_load(f)
        
        security_context = deployment['spec']['template']['spec'].get('securityContext', {})
        assert security_context.get('runAsNonRoot') is True, "Container must run as non-root"
"""
            }
    
    @staticmethod
    def _generate_deployment_procedures(environment: str) -> Dict[str, str]:
        """Generate deployment procedures"""
        return {
            "deploy.sh": f"""#!/bin/bash
set -euo pipefail

# Deployment script for {environment} environment
# Generated following CLAUDE.md methodology

echo "🚀 Deploying to {environment} environment..."

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."
./scripts/pre-deploy-checks.sh

# Deploy infrastructure
echo "🏗️ Deploying infrastructure..."
terraform init
terraform plan -var-file="{environment}.tfvars" -out=tfplan
terraform apply tfplan

# Post-deployment validation
echo "✅ Running post-deployment validation..."
./scripts/post-deploy-validation.sh

# Setup monitoring
echo "📊 Setting up monitoring..."
./scripts/setup-monitoring.sh

echo "✅ Deployment complete!"
""",
            "rollback.sh": """#!/bin/bash
set -euo pipefail

# Rollback procedure
echo "🔄 Initiating rollback procedure..."

# Get previous state
terraform state pull > current-state.backup

# Rollback to previous version
terraform apply -var-file="rollback.tfvars" -auto-approve

echo "✅ Rollback complete!"
"""
        }
    
    @staticmethod
    def _generate_monitoring_setup() -> Dict[str, str]:
        """Generate monitoring and observability setup"""
        return {
            "monitoring.tf": """
# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-${var.environment}"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.main.arn_suffix],
            ["AWS/ApplicationELB", "HTTPCode_Target_2XX_Count", "LoadBalancer", aws_lb.main.arn_suffix],
            ["AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", aws_lb.main.arn_suffix]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Application Performance"
        }
      }
    ]
  })
}

# Alarms
resource "aws_cloudwatch_metric_alarm" "high_response_time" {
  alarm_name          = "${var.project_name}-high-response-time"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = "300"
  statistic           = "Average"
  threshold           = "2"
  alarm_description   = "This metric monitors application response time"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
""",
            "alerts.tf": """
# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
"""
        }
    
    @staticmethod
    def _generate_operational_runbooks() -> Dict[str, str]:
        """Generate operational runbooks"""
        return {
            "runbook_incident_response.md": """
# Incident Response Runbook

## High Response Time Alert

### Immediate Actions (0-5 minutes)
1. Check CloudWatch dashboard for traffic patterns
2. Verify all instances are healthy in target group
3. Check for any recent deployments

### Investigation (5-15 minutes)  
1. Review application logs in CloudWatch Logs
2. Check database performance metrics
3. Verify auto-scaling group is responding

### Escalation
- If issue persists > 15 minutes: Page on-call engineer
- If customer impact: Update status page
- If security related: Follow security incident procedure

## Database Connection Issues

### Immediate Actions
1. Check RDS instance status
2. Verify security group rules
3. Check application connection pool

### Investigation
1. Review RDS performance insights
2. Check for deadlocks or long-running queries
3. Verify database credentials in secrets manager
""",
            "runbook_deployment.md": """
# Deployment Runbook

## Pre-Deployment Checklist
- [ ] All tests passing in CI/CD
- [ ] Security scan completed
- [ ] Database migrations reviewed
- [ ] Rollback plan confirmed
- [ ] Stakeholders notified

## Deployment Steps
1. Deploy to staging environment
2. Run smoke tests
3. Deploy to production during maintenance window
4. Monitor key metrics for 30 minutes
5. Update documentation if needed

## Post-Deployment Verification
- [ ] Health checks passing
- [ ] Key user journeys working
- [ ] Metrics within normal ranges
- [ ] No error rate increase
"""
        }


# MCP Server Configuration
MCP_TOOLS = {
    "iac_analyze_complexity": {
        "description": "Analyze infrastructure request complexity to determine workflow approach",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Infrastructure request to analyze"}
            },
            "required": ["prompt"]
        }
    },
    "iac_explore": {
        "description": "EXPLORE phase - Systematically gather infrastructure requirements",
        "parameters": {
            "type": "object", 
            "properties": {
                "domain": {"type": "string", "enum": ["web_app", "data_platform", "microservices", "api", "batch_processing"]},
                "scale": {"type": "string", "enum": ["small", "medium", "large", "enterprise"]},
                "constraints": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["domain", "scale"]
        }
    },
    "iac_plan": {
        "description": "PLAN phase - Design infrastructure architecture with trade-offs",
        "parameters": {
            "type": "object",
            "properties": {
                "requirements": {"type": "object", "description": "Requirements from explore phase"},
                "preferences": {"type": "array", "items": {"type": "string"}, "description": "User preferences for tools and patterns"}
            },
            "required": ["requirements"]
        }
    },
    "iac_implement": {
        "description": "CODE phase - Generate production-ready IaC code",
        "parameters": {
            "type": "object",
            "properties": {
                "architecture": {"type": "object", "description": "Architecture from plan phase"},
                "format": {"type": "string", "enum": ["terraform", "kubernetes", "docker", "ansible"], "default": "terraform"}
            },
            "required": ["architecture"]
        }
    },
    "iac_validate": {
        "description": "COMMIT phase - Create tests and deployment procedures",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Generated IaC code to validate"},
                "environment": {"type": "string", "enum": ["dev", "staging", "prod"], "default": "prod"}
            },
            "required": ["code"]
        }
    }
}


if __name__ == "__main__":
    # Example usage
    tools = IaCMCPTools()
    
    # Test complexity analysis
    simple_request = "Create an S3 bucket"
    complex_request = "Design a production-ready, scalable web application infrastructure with security, monitoring, and disaster recovery"
    
    print("Simple request complexity:", analyze_request_complexity(simple_request))
    print("Complex request complexity:", analyze_request_complexity(complex_request))
    
    # Test full workflow
    requirements = tools.iac_explore("web_app", "large", ["budget", "security"])
    print("\nRequirements:", requirements)
    
    architecture = tools.iac_plan(requirements, ["aws", "terraform", "microservices"])
    print("\nArchitecture:", architecture.topology)
    
    code_files = tools.iac_implement(architecture, "terraform")
    print("\nGenerated files:", list(code_files.keys()))