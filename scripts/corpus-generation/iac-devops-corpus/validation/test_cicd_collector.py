#!/usr/bin/env python3
"""
TDD Tests for CI/CD Pipeline Collector
Following CLAUDE.md principle: Write failing tests first, then implement
"""

import pytest
import tempfile
import json
from pathlib import Path
from cicd_real_collector import CICDRealCollector

class TestCICDRealCollector:
    """Test cases for CI/CD pipeline collection following CLAUDE.md TDD approach"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = CICDRealCollector(output_dir=self.temp_dir)
    
    def test_cicd_collector_initialization(self):
        """Test collector initializes with required repositories"""
        # Should target production CI/CD repositories
        assert 'actions/starter-workflows' in self.collector.target_repos
        assert 'kubernetes/kubernetes' in self.collector.target_repos
        assert 'docker/build-push-action' in self.collector.target_repos
        
        # Should have empty collected examples initially
        assert len(self.collector.examples) == 0
        assert len(self.collector.collected_hashes) == 0

    def test_identifies_quality_github_actions_content(self):
        """Test quality detection for GitHub Actions workflow content"""
        # Valid GitHub Actions workflow
        valid_workflow = """
name: CI/CD Pipeline
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: user/app:latest
"""
        assert self.collector._is_quality_cicd_content(valid_workflow) == True
        
        # Valid GitLab CI content
        valid_gitlab = """
stages:
  - build
  - test
  - deploy

build-job:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy-job:
  stage: deploy
  script:
    - kubectl apply -f deployment.yaml
  environment:
    name: production
"""
        assert self.collector._is_quality_cicd_content(valid_gitlab) == True
        
        # Invalid content (too short)
        invalid_short = "name: test\nrun: echo hello"
        assert self.collector._is_quality_cicd_content(invalid_short) == False
        
        # Invalid content (template placeholder)
        invalid_template = """
name: Your App CI
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - run: echo "Replace with your commands"
"""
        assert self.collector._is_quality_cicd_content(invalid_template) == False

    def test_categorizes_cicd_content_correctly(self):
        """Test proper categorization of different CI/CD pipeline types"""
        github_actions = "name: CI\non:\n  push:\njobs:\n  build:\n    runs-on: ubuntu-latest"
        gitlab_ci = "stages:\n  - build\n  - test\nbuild-job:\n  stage: build\n  script:\n    - echo test"
        jenkins_file = "pipeline {\n  agent any\n  stages {\n    stage('Build') {\n      steps {\n        sh 'docker build .'\n      }\n    }\n  }\n}"
        
        assert self.collector._categorize_cicd_content(github_actions, "/.github/workflows/ci.yml") == "github_actions_workflow"
        assert self.collector._categorize_cicd_content(gitlab_ci, "/.gitlab-ci.yml") == "gitlab_ci_pipeline"
        assert self.collector._categorize_cicd_content(jenkins_file, "/Jenkinsfile") == "jenkins_pipeline"

    def test_detects_deployment_patterns(self):
        """Test detection of specific deployment patterns in CI/CD"""
        docker_deploy = "docker build -t app:latest .\ndocker push app:latest"
        k8s_deploy = "kubectl apply -f deployment.yaml\nkubectl rollout status deployment/app"
        terraform_deploy = "terraform plan\nterraform apply -auto-approve"
        
        assert "docker" in self.collector._extract_deployment_patterns(docker_deploy)
        assert "kubernetes" in self.collector._extract_deployment_patterns(k8s_deploy)
        assert "terraform" in self.collector._extract_deployment_patterns(terraform_deploy)

    def test_generates_appropriate_prompts(self):
        """Test prompt generation for different CI/CD content types"""
        github_content = "name: Deploy\non:\n  push:\njobs:\n  deploy:\n    runs-on: ubuntu-latest"
        gitlab_content = "stages:\n  - deploy\ndeploy:\n  script:\n    - kubectl apply -f ."
        
        github_prompt = self.collector._generate_cicd_prompt(github_content, "github_actions_workflow", "yaml")
        gitlab_prompt = self.collector._generate_cicd_prompt(gitlab_content, "gitlab_ci_pipeline", "yaml")
        
        assert "github actions" in github_prompt.lower()
        assert "gitlab ci" in gitlab_prompt.lower()
        assert len(github_prompt) > 20
        assert len(gitlab_prompt) > 20

    def test_creates_valid_jsonl_examples(self):
        """Test creation of valid JSONL training examples"""
        content = """
name: CI/CD Pipeline
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: docker build -t app:latest .
"""
        example = self.collector._create_cicd_example(content, "/.github/workflows/ci.yml", "kubernetes/kubernetes")
        
        # Should create valid example structure
        assert example is not None
        assert "prompt" in example
        assert "completion" in example
        assert "metadata" in example
        
        # Metadata should include required fields
        metadata = example["metadata"]
        assert metadata["source"] == "real_production_repository"
        assert metadata["category"] == "github_actions_workflow"
        assert metadata["authentic"] == True
        assert metadata["language"] == "yaml"
        assert metadata["source_repo"] == "kubernetes/kubernetes"

    def test_deduplicates_examples(self):
        """Test deduplication prevents duplicate training examples"""
        content1 = "name: CI\non: push\njobs:\n  build:\n    runs-on: ubuntu-latest"
        content2 = "name: CI\non: push\njobs:\n  build:\n    runs-on: ubuntu-latest"  # Identical
        content3 = "name: Deploy\non: push\njobs:\n  deploy:\n    runs-on: ubuntu-latest"  # Different
        
        example1 = self.collector._create_cicd_example(content1, "/.github/workflows/ci.yml")
        example2 = self.collector._create_cicd_example(content2, "/.github/workflows/ci2.yml")
        example3 = self.collector._create_cicd_example(content3, "/.github/workflows/deploy.yml")
        
        # First example should be created
        assert example1 is not None
        # Second (duplicate) should be rejected
        assert example2 is None
        # Third (different) should be created
        assert example3 is not None

    def test_collects_from_target_repositories(self):
        """Test collection targets correct high-star repositories"""
        target_repos = self.collector.target_repos
        
        # Should include major CI/CD repositories
        production_repos = [
            'actions/starter-workflows',
            'kubernetes/kubernetes',
            'docker/build-push-action',
            'hashicorp/terraform',
            'prometheus/prometheus'
        ]
        
        for repo in production_repos:
            assert repo in target_repos, f"Missing production repository: {repo}"

    def test_identifies_pipeline_complexity(self):
        """Test identification of pipeline complexity levels"""
        simple_pipeline = "name: Simple\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n    - run: echo test"
        complex_pipeline = """
name: Complex Pipeline
on: [push, pull_request]
env:
  REGISTRY: ghcr.io
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [14, 16, 18]
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-node@v3
    - run: npm test
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: docker/build-push-action@v4
  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
    - run: kubectl apply -f k8s/
"""
        
        assert self.collector._determine_pipeline_complexity(simple_pipeline) == "basic"
        assert self.collector._determine_pipeline_complexity(complex_pipeline) == "enterprise"

    def test_saves_corpus_to_jsonl(self):
        """Test corpus saving to JSONL format"""
        # Create sample examples
        examples = [
            {
                "prompt": "Create a GitHub Actions workflow for Docker deployment",
                "completion": "```yaml\nname: Deploy\non:\n  push:\njobs:\n  deploy:\n    runs-on: ubuntu-latest\n```",
                "metadata": {
                    "source": "real_production_repository",
                    "category": "github_actions_workflow",
                    "authentic": True,
                    "language": "yaml"
                }
            }
        ]
        
        # Save corpus
        output_file = "test_cicd_corpus.jsonl"
        self.collector.save_corpus(examples, output_file)
        
        # Verify file was created and contains valid JSONL
        output_path = Path(self.temp_dir) / output_file
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["prompt"] == examples[0]["prompt"]
        assert parsed["metadata"]["authentic"] == True

    def test_collector_finds_minimum_examples(self):
        """Test collector can find minimum number of quality examples"""
        # This test will initially fail until collector is implemented
        examples = self.collector.collect_real_cicd_examples(target_count=15)
        
        # Should collect at least 15 examples
        assert len(examples) >= 15
        
        # All examples should be valid
        for example in examples:
            assert "prompt" in example
            assert "completion" in example
            assert "metadata" in example
            assert example["metadata"]["authentic"] == True

    def test_excludes_template_placeholders(self):
        """Test collector excludes obvious template/placeholder content"""
        template_content = """
name: Your App CI
on:
  push:
    branches: [ main ]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Replace this step
      run: echo "Add your build commands here"
"""
        
        # Should reject template content
        assert self.collector._is_quality_cicd_content(template_content) == False
        
        # Should not create example from template
        example = self.collector._create_cicd_example(template_content, "/.github/workflows/template.yml")
        assert example is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])