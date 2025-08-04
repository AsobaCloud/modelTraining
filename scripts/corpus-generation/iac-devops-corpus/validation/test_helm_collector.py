#!/usr/bin/env python3
"""
TDD Tests for Helm Chart Collector
Following CLAUDE.md principle: Write failing tests first, then implement
"""

import pytest
import tempfile
import json
from pathlib import Path
from helm_real_collector import HelmRealCollector

class TestHelmRealCollector:
    """Test cases for Helm chart collection following CLAUDE.md TDD approach"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = HelmRealCollector(output_dir=self.temp_dir)
    
    def test_helm_collector_initialization(self):
        """Test collector initializes with required repositories"""
        # Should target production Helm repositories
        assert 'bitnami/charts' in self.collector.target_repos
        assert 'prometheus-community/helm-charts' in self.collector.target_repos
        assert 'grafana/helm-charts' in self.collector.target_repos
        
        # Should have empty collected examples initially
        assert len(self.collector.examples) == 0
        assert len(self.collector.collected_hashes) == 0

    def test_identifies_quality_helm_content(self):
        """Test quality detection for Helm chart content"""
        # Valid Chart.yaml content
        valid_chart = """
apiVersion: v2
name: nginx
description: A Helm chart for nginx
version: 1.0.0
appVersion: 1.21.0
dependencies:
  - name: common
    version: 1.x.x
    repository: https://charts.bitnami.com/bitnami
"""
        assert self.collector._is_quality_helm_content(valid_chart) == True
        
        # Valid values.yaml content
        valid_values = """
replicaCount: 2
image:
  repository: nginx
  pullPolicy: IfNotPresent
  tag: "1.21.0"
service:
  type: ClusterIP
  port: 80
ingress:
  enabled: false
  annotations: {}
  hosts:
    - host: chart-example.local
      paths: []
resources:
  limits:
    cpu: 100m
    memory: 128Mi
  requests:
    cpu: 100m
    memory: 128Mi
autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 100
"""
        assert self.collector._is_quality_helm_content(valid_values) == True
        
        # Invalid content (too short)
        invalid_short = "name: test"
        assert self.collector._is_quality_helm_content(invalid_short) == False
        
        # Invalid content (template placeholder)
        invalid_template = """
name: your-app
description: Template for your application
"""
        assert self.collector._is_quality_helm_content(invalid_template) == False

    def test_categorizes_helm_content_correctly(self):
        """Test proper categorization of different Helm file types"""
        chart_yaml = "apiVersion: v2\nname: nginx\nversion: 1.0.0"
        values_yaml = "replicaCount: 2\nimage:\n  repository: nginx"
        template_yaml = "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {{ .Values.name }}"
        
        assert self.collector._categorize_helm_content(chart_yaml, "/charts/nginx/Chart.yaml") == "helm_chart_definition"
        assert self.collector._categorize_helm_content(values_yaml, "/charts/nginx/values.yaml") == "helm_values_config"
        assert self.collector._categorize_helm_content(template_yaml, "/charts/nginx/templates/deployment.yaml") == "helm_template"

    def test_generates_appropriate_prompts(self):
        """Test prompt generation for different Helm content types"""
        chart_content = "apiVersion: v2\nname: web-app\nversion: 1.0.0"
        values_content = "replicaCount: 3\nimage:\n  repository: nginx"
        
        chart_prompt = self.collector._generate_helm_prompt(chart_content, "helm_chart_definition", "yaml")
        values_prompt = self.collector._generate_helm_prompt(values_content, "helm_values_config", "yaml")
        
        assert "helm chart definition" in chart_prompt.lower()
        assert "values configuration" in values_prompt.lower()
        assert len(chart_prompt) > 20
        assert len(values_prompt) > 20

    def test_creates_valid_jsonl_examples(self):
        """Test creation of valid JSONL training examples"""
        content = """
apiVersion: v2
name: nginx
description: A Helm chart for nginx deployment
version: 1.0.0
appVersion: 1.21.0
"""
        example = self.collector._create_helm_example(content, "/charts/nginx/Chart.yaml", "bitnami/charts")
        
        # Should create valid example structure
        assert example is not None
        assert "prompt" in example
        assert "completion" in example
        assert "metadata" in example
        
        # Metadata should include required fields
        metadata = example["metadata"]
        assert metadata["source"] == "real_production_repository"
        assert metadata["category"] == "helm_chart_definition"
        assert metadata["authentic"] == True
        assert metadata["language"] == "yaml"
        assert metadata["source_repo"] == "bitnami/charts"

    def test_deduplicates_examples(self):
        """Test deduplication prevents duplicate training examples"""
        content1 = "apiVersion: v2\nname: test\nversion: 1.0.0"
        content2 = "apiVersion: v2\nname: test\nversion: 1.0.0"  # Identical
        content3 = "apiVersion: v2\nname: different\nversion: 1.0.0"  # Different
        
        example1 = self.collector._create_helm_example(content1, "/chart1/Chart.yaml")
        example2 = self.collector._create_helm_example(content2, "/chart2/Chart.yaml")
        example3 = self.collector._create_helm_example(content3, "/chart3/Chart.yaml")
        
        # First example should be created
        assert example1 is not None
        # Second (duplicate) should be rejected
        assert example2 is None
        # Third (different) should be created
        assert example3 is not None

    def test_collects_from_target_repositories(self):
        """Test collection targets correct high-star repositories"""
        # This test verifies the collector targets production repositories
        target_repos = self.collector.target_repos
        
        # Should include major Helm chart repositories
        production_repos = [
            'bitnami/charts',
            'prometheus-community/helm-charts',
            'grafana/helm-charts',
            'elastic/helm-charts',
            'kubernetes/ingress-nginx'
        ]
        
        for repo in production_repos:
            assert repo in target_repos, f"Missing production repository: {repo}"

    def test_saves_corpus_to_jsonl(self):
        """Test corpus saving to JSONL format"""
        # Create sample examples
        examples = [
            {
                "prompt": "Create a Helm chart for nginx deployment",
                "completion": "```yaml\napiVersion: v2\nname: nginx\n```",
                "metadata": {
                    "source": "real_production_repository",
                    "category": "helm_chart_definition",
                    "authentic": True,
                    "language": "yaml"
                }
            }
        ]
        
        # Save corpus
        output_file = "test_helm_corpus.jsonl"
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
        examples = self.collector.collect_real_helm_examples(target_count=10)
        
        # Should collect at least 10 examples
        assert len(examples) >= 10
        
        # All examples should be valid
        for example in examples:
            assert "prompt" in example
            assert "completion" in example
            assert "metadata" in example
            assert example["metadata"]["authentic"] == True

    def test_excludes_template_placeholders(self):
        """Test collector excludes obvious template/placeholder content"""
        template_content = """
apiVersion: v2
name: your-app
description: A template for your application
version: 0.1.0
"""
        
        # Should reject template content
        assert self.collector._is_quality_helm_content(template_content) == False
        
        # Should not create example from template
        example = self.collector._create_helm_example(template_content, "/template/Chart.yaml")
        assert example is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])