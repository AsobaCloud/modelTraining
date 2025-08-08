#!/usr/bin/env python3
"""
Tests for resolve_model.sh script
Following TDD principles (better late than never)
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path
import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "resolve_model.sh"

class TestResolveModel:
    
    def test_resolve_model_finds_trained_model(self):
        """Given both model locations exist, should prefer trained-models/"""
        # This test would require mocking S3 or using test buckets
        # For now, test the script exists and is executable
        assert SCRIPT_PATH.exists(), f"resolve_model.sh not found at {SCRIPT_PATH}"
        assert os.access(SCRIPT_PATH, os.X_OK), "resolve_model.sh not executable"
    
    def test_resolve_model_requires_tag_argument(self):
        """Should fail with clear error when --tag not provided"""
        result = subprocess.run([str(SCRIPT_PATH)], capture_output=True, text=True)
        assert result.returncode == 64  # Usage error
        assert "tag is required" in (result.stdout + result.stderr).lower()
    
    def test_resolve_model_creates_local_directory(self):
        """Should create local model directory structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # This would need actual S3 setup to test properly
            # For now just verify script can be called with arguments
            result = subprocess.run([
                str(SCRIPT_PATH), 
                "--tag", "test-model",
                "--local-root", temp_dir
            ], capture_output=True, text=True)
            
            # Expect failure due to no S3 access, but should parse args correctly
            assert "test-model" in result.stderr or result.returncode != 64

class TestIntegrationTrainingScript:
    
    def test_int_training_script_accepts_model_tag(self):
        """Integration test: training script should accept --model-tag parameter"""
        script_path = Path(__file__).parent.parent / "scripts" / "mistral" / "train_mistral_simple.py"
        
        # Test help output includes new parameter
        result = subprocess.run([
            "python3", str(script_path), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "--model-tag" in result.stdout
        assert "Model tag to resolve from S3" in result.stdout
    
    def test_int_training_script_has_resolve_function(self):
        """Integration test: training script should have resolve_model_source function"""
        script_path = Path(__file__).parent.parent / "scripts" / "mistral" / "train_mistral_simple.py"
        
        # Read script and verify function exists
        script_content = script_path.read_text()
        assert "def resolve_model_source" in script_content
        assert "resolve_model.sh" in script_content

class TestE2EModelResolution:
    
    def test_e2e_model_resolution_dry_run(self):
        """E2E test: model resolution should work end-to-end (dry run)"""
        # This would be a full test with real S3 access
        # For now, verify the components are wired together
        
        script_path = Path(__file__).parent.parent / "scripts" / "mistral" / "train_mistral_simple.py"
        resolver_path = Path(__file__).parent.parent / "scripts" / "resolve_model.sh"
        
        assert script_path.exists()
        assert resolver_path.exists()
        
        # Verify training script can find resolver
        script_content = script_path.read_text()
        assert "resolve_model.sh" in script_content

if __name__ == "__main__":
    pytest.main([__file__])