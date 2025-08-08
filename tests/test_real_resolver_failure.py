#!/usr/bin/env python3
"""
Test the actual resolver script to reproduce the production failure
"""
import subprocess
import tempfile
from pathlib import Path
import pytest

def test_actual_resolver_script_fails_at_provenance():
    """Run the actual resolver script to reproduce the heredoc failure"""
    SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "resolve_model.sh"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a fake model directory with config.json to get past the verification
        model_dir = Path(temp_dir) / "test-model" 
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}')
        (model_dir / "model.safetensors").write_text("fake weights")
        
        # Run the actual resolver script - this should reproduce the production error
        result = subprocess.run([
            str(SCRIPT_PATH),
            "--tag", "test-model",
            "--local-root", temp_dir,
            "--bucket", "nonexistent-bucket"  # Will fail at S3, but we want to see heredoc error
        ], capture_output=True, text=True)
        
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        print(f"Return code: {result.returncode}")
        
        # This test should FAIL and show us the heredoc syntax error
        assert "here-document" in result.stderr or "SyntaxError" in result.stderr, \
            "Expected heredoc syntax error not found"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output