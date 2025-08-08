#!/usr/bin/env python3
"""
Test the specific heredoc issue in resolver script
"""
import subprocess
import tempfile
import json
from pathlib import Path
import pytest

def test_python_heredoc_with_args_should_work():
    """Test that Python heredoc with arguments works correctly"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "config.json"
        source_file = Path(temp_dir) / ".source_s3"
        manifest_file = Path(temp_dir) / ".manifest.json"
        
        # Setup test files
        config_file.write_text('{"model_type": "test"}')
        source_file.write_text("s3://test-bucket/test-model/")
        
        # Test the exact heredoc pattern from the resolver
        script = f"""
export TAG="test-model"
python3 - "{config_file}" <<'PY'
import hashlib, json, os, sys
cfg = sys.argv[1]
local_dir = os.path.dirname(cfg)
sha = hashlib.sha256(open(cfg,'rb').read()).hexdigest()
manifest = {{
  "tag": os.environ["TAG"],
  "source_s3": open(os.path.join(local_dir,".source_s3")).read().strip(),
  "config_sha256": sha
}}
with open(os.path.join(local_dir,'.manifest.json'),'w') as f:
  json.dump(manifest,f,indent=2)
print(f"[INFO] Wrote provenance to {{local_dir}}/.manifest.json")
PY
"""
        
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            cwd=temp_dir
        )
        
        # This test will fail and show us exactly what's wrong
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert manifest_file.exists(), "Manifest file not created"
        
        # Verify manifest contents
        manifest = json.loads(manifest_file.read_text())
        assert manifest["tag"] == "test-model"
        assert "s3://test-bucket/test-model/" in manifest["source_s3"]
        assert "config_sha256" in manifest
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])