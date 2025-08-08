#!/usr/bin/env python3
"""
Test the provenance generation heredoc specifically
"""
import subprocess
import tempfile
from pathlib import Path
import json

def test_provenance_heredoc_directly():
    """Test just the provenance generation heredoc from the resolver"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "config.json"
        source_file = Path(temp_dir) / ".source_s3"
        
        # Create the files the provenance script expects
        config_file.write_text('{"model_type": "test"}')
        source_file.write_text("s3://test-bucket/test-model/")
        
        # Extract and test just the provenance generation part from the resolver
        script = f'''
export TAG="test-model"
CONFIG="{config_file}"

# This is the exact code from the resolver script
cat > /tmp/gen_manifest.py <<'EOF'
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
EOF

python3 /tmp/gen_manifest.py "$CONFIG"
rm -f /tmp/gen_manifest.py
'''
        
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True
        )
        
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")  
        print(f"Return code: {result.returncode}")
        
        if result.returncode != 0:
            print("PROVENANCE GENERATION FAILED!")
            assert False, f"Provenance generation failed: {result.stderr}"
        
        # Check if manifest was created
        manifest_file = Path(temp_dir) / ".manifest.json"
        assert manifest_file.exists(), "Manifest file not created"
        
        manifest = json.loads(manifest_file.read_text())
        assert manifest["tag"] == "test-model"
        
        print("✅ Provenance generation works!")

if __name__ == "__main__":
    test_provenance_heredoc_directly()