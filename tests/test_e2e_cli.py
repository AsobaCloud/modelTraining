# E2E CLI tests - no mocks, real entrypoint
import os, shlex, subprocess, pytest

# Configuration from environment
E2E_CMD = os.getenv("E2E_CMD", "python3 scripts/preflight_check.py --instance-id {instance_id} --instance-ip {instance_ip} --ssh-key {ssh_key}")
E2E_GOOD_CONFIG = os.getenv("E2E_GOOD_CONFIG", "i-01fa5b57d64c6196a 54.197.142.172 config/mistral-base.pem")
E2E_BAD_CONFIG = os.getenv("E2E_BAD_CONFIG", "i-invalid 192.168.1.1 nonexistent.pem")

need_config = pytest.mark.skipif(
    not all([E2E_CMD, E2E_GOOD_CONFIG, E2E_BAD_CONFIG]), 
    reason="set E2E_CMD, E2E_GOOD_CONFIG, E2E_BAD_CONFIG environment variables"
)

@need_config
def test_e2e_preflight_valid():
    """E2E test with valid configuration should pass"""
    if E2E_GOOD_CONFIG == "i-01fa5b57d64c6196a 54.197.142.172 config/mistral-base.pem":
        pytest.skip("Using default config - requires real instance access")
    
    parts = E2E_GOOD_CONFIG.split()
    cmd = E2E_CMD.format(instance_id=parts[0], instance_ip=parts[1], ssh_key=parts[2])
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=120)
    
    # Should exit 0 for valid config
    assert result.returncode == 0, f"Valid config failed: {result.stderr}"

@need_config  
def test_e2e_preflight_invalid():
    """E2E test with invalid configuration should fail"""
    parts = E2E_BAD_CONFIG.split()
    cmd = E2E_CMD.format(instance_id=parts[0], instance_ip=parts[1], ssh_key=parts[2])
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=60)
    
    # Should exit non-zero for invalid config
    assert result.returncode != 0, f"Invalid config should have failed but passed: {result.stdout}"

def test_e2e_deployment_script_exists():
    """E2E test that deployment script exists and is executable"""
    import pathlib
    
    script_path = pathlib.Path("scripts/deploy_with_validation.sh")
    assert script_path.exists(), "Deployment script not found"
    assert script_path.stat().st_mode & 0o111, "Deployment script is not executable"

def test_e2e_validation_framework_complete():
    """E2E test that validation framework components exist"""
    import pathlib
    
    required_files = [
        "scripts/preflight_check.py",
        "scripts/test_runner.py", 
        "docs/complete-pipeline-test-matrix.md",
        "tests/test_data_validation_failures.py"
    ]
    
    for file_path in required_files:
        path = pathlib.Path(file_path)
        assert path.exists(), f"Required validation component missing: {file_path}"