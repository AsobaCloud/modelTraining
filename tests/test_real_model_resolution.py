#!/usr/bin/env python3
"""
REAL tests for model resolution - actually test the functionality
"""
import os
import json
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest
import boto3
from moto.s3 import mock_s3

SCRIPT_DIR = Path(__file__).parent.parent
RESOLVE_SCRIPT = SCRIPT_DIR / "scripts" / "resolve_model.sh"
TRAINING_SCRIPT = SCRIPT_DIR / "scripts" / "mistral" / "train_mistral_simple.py"

class TestIntModelResolution:
    """Integration tests that actually test S3 operations"""
    
    @mock_s3
    def test_int_resolve_trained_model_wins(self):
        """Integration: When both model locations exist, trained-models/ should win"""
        # Setup real S3 mock with actual files
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='test-bucket')
        
        # Create files in both locations
        s3.put_object(Bucket='test-bucket', Key='trained-models/mistral-7b-v0.3/config.json', 
                     Body=json.dumps({"model_type": "mistral", "trained": True}))
        s3.put_object(Bucket='test-bucket', Key='trained-models/mistral-7b-v0.3/model.safetensors', 
                     Body=b'fake_model_weights_trained')
        
        s3.put_object(Bucket='test-bucket', Key='models/mistral-7b-v0.3/config.json',
                     Body=json.dumps({"model_type": "mistral", "trained": False}))
        s3.put_object(Bucket='test-bucket', Key='models/mistral-7b-v0.3/model.safetensors',
                     Body=b'fake_model_weights_base')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run resolver
            env = os.environ.copy()
            env['AWS_ACCESS_KEY_ID'] = 'testing'
            env['AWS_SECRET_ACCESS_KEY'] = 'testing'
            
            result = subprocess.run([
                str(RESOLVE_SCRIPT),
                '--tag', 'mistral-7b-v0.3',
                '--bucket', 'test-bucket',
                '--region', 'us-east-1',
                '--local-root', temp_dir
            ], capture_output=True, text=True, env=env)
            
            assert result.returncode == 0, f"Resolver failed: {result.stderr}"
            
            # Verify trained model was chosen
            source_file = Path(temp_dir) / 'mistral-7b-v0.3' / '.source_s3'
            assert source_file.exists()
            source = source_file.read_text().strip()
            assert 'trained-models' in source
            
            # Verify actual files were downloaded
            config_file = Path(temp_dir) / 'mistral-7b-v0.3' / 'config.json'
            assert config_file.exists()
            config = json.loads(config_file.read_text())
            assert config['trained'] is True  # Trained model, not base model
    
    @mock_s3 
    def test_int_resolve_base_model_fallback(self):
        """Integration: When only base model exists, should use models/"""
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='test-bucket')
        
        # Only create base model
        s3.put_object(Bucket='test-bucket', Key='models/mistral-7b-v0.3/config.json',
                     Body=json.dumps({"model_type": "mistral"}))
        s3.put_object(Bucket='test-bucket', Key='models/mistral-7b-v0.3/model.safetensors',
                     Body=b'fake_model_weights')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env['AWS_ACCESS_KEY_ID'] = 'testing'
            env['AWS_SECRET_ACCESS_KEY'] = 'testing'
            
            result = subprocess.run([
                str(RESOLVE_SCRIPT),
                '--tag', 'mistral-7b-v0.3', 
                '--bucket', 'test-bucket',
                '--region', 'us-east-1',
                '--local-root', temp_dir
            ], capture_output=True, text=True, env=env)
            
            assert result.returncode == 0
            
            # Verify base model source
            source_file = Path(temp_dir) / 'mistral-7b-v0.3' / '.source_s3'
            source = source_file.read_text().strip()
            assert 'models/mistral-7b-v0.3' in source
            assert 'trained-models' not in source
    
    @mock_s3
    def test_int_resolve_fails_no_models(self):
        """Integration: Should fail cleanly when no models exist"""
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='test-bucket')
        # Don't create any model files
        
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env['AWS_ACCESS_KEY_ID'] = 'testing'
            env['AWS_SECRET_ACCESS_KEY'] = 'testing'
            
            result = subprocess.run([
                str(RESOLVE_SCRIPT),
                '--tag', 'nonexistent-model',
                '--bucket', 'test-bucket', 
                '--region', 'us-east-1',
                '--local-root', temp_dir
            ], capture_output=True, text=True, env=env)
            
            assert result.returncode == 1  # Should fail
            assert 'No model found' in result.stderr
    
    @mock_s3
    def test_int_model_verification_catches_incomplete_sync(self):
        """Integration: Should fail when model files are incomplete"""
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='test-bucket')
        
        # Create config but no weights - incomplete model
        s3.put_object(Bucket='test-bucket', Key='models/mistral-7b-v0.3/config.json',
                     Body=json.dumps({"model_type": "mistral"}))
        # Missing model.safetensors intentionally
        
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env['AWS_ACCESS_KEY_ID'] = 'testing'
            env['AWS_SECRET_ACCESS_KEY'] = 'testing'
            
            result = subprocess.run([
                str(RESOLVE_SCRIPT),
                '--tag', 'mistral-7b-v0.3',
                '--bucket', 'test-bucket',
                '--region', 'us-east-1', 
                '--local-root', temp_dir
            ], capture_output=True, text=True, env=env)
            
            assert result.returncode == 3  # Verification error
            assert 'No weight files' in result.stderr

class TestIntTrainingScriptIntegration:
    """Integration tests for training script model resolution"""
    
    def test_int_training_script_uses_resolved_path(self):
        """Integration: Training script should call resolver and use returned path"""
        
        with patch('subprocess.run') as mock_subprocess:
            # Mock successful resolver execution
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stderr = ""
            
            # Import the function to test it directly
            import sys
            sys.path.append(str(TRAINING_SCRIPT.parent))
            from train_mistral_simple import resolve_model_source
            
            result_path = resolve_model_source(
                model_tag='test-model',
                bucket='test-bucket',
                region='us-west-2'
            )
            
            # Verify subprocess was called correctly
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]  # First positional arg (command list)
            
            assert '--tag' in call_args
            assert 'test-model' in call_args
            assert '--bucket' in call_args  
            assert 'test-bucket' in call_args
            assert '--region' in call_args
            assert 'us-west-2' in call_args
            
            # Verify returned path
            assert result_path == '/mnt/training/models/test-model'
    
    def test_int_training_script_handles_resolver_failure(self):
        """Integration: Training script should handle resolver failures properly"""
        
        with patch('subprocess.run') as mock_subprocess:
            # Mock failed resolver execution
            mock_subprocess.return_value.returncode = 1
            mock_subprocess.return_value.stderr = "Model not found in S3"
            
            import sys
            sys.path.append(str(TRAINING_SCRIPT.parent))
            from train_mistral_simple import resolve_model_source
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Model resolution failed"):
                resolve_model_source('nonexistent-model')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])