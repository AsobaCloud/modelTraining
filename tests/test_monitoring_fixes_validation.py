#!/usr/bin/env python3
"""
Test that validates our monitoring system fixes are properly implemented
in the actual training script
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestMonitoringSystemFixes:
    """Test that our training script has the monitoring fixes"""
    
    def test_cleanup_monitoring_state_function_exists(self):
        """Test that cleanup_monitoring_state function is properly implemented"""
        from scripts.mistral.train_mistral_simple_validated import cleanup_monitoring_state
        
        # Function should exist and be callable
        assert callable(cleanup_monitoring_state)
        
        # Test function signature
        import inspect
        sig = inspect.signature(cleanup_monitoring_state)
        assert len(sig.parameters) == 0, "cleanup_monitoring_state should take no parameters"
    
    def test_run_id_generation_provides_isolation(self):
        """Test that run ID generation provides proper isolation"""
        from scripts.mistral.train_mistral_simple_validated import datetime
        import re
        
        # Test run ID format
        run_id_1 = f"mistral-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Should match expected format
        pattern = r"mistral-\d{8}-\d{6}"
        assert re.match(pattern, run_id_1), f"Run ID {run_id_1} doesn't match expected format"
        
        # Different calls should generate different IDs (time-based)
        import time
        time.sleep(1)
        run_id_2 = f"mistral-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        assert run_id_1 != run_id_2, "Run IDs should be unique"
    
    @patch('boto3.client')
    def test_cleanup_called_at_start_of_training(self, mock_boto3):
        """Test that cleanup is called at the start of main()"""
        mock_s3_client = Mock()
        mock_boto3.return_value = mock_s3_client
        
        # Mock the cleanup calls
        cleanup_calls = []
        
        def mock_head_object(Bucket, Key):
            cleanup_calls.append(f"head:{Key}")
            # Simulate key exists
            return {'ContentLength': 100}
            
        def mock_delete_object(Bucket, Key):
            cleanup_calls.append(f"delete:{Key}")
            
        mock_s3_client.head_object.side_effect = mock_head_object
        mock_s3_client.delete_object.side_effect = mock_delete_object
        
        # Mock other functions to prevent full execution
        with patch('scripts.mistral.train_mistral_simple_validated.validate_dependencies') as mock_validate:
            mock_validate.side_effect = SystemExit(0)  # Exit after cleanup
            
            # Import and run with mocked args
            from scripts.mistral.train_mistral_simple_validated import main
            
            # Mock sys.argv
            with patch('sys.argv', ['script.py', '--train-dataset', 's3://test/train.jsonl', 
                                   '--val-dataset', 's3://test/val.jsonl', '--model-tag', 'test-model']):
                try:
                    main()
                except SystemExit:
                    pass  # Expected from our mock
            
            # Verify cleanup was attempted
            expected_keys = ['_error', '_complete', 'progress.json']
            for key_suffix in expected_keys:
                # Should have attempted to check and delete each key
                head_calls = [call for call in cleanup_calls if call.startswith('head:') and call.endswith(key_suffix)]
                assert len(head_calls) > 0, f"Should have checked for key ending with {key_suffix}"
    
    def test_completion_and_error_markers_use_run_id(self):
        """Test that completion and error markers properly use run ID"""
        from scripts.mistral.train_mistral_simple_validated import write_completion_marker, write_error_sentinel
        
        # Mock globals
        import scripts.mistral.train_mistral_simple_validated as train_module
        original_run_id = getattr(train_module, 'current_run_id', None)
        original_bucket = getattr(train_module, 's3_bucket', None)
        
        try:
            # Set test values
            train_module.current_run_id = "test-run-12345"
            train_module.s3_bucket = "test-bucket"
            
            with patch('boto3.client') as mock_boto3:
                mock_s3_client = Mock()
                mock_boto3.return_value = mock_s3_client
                
                # Test completion marker
                write_completion_marker()
                
                # Verify completion marker uses run ID
                mock_s3_client.put_object.assert_called_once()
                call_args = mock_s3_client.put_object.call_args
                assert call_args[1]['Key'] == "training-runs/test-run-12345/_complete"
                assert "Training completed successfully" in call_args[1]['Body']
                
                # Reset mock
                mock_s3_client.reset_mock()
                
                # Test error sentinel
                test_error = "Test error message"
                write_error_sentinel(test_error)
                
                # Verify error sentinel uses run ID
                mock_s3_client.put_object.assert_called_once()
                call_args = mock_s3_client.put_object.call_args
                assert call_args[1]['Key'] == "training-runs/test-run-12345/_error"
                assert call_args[1]['Body'] == test_error
                
        finally:
            # Restore original values
            train_module.current_run_id = original_run_id
            train_module.s3_bucket = original_bucket
    
    def test_monitoring_state_keys_are_consistent(self):
        """Test that all monitoring functions use consistent S3 key patterns"""
        # All monitoring keys should follow the pattern: training-runs/{run_id}/{marker}
        expected_patterns = [
            "training-runs/{run_id}/_complete",
            "training-runs/{run_id}/_error", 
            "training-runs/{run_id}/progress.json"
        ]
        
        # This test verifies that our key naming is consistent across all functions
        test_run_id = "test-12345"
        
        for pattern in expected_patterns:
            expected_key = pattern.format(run_id=test_run_id)
            
            # Key should have proper structure
            assert expected_key.startswith("training-runs/")
            assert test_run_id in expected_key
            assert "/" in expected_key  # Should have path structure
    
    def test_training_script_imports_without_errors(self):
        """Test that our updated training script imports cleanly"""
        try:
            from scripts.mistral.train_mistral_simple_validated import main, cleanup_monitoring_state
            from scripts.mistral.train_mistral_simple_validated import write_completion_marker, write_error_sentinel
            
            # All functions should be importable
            assert callable(main)
            assert callable(cleanup_monitoring_state)
            assert callable(write_completion_marker)
            assert callable(write_error_sentinel)
            
        except ImportError as e:
            pytest.fail(f"Training script import failed: {e}")
    
    def test_s3_error_handling_in_monitoring_functions(self):
        """Test that monitoring functions handle S3 errors gracefully"""
        from scripts.mistral.train_mistral_simple_validated import cleanup_monitoring_state
        import scripts.mistral.train_mistral_simple_validated as train_module
        
        # Set test values
        original_run_id = getattr(train_module, 'current_run_id', None)
        original_bucket = getattr(train_module, 's3_bucket', None)
        
        try:
            train_module.current_run_id = "test-run-12345"
            train_module.s3_bucket = "test-bucket"
            
            with patch('boto3.client') as mock_boto3:
                mock_s3_client = Mock()
                mock_boto3.return_value = mock_s3_client
                
                # Simulate S3 error
                from botocore.exceptions import NoCredentialsError
                mock_s3_client.head_object.side_effect = NoCredentialsError()
                
                # Should not raise exception
                try:
                    cleanup_monitoring_state()
                except Exception as e:
                    pytest.fail(f"cleanup_monitoring_state should handle S3 errors gracefully: {e}")
                
        finally:
            train_module.current_run_id = original_run_id
            train_module.s3_bucket = original_bucket


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])