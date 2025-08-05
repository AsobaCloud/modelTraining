#!/usr/bin/env python3
"""
Data Validation Failure Tests
Tests that should have been written BEFORE the validation failure surprised us
"""

import json
import tempfile
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project paths
sys.path.append(str(Path(__file__).parent.parent / "scripts" / "mistral"))
sys.path.append(str(Path(__file__).parent.parent / "scripts" / "shared"))

from prepare_mistral_dataset import DatasetPipeline
from dataset_utils import validate_dataset_format
from heartbeat_manager import HeartbeatManager

class TestDataValidationFailures:
    """Test all the validation failures that should have been anticipated"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.work_dir = Path(self.temp_dir) / "test_work"
        self.s3_bucket = "test-bucket"
        self.s3_prefix = "test-prefix"
        
    def create_test_jsonl(self, records):
        """Helper to create test JSONL files"""
        test_file = self.work_dir / "test.jsonl"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        return test_file
    
    def test_missing_text_field_validation_failure(self):
        """Test: Records missing 'text' field should be handled gracefully"""
        
        # This is the EXACT failure we just experienced
        malformed_records = [
            {"source": "test1.pdf", "domain": "policy_analysis"},  # Missing 'text'
            {"text": "", "source": "test2.pdf"},  # Empty 'text'
            {"text": None, "source": "test3.pdf"},  # Null 'text'
        ]
        
        test_file = self.create_test_jsonl(malformed_records)
        
        # This should NOT crash the pipeline
        with patch('boto3.client'):
            pipeline = DatasetPipeline(
                work_dir=str(self.work_dir),
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
                run_id="test-validation-failure"
            )
            
            # Mock heartbeat to capture failure status
            mock_heartbeat = MagicMock()
            pipeline.heartbeat = mock_heartbeat
            
            # Validation should fail gracefully
            is_valid = validate_dataset_format(str(test_file))
            
            # REQUIREMENT: Should return False, not crash
            assert is_valid == False
            
            # REQUIREMENT: Should update monitoring with failure details
            mock_heartbeat.set_status.assert_called_with("failed")
            mock_heartbeat.update_phase.assert_called_with("data_prep", "error", "Validation failed: Missing or empty 'text' field")
    
    def test_mixed_valid_invalid_records(self):
        """Test: Mix of valid and invalid records should process valid ones"""
        
        mixed_records = [
            {"text": "Valid record 1", "source": "valid1.pdf"},
            {"source": "invalid1.pdf"},  # Missing text
            {"text": "Valid record 2", "source": "valid2.pdf"},
            {"text": "", "source": "invalid2.pdf"},  # Empty text
            {"text": "Valid record 3", "source": "valid3.pdf"},
        ]
        
        test_file = self.create_test_jsonl(mixed_records)
        
        # Should process valid records and skip invalid ones
        with patch('boto3.client'):
            pipeline = DatasetPipeline(
                work_dir=str(self.work_dir),
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix
            )
            
            # TODO: Implement logic to skip invalid records instead of failing
            # This test documents the EXPECTED behavior we should implement
            
            # REQUIREMENT: Should process 3 valid records
            # REQUIREMENT: Should log 2 skipped invalid records
            # REQUIREMENT: Should complete successfully with warnings
            
            # Currently this would fail - which is the problem
            
    def test_completely_invalid_dataset(self):
        """Test: Dataset with zero valid records should fail with clear message"""
        
        all_invalid = [
            {"source": "invalid1.pdf"},  # Missing text
            {"source": "invalid2.pdf"},  # Missing text
            {"text": "", "source": "invalid3.pdf"},  # Empty text
        ]
        
        test_file = self.create_test_jsonl(all_invalid)
        
        with patch('boto3.client'):
            pipeline = DatasetPipeline(
                work_dir=str(self.work_dir),
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
                run_id="test-all-invalid"
            )
            
            mock_heartbeat = MagicMock()
            pipeline.heartbeat = mock_heartbeat
            
            # Should fail validation
            is_valid = validate_dataset_format(str(test_file))
            assert is_valid == False
            
            # REQUIREMENT: Should provide actionable error message
            # REQUIREMENT: Should update monitoring with specific failure reason
            mock_heartbeat.set_status.assert_called_with("failed")
            
            # Should indicate it's a data quality issue, not infrastructure
            error_call = mock_heartbeat.update_phase.call_args[0]
            assert "validation" in error_call[1].lower()
            assert "text" in error_call[2].lower()
    
    def test_validation_failure_monitoring_integration(self):
        """Test: Validation failures should immediately update monitoring status"""
        
        # Mock S3 client for heartbeat
        with patch('boto3.client') as mock_s3:
            mock_s3_instance = MagicMock()
            mock_s3.return_value = mock_s3_instance
            
            # Create pipeline with monitoring
            pipeline = DatasetPipeline(
                work_dir=str(self.work_dir),
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
                run_id="test-monitoring-integration"
            )
            
            # Simulate validation failure
            invalid_records = [{"source": "test.pdf"}]  # Missing text
            test_file = self.create_test_jsonl(invalid_records)
            
            # Start heartbeat
            if pipeline.heartbeat:
                pipeline.heartbeat.start()
            
            # Validation should fail AND update monitoring
            is_valid = validate_dataset_format(str(test_file))
            assert is_valid == False
            
            # REQUIREMENT: S3 should have been called to update status
            # This verifies the monitoring integration works
            assert mock_s3_instance.put_object.called
            
            # Check that error status was written
            put_calls = mock_s3_instance.put_object.call_args_list
            error_update = None
            for call in put_calls:
                if 'failed' in str(call):
                    error_update = call
                    break
            
            assert error_update is not None, "Monitoring should have been updated with failure status"
    
    def test_s3_monitoring_failure_fallback(self):
        """Test: If monitoring update fails, pipeline should still report error locally"""
        
        with patch('boto3.client') as mock_s3:
            # Make S3 calls fail
            mock_s3_instance = MagicMock()
            mock_s3_instance.put_object.side_effect = Exception("S3 unavailable")
            mock_s3.return_value = mock_s3_instance
            
            pipeline = DatasetPipeline(
                work_dir=str(self.work_dir),
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
                run_id="test-s3-failure"
            )
            
            # Even if S3 monitoring fails, should still handle validation failure gracefully
            invalid_records = [{"source": "test.pdf"}]
            test_file = self.create_test_jsonl(invalid_records)
            
            # Should not crash even if monitoring update fails
            is_valid = validate_dataset_format(str(test_file))
            assert is_valid == False
            
            # REQUIREMENT: Should log error locally as fallback
            # REQUIREMENT: Should not crash due to monitoring failure
    
    def test_partial_download_validation_recovery(self):
        """Test: If download is interrupted, partial data should be validated correctly"""
        
        # Simulate scenario where download was interrupted
        partial_records = [
            {"text": "Complete record", "source": "complete.pdf"},
            {"text": "Partial rec",  # Truncated during download
        ]
        
        test_file = self.create_test_jsonl(partial_records)
        
        # Should handle malformed JSON gracefully
        # This tests recovery from network/download issues
        
        # REQUIREMENT: Should process valid records
        # REQUIREMENT: Should skip/report malformed entries
        # REQUIREMENT: Should not crash entire pipeline
        
    def test_large_dataset_validation_performance(self):
        """Test: Validation should work efficiently on large datasets"""
        
        # Create large dataset to test validation performance
        large_dataset = []
        for i in range(10000):
            if i % 1000 == 0:  # 1% invalid records
                large_dataset.append({"source": f"invalid_{i}.pdf"})  # Missing text
            else:
                large_dataset.append({
                    "text": f"Sample text content {i}",
                    "source": f"valid_{i}.pdf"
                })
        
        test_file = self.create_test_jsonl(large_dataset)
        
        # Should complete validation in reasonable time
        import time
        start_time = time.time()
        
        is_valid = validate_dataset_format(str(test_file))
        
        validation_time = time.time() - start_time
        
        # REQUIREMENT: Should complete validation in under 30 seconds
        assert validation_time < 30, f"Validation took {validation_time:.2f}s, should be < 30s"
        
        # REQUIREMENT: Should fail due to invalid records
        assert is_valid == False

class TestMonitoringFailureScenarios:
    """Test monitoring system behavior during various failure scenarios"""
    
    def test_heartbeat_survives_validation_failure(self):
        """Test: Heartbeat should continue working even when validation fails"""
        
        with patch('boto3.client') as mock_s3:
            mock_s3_instance = MagicMock()
            mock_s3.return_value = mock_s3_instance
            
            heartbeat = HeartbeatManager("test-bucket", "test-key", interval_seconds=1)
            heartbeat.start()
            
            # Simulate validation failure
            heartbeat.set_status("failed")
            heartbeat.update_phase("data_prep", "error", "Validation failed")
            
            # Heartbeat should still be running
            assert heartbeat.running == True
            
            # Should have sent error status
            heartbeat.stop()
            
            # Verify S3 was called with error status
            put_calls = mock_s3_instance.put_object.call_args_list
            assert len(put_calls) > 0
            
            # Check final status was 'failed'
            final_call = put_calls[-1]
            body_data = json.loads(final_call[1]['Body'])
            assert body_data['status'] == 'failed'
    
    def test_monitoring_captures_all_error_types(self):
        """Test: Different error types should be captured distinctly in monitoring"""
        
        error_scenarios = [
            ("validation_error", "data_prep", "error", "Missing text fields"),
            ("network_error", "data_prep", "error", "S3 connection timeout"),
            ("disk_full", "data_prep", "error", "No space left on device"),
            ("permission_error", "data_prep", "error", "Access denied to S3 bucket"),
        ]
        
        for error_type, phase, sub_phase, message in error_scenarios:
            with patch('boto3.client') as mock_s3:
                mock_s3_instance = MagicMock()
                mock_s3.return_value = mock_s3_instance
                
                heartbeat = HeartbeatManager("test-bucket", f"test-{error_type}", interval_seconds=1)
                heartbeat.start()
                
                # Simulate specific error
                heartbeat.set_status("failed")
                heartbeat.update_phase(phase, sub_phase, message)
                heartbeat.stop()
                
                # Verify error was captured with correct details
                put_calls = mock_s3_instance.put_object.call_args_list
                final_call = put_calls[-1]
                body_data = json.loads(final_call[1]['Body'])
                
                assert body_data['status'] == 'failed'
                assert body_data['phase'] == phase
                assert body_data['sub_phase'] == sub_phase
                assert message in body_data['current_operation']

if __name__ == "__main__":
    # Run the tests that would have caught our validation failure
    pytest.main([__file__, "-v"])