#!/usr/bin/env python3
"""
Tests for production monitoring system
Uses moto for S3 mocking
"""

import pytest
import json
import boto3
from datetime import datetime, timezone, timedelta
from moto import mock_s3
from core import TrainingMonitor, RunState


@mock_s3
def test_monitor_no_files():
    """Test monitoring run with no S3 files"""
    # Setup mock S3
    s3_client = boto3.client('s3', region_name='us-east-1')
    s3_client.create_bucket(Bucket='test-bucket')
    
    monitor = TrainingMonitor('test-bucket', 'us-east-1', stale_minutes=10)
    state = monitor.evaluate_run('test-run-123')
    
    assert state.status == "TIMED_OUT"
    assert "No heartbeat files found" in state.detail
    assert state.last_meta is None
    assert state.last_progress is None


@mock_s3
def test_monitor_error_sentinel():
    """Test monitoring run with error sentinel"""
    # Setup mock S3
    s3_client = boto3.client('s3', region_name='us-east-1')
    s3_client.create_bucket(Bucket='test-bucket')
    
    # Create error sentinel
    error_msg = "Training failed: CUDA out of memory"
    s3_client.put_object(
        Bucket='test-bucket',
        Key='training-runs/test-run-123/_error',
        Body=error_msg
    )
    
    monitor = TrainingMonitor('test-bucket', 'us-east-1', stale_minutes=10)
    state = monitor.evaluate_run('test-run-123')
    
    assert state.status == "FAILED"
    assert error_msg in state.detail


@mock_s3
def test_monitor_completion_marker():
    """Test monitoring run with completion marker"""
    # Setup mock S3
    s3_client = boto3.client('s3', region_name='us-east-1')
    s3_client.create_bucket(Bucket='test-bucket')
    
    # Create completion marker
    s3_client.put_object(
        Bucket='test-bucket',
        Key='training-runs/test-run-123/_complete',
        Body='Training completed successfully'
    )
    
    monitor = TrainingMonitor('test-bucket', 'us-east-1', stale_minutes=10)
    state = monitor.evaluate_run('test-run-123')
    
    assert state.status == "SUCCEEDED"
    assert "Completion marker found" in state.detail


@mock_s3
def test_monitor_fresh_heartbeats():
    """Test monitoring run with fresh heartbeats"""
    # Setup mock S3
    s3_client = boto3.client('s3', region_name='us-east-1')
    s3_client.create_bucket(Bucket='test-bucket')
    
    # Create fresh metadata
    now = datetime.now(timezone.utc)
    metadata = {
        "run_id": "test-run-123",
        "status": "running",
        "last_update": now.isoformat(),
        "phase": "training"
    }
    
    s3_client.put_object(
        Bucket='test-bucket',
        Key='training-runs/test-run-123/metadata.json',
        Body=json.dumps(metadata)
    )
    
    # Create fresh progress
    progress = {
        "step": 150,
        "loss": 1.23,
        "last_updated": now.isoformat()
    }
    
    s3_client.put_object(
        Bucket='test-bucket',
        Key='training-runs/test-run-123/progress.json',
        Body=json.dumps(progress)
    )
    
    monitor = TrainingMonitor('test-bucket', 'us-east-1', stale_minutes=10)
    state = monitor.evaluate_run('test-run-123')
    
    assert state.status == "RUNNING"
    assert state.detail == "Active"
    assert state.last_meta is not None
    assert state.last_progress is not None


@mock_s3
def test_monitor_stale_heartbeats():
    """Test monitoring run with stale heartbeats"""
    # Setup mock S3
    s3_client = boto3.client('s3', region_name='us-east-1')
    s3_client.create_bucket(Bucket='test-bucket')
    
    # Create stale metadata (20 minutes ago)
    stale_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    metadata = {
        "run_id": "test-run-123",
        "status": "running",
        "last_update": stale_time.isoformat(),
        "phase": "training"
    }
    
    s3_client.put_object(
        Bucket='test-bucket',
        Key='training-runs/test-run-123/metadata.json',
        Body=json.dumps(metadata)
    )
    
    monitor = TrainingMonitor('test-bucket', 'us-east-1', stale_minutes=10)
    state = monitor.evaluate_run('test-run-123')
    
    assert state.status == "TIMED_OUT"
    assert "Both heartbeats stale" in state.detail


@mock_s3
def test_state_transitions_and_alerts():
    """Test state transitions trigger alerts correctly"""
    # Setup mock S3
    s3_client = boto3.client('s3', region_name='us-east-1')
    s3_client.create_bucket(Bucket='test-bucket')
    
    monitor = TrainingMonitor('test-bucket', 'us-east-1', stale_minutes=10)
    
    # Mock alert manager
    alerts_sent = []
    
    class MockAlertManager:
        def send(self, run_id, new_state, reason):
            alerts_sent.append((run_id, new_state, reason))
            return True
    
    monitor.alert_manager = MockAlertManager()
    
    # First check - no files (should be TIMED_OUT)
    state1 = monitor.check_and_alert('test-run-123')
    assert state1.status == "TIMED_OUT"
    assert len(alerts_sent) == 1
    assert alerts_sent[0][1] == "TIMED_OUT"
    
    # Second check - same state (no new alert)
    state2 = monitor.check_and_alert('test-run-123')
    assert state2.status == "TIMED_OUT"
    assert len(alerts_sent) == 1  # No new alert
    
    # Add error sentinel - should trigger FAILED alert
    s3_client.put_object(
        Bucket='test-bucket',
        Key='training-runs/test-run-123/_error',
        Body='Process crashed'
    )
    
    state3 = monitor.check_and_alert('test-run-123')
    assert state3.status == "FAILED"
    assert len(alerts_sent) == 2
    assert alerts_sent[1][1] == "FAILED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])