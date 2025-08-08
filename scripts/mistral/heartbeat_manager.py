#!/usr/bin/env python3
"""
Heartbeat Manager for Long-Running Processes
Provides regular S3 metadata updates during extended operations
"""

import json
import time
import threading
import boto3
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from botocore.config import Config

logger = logging.getLogger(__name__)

class HeartbeatManager:
    """Manages periodic heartbeat updates to S3 for long-running processes"""
    
    def __init__(self, s3_bucket: str, s3_key: str, interval_seconds: int = 180):
        """
        Initialize heartbeat manager
        
        Args:
            s3_bucket: S3 bucket for metadata storage
            s3_key: S3 key path for metadata file
            interval_seconds: Heartbeat interval (default 3 minutes)
        """
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.interval = interval_seconds
        self.running = False
        self.thread = None
        self.consecutive_failures = 0
        self.max_backoff = 300  # 5 minutes max backoff
        
        # S3 client with retry config
        config = Config(
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=10
        )
        self.s3_client = boto3.client('s3', config=config)
        
        # Current state
        self.metadata = {
            "status": "running",
            "phase": "unknown",
            "sub_phase": "",
            "current_operation": "",
            "progress": {},
            "last_update": "",
            "heartbeat_interval": interval_seconds
        }
        
        logger.info(f"Initialized heartbeat manager: bucket={s3_bucket}, key={s3_key}, interval={interval_seconds}s")
    
    def update_metadata(self, **kwargs):
        """Update metadata fields"""
        for key, value in kwargs.items():
            if key in self.metadata:
                self.metadata[key] = value
            elif key == "progress_update":
                # Special handling for progress updates
                self.metadata["progress"].update(value)
            else:
                logger.warning(f"Unknown metadata field: {key}")
    
    def _get_backoff_delay(self):
        """Calculate exponential backoff delay based on consecutive failures"""
        if self.consecutive_failures == 0:
            return self.interval
        
        # Exponential backoff: 2^failures * base_delay, capped at max_backoff
        backoff_delay = min(self.interval * (2 ** min(self.consecutive_failures - 1, 8)), self.max_backoff)
        return min(backoff_delay, self.max_backoff)
    
    def _heartbeat_loop(self):
        """Background heartbeat loop with resilient error handling"""
        while self.running:
            try:
                self._send_heartbeat()
                self.consecutive_failures = 0  # Reset on success
                time.sleep(self.interval)
            except Exception as e:
                self.consecutive_failures += 1
                delay = self._get_backoff_delay()
                
                logger.error(f"Heartbeat failed (attempt {self.consecutive_failures}): {e}")
                if self.consecutive_failures <= 3:
                    logger.info(f"Retrying heartbeat in {delay:.1f} seconds...")
                else:
                    logger.warning(f"Heartbeat failing repeatedly ({self.consecutive_failures} failures), backing off to {delay:.1f}s")
                
                # Continue running even if individual heartbeats fail
                time.sleep(delay)
    
    def _send_heartbeat(self):
        """Send single heartbeat to S3"""
        try:
            # Update timestamp
            self.metadata["last_update"] = datetime.now(timezone.utc).isoformat()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=self.s3_key,
                Body=json.dumps(self.metadata, indent=2),
                ContentType='application/json'
            )
            
            logger.debug(f"Heartbeat sent: {self.metadata['phase']}/{self.metadata['sub_phase']}")
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            raise
    
    def start(self):
        """Start heartbeat manager"""
        if self.running:
            logger.warning("Heartbeat manager already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()
        
        # Try to send initial heartbeat, but don't fail if it doesn't work
        try:
            self._send_heartbeat()
            logger.info("Heartbeat manager started successfully")
        except Exception as e:
            logger.warning(f"Initial heartbeat failed, but manager started: {e}")
            # The background thread will continue trying
    
    def stop(self):
        """Stop heartbeat manager"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        # Try to send final heartbeat, but don't fail if it doesn't work
        try:
            self._send_heartbeat()
            logger.info("Heartbeat manager stopped with final heartbeat")
        except Exception as e:
            logger.warning(f"Failed to send final heartbeat, but manager stopped: {e}")
    
    def update_phase(self, phase: str, sub_phase: str = "", operation: str = ""):
        """Update current phase information"""
        self.update_metadata(
            phase=phase,
            sub_phase=sub_phase,
            current_operation=operation
        )
        logger.info(f"Phase updated: {phase}/{sub_phase} - {operation}")
    
    def update_progress(self, **progress_data):
        """Update progress information"""
        current_progress = self.metadata.get("progress", {})
        current_progress.update(progress_data)
        self.update_metadata(progress=current_progress)
    
    def set_status(self, status: str):
        """Update overall status"""
        self.update_metadata(status=status)
        logger.info(f"Status updated: {status}")

class ProgressTracker:
    """Helper class for tracking progress during operations"""
    
    def __init__(self, heartbeat_manager: HeartbeatManager, operation_name: str, total_items: int):
        self.heartbeat = heartbeat_manager
        self.operation = operation_name
        self.total = total_items
        self.completed = 0
        self.start_time = time.time()
    
    def update(self, completed: int = None, increment: int = 1):
        """Update progress"""
        if completed is not None:
            self.completed = completed
        else:
            self.completed += increment
        
        # Calculate progress metrics
        elapsed = time.time() - self.start_time
        if self.completed > 0 and elapsed > 0:
            rate = self.completed / elapsed
            remaining = self.total - self.completed
            eta_seconds = remaining / rate if rate > 0 else 0
            eta = datetime.now(timezone.utc).timestamp() + eta_seconds
            eta_iso = datetime.fromtimestamp(eta, timezone.utc).isoformat()
        else:
            eta_iso = None
        
        # Update heartbeat with progress
        self.heartbeat.update_progress(
            current_operation=self.operation,
            completed_items=self.completed,
            total_items=self.total,
            completion_percentage=round((self.completed / self.total) * 100, 1) if self.total > 0 else 0,
            estimated_completion=eta_iso
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Mark as completed
        self.update(completed=self.total)

# Context manager for easy heartbeat management
class HeartbeatContext:
    """Context manager for automatic heartbeat lifecycle"""
    
    def __init__(self, s3_bucket: str, run_id: str, monitoring_prefix: str = "training-runs", **kwargs):
        s3_key = f"{monitoring_prefix}/{run_id}/metadata.json"
        self.heartbeat = HeartbeatManager(s3_bucket, s3_key, **kwargs)
    
    def __enter__(self):
        self.heartbeat.start()
        return self.heartbeat
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Set final status based on exception
        if exc_type is not None:
            self.heartbeat.set_status("failed")
            self.heartbeat.update_phase("error", "exception", str(exc_val))
        else:
            self.heartbeat.set_status("completed")
        
        self.heartbeat.stop()

# Export classes
__all__ = ['HeartbeatManager', 'ProgressTracker', 'HeartbeatContext']