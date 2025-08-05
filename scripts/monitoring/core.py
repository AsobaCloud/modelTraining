#!/usr/bin/env python3
"""
Production Training Monitor Core
Detects silent failures, timeouts, and sends alerts
"""
import boto3
import json
import requests
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Literal, Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class RunState:
    run_id: str
    status: Literal["RUNNING", "SUCCEEDED", "FAILED", "TIMED_OUT"]
    last_meta: Optional[datetime]
    last_progress: Optional[datetime]
    detail: str = ""
    
    def to_dict(self):
        return {
            "run_id": self.run_id,
            "status": self.status,
            "last_meta": self.last_meta.isoformat() if self.last_meta else None,
            "last_progress": self.last_progress.isoformat() if self.last_progress else None,
            "detail": self.detail
        }


class AlertManager:
    """Handle alerts via Slack and/or SNS"""
    
    def __init__(self, slack_webhook: Optional[str] = None, sns_topic_arn: Optional[str] = None, 
                 region: str = "us-east-1", bucket: str = "asoba-llm-cache"):
        self.slack_webhook = slack_webhook
        self.sns_topic_arn = sns_topic_arn
        self.sns_client = boto3.client('sns', region_name=region) if sns_topic_arn else None
        self.bucket = bucket
    
    def send(self, run_id: str, new_state: str, reason: str, context: Optional[dict] = None) -> bool:
        """Send alert for state transition"""
        success = True
        
        if self.slack_webhook:
            success &= self._send_slack(run_id, new_state, reason, context)
        
        if self.sns_topic_arn:
            success &= self._send_sns(run_id, new_state, reason)
        
        return success
    
    def _send_slack(self, run_id: str, new_state: str, reason: str, context: Optional[dict] = None) -> bool:
        """Send Slack notification with detailed context"""
        try:
            emoji = "🔥" if new_state == "FAILED" else "⏰" if new_state == "TIMED_OUT" else "✅"
            color = "danger" if new_state in ["FAILED", "TIMED_OUT"] else "good"
            
            # Build actionable message
            action_text = "Unknown issue"
            if "CUDA out of memory" in reason:
                action_text = "Reduce batch size or sequence length"
            elif "heartbeats stale" in reason:
                action_text = "SSH to instance and check: training.log, nvidia-smi, disk space"
            elif "No heartbeat files" in reason:
                action_text = "Training never started - check setup phase logs"
            elif "Error sentinel" in reason:
                action_text = "Check error details in S3 _error file"
            
            fields = [
                {"title": "Run ID", "value": f"`{run_id}`", "short": True},
                {"title": "Status", "value": new_state, "short": True},
                {"title": "Reason", "value": reason, "short": False},
                {"title": "🔧 Action Required", "value": action_text, "short": False}
            ]
            
            # Add context if available
            if context:
                if context.get("instance_id"):
                    fields.append({"title": "Instance", "value": context["instance_id"], "short": True})
                if context.get("phase"):
                    fields.append({"title": "Phase", "value": context["phase"], "short": True})
                if context.get("duration_minutes"):
                    fields.append({"title": "Duration", "value": f"{context['duration_minutes']} minutes", "short": True})
                if context.get("last_log_line"):
                    fields.append({"title": "Last Log", "value": f"```{context['last_log_line']}```", "short": False})
            
            # Add helpful links
            s3_url = f"https://s3.console.aws.amazon.com/s3/buckets/{self.bucket}?prefix=training-runs/{run_id}/"
            fields.append({"title": "📊 S3 Logs", "value": f"<{s3_url}|View in S3 Console>", "short": False})
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"{emoji} Mistral Training {new_state}",
                    "fields": fields,
                    "ts": int(datetime.now().timestamp())
                }]
            }
            
            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Slack alert sent for {run_id}: {new_state}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _send_sns(self, run_id: str, new_state: str, reason: str) -> bool:
        """Send SNS notification"""
        try:
            message = {
                "run_id": run_id,
                "status": new_state,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Subject=f"Training Alert: {run_id} {new_state}",
                Message=json.dumps(message, indent=2)
            )
            
            logger.info(f"SNS alert sent for {run_id}: {new_state}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SNS alert: {e}")
            return False


class TrainingMonitor:
    """Production training monitor with failure detection"""
    
    def __init__(self, bucket: str, region: str = "us-east-1", stale_minutes: int = 10):
        self.bucket = bucket
        self.region = region
        self.stale_minutes = stale_minutes
        
        # Persistent S3 client with retries
        from botocore.config import Config
        config = Config(
            region_name=region,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        self.s3_client = boto3.client('s3', config=config)
        
        self.alert_manager = None
        self._last_states = {}  # Track state transitions
    
    def configure_alerts(self, slack_webhook: Optional[str] = None, sns_topic_arn: Optional[str] = None):
        """Configure alert destinations"""
        self.alert_manager = AlertManager(slack_webhook, sns_topic_arn, self.region, self.bucket)
    
    def _get_timestamp(self, run_id: str, filename: str) -> Optional[datetime]:
        """Get timestamp from S3 object metadata or content"""
        try:
            key = f"training-runs/{run_id}/{filename}"
            
            # Try to get object
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            
            # For JSON files, try to parse timestamp from content
            if filename.endswith('.json'):
                try:
                    content = json.loads(response['Body'].read().decode('utf-8'))
                    
                    # Look for timestamp fields
                    for field in ['last_update', 'last_updated', 'timestamp']:
                        if field in content:
                            return datetime.fromisoformat(content[field].replace('Z', '+00:00'))
                except:
                    pass
            
            # Fall back to S3 LastModified
            return response['LastModified']
            
        except self.s3_client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.warning(f"Failed to get timestamp for {run_id}/{filename}: {e}")
            return None
    
    def _exists(self, run_id: str, filename: str) -> bool:
        """Check if S3 object exists"""
        try:
            key = f"training-runs/{run_id}/{filename}"
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3_client.exceptions.NoSuchKey:
            return False
        except Exception as e:
            logger.warning(f"Failed to check existence of {run_id}/{filename}: {e}")
            return False
    
    def _is_stale(self, timestamp: Optional[datetime]) -> bool:
        """Check if timestamp is stale"""
        if not timestamp:
            return True
        
        now = datetime.now(timezone.utc)
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        minutes_ago = (now - timestamp).total_seconds() / 60
        return minutes_ago > self.stale_minutes
    
    def evaluate_run(self, run_id: str) -> RunState:
        """Evaluate current state of a training run"""
        
        # Check for explicit completion markers
        if self._exists(run_id, "_error"):
            try:
                # Try to get error details
                response = self.s3_client.get_object(
                    Bucket=self.bucket, 
                    Key=f"training-runs/{run_id}/_error"
                )
                error_detail = response['Body'].read().decode('utf-8').strip()
                detail = f"Error: {error_detail}" if error_detail else "Error sentinel found"
            except:
                detail = "Error sentinel found"
            
            return RunState(
                run_id=run_id,
                status="FAILED",
                last_meta=None,
                last_progress=None,
                detail=detail
            )
        
        if self._exists(run_id, "_complete"):
            return RunState(
                run_id=run_id,
                status="SUCCEEDED",
                last_meta=None,
                last_progress=None,
                detail="Completion marker found"
            )
        
        # Check heartbeats
        meta_ts = self._get_timestamp(run_id, "metadata.json")
        progress_ts = self._get_timestamp(run_id, "progress.json")
        
        # Determine if stale
        meta_stale = self._is_stale(meta_ts)
        progress_stale = self._is_stale(progress_ts)
        
        if meta_stale and progress_stale:
            detail = f"Both heartbeats stale (>{self.stale_minutes}m)"
            if not meta_ts and not progress_ts:
                detail = "No heartbeat files found"
            
            return RunState(
                run_id=run_id,
                status="TIMED_OUT",
                last_meta=meta_ts,
                last_progress=progress_ts,
                detail=detail
            )
        
        # Still running
        detail = "Active"
        if meta_stale:
            detail += " (metadata stale)"
        if progress_stale and progress_ts:  # Only warn if progress file exists but stale
            detail += " (progress stale)"
        
        return RunState(
            run_id=run_id,
            status="RUNNING",
            last_meta=meta_ts,
            last_progress=progress_ts,
            detail=detail
        )
    
    def check_and_alert(self, run_id: str) -> RunState:
        """Check run state and send alerts on transitions"""
        current_state = self.evaluate_run(run_id)
        
        # Check for state transition
        last_status = self._last_states.get(run_id)
        if last_status != current_state.status:
            logger.info(f"State transition for {run_id}: {last_status} -> {current_state.status}")
            
            # Send alert for terminal states
            if current_state.status in ["FAILED", "TIMED_OUT"] and self.alert_manager:
                # Build context for alert
                context = self._build_alert_context(run_id, current_state)
                self.alert_manager.send(run_id, current_state.status, current_state.detail, context)
            
            self._last_states[run_id] = current_state.status
        
        return current_state
    
    def _build_alert_context(self, run_id: str, state: RunState) -> Dict:
        """Build rich context for alerts"""
        context = {}
        
        # Try to get metadata for additional context
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=f"training-runs/{run_id}/metadata.json"
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            
            context["instance_id"] = metadata.get("instance_id", "unknown")
            context["phase"] = metadata.get("phase", "unknown")
            
            # Calculate duration if start_time exists
            if metadata.get("start_time"):
                start_time = datetime.fromisoformat(metadata["start_time"].replace('Z', '+00:00'))
                duration = (datetime.now(timezone.utc) - start_time).total_seconds() / 60
                context["duration_minutes"] = int(duration)
        except:
            pass
        
        # Try to get last training log line
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=f"training-runs/{run_id}/logs/training_log_latest.txt"
            )
            log_content = response['Body'].read().decode('utf-8')
            lines = log_content.strip().split('\n')
            if lines:
                context["last_log_line"] = lines[-1][:200]  # Last line, max 200 chars
        except:
            pass
        
        return context