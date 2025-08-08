#!/usr/bin/env python3
"""
HeartbeatCallback - Real training progress integration with HeartbeatManager
"""

import logging
from typing import Optional, Dict, Any

try:
    from heartbeat_manager import HeartbeatManager
except ImportError:
    HeartbeatManager = None

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)

class HeartbeatCallback(TrainerCallback):
    """Updates HeartbeatManager with real training progress"""
    
    def __init__(self, heartbeat_manager: "HeartbeatManager", update_every_steps: int = 100):
        if heartbeat_manager is None:
            raise ValueError("heartbeat_manager must not be None")
        self.hb = heartbeat_manager
        self.update_every_steps = update_every_steps
        self._last_step_pushed = -1
        self._total_steps = None
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._total_steps = state.max_steps or args.max_steps or 0
        try:
            self.hb.set_status("training")
            self.hb.update_phase("training", "active", "Training started")
        except Exception as e:
            logger.warning(f"Heartbeat start failed: {e}")
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        step = int(state.global_step or 0)
        if step == 0 or step - self._last_step_pushed < self.update_every_steps:
            return
            
        self._last_step_pushed = step
        total = self._total_steps or 0
        loss = logs.get('loss', 0)
        message = f"step={step}/{total} loss={loss:.4f}" if loss else f"step={step}/{total}"
        
        try:
            self.hb.update_phase("training", "active", message)
        except Exception as e:
            logger.warning(f"Heartbeat update failed: {e}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            self.hb.set_status("completed")
            self.hb.update_phase("complete", "success", "Training ended")
            self.hb.stop()
        except Exception as e:
            logger.warning(f"Heartbeat stop failed: {e}")