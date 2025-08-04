#!/usr/bin/env python3
"""
Integration test for the monitoring system.
Simulates a training run to validate end-to-end functionality.
"""

import os
import sys
import tempfile
import json
import time
from pathlib import Path

# Add monitoring to path
sys.path.append(os.path.dirname(__file__))
from training_monitor import S3ProgressSync, create_run_metadata, parse_training_log


def test_integration():
    """Test the complete monitoring workflow"""
    print("🧪 Testing monitoring system integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_id = "test-integration-run"
        
        print(f"📁 Using temp directory: {temp_dir}")
        
        # Step 1: Create metadata
        print("1️⃣ Creating run metadata...")
        metadata = create_run_metadata(run_id, "test-model", temp_dir)
        assert metadata["run_id"] == run_id
        print("✅ Metadata created successfully")
        
        # Step 2: Simulate training log output
        print("2️⃣ Simulating training logs...")
        log_file = Path(temp_dir) / "training.log"
        
        # Write realistic training log entries
        training_logs = [
            "Starting training...",
            "{'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.1}",
            "Step 100 completed",
            "{'loss': 0.987, 'learning_rate': 0.00018, 'epoch': 0.3}",
            "{'loss': 0.756, 'learning_rate': 0.00015, 'epoch': 0.5}",
            "Training checkpoint saved",
            "{'loss': 0.543, 'learning_rate': 0.00012, 'epoch': 0.8}",
        ]
        
        with open(log_file, 'w') as f:
            for log_line in training_logs:
                f.write(log_line + "\n")
        
        print("✅ Training logs created")
        
        # Step 3: Parse logs and create progress
        print("3️⃣ Parsing training progress...")
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        progress = parse_training_log(log_content)
        
        # Validate progress extraction
        assert progress["loss"] == 0.543  # Latest loss
        assert progress["learning_rate"] == 0.00012
        assert progress["current_step"] == 800  # From epoch 0.8
        print("✅ Progress parsed successfully")
        
        # Step 4: Write progress file
        progress_file = Path(temp_dir) / "progress.json"
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Step 5: Test S3 sync (dry run)
        print("4️⃣ Testing S3 sync configuration...")
        sync = S3ProgressSync(run_id, "test-bucket", temp_dir)
        
        # Verify S3 path construction
        expected_prefix = f"logs/training-runs/{run_id}/"
        assert sync.s3_prefix == expected_prefix
        print("✅ S3 sync configured correctly")
        
        # Step 6: Verify file structure
        print("5️⃣ Verifying file structure...")
        expected_files = ["metadata.json", "training.log", "progress.json"]
        
        for file_name in expected_files:
            file_path = Path(temp_dir) / file_name
            assert file_path.exists(), f"Missing file: {file_name}"
        
        print("✅ All required files present")
        
        # Step 7: Test monitoring display format
        print("6️⃣ Testing progress display...")
        from training_monitor import TrainingViewer
        
        viewer = TrainingViewer("test-bucket")
        display = viewer.format_progress(progress)
        
        # Should contain progress bar and metrics
        assert "Step" in display
        assert "0.543" in display  # Loss value
        assert "[" in display  # Progress bar
        print("✅ Progress display working")
        
        print("\n🎉 All integration tests passed!")
        print(f"📊 Final progress: {display}")
        
        return True


if __name__ == "__main__":
    try:
        test_integration()
        print("\n✅ Integration test PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)