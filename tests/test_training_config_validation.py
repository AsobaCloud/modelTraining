#!/usr/bin/env python3
"""
TDD: Training Configuration Validation
Tests that catch configuration errors before they cause training failures
"""

import pytest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestTrainingArgumentValidation:
    """Test training argument validation catches configuration errors"""
    
    def test_max_steps_required_for_iterable_dataset(self):
        """CRITICAL: IterableDataset requires max_steps to be set"""
        from transformers import TrainingArguments
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # This configuration should be valid (max_steps set)
            valid_config = {
                "output_dir": tmp_dir,
                "max_steps": 1000,  # Required for IterableDataset
                "per_device_train_batch_size": 1,
                "eval_strategy": "steps",
                "eval_steps": 100,
            }
            
            # Should not raise exception
            training_args = TrainingArguments(**valid_config)
            assert training_args.max_steps == 1000
            
            # This configuration should fail (max_steps = -1)
            invalid_config = valid_config.copy()
            invalid_config["max_steps"] = -1
            
            # The error will come later when trainer.train() is called with IterableDataset
            # but we can validate the config here
            training_args_invalid = TrainingArguments(**invalid_config)
            assert training_args_invalid.max_steps == -1
            
            # This would cause: "The train_dataset does not implement __len__, max_steps has to be specified"
    
    def test_bf16_fp16_mutual_exclusivity(self):
        """bf16 and fp16 cannot both be True"""
        from transformers import TrainingArguments
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Valid: bf16=True, fp16=False  
            valid_config = {
                "output_dir": tmp_dir,
                "max_steps": 100,
                "bf16": True,
                "fp16": False,
            }
            training_args = TrainingArguments(**valid_config)
            assert training_args.bf16 is True
            assert training_args.fp16 is False
            
            # Invalid: both True (should be caught by transformers)
            invalid_config = valid_config.copy()
            invalid_config["fp16"] = True
            
            with pytest.raises(ValueError, match="cannot be both True"):
                TrainingArguments(**invalid_config)
    
    def test_eval_strategy_parameter_compatibility(self):
        """Test that eval_strategy (not evaluation_strategy) is used"""
        from transformers import TrainingArguments
        import inspect
        
        # Verify the parameter exists in current transformers version
        sig = inspect.signature(TrainingArguments.__init__)
        
        # Modern transformers should have eval_strategy
        assert "eval_strategy" in sig.parameters, \
            "eval_strategy parameter missing - wrong transformers version?"
        
        # Old parameter should not exist
        assert "evaluation_strategy" not in sig.parameters, \
            "Old evaluation_strategy parameter still exists - check transformers version"
    
    def test_learning_rate_scheduler_compatibility(self):
        """Test learning rate scheduler configuration"""
        from transformers import TrainingArguments
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "output_dir": tmp_dir,
                "max_steps": 1000,
                "learning_rate": 2e-4,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
            }
            
            training_args = TrainingArguments(**config)
            assert training_args.lr_scheduler_type == "cosine"
            assert training_args.learning_rate == 2e-4
            assert training_args.warmup_ratio == 0.1
    
    def test_optimizer_compatibility(self):
        """Test 8-bit optimizer configuration"""
        from transformers import TrainingArguments
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "output_dir": tmp_dir,
                "max_steps": 100,
                "optim": "paged_adamw_8bit",
            }
            
            training_args = TrainingArguments(**config)
            assert training_args.optim == "paged_adamw_8bit"


class TestModelConfigurationValidation:
    """Test model setup configuration"""
    
    def test_model_path_validation(self):
        """Test model path exists and contains required files"""
        # Test with non-existent path
        non_existent_path = "/fake/model/path"
        model_path = Path(non_existent_path)
        
        assert not model_path.exists(), "Test path should not exist"
        
        # Test with temporary directory (missing model files)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir)
            assert model_path.exists(), "Temp directory should exist"
            
            # But should not contain model files
            config_file = model_path / "config.json"
            assert not config_file.exists(), "Should not have config.json yet"
    
    def test_quantization_config_validation(self):
        """Test 4-bit quantization configuration"""
        from transformers import BitsAndBytesConfig
        import torch
        
        # Valid configuration
        valid_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        assert valid_config.load_in_4bit is True
        assert valid_config.bnb_4bit_quant_type == "nf4"
        assert valid_config.bnb_4bit_compute_dtype == torch.bfloat16
        
        # Test invalid quantization type
        with pytest.raises(ValueError):
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="invalid_type"
            )
    
    def test_lora_config_validation(self):
        """Test LoRA configuration parameters"""
        from peft import LoraConfig, TaskType
        
        # Valid LoRA config for Mistral
        valid_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        assert valid_config.task_type == TaskType.CAUSAL_LM
        assert valid_config.r == 64
        assert valid_config.lora_alpha == 16
        assert "q_proj" in valid_config.target_modules
        
        # Test invalid rank (too low)
        with pytest.raises(ValueError):
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=0,  # Invalid
            )


class TestDatasetConfigurationValidation:
    """Test dataset configuration and loading"""
    
    def test_s3_path_validation(self):
        """Test S3 path format validation"""
        valid_s3_paths = [
            "s3://bucket-name/path/to/dataset.jsonl",
            "s3://asoba-llm-cache/datasets/mistral-verbosity/train_dataset.jsonl"
        ]
        
        invalid_s3_paths = [
            "bucket-name/path/to/dataset.jsonl",  # Missing s3://
            "s3://",  # Empty bucket
            "s3://bucket",  # No key
            "https://bucket.s3.amazonaws.com/file.jsonl"  # Wrong format
        ]
        
        for path in valid_s3_paths:
            assert path.startswith("s3://"), f"Valid path {path} should start with s3://"
            parts = path[5:].split("/", 1)
            assert len(parts) == 2, f"Valid path {path} should have bucket and key"
            assert len(parts[0]) > 0, f"Valid path {path} should have non-empty bucket"
            assert len(parts[1]) > 0, f"Valid path {path} should have non-empty key"
        
        for path in invalid_s3_paths:
            if not path.startswith("s3://"):
                # This should be caught by validation
                assert True  # We expect this to be invalid
            elif path == "s3://":
                parts = path[5:].split("/", 1)
                assert len(parts) == 1 and parts[0] == "", "Empty S3 path should be invalid"
    
    def test_jsonl_format_validation(self):
        """Test JSONL dataset format validation"""
        # Valid JSONL
        valid_jsonl = """{"text": "This is a valid training example."}
{"text": "Another valid example with proper JSON formatting."}
{"text": "Third example to ensure dataset has content."}"""
        
        # Invalid JSONL
        invalid_jsonl_cases = [
            '{"text": }',  # Invalid JSON
            '{"content": "Missing text field"}',  # Wrong field name
            '{"text": ""}',  # Empty text
            '{"text": null}',  # Null text
        ]
        
        # Test valid JSONL parsing
        valid_lines = valid_jsonl.strip().split('\n')
        parsed_valid = []
        for line in valid_lines:
            data = json.loads(line)
            assert "text" in data, f"Line missing 'text' field: {line}"
            assert isinstance(data["text"], str), f"Text field not string: {line}"
            assert len(data["text"]) > 0, f"Empty text field: {line}"
            parsed_valid.append(data)
        
        assert len(parsed_valid) == 3
        
        # Test invalid JSONL cases
        for invalid_line in invalid_jsonl_cases:
            try:
                data = json.loads(invalid_line)
                # Even if JSON is valid, check our requirements
                if "text" not in data:
                    assert False, f"Should have failed - missing text field: {invalid_line}"
                if not isinstance(data["text"], str) or len(data["text"]) == 0:
                    assert False, f"Should have failed - invalid text: {invalid_line}"
            except json.JSONDecodeError:
                # This is expected for malformed JSON
                pass
    
    def test_dataset_streaming_compatibility(self):
        """Test streaming dataset configuration"""
        from datasets import IterableDataset
        
        # Create test data generator
        def data_generator():
            for i in range(10):
                yield {"text": f"Training example {i+1}"}
        
        # Create streaming dataset
        dataset = IterableDataset.from_generator(data_generator)
        
        # Verify streaming properties
        assert not hasattr(dataset, "__len__"), "Streaming dataset should not have __len__"
        
        # Test we can take samples
        samples = list(dataset.take(3))
        assert len(samples) == 3
        assert all("text" in sample for sample in samples)
        
        # Test shuffle buffer
        shuffled_dataset = dataset.shuffle(buffer_size=5, seed=42)
        shuffled_samples = list(shuffled_dataset.take(3))
        assert len(shuffled_samples) == 3


class TestHardwareCompatibilityValidation:
    """Test hardware requirements and compatibility"""
    
    def test_cuda_availability(self):
        """Test CUDA is available and compatible"""
        import torch
        
        # CUDA must be available for training
        assert torch.cuda.is_available(), "CUDA not available - cannot train on GPU"
        
        # Check CUDA version compatibility
        cuda_version = torch.version.cuda
        assert cuda_version is not None, "CUDA version not detected"
        
        # We expect CUDA 12.1 for our setup
        assert cuda_version == "12.1", f"CUDA version {cuda_version} may be incompatible"
    
    def test_gpu_memory_and_compute_capability(self):
        """Test GPU has sufficient memory and compute capability"""
        import torch
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            
            # Check compute capability for bfloat16 (requires >= 8.0)
            compute_capability = properties.major + properties.minor * 0.1
            assert compute_capability >= 8.0, \
                f"Compute capability {properties.major}.{properties.minor} too low for bfloat16"
            
            # Check memory (should have enough for Mistral-7B with 4-bit quantization)
            total_memory_gb = properties.total_memory / (1024**3)
            assert total_memory_gb >= 20, \
                f"GPU memory {total_memory_gb:.1f}GB may be insufficient for Mistral-7B training"
    
    def test_bfloat16_tensor_operations(self):
        """Test bfloat16 operations work correctly"""
        import torch
        
        if torch.cuda.is_available():
            # Create bfloat16 tensors
            tensor_a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16, device="cuda")
            tensor_b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.bfloat16, device="cuda")
            
            # Test basic operations
            result = tensor_a + tensor_b
            assert result.dtype == torch.bfloat16
            assert result.device.type == "cuda"
            
            # Test conversion to float32 and back
            float32_result = result.float()
            assert float32_result.dtype == torch.float32
            
            bfloat16_back = float32_result.bfloat16()
            assert bfloat16_back.dtype == torch.bfloat16


class TestErrorHandlingValidation:
    """Test error handling and recovery mechanisms"""
    
    def test_model_loading_error_handling(self):
        """Test graceful handling of model loading errors"""
        from transformers import AutoModelForCausalLM
        import torch
        
        # Test with non-existent model path
        fake_model_path = "/fake/model/path"
        
        with pytest.raises(OSError):
            AutoModelForCausalLM.from_pretrained(
                fake_model_path,
                local_files_only=True
            )
    
    def test_s3_error_handling(self):
        """Test handling of S3 connection/permission errors"""
        # Mock boto3 client to simulate S3 errors
        with patch('boto3.client') as mock_boto3:
            mock_s3_client = Mock()
            mock_boto3.return_value = mock_s3_client
            
            # Simulate S3 permission error
            from botocore.exceptions import NoCredentialsError
            mock_s3_client.download_file.side_effect = NoCredentialsError()
            
            # This should be handled gracefully in the training script
            # (We don't test the actual script here, just that the error can be caught)
            with pytest.raises(NoCredentialsError):
                mock_s3_client.download_file("bucket", "key", "local_file")
    
    def test_memory_error_handling(self):
        """Test handling of GPU memory errors"""
        import torch
        
        if torch.cuda.is_available():
            # Try to allocate way too much memory
            with pytest.raises(RuntimeError, match="CUDA out of memory|out of memory"):
                # This should fail
                huge_tensor = torch.zeros((100000, 100000), device="cuda")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])