#!/usr/bin/env python3
"""
Test suite for memory-efficient training fixes.
Tests the four HF-Datasets best practice deviations that cause OOM.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add the scripts directory to path so we can import the training module
sys.path.insert(0, '/home/shingai/sort/llm-training/scripts/mistral')

class TestMemoryEfficientFixes:
    """Test the four HF-Datasets memory efficiency fixes"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_data = [
            {"text": "This is test example 1"},
            {"text": "This is test example 2"}, 
            {"text": "This is test example 3"}
        ]
        
        # Create temporary JSONL file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        for item in self.test_data:
            self.temp_file.write(json.dumps(item) + '\n')
        self.temp_file.close()
    
    def teardown_method(self):
        """Cleanup test files"""
        os.unlink(self.temp_file.name)

    def test_map_uses_memory_inefficient_defaults(self):
        """FAILING TEST: Current map() call should use memory-inefficient defaults"""
        from datasets import load_dataset
        
        # Load test dataset
        dataset = load_dataset('json', data_files=self.temp_file.name, split='train')
        
        # Mock the map method to capture its arguments
        with patch.object(dataset, 'map') as mock_map:
            mock_map.return_value = dataset
            
            # Import and call the current preprocessing function
            try:
                import train_mistral_simple
                # This should fail because map() doesn't use keep_in_memory=False
                train_mistral_simple.memory_safe_map(
                    dataset,
                    lambda x: {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
                    remove_columns=dataset.column_names
                )
                
                # Check if memory-safe parameters were used
                mock_map.assert_called_once()
                call_kwargs = mock_map.call_args.kwargs
                
                # These should FAIL in current implementation
                assert call_kwargs.get('keep_in_memory') == False, "keep_in_memory should be False"
                assert call_kwargs.get('writer_batch_size') is not None, "writer_batch_size should be set"
                assert call_kwargs.get('num_proc') == 1, "num_proc should be 1 to avoid crashes"
                
            except ImportError:
                pytest.fail("Cannot import train_mistral_simple - script may not exist")

    def test_preprocess_dataset_creates_intermediate_lists(self):
        """FAILING TEST: preprocess_dataset should NOT create intermediate Python lists"""
        try:
            import train_mistral_simple
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {
                "input_ids": [[1, 2, 3], [4, 5, 6]],
                "attention_mask": [[1, 1, 1], [1, 1, 1]]
            }
            
            # Test current implementation
            examples = {"text": ["hello", "world"]}
            result = train_mistral_simple.preprocess_dataset(examples, mock_tokenizer)
            
            # Check tokenizer was called correctly (should NOT create intermediate list)
            # Current implementation creates texts=[] list - this should FAIL
            mock_tokenizer.assert_called_once()
            call_args = mock_tokenizer.call_args[0][0]
            
            # This should be examples["text"] directly, not a Python list copy
            assert call_args == examples["text"], "Tokenizer should receive examples['text'] directly"
            
        except ImportError:
            pytest.fail("Cannot import train_mistral_simple")

    def test_preprocess_dataset_copies_labels_unnecessarily(self):
        """FAILING TEST: preprocess_dataset should NOT copy input_ids for labels"""
        try:
            import train_mistral_simple
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_input_ids = [[1, 2, 3], [4, 5, 6]]
            mock_tokenizer.return_value = {
                "input_ids": mock_input_ids,
                "attention_mask": [[1, 1, 1], [1, 1, 1]]
            }
            
            examples = {"text": ["hello", "world"]}
            result = train_mistral_simple.preprocess_dataset(examples, mock_tokenizer)
            
            # Labels should be a VIEW of input_ids, not a COPY
            assert result["labels"] is result["input_ids"], "Labels should be same object as input_ids (view, not copy)"
            
        except ImportError:
            pytest.fail("Cannot import train_mistral_simple")

    def test_model_loaded_before_tokenization(self):
        """FAILING TEST: Model should NOT be loaded before tokenization"""
        try:
            import train_mistral_simple
            
            # Mock the setup_model_and_tokenizer function to track when it's called
            with patch.object(train_mistral_simple, 'setup_model_and_tokenizer') as mock_setup:
                mock_setup.return_value = (Mock(), Mock())
                
                with patch.object(train_mistral_simple, 'memory_safe_map') as mock_map:
                    mock_map.return_value = Mock()
                    
                    with patch('datasets.load_dataset') as mock_load:
                        mock_dataset = Mock()
                        mock_load.return_value = mock_dataset
                        
                        # This test will fail because current implementation loads model BEFORE tokenization
                        # We need to track the order of operations
                        call_order = []
                        
                        def track_model_load(*args, **kwargs):
                            call_order.append('model_load')
                            return (Mock(), Mock())
                            
                        def track_tokenization(*args, **kwargs):
                            call_order.append('tokenization')
                            return mock_dataset
                            
                        mock_setup.side_effect = track_model_load
                        mock_map.side_effect = track_tokenization
                        
                        # This should fail because model is loaded first
                        # Expected order: ['tokenization', 'model_load']
                        # Actual order: ['model_load', 'tokenization'] 
                        
                        # The test expects tokenization before model loading
                        assert len(call_order) >= 2, "Both operations should be called"
                        assert call_order.index('tokenization') < call_order.index('model_load'), \
                            "Tokenization should happen BEFORE model loading for memory efficiency"
                            
        except ImportError:
            pytest.fail("Cannot import train_mistral_simple")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])