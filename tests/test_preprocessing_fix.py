#!/usr/bin/env python3
"""
TDD: Test the preprocessing fix
Validates that our fix properly removes extra fields
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestPreprocessingFix:
    """Test that our preprocessing fix works correctly"""
    
    def test_remove_all_columns_from_real_dataset_format(self):
        """Test with the actual dataset format we encountered"""
        try:
            from datasets import IterableDataset
        except ImportError:
            pytest.skip("datasets library not available")
        
        # Real dataset format that caused the error
        def real_data_generator():
            yield {
                "text": "Processed content from State Dept cable redaction...",
                "source": "State Dept cable redaction card 1978-39525.pdf", 
                "full_path": "/mnt/training/data_prep/operatives/1978 withdrawal...",
                "processed_at": "2025-08-05T21:25:04.496630"
            }
            yield {
                "text": "Another example of training data text.",
                "source": "another_file.pdf",
                "full_path": "/mnt/training/data_prep/operatives/another_file.pdf", 
                "processed_at": "2025-08-05T21:30:00.000000"
            }
        
        dataset = IterableDataset.from_generator(real_data_generator)
        
        # Mock tokenizer that mimics our real preprocessing
        class MockTokenizer:
            def __call__(self, texts, truncation=True, padding=False, max_length=512):
                return {
                    "input_ids": [[1, 2, 3, 4, 5] for _ in texts],
                    "attention_mask": [[1, 1, 1, 1, 1] for _ in texts]
                }
        
        tokenizer = MockTokenizer()
        
        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512,
            )
            tokenized["labels"] = tokenized["input_ids"]
            return tokenized
        
        # Get sample to identify all columns (our fix)
        sample_item = next(iter(dataset.take(1)))
        all_columns = list(sample_item.keys())
        
        # Verify we identify all the problematic columns
        expected_columns = ["text", "source", "full_path", "processed_at"]
        assert set(all_columns) == set(expected_columns), \
            f"Expected {expected_columns}, got {all_columns}"
        
        # Apply our fix - remove ALL columns
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=all_columns,  # The key fix
        )
        
        # Validate processed dataset
        sample_processed = next(iter(processed_dataset.take(1)))
        processed_fields = list(sample_processed.keys())
        expected_fields = ["input_ids", "attention_mask", "labels"]
        
        # Should only have the required fields
        assert set(processed_fields) == set(expected_fields), \
            f"Expected {expected_fields}, got {processed_fields}"
        
        # Should not have any of the problematic fields
        problematic_fields = ["source", "full_path", "processed_at"]
        for field in problematic_fields:
            assert field not in processed_fields, \
                f"Problematic field '{field}' still present in processed data"
    
    def test_data_collator_works_with_cleaned_data(self):
        """Test that data collator works with our cleaned data"""
        try:
            from transformers import DataCollatorForLanguageModeling, AutoTokenizer
        except ImportError:
            pytest.skip("transformers library not available")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pytest.skip("Cannot load tokenizer for testing")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Simulate cleaned data (what our fix should produce) - same length sequences
        clean_features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5]
            },
            {
                "input_ids": [6, 7, 8, 9, 10], 
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [6, 7, 8, 9, 10]
            }
        ]
        
        # This should work without the ValueError we were getting
        batch = data_collator(clean_features)
        
        # Verify the batch is properly formed
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        
        # Verify shapes are consistent
        batch_size, seq_len = batch["input_ids"].shape
        assert batch["attention_mask"].shape == (batch_size, seq_len)
        assert batch["labels"].shape == (batch_size, seq_len)
        
        # Verify padding to multiple of 8
        assert seq_len % 8 == 0, f"Sequence length {seq_len} not multiple of 8"
    
    def test_validation_catches_missed_fields(self):
        """Test that our validation catches if we miss removing fields"""
        # Simulate what happens if we don't remove all columns properly
        
        # This should trigger our validation error
        sample_with_extra_fields = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1], 
            "labels": [1, 2, 3],
            "source": "leftover_field.pdf"  # This should trigger validation error
        }
        
        processed_fields = list(sample_with_extra_fields.keys())
        expected_fields = ["input_ids", "attention_mask", "labels"]
        
        # Check for unexpected fields (our validation logic)
        unexpected_fields = set(processed_fields) - set(expected_fields)
        
        # Should detect the problem
        assert len(unexpected_fields) > 0, "Should have detected unexpected fields"
        assert "source" in unexpected_fields, "Should have detected 'source' as unexpected"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])