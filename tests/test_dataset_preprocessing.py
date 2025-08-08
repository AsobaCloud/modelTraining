#!/usr/bin/env python3
"""
TDD: Dataset Preprocessing Validation
Tests that catch dataset preprocessing errors before they cause training failures
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestDatasetFieldValidation:
    """Test dataset field validation and cleaning"""
    
    def test_dataset_contains_only_required_fields_after_preprocessing(self):
        """Dataset should only contain fields needed by the model after preprocessing"""
        # Sample dataset with extra fields (like our real data)
        sample_data = [
            {
                "text": "This is the training text.",
                "source": "some_source_file.jsonl", 
                "domain": "policy_analysis",
                "metadata": {"author": "test"}
            },
            {
                "text": "Another training example.",
                "source": "another_file.jsonl",
                "domain": "policy_analysis", 
                "timestamp": "2025-01-01"
            }
        ]
        
        # After preprocessing, should only have required fields
        expected_fields = {"input_ids", "attention_mask", "labels"}
        
        # Mock tokenizer for testing
        class MockTokenizer:
            def __call__(self, texts, truncation=True, padding=False, max_length=512):
                # Simulate tokenization
                return {
                    "input_ids": [[1, 2, 3] for _ in texts],
                    "attention_mask": [[1, 1, 1] for _ in texts]
                }
        
        tokenizer = MockTokenizer()
        
        # Test preprocessing function
        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512,
            )
            tokenized["labels"] = tokenized["input_ids"]  # For causal LM
            return tokenized
        
        # Apply preprocessing
        batch = {"text": [item["text"] for item in sample_data]}
        processed = preprocess_function(batch)
        
        # Verify only required fields remain
        for field in processed.keys():
            assert field in expected_fields, f"Unexpected field {field} in processed data"
        
        # Verify required fields are present
        for field in expected_fields:
            assert field in processed, f"Missing required field {field}"
    
    def test_dataset_map_removes_extra_columns(self):
        """Test that dataset.map properly removes extra columns"""
        try:
            from datasets import Dataset
        except ImportError:
            pytest.skip("datasets library not available")
        
        # Create test dataset with extra columns
        test_data = {
            "text": ["Example 1", "Example 2"],
            "source": ["file1.jsonl", "file2.jsonl"],
            "domain": ["policy", "policy"],
            "extra_field": ["value1", "value2"]
        }
        
        dataset = Dataset.from_dict(test_data)
        
        # Mock preprocessing function
        def preprocess_function(examples):
            return {
                "input_ids": [[1, 2] for _ in examples["text"]],
                "attention_mask": [[1, 1] for _ in examples["text"]],
                "labels": [[1, 2] for _ in examples["text"]]
            }
        
        # Apply preprocessing with remove_columns
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,  # Remove ALL original columns
        )
        
        # Verify only processed columns remain
        expected_columns = {"input_ids", "attention_mask", "labels"}
        actual_columns = set(processed_dataset.column_names)
        
        assert actual_columns == expected_columns, \
            f"Expected {expected_columns}, got {actual_columns}"
    
    def test_iterable_dataset_column_removal(self):
        """Test column removal for IterableDataset"""
        try:
            from datasets import IterableDataset
        except ImportError:
            pytest.skip("datasets library not available")
        
        # Create test data generator with extra fields
        def data_generator():
            for i in range(3):
                yield {
                    "text": f"Training example {i+1}",
                    "source": f"file_{i}.jsonl",
                    "domain": "policy_analysis",
                    "extra": f"extra_{i}"
                }
        
        dataset = IterableDataset.from_generator(data_generator)
        
        # Mock preprocessing function
        def preprocess_function(examples):
            return {
                "input_ids": [[1, 2, 3] for _ in examples["text"]],
                "attention_mask": [[1, 1, 1] for _ in examples["text"]],
                "labels": [[1, 2, 3] for _ in examples["text"]]
            }
        
        # Apply preprocessing with remove_columns
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["text", "source", "domain", "extra"],
        )
        
        # Test that we can iterate and get expected fields
        sample_batch = next(iter(processed_dataset))
        
        expected_fields = {"input_ids", "attention_mask", "labels"}
        actual_fields = set(sample_batch.keys())
        
        assert actual_fields == expected_fields, \
            f"Expected {expected_fields}, got {actual_fields}"


class TestDataCollatorCompatibility:
    """Test data collator handles processed data correctly"""
    
    def test_data_collator_with_clean_data(self):
        """Test DataCollatorForLanguageModeling with properly cleaned data"""
        try:
            from transformers import DataCollatorForLanguageModeling, AutoTokenizer
        except ImportError:
            pytest.skip("transformers library not available")
        
        # Use a simple tokenizer for testing
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
        
        # Test data that should work (only required fields)
        clean_features = [
            {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "labels": [1, 2, 3, 4]
            },
            {
                "input_ids": [5, 6, 7],
                "attention_mask": [1, 1, 1],
                "labels": [5, 6, 7]
            }
        ]
        
        # This should work without error
        batch = data_collator(clean_features)
        
        assert "input_ids" in batch
        assert "attention_mask" in batch  
        assert "labels" in batch
        
        # Verify padding worked
        assert batch["input_ids"].shape[1] % 8 == 0  # pad_to_multiple_of=8
    
    def test_data_collator_fails_with_dirty_data(self):
        """Test that data collator fails with extra string fields (reproducing our error)"""
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
            mlm=False
        )
        
        # Test data with extra string fields (like our original problem)
        dirty_features = [
            {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "labels": [1, 2, 3, 4],
                "source": "file1.jsonl",  # This causes the error
                "domain": "policy"
            }
        ]
        
        # This should fail with ValueError about tensor conversion
        with pytest.raises(ValueError, match="Unable to create tensor|too many dimensions"):
            data_collator(dirty_features)


class TestJSONLDataValidation:
    """Test JSONL data validation catches problematic fields"""
    
    def test_identify_extra_fields_in_jsonl(self):
        """Test that we can identify extra fields in JSONL data"""
        # Sample JSONL with extra fields (simulating our real data)
        jsonl_content = """{"text": "Training example 1", "source": "file1.jsonl", "domain": "policy"}
{"text": "Training example 2", "source": "file2.jsonl", "domain": "policy", "timestamp": "2025-01-01"}
{"text": "Training example 3", "source": "file3.jsonl"}"""
        
        lines = jsonl_content.strip().split('\n')
        all_fields = set()
        required_fields = {"text"}
        
        for line in lines:
            data = json.loads(line)
            all_fields.update(data.keys())
        
        extra_fields = all_fields - required_fields
        
        # Should identify extra fields
        assert len(extra_fields) > 0, "Should have found extra fields"
        assert "source" in extra_fields, "Should have identified 'source' as extra field"
        assert "domain" in extra_fields, "Should have identified 'domain' as extra field"
        
        # Log what we found for debugging
        print(f"Found extra fields: {extra_fields}")
    
    def test_clean_jsonl_data_function(self):
        """Test function to clean JSONL data of extra fields"""
        # Sample data with extra fields
        raw_data = [
            {"text": "Example 1", "source": "file1.jsonl", "domain": "policy"},
            {"text": "Example 2", "source": "file2.jsonl", "extra": "value"}
        ]
        
        def clean_data(data_list, keep_fields={"text"}):
            """Clean data to keep only specified fields"""
            cleaned = []
            for item in data_list:
                cleaned_item = {k: v for k, v in item.items() if k in keep_fields}
                cleaned.append(cleaned_item)
            return cleaned
        
        cleaned_data = clean_data(raw_data, keep_fields={"text"})
        
        # Verify cleaning worked
        assert len(cleaned_data) == 2
        for item in cleaned_data:
            assert list(item.keys()) == ["text"], f"Expected only 'text' field, got {list(item.keys())}"
            assert "source" not in item, "Should have removed 'source' field"
            assert "domain" not in item, "Should have removed 'domain' field"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])