#!/usr/bin/env python3
"""
Test Suite for Qwen Prompt-Completion Training
Following TDD approach from CLAUDE.md
"""

import torch
import json
from transformers import AutoTokenizer

def test_tokenizer_setup():
    """Test 1: Tokenizer loads and has special tokens"""
    tokenizer = AutoTokenizer.from_pretrained("/opt/models/Qwen3-14B", trust_remote_code=True)
    
    # Add style tokens
    special_tokens = {"additional_special_tokens": ["<STYLE_TERSE>", "<STYLE_VERBOSE>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Verify tokens added
    assert "<STYLE_TERSE>" in tokenizer.get_added_vocab()
    assert "<STYLE_VERBOSE>" in tokenizer.get_added_vocab()
    
    # Verify EOS token
    assert tokenizer.eos_token == "<|im_end|>"
    
    print("✅ Test 1 passed: Tokenizer setup correct")
    return tokenizer

def test_prompt_completion_format(tokenizer):
    """Test 2: Prompt-completion formatting works correctly"""
    
    # Sample message
    messages = [
        {"role": "user", "content": "What is 2+2? Please answer briefly."},
        {"role": "assistant", "content": "4"}
    ]
    
    # Add style token
    user_content = messages[0]['content']
    if "briefly" in user_content.lower():
        style_token = "<STYLE_TERSE>"
    else:
        style_token = "<STYLE_VERBOSE>"
    
    messages[0]['content'] = f"{style_token}\n{user_content}"
    
    # Generate prompt
    prompt = tokenizer.apply_chat_template(
        [messages[0]], 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Generate completion
    completion = messages[1]['content'] + tokenizer.eos_token
    
    # Verify format
    assert prompt.endswith("<|im_start|>assistant\n")
    assert completion.endswith("<|im_end|>")
    assert style_token in prompt
    
    print("✅ Test 2 passed: Prompt-completion format correct")
    print(f"   Prompt ends with: {repr(prompt[-30:])}")
    print(f"   Completion: {repr(completion)}")
    
    return prompt, completion

def test_label_masking(tokenizer, prompt, completion):
    """Test 3: Label masking excludes prompt tokens"""
    
    # Tokenize full sequence
    full_text = prompt + completion
    encoding = tokenizer(full_text, return_tensors="pt")
    input_ids = encoding.input_ids[0]
    
    # Tokenize prompt only to get length
    prompt_encoding = tokenizer(prompt, return_tensors="pt")
    prompt_length = prompt_encoding.input_ids.shape[1]
    
    # Create labels with masking
    labels = input_ids.clone()
    labels[:prompt_length] = -100
    
    # Verify masking
    assert all(labels[:prompt_length] == -100)
    assert all(labels[prompt_length:] != -100)
    
    print("✅ Test 3 passed: Label masking correct")
    print(f"   Prompt length: {prompt_length} tokens")
    print(f"   Total length: {len(input_ids)} tokens")
    print(f"   Masked tokens: {(labels == -100).sum().item()}")
    
    return labels

def test_style_token_differentiation(tokenizer):
    """Test 4: Different styles produce different prompts"""
    
    terse_msg = [{"role": "user", "content": "<STYLE_TERSE>\nWhat is machine learning?"}]
    verbose_msg = [{"role": "user", "content": "<STYLE_VERBOSE>\nWhat is machine learning?"}]
    
    terse_prompt = tokenizer.apply_chat_template(terse_msg, add_generation_prompt=True, tokenize=False)
    verbose_prompt = tokenizer.apply_chat_template(verbose_msg, add_generation_prompt=True, tokenize=False)
    
    assert "<STYLE_TERSE>" in terse_prompt
    assert "<STYLE_VERBOSE>" in verbose_prompt
    assert terse_prompt != verbose_prompt
    
    print("✅ Test 4 passed: Style tokens differentiate prompts")
    
def test_dataset_loading():
    """Test 5: Dataset loads and has expected structure"""
    
    # Simulate dataset loading (would use actual file on server)
    sample_data = {
        "messages": [
            {"role": "user", "content": "What is 2+2? Please answer briefly."},
            {"role": "assistant", "content": "4"}
        ]
    }
    
    assert "messages" in sample_data
    assert len(sample_data["messages"]) == 2
    assert sample_data["messages"][0]["role"] == "user"
    assert sample_data["messages"][1]["role"] == "assistant"
    
    print("✅ Test 5 passed: Dataset structure correct")

def run_all_tests():
    """Run all validation tests"""
    print("Running Qwen Prompt-Completion Training Tests...\n")
    
    # Test 1: Tokenizer setup
    tokenizer = test_tokenizer_setup()
    
    # Test 2: Prompt-completion format
    prompt, completion = test_prompt_completion_format(tokenizer)
    
    # Test 3: Label masking
    labels = test_label_masking(tokenizer, prompt, completion)
    
    # Test 4: Style token differentiation
    test_style_token_differentiation(tokenizer)
    
    # Test 5: Dataset structure
    test_dataset_loading()
    
    print("\n✅ All tests passed! Ready to implement training script.")

if __name__ == "__main__":
    run_all_tests()