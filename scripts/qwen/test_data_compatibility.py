#!/usr/bin/env python3
"""
Test script to verify data compatibility with the updated training script.
This ensures both messages and prompt/completion formats work correctly.
"""

import json
import sys
from pathlib import Path


def test_data_loading(file_path, expected_format="auto"):
    """Test data loading with the same logic as the training script"""
    
    print(f"\n🔍 Testing: {file_path}")
    print(f"Expected format: {expected_format}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    rows = []
    stats = {
        "total_lines": 0,
        "empty_lines": 0,
        "json_errors": 0,
        "prompt_completion": 0,
        "messages_converted": 0,
        "messages_failed": 0,
        "missing_keys": 0
    }
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line_num > 50:  # Test first 50 lines for speed
                break
                
            stats["total_lines"] += 1
            line = line.strip()
            if not line:
                stats["empty_lines"] += 1
                continue
                
            try:
                obj = json.loads(line)
                
                # Handle both prompt/completion and messages format
                if "prompt" in obj and "completion" in obj:
                    rows.append({"prompt": obj["prompt"], "completion": obj["completion"]})
                    stats["prompt_completion"] += 1
                    
                elif "messages" in obj and len(obj["messages"]) >= 2:
                    # Convert messages format to prompt/completion
                    messages = obj["messages"]
                    user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
                    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
                    
                    if user_msg and assistant_msg:
                        rows.append({"prompt": user_msg, "completion": assistant_msg})
                        stats["messages_converted"] += 1
                    else:
                        stats["messages_failed"] += 1
                        print(f"  Line {line_num}: Missing user/assistant in messages")
                        
                else:
                    stats["missing_keys"] += 1
                    print(f"  Line {line_num}: Missing required keys. Has: {list(obj.keys())}")
                    
            except json.JSONDecodeError as e:
                stats["json_errors"] += 1
                print(f"  Line {line_num}: JSON error: {e}")
    
    # Print results
    success = len(rows) > 0
    status = "✅ SUCCESS" if success else "❌ FAILED"
    
    print(f"\n{status} - {len(rows)} examples loaded")
    print(f"📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if len(rows) > 0:
        sample_prompt_len = len(rows[0]["prompt"])
        sample_completion_len = len(rows[0]["completion"])
        print(f"📝 Sample lengths: prompt={sample_prompt_len}, completion={sample_completion_len}")
    
    return success


def main():
    print("🧪 Data Compatibility Test for Qwen3-14B SFT Training")
    print("=" * 60)
    
    # Test files that should exist on the training instance
    test_files = [
        ("/home/ubuntu/verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl", "messages"),
        ("/home/ubuntu/data/final_enhanced_iac_corpus.jsonl", "prompt/completion"),
    ]
    
    all_passed = True
    results = {}
    
    for file_path, expected_format in test_files:
        success = test_data_loading(file_path, expected_format)
        results[file_path] = success
        if not success:
            all_passed = False
    
    print(f"\n🎯 Final Results:")
    print("=" * 60)
    
    for file_path, success in results.items():
        filename = Path(file_path).name
        status = "✅ COMPATIBLE" if success else "❌ INCOMPATIBLE"
        print(f"{status}: {filename}")
    
    if all_passed:
        print(f"\n🎉 All datasets are compatible!")
        print(f"✅ Ready to run multi-domain SFT training")
        return 0
    else:
        print(f"\n⚠️  Some datasets have issues")
        print(f"❌ Training may fail - fix data format issues first")
        return 1


if __name__ == "__main__":
    sys.exit(main())