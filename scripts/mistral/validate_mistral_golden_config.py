#!/usr/bin/env python3
"""
Validation Script for Mistral-7B Golden Config
Tests all components before full training execution
"""

import json
import torch
import sys
import os
from pathlib import Path
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
MODEL_DIR = "/home/ubuntu/mistral-7b-v0.3"
DATASET_PATH = "verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl"
EXPECTED_DATASET_SIZE = 3200

class MistralGoldenConfigValidator:
    """Comprehensive validation for Mistral-7B golden config"""
    
    def __init__(self):
        self.validation_results = {}
        self.model = None
        self.tokenizer = None
        
    def log_result(self, test_name, passed, details=""):
        """Log validation result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"   Details: {details}")
        self.validation_results[test_name] = {"passed": passed, "details": details}
        return passed
    
    def validate_environment(self):
        """Validate environment setup"""
        logger.info("=== ENVIRONMENT VALIDATION ===")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        self.log_result("CUDA Available", cuda_available, 
                       f"Device count: {torch.cuda.device_count()}")
        
        # Check GPU memory
        if cuda_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            sufficient_memory = gpu_memory >= 20.0  # Need at least 20GB for g5.2xlarge
            self.log_result("GPU Memory Sufficient", sufficient_memory,
                           f"{gpu_memory:.1f}GB available")
        
        # Check required libraries
        required_libs = {
            'transformers': '4.54.0',
            'torch': '2.7.1',
            'peft': '0.16.0',
            'bitsandbytes': '0.46.1'
        }
        
        for lib, expected_version in required_libs.items():
            try:
                if lib == 'torch':
                    import torch
                    version = torch.__version__.split('+')[0]  # Remove +cu126 suffix
                elif lib == 'transformers':
                    import transformers
                    version = transformers.__version__
                elif lib == 'peft':
                    import peft
                    version = peft.__version__
                elif lib == 'bitsandbytes':
                    import bitsandbytes
                    version = bitsandbytes.__version__
                
                version_ok = version.startswith(expected_version.split('.')[0])  # Major version match
                self.log_result(f"Library {lib}", version_ok,
                               f"Expected: {expected_version}, Found: {version}")
            except ImportError:
                self.log_result(f"Library {lib}", False, "Not installed")
        
        return True
    
    def validate_model_loading(self):
        """Validate model and tokenizer loading"""
        logger.info("=== MODEL LOADING VALIDATION ===")
        
        # Check if model directory exists
        model_exists = os.path.exists(MODEL_DIR)
        if not self.log_result("Model Directory Exists", model_exists, MODEL_DIR):
            return False
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
            self.log_result("Tokenizer Loading", True, f"Vocab size: {len(self.tokenizer)}")
            
            # Add special tokens
            special_tokens = {"additional_special_tokens": ["<STYLE_TERSE>", "<STYLE_VERBOSE>"]}
            added_tokens = self.tokenizer.add_special_tokens(special_tokens)
            self.log_result("Special Tokens Added", added_tokens == 2,
                           f"Added {added_tokens} tokens")
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            pad_token_set = self.tokenizer.pad_token is not None
            self.log_result("Pad Token Set", pad_token_set,
                           f"Pad token: {self.tokenizer.pad_token}")
            
            # Load model with quantization
            logger.info("Loading model with 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True
            )
            
            self.log_result("Model Loading", True, f"Device: {next(self.model.parameters()).device}")
            
            # Check 4-bit quantization
            is_4bit = getattr(self.model, 'is_loaded_in_4bit', False)
            self.log_result("4-bit Quantization", is_4bit, 
                           f"is_loaded_in_4bit: {is_4bit}")
            
            # Resize embeddings
            original_size = self.model.get_input_embeddings().weight.shape[0]
            self.model.resize_token_embeddings(len(self.tokenizer))
            new_size = self.model.get_input_embeddings().weight.shape[0]
            
            embedding_resized = new_size == len(self.tokenizer)
            self.log_result("Token Embeddings Resized", embedding_resized,
                           f"{original_size} -> {new_size} tokens")
            
            # Check GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                
                memory_reasonable = memory_allocated < 12.0  # Should be < 12GB for 7B model
                self.log_result("GPU Memory Usage", memory_reasonable,
                               f"Allocated: {memory_allocated:.1f}GB, Reserved: {memory_reserved:.1f}GB")
            
            return True
            
        except Exception as e:
            self.log_result("Model Loading", False, f"Error: {str(e)}")
            return False
    
    def validate_dataset(self):
        """Validate dataset loading and formatting"""
        logger.info("=== DATASET VALIDATION ===")
        
        # Check dataset file exists
        dataset_exists = os.path.exists(DATASET_PATH)
        if not self.log_result("Dataset File Exists", dataset_exists, DATASET_PATH):
            return False
        
        try:
            # Load dataset
            data = []
            with open(DATASET_PATH, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            
            # Check dataset size
            correct_size = len(data) == EXPECTED_DATASET_SIZE
            self.log_result("Dataset Size", correct_size,
                           f"Expected: {EXPECTED_DATASET_SIZE}, Found: {len(data)}")
            
            # Check data format
            first_item = data[0]
            has_messages = 'messages' in first_item
            self.log_result("Dataset Format", has_messages, 
                           f"Sample keys: {list(first_item.keys())}")
            
            if has_messages:
                messages = first_item['messages']
                has_user_assistant = (len(messages) == 2 and 
                                    messages[0]['role'] == 'user' and 
                                    messages[1]['role'] == 'assistant')
                self.log_result("Message Structure", has_user_assistant,
                               f"Roles: {[msg['role'] for msg in messages]}")
            
            # Check style detection
            terse_count = 0
            verbose_count = 0
            
            for item in data[:100]:  # Sample first 100
                content = item['messages'][0]['content'].lower()
                if 'briefly' in content or 'concise' in content:
                    terse_count += 1
                elif 'detailed' in content or 'well-structured' in content:
                    verbose_count += 1
            
            style_balance = abs(terse_count - verbose_count) < 20  # Should be roughly balanced
            self.log_result("Style Balance", style_balance,
                           f"TERSE: {terse_count}, VERBOSE: {verbose_count} (in first 100)")
            
            return True
            
        except Exception as e:
            self.log_result("Dataset Validation", False, f"Error: {str(e)}")
            return False
    
    def validate_data_processing(self):
        """Validate data processing pipeline"""
        logger.info("=== DATA PROCESSING VALIDATION ===")
        
        if not self.tokenizer:
            self.log_result("Data Processing", False, "Tokenizer not available")
            return False
        
        try:
            # Load sample data
            with open(DATASET_PATH, 'r') as f:
                sample_item = json.loads(f.readline())
            
            # Test style token addition
            def add_style_token(messages):
                user_content = messages[0]['content']
                if "briefly" in user_content.lower() or "concise" in user_content.lower():
                    style_token = "<STYLE_TERSE>"
                elif "detailed" in user_content.lower() or "well-structured" in user_content.lower():
                    style_token = "<STYLE_VERBOSE>"
                else:
                    style_token = "<STYLE_TERSE>"
                
                messages[0]['content'] = f"{style_token}\n{user_content}"
                return messages
            
            # Test prompt-completion formatting
            def to_prompt_completion(example):
                messages = add_style_token(example['messages'].copy())
                user_side = [messages[0]]
                
                prompt = self.tokenizer.apply_chat_template(
                    user_side, add_generation_prompt=True, tokenize=False
                )
                
                completion = messages[1]['content'].rstrip("\n") + self.tokenizer.eos_token
                
                return {"prompt": prompt, "completion": completion}
            
            # Process sample
            processed = to_prompt_completion(sample_item)
            
            # Validate processed format
            has_prompt_completion = 'prompt' in processed and 'completion' in processed
            self.log_result("Prompt-Completion Format", has_prompt_completion,
                           f"Keys: {list(processed.keys())}")
            
            # Check style token presence
            has_style_token = ('<STYLE_TERSE>' in processed['prompt'] or 
                             '<STYLE_VERBOSE>' in processed['prompt'])
            self.log_result("Style Token Injection", has_style_token,
                           f"Style token found in prompt")
            
            # Check tokenization
            prompt_tokens = self.tokenizer.encode(processed['prompt'])
            completion_tokens = self.tokenizer.encode(processed['completion'])
            
            reasonable_length = len(prompt_tokens) < 500 and len(completion_tokens) < 200
            self.log_result("Token Lengths", reasonable_length,
                           f"Prompt: {len(prompt_tokens)}, Completion: {len(completion_tokens)}")
            
            logger.info(f"Sample prompt (last 100 chars): ...{processed['prompt'][-100:]}")
            logger.info(f"Sample completion: {processed['completion'][:50]}...")
            
            return True
            
        except Exception as e:
            self.log_result("Data Processing", False, f"Error: {str(e)}")
            return False
    
    def validate_lora_config(self):
        """Validate LoRA configuration"""
        logger.info("=== LORA CONFIGURATION VALIDATION ===")
        
        try:
            # Create LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                modules_to_save=["embed_tokens", "lm_head"],
                inference_mode=False,
            )
            
            self.log_result("LoRA Config Creation", True,
                           f"Rank: {peft_config.r}, Alpha: {peft_config.lora_alpha}")
            
            # Check target modules are valid for Mistral
            if self.model:
                model_modules = set()
                for name, _ in self.model.named_modules():
                    if any(target in name for target in peft_config.target_modules):
                        model_modules.add(name.split('.')[-1])
                
                expected_modules = set(peft_config.target_modules)
                modules_found = expected_modules.issubset(model_modules)
                self.log_result("Target Modules Valid", modules_found,
                               f"Found: {model_modules & expected_modules}")
            
            return True
            
        except Exception as e:
            self.log_result("LoRA Config", False, f"Error: {str(e)}")
            return False
    
    def validate_inference_presets(self):
        """Validate inference generation presets"""
        logger.info("=== INFERENCE PRESETS VALIDATION ===")
        
        if not self.model or not self.tokenizer:
            self.log_result("Inference Validation", False, "Model/tokenizer not available")
            return False
        
        try:
            # Test prompts
            test_prompts = [
                "<STYLE_TERSE>\nWhat is machine learning?",
                "<STYLE_VERBOSE>\nWhat is machine learning?"
            ]
            
            # Generation presets
            presets = {
                "TERSE": {"temperature": 0.1, "top_p": 0.95, "max_new_tokens": 32},
                "VERBOSE": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 256}
            }
            
            for i, prompt in enumerate(test_prompts):
                style = "TERSE" if "<STYLE_TERSE>" in prompt else "VERBOSE"
                
                # Apply chat template
                inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                # Test generation (with small max_new_tokens for validation)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=10,  # Small for validation
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs.shape[-1]:], 
                    skip_special_tokens=True
                )
                
                generation_works = len(response.strip()) > 0
                self.log_result(f"Generation {style}", generation_works,
                               f"Response: '{response.strip()[:30]}...'")
            
            return True
            
        except Exception as e:
            self.log_result("Inference Validation", False, f"Error: {str(e)}")
            return False
    
    def run_full_validation(self):
        """Run complete validation suite"""
        logger.info("🚀 Starting Mistral-7B Golden Config Validation")
        logger.info("=" * 60)
        
        # Run all validations
        validations = [
            self.validate_environment,
            self.validate_model_loading,
            self.validate_dataset,
            self.validate_data_processing,
            self.validate_lora_config,
            self.validate_inference_presets
        ]
        
        for validation in validations:
            try:
                validation()
            except Exception as e:
                logger.error(f"Validation failed with exception: {e}")
            print()  # Blank line between sections
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results.values() if result['passed'])
        
        for test_name, result in self.validation_results.items():
            status = "✅" if result['passed'] else "❌"
            logger.info(f"{status} {test_name}")
        
        logger.info("=" * 60)
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"🎯 RESULT: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            logger.info("🟢 READY FOR TRAINING - Golden config validated!")
            return True
        elif success_rate >= 70:
            logger.info("🟡 MOSTLY READY - Some issues to address")
            return False
        else:
            logger.info("🔴 NOT READY - Significant issues found")
            return False

def main():
    """Main validation entry point"""
    validator = MistralGoldenConfigValidator()
    
    try:
        success = validator.run_full_validation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()