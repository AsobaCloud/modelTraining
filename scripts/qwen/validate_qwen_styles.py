#!/usr/bin/env python3
"""
Validate Qwen style differentiation with proper decoding configs
Following user's triage guidance
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_style(model, tokenizer, question, style):
    """Generate with style-specific decoding parameters"""
    messages = [{"role": "user", "content": f"<STYLE_{style}>\n{question}"}]
    
    # Apply chat template
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Log token IDs to confirm style token presence
    token_list = input_ids[0].tolist()
    terse_id = tokenizer.convert_tokens_to_ids("<STYLE_TERSE>")
    verbose_id = tokenizer.convert_tokens_to_ids("<STYLE_VERBOSE>")
    
    if style == "TERSE":
        assert terse_id in token_list, f"TERSE token {terse_id} not found in {token_list[:20]}..."
    else:
        assert verbose_id in token_list, f"VERBOSE token {verbose_id} not found in {token_list[:20]}..."
    
    # Style-specific generation parameters
    if style == "TERSE":
        gen_kwargs = {
            "max_new_tokens": 20,  # Allow some room but expect short
            "temperature": 0.1,
            "top_p": 0.95,
            "do_sample": True,
        }
    else:  # VERBOSE
        gen_kwargs = {
            "min_new_tokens": 40,  # Force longer generation
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
    
    # Common parameters
    gen_kwargs.update({
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    })
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)
    
    # Decode only generated part
    generated = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=False)
    
    # Remove EOS token if present
    if tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    
    return response, len(tokenizer.encode(response))

def main():
    logger.info("Loading model and tokenizer...")
    
    # Load base model and adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        "/opt/models/Qwen3-14B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        "qwen3_14b_verbosity_pc_lora",
        torch_dtype=torch.bfloat16
    )
    
    # Load tokenizer with style tokens
    tokenizer = AutoTokenizer.from_pretrained("qwen3_14b_verbosity_pc_lora", trust_remote_code=True)
    
    # Verify style tokens
    terse_id = tokenizer.convert_tokens_to_ids("<STYLE_TERSE>")
    verbose_id = tokenizer.convert_tokens_to_ids("<STYLE_VERBOSE>")
    logger.info(f"Style token IDs - TERSE: {terse_id}, VERBOSE: {verbose_id}")
    
    # Test questions
    test_questions = [
        "What is 2+2?",
        "What is machine learning?",
        "Explain photosynthesis",
        "What is Python?",
        "Define gravity"
    ]
    
    # Track token counts
    terse_counts = []
    verbose_counts = []
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION WITH STYLE-SPECIFIC DECODING")
    logger.info("="*60 + "\n")
    
    for question in test_questions:
        logger.info(f"\nQuestion: {question}")
        logger.info("-" * 40)
        
        # Generate TERSE
        terse_response, terse_tokens = generate_style(model, tokenizer, question, "TERSE")
        logger.info(f"TERSE ({terse_tokens} tokens): {terse_response}")
        terse_counts.append(terse_tokens)
        
        # Generate VERBOSE
        verbose_response, verbose_tokens = generate_style(model, tokenizer, question, "VERBOSE")
        logger.info(f"VERBOSE ({verbose_tokens} tokens): {verbose_response}")
        verbose_counts.append(verbose_tokens)
        
        # Show ratio
        ratio = verbose_tokens / max(terse_tokens, 1)
        logger.info(f"Token ratio (VERBOSE/TERSE): {ratio:.1f}x")
    
    # Summary statistics
    avg_terse = sum(terse_counts) / len(terse_counts)
    avg_verbose = sum(verbose_counts) / len(verbose_counts)
    overall_ratio = avg_verbose / avg_terse
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    logger.info(f"Average TERSE tokens: {avg_terse:.1f}")
    logger.info(f"Average VERBOSE tokens: {avg_verbose:.1f}")
    logger.info(f"Overall ratio: {overall_ratio:.1f}x")
    logger.info(f"Target ratio: ≥3.0x")
    logger.info(f"SUCCESS: {overall_ratio >= 3.0}")

if __name__ == "__main__":
    main()