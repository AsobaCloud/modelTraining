#!/usr/bin/env python3
"""
Qwen Inference Server with LoRA - Enhanced with verbosity control fine-tuning
"""

import asyncio
import json
import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen_inference_server_lora")

# Global model and tokenizer
model = None
tokenizer = None
system_prompt = None

# Configuration
CONFIG_PATH = "qwen_config.json"
SYSTEM_PROMPT_PATH = "qwen_claude_md_system_prompt_concise.txt"
LORA_ADAPTER_PATH = "./qwen_verbosity_lora"  # LoRA adapter directory

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for code generation")
    max_length: int = Field(2048, ge=1, le=4096, description="Maximum length of generated text")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    complexity: Optional[str] = Field("auto", description="Request complexity (simple, medium, complex, auto)")
    include_tests: Optional[bool] = Field(True, description="Include test code generation")
    format: Optional[str] = Field("auto", description="Response format preference")
    use_lora: Optional[bool] = Field(True, description="Use LoRA verbosity control adapter")

class GenerationResponse(BaseModel):
    code: str = Field(..., description="Generated code")
    methodology_applied: str = Field(..., description="CLAUDE.md methodology phase applied")
    complexity_detected: str = Field(..., description="Detected request complexity")
    generation_time: float = Field(..., description="Time taken to generate the response")
    includes_tests: bool = Field(..., description="Whether response includes test code")
    lora_used: bool = Field(..., description="Whether LoRA adapter was used")
    metadata: Dict = Field(..., description="Additional metadata about the generation")

class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    gpu_memory_total: float
    gpu_memory_used: float
    system_prompt_loaded: bool
    claude_md_enabled: bool
    lora_adapter_loaded: bool
    endpoints: List[str]

def load_config():
    """Load configuration from JSON file"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def load_system_prompt():
    """Load system prompt from file"""
    try:
        with open(SYSTEM_PROMPT_PATH, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return None

def get_gpu_memory_usage():
    """Get GPU memory usage statistics"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return memory_reserved, memory_allocated
    return 0.0, 0.0

def detect_complexity(prompt: str, config: dict) -> str:
    """Detect request complexity based on keywords and length"""
    keywords = config.get("claude_md_integration", {}).get("workflow_keywords", [])
    thresholds = config.get("claude_md_integration", {}).get("complexity_threshold", {})
    
    # Check for complex keywords
    prompt_lower = prompt.lower()
    keyword_matches = sum(1 for keyword in keywords if keyword in prompt_lower)
    
    # Calculate complexity score
    length_score = min(len(prompt) / 200, 3)  # Max 3 points for length
    keyword_score = min(keyword_matches * 2, 5)  # Max 5 points for keywords
    complexity_score = length_score + keyword_score
    
    if complexity_score >= thresholds.get("complex", 8):
        return "complex"
    elif complexity_score >= thresholds.get("medium", 6):
        return "medium"
    else:
        return "simple"

def build_prompt_with_methodology(user_prompt: str, complexity: str, system_prompt: str) -> str:
    """Build the complete prompt with appropriate methodology"""
    if complexity == "simple":
        methodology_instruction = "\nGenerate code directly with brief explanation and basic testing suggestions."
    elif complexity == "medium":
        methodology_instruction = "\nApply PLAN and CODE phases: analyze requirements, design approach, then implement with tests."
    else:  # complex
        methodology_instruction = "\nApply full CLAUDE.md methodology: EXPLORE → PLAN → CODE → COMMIT. Be systematic and comprehensive."
    
    return f"{system_prompt}\n\n{methodology_instruction}\n\nUser Request: {user_prompt}"

async def load_model():
    """Load the Qwen model with optional LoRA adapter"""
    global model, tokenizer, system_prompt
    
    config = load_config()
    system_prompt = load_system_prompt()
    
    if not system_prompt:
        raise Exception("Failed to load system prompt")
    
    logger.info("Loading Qwen3-14B model with 8-bit quantization...")
    
    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["lm_head"],
        llm_int8_enable_fp32_cpu_offload=False
    )
    
    # Download from S3 if needed
    model_path = config.get("model_path", "s3://asoba-llm-cache/models/Qwen/Qwen3-14B")
    if model_path.startswith("s3://"):
        # Use HuggingFace Hub to load directly from S3 or use local cache
        model_path = "Qwen/Qwen3-14B"  # Use Qwen3-14B from S3 cache
        logger.info(f"Using HuggingFace model: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Try to load LoRA adapter
    lora_loaded = False
    if os.path.exists(LORA_ADAPTER_PATH):
        try:
            logger.info(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
            lora_loaded = True
            logger.info("LoRA adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LoRA adapter: {e}")
            logger.info("Using base model without LoRA")
            model = base_model
    else:
        logger.info("No LoRA adapter found, using base model")
        model = base_model
    
    logger.info("Model loaded successfully")
    memory_total, memory_used = get_gpu_memory_usage()
    logger.info(f"GPU Memory - Total: {memory_total:.2f}GB, Used: {memory_used:.2f}GB")
    logger.info(f"LoRA adapter: {'Loaded' if lora_loaded else 'Not loaded'}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Qwen inference server with LoRA support...")
    await load_model()
    yield
    logger.info("Shutting down Qwen inference server...")

# Initialize FastAPI app
app = FastAPI(
    title="Qwen CLAUDE.md Inference Server with LoRA",
    description="FastAPI server for Qwen3-14B with integrated CLAUDE.md methodology and verbosity control",
    version="1.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    memory_total, memory_used = get_gpu_memory_usage()
    
    # Check if LoRA is loaded
    lora_loaded = hasattr(model, 'peft_config') if model else False
    
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model="Qwen3-14B" + (" + LoRA" if lora_loaded else ""),
        model_loaded=model is not None,
        gpu_memory_total=memory_total,
        gpu_memory_used=memory_used,
        system_prompt_loaded=system_prompt is not None,
        claude_md_enabled=True,
        lora_adapter_loaded=lora_loaded,
        endpoints=["/generate", "/health", "/docs"]
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_code(request: GenerationRequest):
    """Generate code using Qwen with CLAUDE.md methodology and optional LoRA"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    config = load_config()
    
    # Check if LoRA is available and requested
    lora_available = hasattr(model, 'peft_config')
    use_lora = request.use_lora and lora_available
    
    # Detect complexity if auto
    if request.complexity == "auto":
        complexity = detect_complexity(request.prompt, config)
    else:
        complexity = request.complexity
    
    # Build prompt with methodology
    full_prompt = build_prompt_with_methodology(request.prompt, complexity, system_prompt)
    
    try:
        # Tokenize input
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        # Set appropriate max_new_tokens based on complexity
        if complexity == "simple":
            max_new_tokens = 10  # Very short for simple questions
            temperature = 0.1    # Low temperature for factual answers
            do_sample = False    # Greedy decoding (fastest)
        elif complexity == "medium":
            max_new_tokens = 100
            temperature = request.temperature
            do_sample = True
        else:  # complex
            max_new_tokens = 300
            temperature = request.temperature
            do_sample = True
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=0.9 if do_sample else None,
                top_k=50 if do_sample else None,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                use_cache=True
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(full_prompt):].strip()
        
        generation_time = time.time() - start_time
        
        # Determine methodology applied
        methodology_map = {
            "simple": "Direct Code Generation",
            "medium": "PLAN → CODE",
            "complex": "EXPLORE → PLAN → CODE → COMMIT"
        }
        
        return GenerationResponse(
            code=response_text,
            methodology_applied=methodology_map.get(complexity, "Direct"),
            complexity_detected=complexity,
            generation_time=generation_time,
            includes_tests=request.include_tests and ("test" in response_text.lower() or "assert" in response_text.lower()),
            lora_used=use_lora,
            metadata={
                "model": "Qwen3-14B",
                "quantization": "8-bit",
                "lora_adapter": "verbosity_control" if use_lora else "none",
                "prompt_length": len(full_prompt),
                "response_length": len(response_text),
                "max_new_tokens": max_new_tokens,
                "gpu_memory_used": get_gpu_memory_usage()[1]
            }
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    lora_status = " + LoRA" if (model and hasattr(model, 'peft_config')) else ""
    return {
        "message": "Qwen CLAUDE.md Inference Server with LoRA Support",
        "model": f"Qwen3-14B{lora_status}",
        "methodology": "CLAUDE.md integrated",
        "verbosity_control": "LoRA fine-tuned" if lora_status else "Basic",
        "endpoints": ["/generate", "/health", "/docs"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "qwen_inference_server_lora:app",
        host="0.0.0.0",
        port=8002,
        workers=1,
        loop="asyncio"
    )