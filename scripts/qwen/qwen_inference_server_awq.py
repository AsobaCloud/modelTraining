#!/usr/bin/env python3
"""
Qwen Inference Server - AWQ Optimized Version
FastAPI server for Qwen3-14B-AWQ with CLAUDE.md methodology
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen_inference_server_awq")

# Global model and tokenizer
model = None
tokenizer = None
system_prompt = None

# Configuration
CONFIG_PATH = "qwen_config.json"
SYSTEM_PROMPT_PATH = "qwen_claude_md_system_prompt_concise.txt"

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for code generation")
    max_length: int = Field(2048, ge=1, le=4096, description="Maximum length of generated text")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    complexity: Optional[str] = Field("auto", description="Request complexity (simple, medium, complex, auto)")
    include_tests: Optional[bool] = Field(True, description="Include test code generation")
    format: Optional[str] = Field("auto", description="Response format preference")

class GenerationResponse(BaseModel):
    code: str = Field(..., description="Generated code")
    methodology_applied: str = Field(..., description="CLAUDE.md methodology phase applied")
    complexity_detected: str = Field(..., description="Detected request complexity")
    generation_time: float = Field(..., description="Time taken to generate the response")
    includes_tests: bool = Field(..., description="Whether response includes test code")
    metadata: Dict = Field(..., description="Additional metadata about the generation")

class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    gpu_memory_total: float
    gpu_memory_used: float
    system_prompt_loaded: bool
    claude_md_enabled: bool
    quantization: str
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
    """Load the Qwen AWQ model"""
    global model, tokenizer, system_prompt
    
    config = load_config()
    system_prompt = load_system_prompt()
    
    if not system_prompt:
        raise Exception("Failed to load system prompt")
    
    logger.info("Loading Qwen3-14B-AWQ model...")
    
    model_path = "/opt/models/Qwen3-14B-AWQ"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load AWQ model - no quantization config needed, it's pre-quantized
    model = AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=True,
        trust_remote_code=True,
        safetensors=True
    )
    
    logger.info("AWQ model loaded successfully")
    memory_total, memory_used = get_gpu_memory_usage()
    logger.info(f"GPU Memory - Total: {memory_total:.2f}GB, Used: {memory_used:.2f}GB")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Qwen AWQ inference server...")
    await load_model()
    yield
    logger.info("Shutting down Qwen AWQ inference server...")

# Initialize FastAPI app
app = FastAPI(
    title="Qwen CLAUDE.md Inference Server (AWQ)",
    description="FastAPI server for Qwen3-14B-AWQ with integrated CLAUDE.md methodology",
    version="2.0.0",
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
    
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model="Qwen3-14B-AWQ",
        model_loaded=model is not None,
        gpu_memory_total=memory_total,
        gpu_memory_used=memory_used,
        system_prompt_loaded=system_prompt is not None,
        claude_md_enabled=True,
        quantization="AWQ 4-bit",
        endpoints=["/generate", "/health", "/docs"]
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_code(request: GenerationRequest):
    """Generate code using Qwen AWQ with CLAUDE.md methodology"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    config = load_config()
    
    # Detect complexity if auto
    if request.complexity == "auto":
        complexity = detect_complexity(request.prompt, config)
    else:
        complexity = request.complexity
    
    # Build prompt with methodology
    full_prompt = build_prompt_with_methodology(request.prompt, complexity, system_prompt)
    
    try:
        # Tokenize input - AWQ models are automatically on GPU
        inputs = tokenizer(full_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Set appropriate max_new_tokens based on complexity
        if complexity == "simple":
            max_new_tokens = 5      # Very short for simple questions
            temperature = 0.1       # Near-deterministic
            do_sample = False       # Greedy decoding
        elif complexity == "medium":
            max_new_tokens = 100
            temperature = request.temperature
            do_sample = True
        else:  # complex
            max_new_tokens = 300
            temperature = request.temperature
            do_sample = True
        
        # Generate response with AWQ optimizations
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
            metadata={
                "model": "Qwen3-14B-AWQ",
                "quantization": "AWQ 4-bit",
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
    return {
        "message": "Qwen CLAUDE.md Inference Server (AWQ Optimized)",
        "model": "Qwen3-14B-AWQ",
        "quantization": "AWQ 4-bit",
        "methodology": "CLAUDE.md integrated",
        "endpoints": ["/generate", "/health", "/docs"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "qwen_inference_server_awq:app",
        host="0.0.0.0",
        port=8002,
        workers=1,
        loop="asyncio"
    )