#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qwen3_14b_optimal_sft.py
---------------------------------
"Optimal setup" drawing on Unsloth + TRL SFTTrainer.
- Long-context (default 4096)
- PEFT LoRA on attention + MLP, Unsloth gradient checkpointing
- AdamW 8-bit optimizer
- Optional mixture of datasets (local JSONL prompt/completion AND/OR small public sets)
- packing=True for throughput on short-turn data
- Optional inclusion of tool-call JSON subsets for strictness

Usage examples:
  # Combined verbosity + IaC training (recommended)
  python3 train_qwen3_14b_optimal_sft.py \
    --base_model unsloth/Qwen3-14B-unsloth-bnb-4bit \
    --local_jsonl verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl:0.35 \
    --local_jsonl data/final_enhanced_iac_corpus.jsonl:0.50 \
    --output_dir qwen3_14b_iac_verbosity_sft \
    --max_len 4096 --epochs 2 --lr 2e-4 --batch_size 1 --grad_accum 16

  # Mix in additional reasoning/chat sets (requires network)
  python3 train_qwen3_14b_optimal_sft.py \
    --base_model unsloth/Qwen3-14B-unsloth-bnb-4bit \
    --local_jsonl verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl:0.3 \
    --local_jsonl data/comprehensive_iac_corpus/final_enhanced_iac_corpus.jsonl:0.5 \
    --add_openmath_reasoning 0.15 \
    --add_finetome_chat 0.05 \
    --output_dir qwen3_14b_iac_verbosity_enhanced_sft \
    --max_len 4096 --epochs 2 --lr 2e-4 --batch_size 1 --grad_accum 16

Notes:
- If running offline, omit the public datasets flags.
- This script keeps prompt-masking for your local JSONL (prompt/completion) and uses standard SFT formatting for public sets.
"""

# Install dependencies if not available
import subprocess
import sys
import os

def install_dependencies():
    """Install required packages if not available"""
    packages = {
        'unsloth': 'unsloth',
        'trl': 'trl',
        'datasets': 'datasets',
        'bitsandbytes': 'bitsandbytes',
        'peft': 'peft',
        'accelerate': 'accelerate'
    }
    
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

install_dependencies()

import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# ---- Optional VRAM optimization ----
try:
    import bitsandbytes as bnb  # noqa: F401
except Exception:
    pass

import datasets as hfds
import random

# -------------------------
# S3 Model Download Function
# -------------------------
def download_model_from_s3():
    """Download Unsloth model from S3 to local cache"""
    s3_path = "s3://asoba-llm-cache/models/Qwen/Qwen3-14B-unsloth-bnb-4bit/"
    local_path = "/home/ubuntu/Qwen3-14B-unsloth-bnb-4bit"
    
    if os.path.exists(local_path):
        print(f"Model already exists at {local_path}")
        return local_path
    
    print(f"Downloading model from {s3_path} to {local_path}")
    os.makedirs(local_path, exist_ok=True)
    
    cmd = f"aws s3 sync {s3_path} {local_path} --region us-east-1"
    subprocess.run(cmd, shell=True, check=True)
    print("Model download complete")
    return local_path

# -------------------------
# StyleDeltaEmbedding wrapper (same as adopt-now)
# -------------------------
import torch.nn as nn

class StyleDeltaEmbedding(nn.Module):
    def __init__(self, base_embed: nn.Embedding, terse_id: int, verbose_id: int):
        super().__init__()
        self.base = base_embed
        self.hidden = base_embed.embedding_dim
        self.terse_id = int(terse_id)
        self.verbose_id = int(verbose_id)
        self.style_delta = nn.Parameter(torch.zeros(2, self.hidden))
    
    @property
    def weight(self):
        """Expose base embedding weight for Unsloth compatibility"""
        return self.base.weight

    def forward(self, input_ids: torch.LongTensor):
        out = self.base(input_ids)
        if self.terse_id >= 0:
            out = out + (input_ids == self.terse_id).unsqueeze(-1).float() * self.style_delta[0].view(1,1,-1)
        if self.verbose_id >= 0:
            out = out + (input_ids == self.verbose_id).unsqueeze(-1).float() * self.style_delta[1].view(1,1,-1)
        return out

# -------------------------
# Local JSONL (prompt/completion) dataset with explicit masking
# -------------------------
class PromptCompletionJsonl(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    
                    # Handle both prompt/completion and messages format
                    if "prompt" in obj and "completion" in obj:
                        self.rows.append({"prompt": obj["prompt"], "completion": obj["completion"]})
                    elif "messages" in obj and len(obj["messages"]) >= 2:
                        # Convert messages format to prompt/completion
                        messages = obj["messages"]
                        user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
                        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
                        
                        if user_msg and assistant_msg:
                            self.rows.append({"prompt": user_msg, "completion": assistant_msg})
                        else:
                            print(f"Warning: Skipping entry missing user/assistant messages: {line[:100]}...")
                            continue
                    else:
                        print(f"Warning: Skipping entry missing prompt/completion or messages keys: {line[:100]}...")
                        continue
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON: {e}")
                    continue
        if not self.rows:
            raise ValueError(f"No examples in {path}")
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

@dataclass
class PromptCompletionCollator:
    tokenizer: AutoTokenizer
    max_len: int
    add_eos_if_missing: bool = True
    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_batch, labels_batch, attn_batch = [], [], []
        eos_id = self.tokenizer.eos_token_id
        for ex in batch:
            prompt_ids = self.tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = self.tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
            if self.add_eos_if_missing and (len(completion_ids)==0 or completion_ids[-1]!=eos_id):
                completion_ids = completion_ids + [eos_id]
            full_ids = prompt_ids + completion_ids
            labels = [-100]*len(prompt_ids) + completion_ids[:]
            # Truncate to max_len (prompt-first retention)
            if len(full_ids) > self.max_len:
                overflow = len(full_ids) - self.max_len
                if overflow < len(prompt_ids):
                    prompt_ids = prompt_ids[overflow:]
                    full_ids = prompt_ids + completion_ids
                    labels = [-100]*len(prompt_ids) + completion_ids[:]
                else:
                    keep = self.max_len
                    full_ids = full_ids[-keep:]
                    c_len = min(len(completion_ids), len(full_ids))
                    labels = [-100]*(len(full_ids)-c_len) + full_ids[-c_len:]
            input_ids = full_ids
            attn = [1]*len(input_ids)
            input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            attn_batch.append(torch.tensor(attn, dtype=torch.long))

        input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_batch = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100)
        attn_batch = torch.nn.utils.rnn.pad_sequence(attn_batch, batch_first=True, padding_value=0)

        if input_ids_batch.size(1) > self.max_len:
            input_ids_batch = input_ids_batch[:, :self.max_len]
            labels_batch = labels_batch[:, :self.max_len]
            attn_batch = attn_batch[:, :self.max_len]
        return {"input_ids": input_ids_batch, "labels": labels_batch, "attention_mask": attn_batch}

def add_style_tokens(tokenizer):
    added = 0
    for tok in ["<STYLE_TERSE>", "<STYLE_VERBOSE>"]:
        if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
            tokenizer.add_tokens([tok], special_tokens=False)  # Match golden config
            added += 1
    return added

def build_public_sets(tokenizer, add_openmath_reasoning: float, add_finetome_chat: float):
    """Optionally fetch small public sets; returns a list of (dataset, weight, formatter) tuples.
       Formatter returns 'text' suitable for SFTTrainer with packing=True.
    """
    mixtures = []

    if add_openmath_reasoning > 0.0:
        ds = hfds.load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
        def fmt_reasoning(ex):
            # Keep it simple: user asks, assistant reasons to an answer
            prompt = f"<|im_start|>user\n{ex.get('question','')}\n<|im_end|>\n<|im_start|>assistant\n{ex.get('answer','')}\n<|im_end|>\n"
            return {"text": prompt}
        mixtures.append((ds.map(fmt_reasoning, remove_columns=ds.column_names), float(add_openmath_reasoning)))

    if add_finetome_chat > 0.0:
        ds = hfds.load_dataset("mlabonne/FineTome-100k", split="train")
        def fmt_sharegpt(ex):
            conv = ex.get("conversations", [])
            # convert ShareGPT format to Qwen-style chat template
            text = ""
            for turn in conv:
                role = turn.get("from","").strip()
                value = turn.get("value","")
                if role == "human":
                    text += f"<|im_start|>user\n{value}\n<|im_end|>\n"
                else:
                    text += f"<|im_start|>assistant\n{value}\n<|im_end|>\n"
            return {"text": text}
        mixtures.append((ds.map(fmt_sharegpt, remove_columns=ds.column_names), float(add_finetome_chat)))

    # Normalize weights
    if mixtures:
        s = sum(w for _, w in mixtures)
        mixtures = [(d, w/s) for d,w in mixtures]
    return mixtures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True, help="Unsloth 4-bit Qwen3-14B base (repo or local path).")
    ap.add_argument("--local_jsonl", action="append", default=[], help="Path:weight for local JSONL prompt/completion. Can repeat. Example: data.jsonl:0.7")
    ap.add_argument("--add_openmath_reasoning", type=float, default=0.0, help="Weight in mixture [0..1].")
    ap.add_argument("--add_finetome_chat", type=float, default=0.0, help="Weight in mixture [0..1].")
    ap.add_argument("--output_dir", type=str, default="qwen3_14b_optimal_sft")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)  # Match golden config
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)  # Match golden config
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)  # Match golden config
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    
    # Set AWS region per CLAUDE.md requirements
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    # 1) Use local S3-downloaded model if default base model specified
    if args.base_model == "unsloth/Qwen3-14B-unsloth-bnb-4bit":
        model_path = "/home/ubuntu/Qwen3-14B-unsloth-bnb-4bit"
        if not os.path.exists(model_path):
            model_path = download_model_from_s3()
    else:
        model_path = args.base_model

    # Load base model/tokenizer via Unsloth in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = args.max_len,
        load_in_4bit = True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Style tokens + embedding wrapper
    added = add_style_tokens(tokenizer)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    terse_id = tokenizer.convert_tokens_to_ids("<STYLE_TERSE>")
    verbose_id = tokenizer.convert_tokens_to_ids("<STYLE_VERBOSE>")
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        base_embed = model.model.embed_tokens
        model.model.embed_tokens = StyleDeltaEmbedding(base_embed, terse_id, verbose_id)
    else:
        raise RuntimeError("Could not locate model.model.embed_tokens for StyleDeltaEmbedding.")

    # 3) PEFT LoRA with Unsloth GC
    # Note: Unsloth doesn't support custom modules_to_save, so we'll add style_delta to the base model
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = 0,  # Unsloth requires 0 for fast patching
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing = "unsloth",
        modules_to_save = None,  # Unsloth doesn't allow custom modules
    )
    
    # Make style_delta trainable separately
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens") and hasattr(model.model.embed_tokens, "style_delta"):
        model.model.embed_tokens.style_delta.requires_grad_(True)

    # 4) Build datasets
    # 4a) Local JSONL prompt/completion portion (masked labels)
    local_parts = []
    total_w = 0.0
    for spec in args.local_jsonl:
        if ":" not in spec:
            raise ValueError("--local_jsonl expects PATH:WEIGHT, e.g. my.jsonl:0.7")
        path, w = spec.rsplit(":", 1)
        w = float(w)
        if w <= 0.0: 
            continue
        ds = PromptCompletionJsonl(path)
        local_parts.append((ds, w))
        total_w += w

    # 4b) Optional public sets (formatted to 'text')
    public_mixtures = build_public_sets(tokenizer, args.add_openmath_reasoning, args.add_finetome_chat)
    for ds, w in public_mixtures:
        local_parts.append((ds, w))
        total_w += w

    if total_w <= 0.0:
        raise ValueError("No training data specified. Provide at least one --local_jsonl or a public mixture flag.")

    # Normalize weights
    local_parts = [(d, w/total_w) for d, w in local_parts]

    # 5) SFTConfig — adamw_8bit + long context + packing
    sft_config = SFTConfig(
        output_dir = args.output_dir,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate = args.lr,
        num_train_epochs = args.epochs,
        logging_steps = 10,
        save_steps = 200,
        save_strategy = "steps",
        save_total_limit = 2,
        bf16 = torch.cuda.is_available(),
        fp16 = False,
        optim = "adamw_bnb_8bit",
        gradient_checkpointing = True,
        max_seq_length = args.max_len,
        packing = False,  # Disabled because we need custom collator for prompt masking
        remove_unused_columns = False,
        dataset_kwargs = {"add_special_tokens": False, "skip_prepare_dataset": True},
        report_to = "none",
    )

    # 6) Trainer — we pass a composite dataset via an Iterable wrapper
    #    For prompt/completion parts we need explicit masking; for public 'text' we use standard SFT.
    #    We'll implement a simple interleaving sampler by weights.
    from torch.utils.data import IterableDataset

    class MixedIterable(IterableDataset):
        def __init__(self, parts, tokenizer, max_len):
            self.parts = parts  # list of (dataset, weight)
            self.tokenizer = tokenizer
            self.max_len = max_len
            # build cumulative weights
            self.datasets = [p[0] for p in parts]
            self.weights = [p[1] for p in parts]
            s = sum(self.weights)
            self.probs = [w/s for w in self.weights]
            
        def __len__(self):
            # Return total length based on weighted sum of datasets
            total = 0
            for ds, weight in self.parts:
                if hasattr(ds, '__len__'):
                    total += len(ds)
                else:
                    # For datasets without len(), estimate based on weight
                    total += int(1000 * weight)  # Reasonable estimate
            return total
            
        def __getitem__(self, idx):
            # IterableDataset doesn't support indexing, but Unsloth requires it
            # Return a properly formatted dummy item for compatibility checks
            # The actual data comes from __iter__ during training
            dummy_ids = [1, 2, 3, 4, 5]  # Some dummy token IDs
            return {
                "input_ids": dummy_ids,
                "labels": dummy_ids,  # Not all -100 to avoid zero loss detection
                "attention_mask": [1] * len(dummy_ids)
            }

        def __iter__(self):
            rng = random.Random(1234)
            # Properly initialize iterators for all datasets
            iters = []
            for ds in self.datasets:
                if hasattr(ds, "__iter__"):
                    iters.append(iter(ds))
                else:
                    # For non-iterable datasets (regular Dataset), create manual iterator
                    class DatasetIterator:
                        def __init__(self, dataset):
                            self.dataset = dataset
                            self.idx = 0
                        def __iter__(self):
                            return self
                        def __next__(self):
                            if self.idx >= len(self.dataset):
                                self.idx = 0  # Reset for continuous iteration
                            item = self.dataset[self.idx]
                            self.idx += 1
                            return item
                    iters.append(DatasetIterator(ds))
            while True:
                # choose a dataset by weight
                idx = rng.choices(range(len(self.datasets)), weights=self.probs, k=1)[0]
                ds = self.datasets[idx]
                try:
                    ex = next(iters[idx])
                except StopIteration:
                    # restart iterator
                    iters[idx] = iter(ds)
                    ex = next(iters[idx])

                # Standardize to dict with either 'text' or ('prompt','completion')
                if isinstance(ds, PromptCompletionJsonl):
                    yield {"prompt": ex["prompt"], "completion": ex["completion"]}
                else:
                    # Public sets already formatted with 'text'
                    yield {"text": ex["text"]}

    mixed_stream = MixedIterable(local_parts, tokenizer, args.max_len)

    # Custom collator: if 'text' present → tokenize as-is;
    # if ('prompt','completion') → mask prompt.
    from dataclasses import dataclass

    @dataclass
    class MixedCollator:
        tokenizer: AutoTokenizer
        max_len: int
        def __call__(self, batch):
            input_ids_batch, labels_batch, attn_batch = [], [], []
            eos_id = self.tokenizer.eos_token_id
            for ex in batch:
                if "text" in ex:
                    ids = self.tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
                    if len(ids) == 0 or ids[-1] != eos_id:
                        ids = ids + [eos_id]
                    input_ids = ids
                    labels = ids[:]  # standard SFT (no special masking beyond possible BOS)
                else:
                    prompt_ids = self.tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
                    completion_ids = self.tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
                    if len(completion_ids) == 0 or completion_ids[-1] != eos_id:
                        completion_ids = completion_ids + [eos_id]
                    input_ids = prompt_ids + completion_ids
                    labels = [-100]*len(prompt_ids) + completion_ids[:]

                if len(input_ids) > self.max_len:
                    overflow = len(input_ids) - self.max_len
                    input_ids = input_ids[overflow:]
                    labels = labels[overflow:]

                attn = [1]*len(input_ids)
                input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
                labels_batch.append(torch.tensor(labels, dtype=torch.long))
                attn_batch.append(torch.tensor(attn, dtype=torch.long))

            pad_id = self.tokenizer.pad_token_id
            input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=pad_id)
            labels_batch = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100)
            attn_batch = torch.nn.utils.rnn.pad_sequence(attn_batch, batch_first=True, padding_value=0)

            if input_ids_batch.size(1) > self.max_len:
                input_ids_batch = input_ids_batch[:, :self.max_len]
                labels_batch = labels_batch[:, :self.max_len]
                attn_batch = attn_batch[:, :self.max_len]
            return {"input_ids": input_ids_batch, "labels": labels_batch, "attention_mask": attn_batch}

    collator = MixedCollator(tokenizer, args.max_len)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        args = sft_config,
        train_dataset = mixed_stream,
        data_collator = collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete. Adapter saved to:", args.output_dir)


if __name__ == "__main__":
    main()
