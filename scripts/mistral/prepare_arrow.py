#!/usr/bin/env python3
"""
Offline preprocessing: tokenize + pack JSONL to fixed blocks
Boringly reliable approach for 90K+ docs on 64GB RAM
"""

import os
import multiprocessing as mp
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Environment configuration
IN_PATH  = os.environ.get("TRAIN_JSONL", "train.jsonl")
VAL_PATH = os.environ.get("VAL_JSONL", "val.jsonl")
OUT_DIR  = os.environ.get("OUT_DIR", "data_packed_512")
MODEL    = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.1")
BLOCK    = int(os.environ.get("BLOCK_SIZE", "512"))
NUMPROC  = max(1, mp.cpu_count() - 1)
WBATCH   = int(os.environ.get("WRITER_BATCH_SIZE", "512"))

print(f"🔧 Configuration:")
print(f"  Train: {IN_PATH}")
print(f"  Val: {VAL_PATH}")
print(f"  Model: {MODEL}")
print(f"  Block size: {BLOCK}")
print(f"  Processes: {NUMPROC}")
print(f"  Output: {OUT_DIR}")

# Load tokenizer
print(f"\n📝 Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
tok.model_max_length = BLOCK
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  Tokenizer loaded: {tok.__class__.__name__}")
print(f"  Vocab size: {tok.vocab_size}")

# Prepare dataset paths
raw = {"train": IN_PATH}
if os.path.exists(VAL_PATH):
    raw["validation"] = VAL_PATH
    print(f"  Found validation file: {VAL_PATH}")
else:
    print(f"  No validation file found at: {VAL_PATH}")

# Load datasets
print(f"\n📂 Loading datasets...")
ds = DatasetDict({
    split: load_dataset("json", data_files=path, split="train")
    for split, path in raw.items()
})

for split, dataset in ds.items():
    print(f"  {split}: {len(dataset):,} examples")

def tokenize(batch):
    """Tokenize batch of text examples"""
    return tok(batch["text"], truncation=False)  # no trunc here; we'll pack

def pack(batch):
    """
    Concatenate tokenized sequences then split into fixed BLOCK-sized chunks
    This is the standard approach for causal LM training
    """
    import itertools
    
    # Concatenate all sequences in the batch
    ids = list(itertools.chain.from_iterable(batch["input_ids"]))
    att = list(itertools.chain.from_iterable(batch["attention_mask"]))
    
    # Drop remainder that doesn't fit in complete blocks
    n = (len(ids) // BLOCK) * BLOCK
    ids, att = ids[:n], att[:n]
    
    # Split into BLOCK-sized chunks
    input_ids = [ids[i:i+BLOCK] for i in range(0, n, BLOCK)]
    attention_mask = [att[i:i+BLOCK] for i in range(0, n, BLOCK)]
    
    # For causal LM, labels = input_ids
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": input_ids
    }

# Process each split
for split in ds.keys():
    print(f"\n🔄 Processing {split} split...")
    
    # Step 1: Tokenize to Arrow files on disk
    print(f"  Tokenizing...")
    toked = ds[split].map(
        tokenize,
        batched=True,
        num_proc=NUMPROC,
        remove_columns=[c for c in ds[split].column_names if c != "text"],
        keep_in_memory=False,
        writer_batch_size=WBATCH,
        desc=f"tokenize {split}",
    )
    
    # Step 2: Pack to fixed blocks on disk
    print(f"  Packing to {BLOCK}-token blocks...")
    packed = toked.map(
        pack,
        batched=True,
        batch_size=64,
        remove_columns=[c for c in toked.column_names if c not in ("input_ids", "attention_mask")],
        keep_in_memory=False,
        writer_batch_size=WBATCH,
        desc=f"pack {split} (block={BLOCK})",
    )
    
    # Step 3: Shuffle (creates permutation file; no RAM explosion)
    print(f"  Shuffling...")
    shuffled = packed.shuffle(seed=42)
    
    ds[split] = shuffled
    print(f"  ✅ {split}: {len(shuffled):,} packed blocks")

# Save to disk
print(f"\n💾 Saving to {OUT_DIR}...")
os.makedirs(OUT_DIR, exist_ok=True)
ds.save_to_disk(OUT_DIR)

print(f"\n✅ Preprocessing complete!")
print(f"   Output directory: {OUT_DIR}")
print(f"   To use in training: TOKENIZED_DIR={OUT_DIR} python train_mistral_simple.py [args]")

# Print some stats
total_blocks = sum(len(ds[split]) for split in ds.keys())
total_tokens = total_blocks * BLOCK
print(f"\n📊 Stats:")
print(f"   Total blocks: {total_blocks:,}")
print(f"   Total tokens: {total_tokens:,}")
print(f"   Avg tokens per original example: {total_tokens // sum(len(load_dataset('json', data_files=path, split='train')) for path in raw.values()):,}")