#!/usr/bin/env python3
"""
Finetune 1.5B NeuTTS with V2 Zero-Shot Voice Cloning Template

This script finetunes the extended 1.5B model on pre-encoded data
using the V2 chat template format with reference audio conditioning.

V2 CHAT TEMPLATE:
=================
user: Convert the text to speech:
<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|>
<|REF_SPEECH_START|>{ref_codes}<|REF_SPEECH_END|>
<|TARGET_TEXT_START|>{target_text}<|TARGET_TEXT_END|>
assistant:<|TARGET_CODES_START|>{target_codes}<|TARGET_CODES_END|>

LABEL MASKING:
==============
- Everything before <|TARGET_CODES_START|> is masked with -100
- Train ONLY on: <|TARGET_CODES_START|>{target_codes}<|TARGET_CODES_END|>

INPUT DATA FORMAT:
==================
{
    "__key__": "sample_000000",
    "text": "target text",
    "codes": [1234, 5678, ...],      # NeuCodec codes for target
    "ref_text": "reference text",
    "ref_codes": [9876, 5432, ...]   # NeuCodec codes for reference
}

Usage:
    # Single GPU
    python3 finetune_1.5b_v2.py config_1.5b_v2.yaml

    # Multi-GPU (8x H100)
    torchrun --nproc_per_node=8 finetune_1.5b_v2.py config_1.5b_v2.yaml

    # DeepSpeed ZeRO-2
    deepspeed --num_gpus=8 finetune_1.5b_v2.py config_1.5b_v2.yaml --deepspeed ds_config.json
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import torch
from fire import Fire
from omegaconf import OmegaConf
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from loguru import logger as LOGGER
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# V2 Special Tokens
V2_TOKENS = {
    "REF_TEXT_START": "<|REF_TEXT_START|>",
    "REF_TEXT_END": "<|REF_TEXT_END|>",
    "REF_SPEECH_START": "<|REF_SPEECH_START|>",
    "REF_SPEECH_END": "<|REF_SPEECH_END|>",
    "TARGET_TEXT_START": "<|TARGET_TEXT_START|>",
    "TARGET_TEXT_END": "<|TARGET_TEXT_END|>",
    "TARGET_CODES_START": "<|TARGET_CODES_START|>",
    "TARGET_CODES_END": "<|TARGET_CODES_END|>",
}


@dataclass
class V2DataCollator:
    """
    Data collator for V2 zero-shot voice cloning.
    Handles dynamic padding and attention mask creation.
    """
    tokenizer: Any
    max_length: int = 8192
    pad_to_multiple_of: int = 8
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Get max length in batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        # Pad to multiple
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Cap at max_length
        max_len = min(max_len, self.max_length)
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for f in features:
            input_ids = f["input_ids"][:max_len]
            labels = f["labels"][:max_len]
            
            # Pad
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len  # Don't compute loss on padding
            
            attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in input_ids]
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def verify_v2_tokens(tokenizer) -> Dict[str, int]:
    """Verify all V2 tokens exist and return their IDs."""
    token_ids = {}
    missing = []
    
    for name, token in V2_TOKENS.items():
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid == tokenizer.unk_token_id:
            missing.append(token)
        else:
            token_ids[name] = tid
    
    if missing:
        raise ValueError(f"Missing V2 tokens in tokenizer: {missing}\n"
                        f"Run extend_tokenizer_1.5b.py first!")
    
    return token_ids


def build_v2_sample(
    sample: Dict[str, Any],
    tokenizer,
    token_ids: Dict[str, int],
    max_length: int = 8192,
) -> Optional[Dict[str, Any]]:
    """
    Build V2 training sample from pre-encoded data.
    
    Args:
        sample: Dict with keys: text, codes, ref_text, ref_codes
        tokenizer: Extended tokenizer with V2 tokens
        token_ids: Dict mapping token names to IDs
        max_length: Maximum sequence length
    
    Returns:
        Dict with input_ids, labels (or None if invalid)
    """
    try:
        # Extract data
        target_text = sample.get("text", "").strip()
        target_codes = sample.get("codes", [])
        ref_text = sample.get("ref_text", "").strip()
        ref_codes = sample.get("ref_codes", [])
        
        # Validate
        if not target_text or not target_codes:
            return None
        if not ref_text or not ref_codes:
            return None
        
        # Convert codes to speech tokens
        ref_codes_str = "".join([f"<|speech_{c}|>" for c in ref_codes])
        target_codes_str = "".join([f"<|speech_{c}|>" for c in target_codes])
        
        # Tokenize components
        user_prefix = "user: Convert the text to speech:"
        user_prefix_ids = tokenizer.encode(user_prefix, add_special_tokens=False)
        
        ref_text_ids = tokenizer.encode(ref_text, add_special_tokens=False)
        ref_codes_ids = tokenizer.encode(ref_codes_str, add_special_tokens=False)
        target_text_ids = tokenizer.encode(target_text, add_special_tokens=False)
        
        assistant_prefix = "\nassistant:"
        assistant_prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
        
        target_codes_ids = tokenizer.encode(target_codes_str, add_special_tokens=False)
        
        # Build input_ids with V2 template
        input_ids = (
            user_prefix_ids +
            [token_ids["REF_TEXT_START"]] + ref_text_ids + [token_ids["REF_TEXT_END"]] +
            [token_ids["REF_SPEECH_START"]] + ref_codes_ids + [token_ids["REF_SPEECH_END"]] +
            [token_ids["TARGET_TEXT_START"]] + target_text_ids + [token_ids["TARGET_TEXT_END"]] +
            assistant_prefix_ids +
            [token_ids["TARGET_CODES_START"]] + target_codes_ids + [token_ids["TARGET_CODES_END"]]
        )
        
        # Build labels: mask everything before TARGET_CODES_START
        num_masked = (
            len(user_prefix_ids) +
            1 + len(ref_text_ids) + 1 +      # REF_TEXT
            1 + len(ref_codes_ids) + 1 +      # REF_SPEECH
            1 + len(target_text_ids) + 1 +    # TARGET_TEXT
            len(assistant_prefix_ids)         # assistant:
        )
        
        labels = (
            [-100] * num_masked +
            [token_ids["TARGET_CODES_START"]] + target_codes_ids + [token_ids["TARGET_CODES_END"]]
        )
        
        # Sanity check
        assert len(input_ids) == len(labels), f"Length mismatch: {len(input_ids)} vs {len(labels)}"
        
        # Truncate if too long
        if len(input_ids) > max_length:
            LOGGER.warning(f"Sample {sample.get('__key__', 'unknown')} truncated: {len(input_ids)} -> {max_length}")
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
        
    except Exception as e:
        LOGGER.warning(f"Failed to process sample {sample.get('__key__', 'unknown')}: {e}")
        return None


def load_encoded_dataset(
    data_path: str,
    tokenizer,
    token_ids: Dict[str, int],
    max_length: int = 8192,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load pre-encoded JSON dataset and convert to V2 format.
    
    Args:
        data_path: Path to JSON file with encoded samples
        tokenizer: Extended tokenizer
        token_ids: V2 token ID mapping
        max_length: Max sequence length
        max_samples: Limit number of samples (for debugging)
    
    Returns:
        HuggingFace Dataset ready for training
    """
    LOGGER.info(f"Loading dataset: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    LOGGER.info(f"Loaded {len(raw_data):,} raw samples")
    
    if max_samples:
        raw_data = raw_data[:max_samples]
        LOGGER.info(f"Limited to {len(raw_data):,} samples")
    
    # Convert to V2 format
    processed = []
    skipped = 0
    
    for i, sample in enumerate(raw_data):
        result = build_v2_sample(sample, tokenizer, token_ids, max_length)
        if result:
            processed.append(result)
        else:
            skipped += 1
        
        if (i + 1) % 10000 == 0:
            LOGGER.info(f"Processed {i+1:,}/{len(raw_data):,} samples...")
    
    LOGGER.info(f"Processed: {len(processed):,}, Skipped: {skipped:,}")
    
    if not processed:
        raise ValueError("No valid samples after processing!")
    
    # Calculate statistics
    seq_lengths = [len(s["input_ids"]) for s in processed]
    train_lengths = [sum(1 for l in s["labels"] if l != -100) for s in processed]
    
    LOGGER.info(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.0f}")
    LOGGER.info(f"Training tokens per sample: min={min(train_lengths)}, max={max(train_lengths)}, avg={sum(train_lengths)/len(train_lengths):.0f}")
    
    return Dataset.from_list(processed)


def main(config_path: str, deepspeed: str = None):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML config file
        deepspeed: Optional path to DeepSpeed config JSON
    """
    # Load config
    LOGGER.info(f"Loading config: {config_path}")
    config = OmegaConf.load(config_path)
    
    output_dir = os.path.join(config.save_root, config.run_name)
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info(f"Output directory: {output_dir}")
    
    # Load tokenizer
    LOGGER.info(f"Loading tokenizer: {config.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    LOGGER.info(f"Vocab size: {len(tokenizer):,}")
    
    # Verify V2 tokens
    token_ids = verify_v2_tokens(tokenizer)
    LOGGER.info(f"V2 tokens verified: {list(token_ids.keys())}")
    
    # Load model
    LOGGER.info(f"Loading model: {config.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.get("use_flash_attn", True) else "eager",
    )
    
    # Verify embedding size matches tokenizer
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        LOGGER.warning(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    LOGGER.info(f"Model parameters: {model.num_parameters():,}")
    
    # Load dataset
    dataset = load_encoded_dataset(
        config.data_path,
        tokenizer,
        token_ids,
        max_length=config.get("max_seq_len", 8192),
        max_samples=config.get("max_samples", None),
    )
    
    LOGGER.info(f"Dataset size: {len(dataset):,}")
    
    # Data collator
    data_collator = V2DataCollator(
        tokenizer=tokenizer,
        max_length=config.get("max_seq_len", 8192),
        pad_to_multiple_of=8,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=config.run_name,
        
        # Training
        do_train=True,
        num_train_epochs=config.get("num_epochs", 1),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        
        # Optimizer
        learning_rate=config.lr,
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        
        # Precision
        bf16=True,
        tf32=True,
        
        # Saving
        save_strategy="steps",
        save_steps=config.get("save_steps", 1000),
        save_total_limit=config.get("save_total_limit", 3),
        
        # Logging
        logging_steps=config.get("logging_steps", 10),
        logging_first_step=True,
        report_to=config.get("report_to", "tensorboard"),
        
        # Performance
        dataloader_num_workers=config.get("num_workers", 8),
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        torch_compile=config.get("torch_compile", False),
        
        # DeepSpeed
        deepspeed=deepspeed,
        
        # Other
        seed=config.get("seed", 42),
        remove_unused_columns=False,
        ignore_data_skip=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    LOGGER.info("Starting training...")
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint", None))
    
    # Save final model
    LOGGER.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    LOGGER.info("Training complete!")


if __name__ == "__main__":
    Fire(main)
