#!/usr/bin/env python3
"""
Finetune 1.5B NeuTTS with HuggingFace Dataset (.arrow format)

This script finetunes the extended 1.5B model using pre-converted HuggingFace datasets
saved in .arrow format (created by hf_dataset.py).

Dataset Format (expected columns):
- text: target text
- codes: NeuCodec codes for target audio (list of ints)
- ref_text: reference speaker text
- ref_codes: NeuCodec codes for reference audio (list of ints)

Usage:
    # From local .arrow dataset
    python3 finetune_1.5b_hf.py config_1.5b_hf.yaml

    # From HuggingFace Hub
    python3 finetune_1.5b_hf.py config_1.5b_hf.yaml --dataset-name "username/neutts-v2-encoded"

    # Multi-GPU
    torchrun --nproc_per_node=8 finetune_1.5b_hf.py config_1.5b_hf.yaml
"""

import os
import warnings
warnings.filterwarnings("ignore")

import torch
from fire import Fire
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, load_from_disk, Dataset
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
    """Data collator with dynamic padding."""
    tokenizer: Any
    max_length: int = 8192
    pad_to_multiple_of: int = 8
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        max_len = min(max_len, self.max_length)
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for f in features:
            input_ids = f["input_ids"][:max_len]
            labels = f["labels"][:max_len]
            
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
            
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
        raise ValueError(f"Missing V2 tokens: {missing}\nRun extend_tokenizer_1.5b.py first!")
    
    return token_ids


def build_v2_sample(
    sample: Dict[str, Any],
    tokenizer,
    token_ids: Dict[str, int],
    max_length: int = 8192,
) -> Optional[Dict[str, Any]]:
    """Build V2 training sample from dataset row."""
    try:
        target_text = sample.get("text", "").strip() if sample.get("text") else ""
        target_codes = sample.get("codes", [])
        ref_text = sample.get("ref_text", "").strip() if sample.get("ref_text") else ""
        ref_codes = sample.get("ref_codes", [])
        
        if not target_text or not target_codes or not ref_text or not ref_codes:
            return None
        
        # Convert codes to speech tokens
        ref_codes_str = "".join([f"<|speech_{c}|>" for c in ref_codes])
        target_codes_str = "".join([f"<|speech_{c}|>" for c in target_codes])
        
        # Tokenize components
        user_prefix_ids = tokenizer.encode("user: Convert the text to speech:", add_special_tokens=False)
        ref_text_ids = tokenizer.encode(ref_text, add_special_tokens=False)
        ref_codes_ids = tokenizer.encode(ref_codes_str, add_special_tokens=False)
        target_text_ids = tokenizer.encode(target_text, add_special_tokens=False)
        assistant_prefix_ids = tokenizer.encode("\nassistant:", add_special_tokens=False)
        target_codes_ids = tokenizer.encode(target_codes_str, add_special_tokens=False)
        
        # Build input_ids
        input_ids = (
            user_prefix_ids +
            [token_ids["REF_TEXT_START"]] + ref_text_ids + [token_ids["REF_TEXT_END"]] +
            [token_ids["REF_SPEECH_START"]] + ref_codes_ids + [token_ids["REF_SPEECH_END"]] +
            [token_ids["TARGET_TEXT_START"]] + target_text_ids + [token_ids["TARGET_TEXT_END"]] +
            assistant_prefix_ids +
            [token_ids["TARGET_CODES_START"]] + target_codes_ids + [token_ids["TARGET_CODES_END"]]
        )
        
        # Build labels
        num_masked = (
            len(user_prefix_ids) +
            1 + len(ref_text_ids) + 1 +
            1 + len(ref_codes_ids) + 1 +
            1 + len(target_text_ids) + 1 +
            len(assistant_prefix_ids)
        )
        
        labels = (
            [-100] * num_masked +
            [token_ids["TARGET_CODES_START"]] + target_codes_ids + [token_ids["TARGET_CODES_END"]]
        )
        
        assert len(input_ids) == len(labels)
        
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        
        return {"input_ids": input_ids, "labels": labels}
        
    except Exception as e:
        return None


def load_and_process_dataset(
    data_path: str,
    dataset_name: Optional[str],
    tokenizer,
    token_ids: Dict[str, int],
    max_length: int = 8192,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load dataset from local .arrow or HuggingFace Hub."""
    
    if dataset_name:
        LOGGER.info(f"Loading from HuggingFace Hub: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
    elif os.path.isdir(data_path):
        LOGGER.info(f"Loading from local .arrow: {data_path}")
        dataset = load_from_disk(data_path)
    else:
        LOGGER.info(f"Loading from JSON: {data_path}")
        import json
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        dataset = Dataset.from_list(raw_data)
    
    LOGGER.info(f"Raw dataset: {len(dataset)} samples")
    LOGGER.info(f"Features: {dataset.features}")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        LOGGER.info(f"Limited to {len(dataset)} samples")
    
    # Process with V2 template
    def process_fn(sample):
        result = build_v2_sample(sample, tokenizer, token_ids, max_length)
        if result is None:
            return {"input_ids": [], "labels": []}
        return result
    
    LOGGER.info("Processing with V2 template...")
    processed = dataset.map(
        process_fn,
        remove_columns=dataset.column_names,
        num_proc=min(os.cpu_count() or 1, 16),
        desc="Building V2 samples",
    )
    
    # Filter empty samples
    processed = processed.filter(lambda x: len(x["input_ids"]) > 0)
    LOGGER.info(f"Processed: {len(processed)} valid samples")
    
    return processed


def main(config_path: str, dataset_name: str = None, deepspeed: str = None):
    """Main training function."""
    
    LOGGER.info(f"Loading config: {config_path}")
    config = OmegaConf.load(config_path)
    
    output_dir = os.path.join(config.save_root, config.run_name)
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info(f"Output: {output_dir}")
    
    # Load tokenizer
    LOGGER.info(f"Loading tokenizer: {config.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info(f"Vocab: {len(tokenizer):,}")
    
    # Verify V2 tokens
    token_ids = verify_v2_tokens(tokenizer)
    LOGGER.info(f"V2 tokens verified")
    
    # Load model
    LOGGER.info(f"Loading model: {config.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.get("use_flash_attn", True) else "eager",
    )
    
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        LOGGER.warning(f"Resizing embeddings")
        model.resize_token_embeddings(len(tokenizer))
    
    LOGGER.info(f"Parameters: {model.num_parameters():,}")
    
    # Load dataset
    dataset = load_and_process_dataset(
        config.data_path,
        dataset_name or config.get("dataset_name"),
        tokenizer,
        token_ids,
        max_length=config.get("max_seq_len", 8192),
        max_samples=config.get("max_samples"),
    )
    
    # Data collator
    data_collator = V2DataCollator(
        tokenizer=tokenizer,
        max_length=config.get("max_seq_len", 8192),
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=config.run_name,
        do_train=True,
        num_train_epochs=config.get("num_epochs", 1),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config.lr,
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        bf16=True,
        tf32=True,
        save_strategy="steps",
        save_steps=config.get("save_steps", 1000),
        save_total_limit=config.get("save_total_limit", 3),
        logging_steps=config.get("logging_steps", 10),
        logging_first_step=True,
        report_to=config.get("report_to", "none"),
        dataloader_num_workers=config.get("num_workers", 8),
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        torch_compile=config.get("torch_compile", False),
        deepspeed=deepspeed,
        seed=config.get("seed", 42),
        remove_unused_columns=False,
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
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    
    # Save
    LOGGER.info(f"Saving to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info("Done!")


if __name__ == "__main__":
    Fire(main)
