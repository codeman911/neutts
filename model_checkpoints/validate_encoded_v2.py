#!/usr/bin/env python3
"""
Validate V2 Encoded Training Data for 1.5B NeuTTS

Validates:
1. JSON structure and required fields
2. V2 template token presence and order
3. Label masking correctness (-100 for non-target parts)
4. Token ID ranges (vocab bounds, speech codes)
5. Sequence statistics and anomaly detection

Usage:
    python3 validate_encoded_v2.py training_encoded_v2.json --tokenizer_path pretrained_1.5b_v2
"""

import json
import argparse
import sys
from collections import Counter
from typing import List, Dict, Any, Tuple

# V2 Special Token Names
V2_TOKENS = [
    "<|REF_TEXT_START|>",
    "<|REF_TEXT_END|>",
    "<|REF_SPEECH_START|>",
    "<|REF_SPEECH_END|>",
    "<|TARGET_TEXT_START|>",
    "<|TARGET_TEXT_END|>",
    "<|TARGET_CODES_START|>",
    "<|TARGET_CODES_END|>",
]


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer and get V2 token IDs."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    token_ids = {}
    for token in V2_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(token)
        token_ids[token] = tid
    
    return tokenizer, token_ids


def validate_sample(sample: Dict[str, Any], token_ids: Dict[str, int], 
                    vocab_size: int, idx: int) -> Tuple[bool, List[str]]:
    """
    Validate a single encoded sample.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required fields
    required = ["__key__", "input_ids", "labels", "text", "ref_text"]
    for field in required:
        if field not in sample:
            errors.append(f"Missing field: {field}")
    
    if errors:
        return False, errors
    
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    
    # Check lengths match
    if len(input_ids) != len(labels):
        errors.append(f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}")
    
    # Check input_ids are valid token IDs
    invalid_ids = [tid for tid in input_ids if tid < 0 or tid >= vocab_size]
    if invalid_ids:
        errors.append(f"Invalid token IDs in input_ids: {invalid_ids[:5]}...")
    
    # Check labels are valid (-100 or valid token IDs)
    invalid_labels = [l for l in labels if l != -100 and (l < 0 or l >= vocab_size)]
    if invalid_labels:
        errors.append(f"Invalid labels: {invalid_labels[:5]}...")
    
    # V2 Template validation: check token order
    TARGET_CODES_START = token_ids["<|TARGET_CODES_START|>"]
    TARGET_CODES_END = token_ids["<|TARGET_CODES_END|>"]
    
    # Find TARGET_CODES_START and TARGET_CODES_END positions
    try:
        tcs_pos = input_ids.index(TARGET_CODES_START)
    except ValueError:
        errors.append("Missing <|TARGET_CODES_START|> in input_ids")
        tcs_pos = -1
    
    try:
        tce_pos = input_ids.index(TARGET_CODES_END)
    except ValueError:
        errors.append("Missing <|TARGET_CODES_END|> in input_ids")
        tce_pos = -1
    
    # Validate label masking
    if tcs_pos >= 0 and tce_pos >= 0:
        # Everything before TARGET_CODES_START should be -100
        pre_mask = labels[:tcs_pos]
        non_masked_pre = [i for i, l in enumerate(pre_mask) if l != -100]
        if non_masked_pre:
            errors.append(f"Labels before TARGET_CODES_START not masked: positions {non_masked_pre[:5]}")
        
        # TARGET_CODES_START through TARGET_CODES_END should NOT be -100
        target_labels = labels[tcs_pos:tce_pos+1]
        masked_target = [i for i, l in enumerate(target_labels) if l == -100]
        if masked_target:
            errors.append(f"Target labels incorrectly masked at positions {masked_target[:5]}")
        
        # Verify target_labels match input_ids in target region
        target_input = input_ids[tcs_pos:tce_pos+1]
        if target_labels != target_input:
            errors.append("Target labels don't match input_ids in target region")
    
    # Check for reasonable sequence length
    if len(input_ids) < 50:
        errors.append(f"Sequence too short: {len(input_ids)}")
    if len(input_ids) > 50000:
        errors.append(f"Sequence too long: {len(input_ids)}")
    
    return len(errors) == 0, errors


def validate_dataset(data: List[Dict], tokenizer, token_ids: Dict[str, int]) -> Dict[str, Any]:
    """Validate entire dataset and collect statistics."""
    
    vocab_size = len(tokenizer)
    
    stats = {
        "total_samples": len(data),
        "valid_samples": 0,
        "invalid_samples": 0,
        "errors": [],
        "seq_lengths": [],
        "target_lengths": [],
        "label_distributions": Counter(),
    }
    
    TARGET_CODES_START = token_ids["<|TARGET_CODES_START|>"]
    TARGET_CODES_END = token_ids["<|TARGET_CODES_END|>"]
    
    for idx, sample in enumerate(data):
        is_valid, errors = validate_sample(sample, token_ids, vocab_size, idx)
        
        if is_valid:
            stats["valid_samples"] += 1
            
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            
            stats["seq_lengths"].append(len(input_ids))
            
            # Count target length
            try:
                tcs_pos = input_ids.index(TARGET_CODES_START)
                tce_pos = input_ids.index(TARGET_CODES_END)
                target_len = tce_pos - tcs_pos - 1  # codes between START and END
                stats["target_lengths"].append(target_len)
            except ValueError:
                pass
            
            # Count masked vs unmasked
            masked = sum(1 for l in labels if l == -100)
            unmasked = len(labels) - masked
            stats["label_distributions"]["masked"] += masked
            stats["label_distributions"]["unmasked"] += unmasked
        else:
            stats["invalid_samples"] += 1
            if len(stats["errors"]) < 20:  # Limit error collection
                stats["errors"].append({
                    "index": idx,
                    "key": sample.get("__key__", "unknown"),
                    "errors": errors
                })
        
        if (idx + 1) % 10000 == 0:
            print(f"  Validated {idx + 1:,}/{len(data):,}...")
    
    return stats


def print_report(stats: Dict[str, Any], token_ids: Dict[str, int]):
    """Print validation report."""
    
    print("\n" + "=" * 80)
    print("V2 ENCODED DATA VALIDATION REPORT")
    print("=" * 80)
    
    total = stats["total_samples"]
    valid = stats["valid_samples"]
    invalid = stats["invalid_samples"]
    
    print(f"\nSAMPLE COUNTS:")
    print(f"  Total:   {total:,}")
    print(f"  Valid:   {valid:,} ({100*valid/total:.2f}%)")
    print(f"  Invalid: {invalid:,} ({100*invalid/total:.2f}%)")
    
    if stats["seq_lengths"]:
        lengths = stats["seq_lengths"]
        print(f"\nSEQUENCE LENGTHS:")
        print(f"  Min:    {min(lengths):,}")
        print(f"  Max:    {max(lengths):,}")
        print(f"  Mean:   {sum(lengths)/len(lengths):,.1f}")
        print(f"  Median: {sorted(lengths)[len(lengths)//2]:,}")
    
    if stats["target_lengths"]:
        targets = stats["target_lengths"]
        print(f"\nTARGET CODE LENGTHS (audio frames):")
        print(f"  Min:    {min(targets):,} ({min(targets)/50:.1f}s)")
        print(f"  Max:    {max(targets):,} ({max(targets)/50:.1f}s)")
        print(f"  Mean:   {sum(targets)/len(targets):,.1f} ({sum(targets)/len(targets)/50:.1f}s)")
    
    dist = stats["label_distributions"]
    if dist:
        total_labels = dist["masked"] + dist["unmasked"]
        print(f"\nLABEL MASKING:")
        print(f"  Masked (-100):   {dist['masked']:,} ({100*dist['masked']/total_labels:.1f}%)")
        print(f"  Unmasked (train): {dist['unmasked']:,} ({100*dist['unmasked']/total_labels:.1f}%)")
    
    print(f"\nV2 TOKEN IDS:")
    for token, tid in token_ids.items():
        print(f"  {token}: {tid}")
    
    if stats["errors"]:
        print(f"\nSAMPLE ERRORS (first {len(stats['errors'])}):")
        for err in stats["errors"][:10]:
            print(f"  [{err['key']}] {err['errors']}")
    
    print("\n" + "=" * 80)
    if invalid == 0:
        print("VALIDATION PASSED - All samples valid")
    else:
        print(f"VALIDATION FAILED - {invalid} invalid samples")
    print("=" * 80)
    
    return invalid == 0


def main():
    parser = argparse.ArgumentParser(description="Validate V2 encoded training data")
    parser.add_argument("input_file", type=str, help="Path to encoded JSON file")
    parser.add_argument("--tokenizer_path", type=str, default="pretrained_1.5b_v2",
                        help="Path to extended tokenizer")
    parser.add_argument("--sample", type=int, default=0,
                        help="Validate only first N samples (0 = all)")
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer, token_ids = load_tokenizer(args.tokenizer_path)
    print(f"  Vocab size: {len(tokenizer):,}")
    
    print(f"\nLoading data: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded: {len(data):,} samples")
    
    if args.sample > 0:
        data = data[:args.sample]
        print(f"  Validating first {len(data):,} samples")
    
    print(f"\nValidating...")
    stats = validate_dataset(data, tokenizer, token_ids)
    
    success = print_report(stats, token_ids)
    
    # Print sample for inspection
    if data:
        print(f"\nSAMPLE INSPECTION (first valid sample):")
        print("-" * 60)
        s = data[0]
        print(f"Key: {s.get('__key__')}")
        print(f"Text: {s.get('text', '')[:100]}...")
        print(f"Ref text: {s.get('ref_text', '')[:100]}...")
        print(f"Input IDs length: {len(s.get('input_ids', []))}")
        print(f"Labels length: {len(s.get('labels', []))}")
        
        # Show token breakdown
        input_ids = s.get('input_ids', [])
        labels = s.get('labels', [])
        
        print(f"\nFirst 30 tokens:")
        for i in range(min(30, len(input_ids))):
            tok = tokenizer.decode([input_ids[i]])
            label = labels[i] if i < len(labels) else "?"
            label_str = "-100" if label == -100 else str(label)
            print(f"  {i:4d}: {input_ids[i]:6d} -> '{tok[:20]:<20}' label={label_str}")
        
        print(f"\nLast 20 tokens:")
        for i in range(max(0, len(input_ids)-20), len(input_ids)):
            tok = tokenizer.decode([input_ids[i]])
            label = labels[i] if i < len(labels) else "?"
            label_str = "-100" if label == -100 else str(label)
            print(f"  {i:4d}: {input_ids[i]:6d} -> '{tok[:20]:<20}' label={label_str}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
