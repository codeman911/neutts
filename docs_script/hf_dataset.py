#!/usr/bin/env python3
"""
Convert JSON dataset to HuggingFace format (.arrow) and optionally upload to Hub.

This script:
1. Reads a JSON file (list of samples)
2. Converts to JSONL format
3. Saves as HuggingFace Dataset (.arrow format) locally
4. Optionally uploads to HuggingFace Hub

Usage:
    # Save locally only (creates output_dir/train/data.arrow)
    python3 hf_dataset.py --input-json training_encoded.json --output-dir ./hf_dataset

    # Save locally and upload to Hub
    python3 hf_dataset.py --input-json training_encoded.json --output-dir ./hf_dataset \
        --hf-dataset-name "neutts-v2-encoded" --upload

    # With sample limit
    python3 hf_dataset.py --input-json training_encoded.json --output-dir ./hf_dataset \
        --max-samples 10000 --hf-dataset-name "neutts-v2-encoded" --upload --private
"""
import json
import argparse
import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo, whoami


def convert_to_jsonl(input_file: str, output_file: str, max_samples: int | None) -> int:
    """Read a list-style JSON file and write JSONL with optional truncation."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of samples (list[dict]).")

    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("--max-samples must be >= 0")
        data = data[:max_samples]

    with open(output_file, "w", encoding="utf-8") as out:
        for item in data:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} samples to JSONL: {output_file}")
    return len(data)


def resolve_repo_id(hf_dataset_name: str) -> str:
    """
    If user provided 'org/name' or 'user/name', use it as-is.
    If they provided just 'name', prefix with currently logged-in username.
    """
    if "/" in hf_dataset_name:
        return hf_dataset_name
    me = whoami()
    username = me.get("name")
    if not username:
        raise RuntimeError("Could not determine HF username. Are you logged in via `huggingface-cli login`?")
    return f"{username}/{hf_dataset_name}"


def main():
    parser = argparse.ArgumentParser(description="Convert JSON to HF Dataset (.arrow) and optionally upload to Hub")
    parser.add_argument("--input-json", required=True, help="Path to input JSON (list of samples)")
    parser.add_argument("--output-dir", required=True, help="Directory to save HF dataset (.arrow format)")
    parser.add_argument("--output-jsonl", default=None, help="Path to write JSONL (default: output-dir/data.jsonl)")
    parser.add_argument("--hf-dataset-name", default=None, help="Dataset name for Hub upload (e.g., 'name' or 'org/name')")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples to first N")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    parser.add_argument("--private", action="store_true", help="Create as private dataset (default: public)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default JSONL path
    jsonl_path = args.output_jsonl or os.path.join(args.output_dir, "data.jsonl")

    # 1) Convert list JSON → JSONL (subset if requested)
    print(f"Loading: {args.input_json}")
    count = convert_to_jsonl(args.input_json, jsonl_path, args.max_samples)
    if count == 0:
        print("max-samples=0 (or empty input). Nothing to process. Exiting.")
        return

    # 2) Load JSONL as a HF Dataset
    print(f"Converting to HuggingFace Dataset...")
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    print(f"Loaded {len(dataset)} records")
    print(f"Features: {dataset.features}")

    # 3) Save to disk in Arrow format
    arrow_dir = os.path.join(args.output_dir, "train")
    os.makedirs(arrow_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    print(f"Saved HF Dataset to: {args.output_dir}")
    print(f"Arrow files: {os.listdir(args.output_dir)}")

    # 4) Optionally upload to Hub
    if args.upload:
        if not args.hf_dataset_name:
            print("--hf-dataset-name required for upload. Skipping Hub upload.")
            return
            
        repo_id = resolve_repo_id(args.hf_dataset_name)

        # Create repo if needed
        api = HfApi()
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)
            print(f"Repo ready: {repo_id}")
        except Exception as e:
            print(f"Repo may already exist: {repo_id}\n   Reason: {e}")

        # Push dataset to Hub
        dataset.push_to_hub(repo_id=repo_id)
        print(f"Uploaded {len(dataset)} samples to https://huggingface.co/datasets/{repo_id}")
    
    print(f"\nDataset ready for training:")
    print(f"  Local: {args.output_dir}")
    if args.upload and args.hf_dataset_name:
        print(f"  Hub: {resolve_repo_id(args.hf_dataset_name)}")


if __name__ == "__main__":
    main()
