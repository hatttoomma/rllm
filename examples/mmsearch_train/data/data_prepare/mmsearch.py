#!/usr/bin/env python3
"""Download and prepare MMSearch dataset for verl training.

This script downloads the MMSearch dataset (CaraJ/MMSearch) subset 'end2end',
split 'end2end', and saves it in parquet format compatible with verl training.

Usage:
    python mmsearch.py [--local_dir ./mmsearch_train/data/mmsearch]

The output parquet file contains the following fields per example:
    - query: The question text
    - query_image: PIL Image or list of PIL Images (base64 encoded in parquet)
    - gt_answer: Ground truth answer
    - alternative_gt_answers: Alternative correct answers (optional)
    - prompt: Formatted prompt for verl training
    - reward_model: Reward model configuration
    - extra_info: Additional metadata
"""

import argparse
import base64
import io
import json
import os
from io import BytesIO
from typing import Any

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def process_example(example: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Process a single MMSearch example into verl-compatible format.

    Args:
        example: Raw example from MMSearch dataset
        idx: Index of the example

    Returns:
        Processed example dictionary or None if processing fails
    """
    try:
        query = example.get("query", "")
        if not query:
            print(f"Warning: Empty query at index {idx}")
            return None

        gt_answer = example.get("gt_answer", "")

        alternative_gt_answers = example.get("alternative_gt_answers", [])

        query_image = example.get("query_image", None)
        query_image_b64 = None

        processed = {
            "query": query,
            "images": query_image,
            "gt_answer": gt_answer,
            "alternative_gt_answers": alternative_gt_answers,
            "answer": [gt_answer] + alternative_gt_answers,
            "prompt": [{"role": "user", "content": query}],
            "reward_model": {
                "style": "rule",
                "ground_truth": gt_answer,
            },
            "uid": idx,
        }

        return processed

    except Exception as e:
        print(f"Error processing example {idx}: {e}")
        return None


def download_mmsearch_dataset(
    local_dir: str = "./mmsearch_train/data/mmsearch",
    subset: str = "end2end",
    split: str = "end2end",
) -> str:
    """Download and process MMSearch dataset.

    Args:
        local_dir: Directory to save processed data
        subset: Dataset subset name
        split: Dataset split name

    Returns:
        Path to the saved parquet file
    """
    print(f"Downloading MMSearch dataset (subset: {subset}, split: {split})...")

    os.makedirs(local_dir, exist_ok=True)

    try:
        dataset = load_dataset(
            "CaraJ/MMSearch",
            subset,
            split=split,
            trust_remote_code=True,
        )
        print(f"Downloaded {len(dataset)} examples")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

    print("Processing examples...")
    processed_data = []

    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        processed = process_example(example, idx)
        if processed is not None:
            processed_data.append(processed)

    print(f"Successfully processed {len(processed_data)} examples")

    df = pd.DataFrame(processed_data)

    parquet_path = os.path.join(local_dir, f"{split}.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet file to {parquet_path}")

    verl_parquet_path = os.path.join(local_dir, f"{split}_verl.parquet")
    verl_df = pd.DataFrame([
        {
            "prompt": item["prompt"],
            "reward_model": item["reward_model"],
            "extra_info": {
                "query": item["query"],
                "images": item["images"],
                "answer": item["answer"],
                "uid": item["uid"],
            },
        }
        for item in processed_data
    ])
    verl_df.to_parquet(verl_parquet_path, index=False)
    print(f"Saved verl-compatible parquet file to {verl_parquet_path}")

    summary = {
        "dataset": "CaraJ/MMSearch",
        "subset": subset,
        "split": split,
        "total_examples": len(processed_data),
        "files": {
            "original": parquet_path,
            "verl": verl_parquet_path,
        },
    }

    summary_path = os.path.join(local_dir, "data_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    print("\nDataset Statistics:")
    print(f"  Total examples: {len(processed_data)}")
    images_count = sum(1 for item in processed_data if item.get("query_image"))
    print(f"  Examples with images: {images_count}")

    return parquet_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare MMSearch dataset for verl training"
    )
    parser.add_argument(
        "--local_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "mmsearch"),
        help="Directory to save processed datasets",
    )
    parser.add_argument(
        "--subset",
        default="end2end",
        help="Dataset subset name (default: end2end)",
    )
    parser.add_argument(
        "--split",
        default="end2end",
        help="Dataset split name (default: end2end)",
    )

    args = parser.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    print(f"Output directory: {local_dir}")

    download_mmsearch_dataset(
        local_dir=local_dir,
        subset=args.subset,
        split=args.split,
    )


if __name__ == "__main__":
    main()
