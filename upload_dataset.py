#!/usr/bin/env python3
"""
Upload dataset to HuggingFace Hub for use on RunPod.
Run this on your LOCAL machine before starting RunPod training.

Usage:
    python upload_dataset.py --username YOUR_HF_USERNAME

Prerequisites:
    pip install huggingface-hub
    huggingface-cli login
"""

import argparse
import sys
from pathlib import Path

# try:
from huggingface_hub import HfApi
# except ImportError:
    # print("Error: huggingface-hub not installed. Install with: pip install huggingface-hub")
    # sys.exit(1)


def upload_dataset(username: str, repo_name: str = "snake-repair-data", dataset_dir: str = "../dataset"):
    """Upload train and validation parquet files to HuggingFace Hub."""
    dataset_path = Path(dataset_dir)
    train_file = dataset_path / "train.parquet"
    val_file = dataset_path / "validation.parquet"

    # Verify files exist
    if not train_file.exists():
        print(f"Error: {train_file} not found")
        sys.exit(1)
    if not val_file.exists():
        print(f"Error: {val_file} not found")
        sys.exit(1)

    print(f"Train file: {train_file} ({train_file.stat().st_size / 1e6:.1f} MB)")
    print(f"Validation file: {val_file} ({val_file.stat().st_size / 1e6:.1f} MB)")

    repo_id = f"{username}/{repo_name}"
    print(f"\nUploading to: https://huggingface.co/datasets/{repo_id}")

    api = HfApi()

    # Create repo (private by default)
    try:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        print(f"Repository ready: {repo_id}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        sys.exit(1)

    # Upload files
    print("\nUploading train.parquet...")
    api.upload_file(
        path_or_fileobj=str(train_file),
        path_in_repo="train.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  Done!")

    print("Uploading validation.parquet...")
    api.upload_file(
        path_or_fileobj=str(val_file),
        path_in_repo="validation.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  Done!")

    # Also upload metadata if it exists
    metadata_file = dataset_path / "metadata.json"
    if metadata_file.exists():
        print("Uploading metadata.json...")
        api.upload_file(
            path_or_fileobj=str(metadata_file),
            path_in_repo="metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  Done!")

    print(f"\nUpload complete!")
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"\nOn RunPod, download with:")
    print(f"  python -c \"")
    print(f"  from huggingface_hub import hf_hub_download")
    print(f"  hf_hub_download('{repo_id}', 'train.parquet', repo_type='dataset', local_dir='/workspace/dataset/')")
    print(f"  hf_hub_download('{repo_id}', 'validation.parquet', repo_type='dataset', local_dir='/workspace/dataset/')")
    print(f"  \"")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Snake-RepairLLaMA dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Upload dataset:
    python upload_dataset.py --username myuser

  Upload from custom path:
    python upload_dataset.py --username myuser --dataset_dir /path/to/dataset

  Custom repo name:
    python upload_dataset.py --username myuser --repo_name my-custom-dataset
        """
    )

    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Your HuggingFace username"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default="snake-repair-data",
        help="Name for the HuggingFace dataset repo (default: snake-repair-data)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/",
        help="Path to dataset directory containing train.parquet and validation.parquet"
    )

    args = parser.parse_args()
    upload_dataset(args.username, args.repo_name, args.dataset_dir)


if __name__ == "__main__":
    main()
