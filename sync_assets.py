#!/usr/bin/env python3
"""
Asset Synchronization Utility for Smart Product Pricing.
Downloads dataset files (train/test features in parquet and csv) and trained model weights
directly from the Hugging Face Bucket: hf://buckets/mdumr4/Smart-Product-Pricing
"""

import sys
import os
import argparse
import subprocess
import shutil
from pathlib import Path

BUCKET_URI = "hf://buckets/mdumr4/Smart-Product-Pricing"
REPO_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd_list):
    """Executes a command and streams output."""
    cmd_str = " ".join(cmd_list)
    print(f"Running: {cmd_str}")
    res = subprocess.run(cmd_list, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print(f"Command failed with exit code {res.returncode}")
        return False
    return True


def check_hf_cli():
    """Checks if hf CLI is installed."""
    if shutil.which("hf") is None:
        print("Error: `hf` CLI is not installed or not in PATH.")
        print("Install it via: pip install huggingface_hub[cli]")
        print("And login via: hf auth login")
        return False
    return True


def sync_data():
    """Syncs train and test feature datasets from bucket."""
    print("\n=== Syncing Datasets (Parquet & CSV) from HF Bucket ===")
    os.makedirs(REPO_ROOT / "data", exist_ok=True)
    cmd = ["hf", "buckets", "cp", "-r", f"{BUCKET_URI}/data", str(REPO_ROOT / "data")]
    return run_cmd(cmd)


def sync_models():
    """Syncs trained checkpoints and encoder artifacts from bucket."""
    print("\n=== Syncing Model Checkpoints (NN & LightGBM) from HF Bucket ===")
    os.makedirs(REPO_ROOT / "models", exist_ok=True)
    cmd = ["hf", "buckets", "cp", "-r", f"{BUCKET_URI}/models", str(REPO_ROOT / "models")]
    return run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Sync golden precomputed assets from Hugging Face Bucket (hf://buckets/mdumr4/Smart-Product-Pricing)"
    )
    parser.add_argument("--all", action="store_true", help="Download all assets (data and models)")
    parser.add_argument("--data", action="store_true", help="Download dataset files (train/test features)")
    parser.add_argument("--models", action="store_true", help="Download trained model weights and encoders")
    args = parser.parse_args()

    if not check_hf_cli():
        sys.exit(1)

    if not (args.all or args.data or args.models):
        args.data = True

    success = True
    if args.all or args.data:
        success = sync_data() and success

    if args.all or args.models:
        success = sync_models() and success

    if success:
        print("\nAll requested assets synced successfully!")
    else:
        print("\nSome assets failed to sync. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
