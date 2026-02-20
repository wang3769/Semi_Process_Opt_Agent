#!/usr/bin/env python3
"""
SECOM Dataset Download Script
=============================
Downloads the SECOM dataset from UCI Machine Learning Repository.
This dataset contains sensor data from semiconductor manufacturing.

Dataset: https://archive.ics.uci.edu/dataset/171/secom+data
Labels: https://archive.ics.uci.edu/dataset/171/secom+data

Features:
- 590 features from sensor readings
- Binary labels (pass/fail)
- Used for anomaly detection and yield prediction
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path

# Configuration - SECOM dataset from UCI
# Dataset info: https://archive.ics.uci.edu/dataset/171/secom+data
# The dataset includes sensor data and labels
DATASET_URL = "https://archive.ics.uci.edu/static/public/171/171.tar.gz"
DATASET_NAME = "secom"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "external" / DATASET_NAME


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress indication."""
    print(f"Downloading {url}...")
    print(f"Destination: {dest_path}")
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        sys.stdout.write(f"\rProgress: {percent}% ({downloaded / 1e6:.1f} MB)")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\nDownload complete!")


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract tar.gz or zip archive."""
    import tarfile
    
    print(f"Extracting {archive_path}...")
    
    if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
        # It's a tar.gz file
        with tarfile.open(archive_path, 'r:gz') as tf:
            tf.extractall(extract_to)
    elif archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_to)
    else:
        # Try tarfile anyway
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_to)
    
    print("Extraction complete!")


def process_secom_data(data_dir: Path) -> pd.DataFrame:
    """
    Process SECOM data files into usable format.
    
    The SECOM dataset contains:
    - secom.data: 590 features (sensor readings)
    - secom_labels.data: labels (1 = pass, -1 = fail)
    """
    # Find data files
    data_file = data_dir / "secom.data"
    labels_file = data_dir / "secom_labels.data"
    
    if not data_file.exists():
        # Try alternative paths
        for f in data_dir.rglob("secom*.data"):
            if "label" not in f.name:
                data_file = f
            else:
                labels_file = f
    
    print(f"Loading data from: {data_file}")
    print(f"Loading labels from: {labels_file}")
    
    # Load data
    df = pd.read_csv(data_file, sep=r'\s+', header=None)
    print(f"Data shape: {df.shape}")
    
    # Load labels
    labels_df = pd.read_csv(labels_file, sep=r'\s+', header=None)
    # Format: column 0 = timestamp, column 1 = label
    labels_df.columns = ['timestamp', 'label']
    
    # Merge
    df['label'] = labels_df['label'].values
    df['timestamp'] = labels_df['timestamp'].values
    
    # Convert labels: -1 (fail) -> 0, 1 (pass) -> 1
    df['label'] = df['label'].apply(lambda x: 0 if x == -1 else 1)
    
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    return df


def create_synthetic_secom(output_dir: Path, n_samples: int = 1567) -> pd.DataFrame:
    """Create synthetic SECOM-like data for development."""
    import numpy as np
    
    print("Creating synthetic SECOM-like dataset...")
    
    np.random.seed(42)
    
    # SECOM has 590 features + timestamp + label
    n_features = 590
    
    # Generate features (simulate sensor readings)
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (roughly 6% failure rate like real SECOM)
    # Make labels somewhat correlated with a few features
    y_prob = 0.06 + 0.1 * (X[:, 0] + X[:, 1]) / (np.std(X[:, 0]) * 2)
    y_prob = np.clip(y_prob, 0.02, 0.3)
    y = (np.random.rand(n_samples) < y_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df['label'] = y
    df['timestamp'] = np.arange(n_samples)
    
    # Add some missing values like real SECOM
    mask = np.random.rand(n_samples, n_features) < 0.05
    df.iloc[:, :n_features] = df.iloc[:, :n_features].mask(mask)
    
    print(f"Created {n_samples} samples with {n_features} features")
    print(f"Pass rate: {1 - y.mean():.2%}")
    
    return df


def main():
    """Main download and processing function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    archive_path = OUTPUT_DIR.parent / f"{DATASET_NAME}.tar.gz"
    
    # Try to download
    download_success = False
    if not archive_path.exists():
        try:
            download_file(DATASET_URL, archive_path)
            download_success = True
        except Exception as e:
            print(f"Download failed: {e}")
            print("Will create synthetic data instead")
    else:
        print(f"Archive already exists at {archive_path}")
        download_success = True
    
    # Extract if available
    extracted_dir = OUTPUT_DIR
    if download_success and not (extracted_dir / "secom.data").exists():
        try:
            extract_archive(archive_path, extracted_dir)
            
            # Find the extracted folder
            for item in extracted_dir.iterdir():
                if item.is_dir() and item.name != DATASET_NAME:
                    # Move contents up
                    for subitem in item.iterdir():
                        subitem.rename(extracted_dir / subitem.name)
                    item.rmdir()
        except Exception as e:
            print(f"Extraction failed: {e}")
            download_success = False
    
    # Process data or create synthetic
    print("\n" + "="*50)
    print("Processing SECOM dataset...")
    print("="*50)
    
    if (extracted_dir / "secom.data").exists():
        df = process_secom_data(extracted_dir)
    else:
        print("Using synthetic SECOM-like data")
        df = create_synthetic_secom(extracted_dir)
    
    # Save processed data
    processed_dir = Path(__file__).parent.parent / "data" / "processed" / DATASET_NAME
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "secom_processed.csv"
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    print(f"Shape: {df.shape}")
    
    # Also save just features and labels for easy loading
    features_file = processed_dir / "secom_features.csv"
    labels_file = processed_dir / "secom_labels.csv"
    
    # Drop timestamp for features
    df_features = df.drop(['timestamp'], axis=1)
    df_features.to_csv(features_file, index=False)
    
    # Save labels separately
    df[['timestamp', 'label']].to_csv(labels_file, index=False)
    
    print(f"Features saved to: {features_file}")
    print(f"Labels saved to: {labels_file}")
    
    # Create data manifest
    manifest = {
        "dataset": "SECOM",
        "source": "UCI Machine Learning Repository" if download_success else "Synthetic (development)",
        "url": DATASET_URL if download_success else None,
        "n_samples": len(df),
        "n_features": df.shape[1] - 2,  # excluding timestamp and label
        "n_positive": int(df['label'].sum()),
        "n_negative": int(len(df) - df['label'].sum()),
        "pass_rate": float(df['label'].mean())
    }
    
    import json
    manifest_file = processed_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest saved to: {manifest_file}")
    print("\nSECOM dataset ready!")


if __name__ == "__main__":
    main()
