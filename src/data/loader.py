"""
Data Loader Module
==================
Provides unified interface for loading SECOM and WM-811K datasets.

This module handles:
- SECOM tabular sensor data (anomaly detection)
- WM-811K wafer map data (defect classification)
- Train/test splits with proper preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json


# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str
    data_dir: Path
    test_split: float = 0.2
    random_seed: int = 42
    stratify: bool = True


class SECOMLoader:
    """Loader for SECOM semiconductor manufacturing dataset."""
    
    CONFIG = DatasetConfig(
        name="SECOM",
        data_dir=PROCESSED_DIR / "secom"
    )
    
    @classmethod
    def load_raw(cls) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw SECOM data.
        
        Returns:
            Tuple of (features_df, labels_df)
        """
        data_dir = cls.CONFIG.data_dir
        
        # Load processed data
        features_file = data_dir / "secom_features.csv"
        labels_file = data_dir / "secom_labels.csv"
        
        if not features_file.exists():
            raise FileNotFoundError(
                f"SECOM data not found at {features_file}. "
                "Run scripts/download_secom.py first."
            )
        
        X = pd.read_csv(features_file)
        y = pd.read_csv(labels_file)
        
        return X, y
    
    @classmethod
    def load_train_test(
        cls, 
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load train/test split.
        
        Args:
            test_size: Fraction for test set (default: 0.2)
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X, y = cls.load_raw()
        
        # Extract label column
        y_labels = y['label']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        test_size = test_size or cls.CONFIG.test_split
        random_state = random_state or cls.CONFIG.random_seed
        
        return train_test_split(
            X, y_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_labels if cls.CONFIG.stratify else None
        )
    
    @classmethod
    def get_feature_names(cls) -> list:
        """Get feature names."""
        return [f"feature_{i}" for i in range(590)]
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get dataset statistics."""
        X, y = cls.load_raw()
        
        return {
            "n_samples": len(X),
            "n_features": X.shape[1] - 2,  # Exclude timestamp, label
            "n_positive": int(y['label'].sum()),
            "n_negative": int(len(y) - y['label'].sum()),
            "pass_rate": float(y['label'].mean()),
            "missing_values": int(X.isna().sum().sum())
        }


class WM811KLoader:
    """Loader for WM-811K wafer map dataset."""
    
    CONFIG = DatasetConfig(
        name="WM-811K",
        data_dir=PROCESSED_DIR / "wm811k"
    )
    
    # Class mapping for defect types
    CLASS_MAPPING = {
        0: 'center',
        1: 'donut', 
        2: 'edge-loc',
        3: 'edge-ring',
        4: 'local',
        5: 'random',
        6: 'scratch',
        7: 'none',
        8: 'near-full'
    }
    
    @classmethod
    def load_raw(cls) -> pd.DataFrame:
        """
        Load WM-811K labels.
        
        Returns:
            DataFrame with wafer information
        """
        data_dir = cls.CONFIG.data_dir
        labels_file = data_dir / "wm811k_labels.csv"
        
        if not labels_file.exists():
            raise FileNotFoundError(
                f"WM-811K data not found at {labels_file}. "
                "Run scripts/download_wm811k.py first."
            )
        
        df = pd.read_csv(labels_file)
        return df
    
    @classmethod
    def load_train_test(
        cls,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train/test split.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        df = cls.load_raw()
        
        test_size = test_size or cls.CONFIG.test_split
        random_state = random_state or cls.CONFIG.random_seed
        
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['defect_class'] if cls.CONFIG.stratify else None
        )
    
    @classmethod
    def get_class_distribution(cls) -> Dict[str, int]:
        """Get class distribution."""
        df = cls.load_raw()
        return df['defect_class'].value_counts().to_dict()
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get dataset statistics."""
        df = cls.load_raw()
        
        return {
            "n_samples": len(df),
            "n_classes": df['defect_class'].nunique(),
            "classes": df['defect_class'].unique().tolist(),
            "class_distribution": cls.get_class_distribution(),
            "wafer_sizes": df['diameter'].unique().tolist()
        }


class ExperimentDataLoader:
    """
    Unified loader for all AEOC experiments.
    
    Provides a single interface to load both SECOM and WM-811K
    for training and evaluation.
    """
    
    @staticmethod
    def load_secom() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load SECOM with train/test split."""
        return SECOMLoader.load_train_test()
    
    @staticmethod
    def load_wm811k() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load WM-811K with train/test split."""
        return WM811KLoader.load_train_test()
    
    @staticmethod
    def get_all_stats() -> Dict[str, Dict[str, Any]]:
        """Get statistics for all datasets."""
        return {
            "secom": SECOMLoader.get_stats() if (PROCESSED_DIR / "secom").exists() else None,
            "wm811k": WM811KLoader.get_stats() if (PROCESSED_DIR / "wm811k").exists() else None
        }
    
    @staticmethod
    def check_data_ready() -> Dict[str, bool]:
        """Check which datasets are ready."""
        return {
            "secom": (PROCESSED_DIR / "secom").exists(),
            "wm811k": (PROCESSED_DIR / "wm811k").exists()
        }


def load_experiment_data(dataset: str = "secom") -> Any:
    """
    Convenience function to load experiment data.
    
    Args:
        dataset: Name of dataset ("secom" or "wm811k")
        
    Returns:
        Loaded data based on dataset type
    """
    if dataset.lower() == "secom":
        return SECOMLoader.load_train_test()
    elif dataset.lower() == "wm811k":
        return WM811KLoader.load_train_test()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    # Quick test
    print("Checking data availability...")
    ready = ExperimentDataLoader.check_data_ready()
    print(f"SECOM ready: {ready['secom']}")
    print(f"WM-811K ready: {ready['wm811k']}")
    
    if ready['secom']:
        print("\nSECOM Stats:")
        print(SECOMLoader.get_stats())
    
    if ready['wm811k']:
        print("\nWM-811K Stats:")
        print(WM811KLoader.get_stats())
