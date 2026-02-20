#!/usr/bin/env python3
"""
Download Runner Script
=====================
Downloads all required datasets for the AEOC project.

Usage:
    python scripts/download_all.py
"""

import subprocess
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script_name: str) -> bool:
    """Run a download script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False


def main():
    """Download all datasets."""
    print("="*60)
    print("AEOC Data Download Pipeline")
    print("="*60)
    
    # Track results
    results = {}
    
    # Download SECOM
    print("\n[1/2] Downloading SECOM dataset...")
    results['secom'] = run_script('download_secom.py')
    
    # Download WM-811K
    print("\n[2/2] Downloading WM-811K dataset...")
    results['wm811k'] = run_script('download_wm811k.py')
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    
    # Check data availability
    print("\n" + "="*60)
    print("Verifying data...")
    print("="*60)
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from data.loader import ExperimentDataLoader
        
        ready = ExperimentDataLoader.check_data_ready()
        print(f"  SECOM ready: {ready['secom']}")
        print(f"  WM-811K ready: {ready['wm811k']}")
        
        if ready['secom']:
            stats = ExperimentDataLoader.get_all_stats()['secom']
            if stats:
                print(f"    - Samples: {stats['n_samples']}")
                print(f"    - Features: {stats['n_features']}")
                print(f"    - Pass rate: {stats['pass_rate']:.2%}")
        
        if ready['wm811k']:
            stats = ExperimentDataLoader.get_all_stats()['wm811k']
            if stats:
                print(f"    - Samples: {stats['n_samples']}")
                print(f"    - Classes: {stats['n_classes']}")
        
    except ImportError as e:
        print(f"Could not verify data: {e}")
        print("Run 'python -m src.data.loader' to verify manually.")
    
    # Final message
    print("\n" + "="*60)
    print("Data pipeline complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review data in: data/processed/")
    print("  2. Train models: See src/models/")
    print("  3. Run experiments: See experiments/")


if __name__ == "__main__":
    main()
