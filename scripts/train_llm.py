"""
LLM Training Pipeline Script
============================

Complete pipeline for training the semiconductor domain LLM:
1. Generate training datasets
2. SFT (Supervised Fine-tuning)
3. DPO (Direct Preference Optimization)

Usage:
    python scripts/train_llm.py --phase sft      # Train SFT
    python scripts/train_llm.py --phase dpo      # Train DPO  
    python scripts/train_llm.py --phase all      # Full pipeline
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability."""
    import torch
    
    if not torch.cuda.is_available():
        logger.error("No GPU available! LLM training requires GPU.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"VRAM: {vram_gb:.1f} GB")
    
    if vram_gb < 16:
        logger.warning("Less than 16GB VRAM. LoRA recommended.")
    
    return True


def create_datasets():
    """Create SFT and DPO training datasets."""
    from src.llm.dataset import create_sft_dataset, create_dpo_dataset
    
    logger.info("=" * 60)
    logger.info("Creating Training Datasets")
    logger.info("=" * 60)
    
    # Create SFT dataset
    logger.info("\n[1/2] Creating SFT dataset...")
    sft_data = create_sft_dataset(
        num_examples=100,
        output_file="data/processed/llm/sft_train.jsonl"
    )
    logger.info(f"Created {len(sft_data)} SFT examples")
    
    # Create DPO dataset
    logger.info("\n[2/2] Creating DPO dataset...")
    dpo_data = create_dpo_dataset(
        num_pairs=50,
        output_file="data/processed/llm/dpo_train.jsonl"
    )
    logger.info(f"Created {len(dpo_data)} DPO pairs")
    
    logger.info("\n✅ Dataset creation complete!")


def train_sft(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Run SFT training."""
    from src.llm.sft_trainer import train_sft
    
    logger.info("=" * 60)
    logger.info("Starting SFT Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info("This will take several hours on RTX 5090")
    
    trainer = train_sft(
        model_name=model_name,
        output_dir="models/llm/sft_model",
        train_file="data/processed/llm/sft_train.jsonl",
    )
    
    logger.info("\n✅ SFT training complete!")
    logger.info(f"Model saved to: models/llm/sft_model/final")


def train_dpo(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Run DPO training."""
    from src.llm.dpo_trainer import train_dpo
    
    logger.info("=" * 60)
    logger.info("Starting DPO Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    
    # Check if SFT model exists
    sft_model_path = "models/llm/sft_model/final"
    if os.path.exists(sft_model_path):
        logger.info(f"Using SFT model: {sft_model_path}")
    else:
        logger.warning(f"SFT model not found at {sft_model_path}")
    
    trainer = train_dpo(
        model_name=model_name,
        sft_model_path=sft_model_path,
        output_dir="models/llm/dpo_model",
        train_file="data/processed/llm/dpo_train.jsonl",
    )
    
    logger.info("\n✅ DPO training complete!")
    logger.info(f"Model saved to: models/llm/dpo_model/final")


def main():
    parser = argparse.ArgumentParser(description="Train LLM for Semiconductor RCA")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["data", "sft", "dpo", "all"],
        default="all",
        help="Training phase to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to use"
    )
    
    args = parser.parse_args()
    
    # Check GPU
    if not check_gpu():
        sys.exit(1)
    
    # Create output directories
    os.makedirs("data/processed/llm", exist_ok=True)
    os.makedirs("models/llm", exist_ok=True)
    
    # Run selected phases
    if args.phase in ["data", "all"]:
        create_datasets()
    
    if args.phase in ["sft", "all"]:
        train_sft(args.model)
    
    if args.phase in ["dpo", "all"]:
        train_dpo(args.model)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Run inference: python -c 'from src.llm.inference import *; ...'")
    logger.info("2. Collect feedback: Use UserFeedbackCollector to rate responses")
    logger.info("3. Retrain with feedback for continuous improvement")


if __name__ == "__main__":
    main()
