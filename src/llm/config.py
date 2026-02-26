"""
LLM Configuration Module
=======================
This is practically a container for settings
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass # dataclass is a decorator that automatically generates special methods like __init__() and __repr__() for the class based on the defined fields, more readable
class ModelConfig:
    """Configuration for base model."""
    
    # Model selection - optimized for RTX 5090 (32GB VRAM)
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # or "meta-llama/Llama-3.2-8B-Instruct"
    
    # Quantization (4-bit for efficiency, or None for full precision)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # or "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Device
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"


@dataclass
class SFTConfig:
    """Configuration for SFT (Supervised Fine-tuning)."""
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 4 * 4 = 16
    per_device_eval_batch_size: int = 8
    
    # Optimization
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True  # Use BF16 on RTX 5090
    
    # Data
    train_file: str = "data/processed/llm/sft_train.jsonl"
    eval_file: str = "data/processed/llm/sft_eval.jsonl"
    max_seq_length: int = 2048
    
    # Output
    output_dir: str = "models/llm/sft_model"


@dataclass
class DPOConfig:
    """Configuration for DPO (Direct Preference Optimization)."""
    
    # Training hyperparameters
    learning_rate: float = 1e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # Effective batch = 2 * 4 = 8
    beta: float = 0.1  # DPO beta (temperature for preference loss)
    
    # Optimization
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Data
    train_file: str = "data/processed/llm/dpo_train.jsonl"
    eval_file: str = "data/processed/llm/dpo_eval.jsonl"
    max_seq_length: int = 2048
    
    # Output
    output_dir: str = "models/llm/dpo_model"


@dataclass
class GRPOConfig:
    """Configuration for GRPO (RL from Your Feedback)."""
    
    # Training hyperparameters
    learning_rate: float = 5e-6
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    # Reward settings
    reward_scale: float = 0.1
    clip_ratio: float = 0.2
    
    # Optimization
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 50
    
    # Data
    feedback_file: str = "data/processed/llm/feedback.jsonl"
    max_seq_length: int = 2048
    
    # Output
    output_dir: str = "models/llm/grpo_model"


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    
    # Model path
    model_path: str = "models/llm/sft_model"  # or dpo_model or grpo_model
    
    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_default_model_config() -> ModelConfig:
    """Get default model configuration."""
    return ModelConfig()


def get_default_sft_config() -> SFTConfig:
    """Get default SFT configuration."""
    return SFTConfig()


def get_default_dpo_config() -> DPOConfig:
    """Get default DPO configuration."""
    return DPOConfig()


def get_default_grpo_config() -> GRPOConfig:
    """Get default GRPO configuration."""
    return GRPOConfig()


def get_default_inference_config() -> InferenceConfig:
    """Get default inference configuration."""
    return InferenceConfig()
