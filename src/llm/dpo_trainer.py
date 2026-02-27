"""
DPO (Direct Preference Optimization) Trainer
==============================================

Trains the LLM to align with user preferences using TRL's DPOTrainer.
Optimized for RTX 5090 (32GB VRAM).
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.config import ModelConfig, DPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPOTrainerWrapper:
    """DPO Trainer for preference optimization."""
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        dpo_config: Optional[DPOConfig] = None,
    ):
        """
        Initialize DPO Trainer.
        
        Args:
            model_config: Model configuration
            dpo_config: DPO training configuration
        """
        self.model_config = model_config or ModelConfig()
        self.dpo_config = dpo_config or DPOConfig()
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_model(self):
        """Load and prepare the model with LoRA."""
        logger.info(f"Loading model for DPO: {self.model_config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model (reference model for DPO)
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_config.model_name,
            "torch_dtype": torch.bfloat16,
            "device_map": self.model_config.device_map,
            "trust_remote_code": True,
        }
        
        #DPO RL has two copies of the model: the "reference" (frozen) and the "policy" (trainable). 4bit quantization is efficient
        if self.model_config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        
        # Apply LoRA if enabled
        if self.model_config.use_lora:
            logger.info("Applying LoRA adapters...")
            lora_config = LoraConfig(
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
                target_modules=self.model_config.target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        return self.model
    
    def prepare_dataset(self):
        """Load and prepare the DPO dataset."""
        logger.info("Loading DPO dataset...")
        
        # Load dataset
        dataset = {}
        if os.path.exists(self.dpo_config.train_file):
            dataset["train"] = self.dpo_config.train_file
        if os.path.exists(self.dpo_config.eval_file):
            dataset["eval"] = self.dpo_config.eval_file
        
        if not dataset:
            raise ValueError("No DPO training data found!")
        
        from datasets import load_dataset
        dpo_dataset = load_dataset("json", data_files=dataset)
        
        logger.info(f"Loaded DPO dataset: {dpo_dataset}")
        
        # Format for DPO (needs prompt, chosen, rejected)
        def format_dpo(example):
            return {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }
        
        for split in dpo_dataset:
            dpo_dataset[split] = dpo_dataset[split].map(
                format_dpo,
                desc="Formatting DPO dataset"
            )
        
        return dpo_dataset
    
    def train(self):
        """Run DPO training."""
        # Setup model
        self.setup_model()
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.dpo_config.output_dir,
            num_train_epochs=self.dpo_config.num_train_epochs,
            per_device_train_batch_size=self.dpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.dpo_config.gradient_accumulation_steps,
            learning_rate=self.dpo_config.learning_rate,
            warmup_ratio=self.dpo_config.warmup_ratio,
            weight_decay=self.dpo_config.weight_decay,
            max_grad_norm=self.dpo_config.max_grad_norm,
            logging_steps=self.dpo_config.logging_steps,
            save_steps=self.dpo_config.save_steps,
            eval_steps=self.dpo_config.eval_steps,
            save_total_limit=self.dpo_config.save_total_limit,
            fp16=self.dpo_config.fp16,
            bf16=self.dpo_config.bf16,
            save_strategy="steps",
            evaluation_strategy="steps" if "eval" in dataset else "no",
            report_to=["tensorboard"],
            logging_dir=f"{self.dpo_config.output_dir}/logs",
        )
        
        # Initialize DPO trainer
        eval_dataset = dataset.get("eval")
        
        self.trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            beta=self.dpo_config.beta,
            max_length=self.dpo_config.max_seq_length,
            max_prompt_length=self.dpo_config.max_seq_length // 2,
            max_target_length=self.dpo_config.max_seq_length // 2,
        )
        
        # Train
        logger.info("Starting DPO training...")
        self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model(f"{self.dpo_config.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.dpo_config.output_dir}/final")
        
        return self.trainer
    
    def load_trained_model(self, model_path: str):
        """Load a trained model for inference."""
        logger.info(f"Loading trained model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        return self.model


def train_dpo(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    sft_model_path: str = "models/llm/sft_model/final",
    output_dir: str = "models/llm/dpo_model",
    train_file: str = "data/processed/llm/dpo_train.jsonl",
    eval_file: str = "data/processed/llm/dpo_eval.jsonl",
):
    """
    Convenience function to run DPO training.
    
    Args:
        model_name: Base model name
        sft_model_path: Path to SFT trained model (for initialization)
        output_dir: Output directory
        train_file: Training data file
        eval_file: Evaluation data file
    """
    # Check if SFT model exists, use it as base
    if os.path.exists(sft_model_path):
        model_config = ModelConfig(model_name=sft_model_path)
        logger.info(f"Using SFT model as base: {sft_model_path}")
    else:
        model_config = ModelConfig(model_name=model_name)
        logger.info(f"SFT model not found, using base model: {model_name}")
    
    dpo_config = DPOConfig(
        output_dir=output_dir,
        train_file=train_file,
        eval_file=eval_file,
    )
    
    # Train
    trainer = DPOTrainerWrapper(model_config, dpo_config)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available!")
    
    # Train
    trainer = train_dpo()
    logger.info("DPO training complete!")
