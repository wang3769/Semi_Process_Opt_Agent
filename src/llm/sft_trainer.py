"""
SFT (Supervised Fine) Trainer
================================-tuning====

Trains the LLM on instruction-response pairs using TRL's SFTTrainer.
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
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType #paraneter efficient fine-tuning (PEFT) library for LoRA
from trl import SFTTrainer # TRL (Transformer Reinforcement Learning) library for SFT training

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.config import ModelConfig, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTTrainerWrapper:
    """SFT Trainer with LoRA optimization for RTX 5090."""
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        sft_config: Optional[SFTConfig] = None,
    ):
        """
        Initialize SFT Trainer.
        
        Args:
            model_config: Model configuration
            sft_config: SFT training configuration
        """
        self.model_config = model_config or ModelConfig()
        self.sft_config = sft_config or SFTConfig()
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_model(self):
        """Load and prepare the model with LoRA."""
        logger.info(f"Loading model: {self.model_config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        load_kwargs = {
            "pretrained_model_name_or_path": self.model_config.model_name,
            "torch_dtype": torch.bfloat16 if self.model_config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16,
            "device_map": self.model_config.device_map,
            "trust_remote_code": True,
        }
        
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
        """Load and tokenize the training dataset."""
        logger.info("Loading dataset...")
        
        # Load dataset
        dataset = load_dataset(
            "json",
            data_files={
                "train": self.sft_config.train_file,
                "eval": self.sft_config.eval_file if os.path.exists(self.sft_config.eval_file) else self.sft_config.train_file,
            }
        )
        
        logger.info(f"Loaded dataset: {dataset}")
        
        # Tokenize function
        def tokenize_function(examples):
            # Format as instruction-response
            texts = []
            for instruction, context, response in zip(
                examples["instruction"], 
                examples.get("context", [""] * len(examples["instruction"])),
                examples["response"]
            ):
                #  this is qwen format
                if context:
                    text = f"""<|im_start|>system
You are an expert semiconductor process engineer specializing in yield optimization, defect analysis, and root cause analysis.<|im_end|>
<|im_start|>user
Context:
{context}

Instruction: {instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""
                else:
                    text = f"""<|im_start|>system
You are an expert semiconductor process engineer specializing in yield optimization, defect analysis, and root cause analysis.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""
                texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.sft_config.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            
            # Set labels (same as input for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Tokenize dataset
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
        )
        
        return dataset
    
    def train(self):
        """Run SFT training."""
        # Setup model
        self.setup_model()
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.sft_config.output_dir,
            num_train_epochs=self.sft_config.num_train_epochs,
            per_device_train_batch_size=self.sft_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.sft_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.sft_config.gradient_accumulation_steps,
            learning_rate=self.sft_config.learning_rate,
            warmup_ratio=self.sft_config.warmup_ratio,
            weight_decay=self.sft_config.weight_decay,
            max_grad_norm=self.sft_config.max_grad_norm,
            logging_steps=self.sft_config.logging_steps,
            save_steps=self.sft_config.save_steps,
            eval_steps=self.sft_config.eval_steps,
            save_total_limit=self.sft_config.save_total_limit,
            fp16=self.sft_config.fp16,
            bf16=self.sft_config.bf16,
            save_strategy="steps",
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            report_to=["tensorboard"],
            logging_dir=f"{self.sft_config.output_dir}/logs",
        )
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("eval", dataset["train"].train_test_split(test_size=0.1)["test"]),
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            max_seq_length=self.sft_config.max_seq_length,
            dataset_text_field="input_ids",
        )
        
        # Train
        logger.info("Starting SFT training...")
        self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model(f"{self.sft_config.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.sft_config.output_dir}/final")
        
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


def train_sft(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "models/llm/sft_model",
    train_file: str = "data/processed/llm/sft_train.jsonl",
    eval_file: str = "data/processed/llm/sft_eval.jsonl",
):
    """
    Convenience function to run SFT training.
    
    Args:
        model_name: Base model to fine-tune
        output_dir: Output directory
        train_file: Training data file
        eval_file: Evaluation data file
    """
    # Update configs
    model_config = ModelConfig(model_name=model_name)
    sft_config = SFTConfig(
        output_dir=output_dir,
        train_file=train_file,
        eval_file=eval_file,
    )
    
    # Train
    trainer = SFTTrainerWrapper(model_config, sft_config)
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
    trainer = train_sft()
    logger.info("SFT training complete!")
