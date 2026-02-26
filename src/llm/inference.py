"""
LLM Inference Module
===================

Generates RCA reports using the fine-tuned LLM.
Integrates with RAG for context-aware generation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm.config import ModelConfig, InferenceConfig
from src.rag.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RCAInferenceEngine:
    """Inference engine for Root Cause Analysis reports."""
    
    def __init__(
        self,
        model_path: str = "models/llm/sft_model/final",
        config: Optional[InferenceConfig] = None,
        retriever: Optional[Retriever] = None,
    ):
        """Initialize inference engine."""
        self.model_path = model_path
        self.config = config or InferenceConfig()
        self.retriever = retriever
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Model loaded successfully!")
        return self
    
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate text from prompt."""
        if self.pipeline is None:
            self.load_model()
        
        # Build full prompt with context
        if context:
            full_prompt = f"""<|im_start|>system
You are an expert semiconductor process engineer.<|im_end|>
<|im_start|>user
Context:
{context}

Instruction: {prompt}<|im_end|>
<|im_start|>assistant
"""
        else:
            full_prompt = f"""<|im_start|>system
You are an expert semiconductor process engineer.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": self.config.do_sample,
        }
        
        outputs = self.pipeline(full_prompt, **gen_kwargs)
        generated_text = outputs[0]["generated_text"]
        response = generated_text[len(full_prompt):].strip()
        
        return response
    
    def analyze_defect_pattern(
        self,
        defect_description: str,
        wafer_map_data: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze a defect pattern from description."""
        prompt = f"Analyze: {defect_description}"
        
        context = None
        if self.retriever:
            context = self.retriever.retrieve_with_context(
                f"defect pattern {defect_description}", n_results=3
            )
        
        response = self.generate(prompt, context)
        
        return {
            "defect_description": defect_description,
            "analysis": response,
            "context_used": context is not None,
        }
    
    def generate_full_rca_report(
        self,
        wafer_data: Dict[str, Any],
    ) -> str:
        """Generate a comprehensive RCA report."""
        defect_pattern = wafer_data.get("defect_pattern", "Unknown")
        defect_count = wafer_data.get("defect_count", 0)
        yield_percentage = wafer_data.get("yield_percentage", 0)
        
        prompt = f"""Generate RCA report:
- Defect Pattern: {defect_pattern}
- Defect Count: {defect_count}
- Yield: {yield_percentage}%"""
        
        context = None
        if self.retriever:
            context = self.retriever.retrieve_with_context(
                f"{defect_pattern} yield analysis", n_results=3
            )
        
        return self.generate(prompt, context)


def create_inference_engine(
    model_path: str = "models/llm/sft_model/final",
    retriever=None,
) -> RCAInferenceEngine:
    """Create inference engine."""
    return RCAInferenceEngine(model_path=model_path, retriever=retriever)


class UserFeedbackCollector:
    """Collects user feedback for GRPO training."""
    
    def __init__(self, feedback_file: str = "data/processed/llm/feedback.jsonl"):
        self.feedback_file = feedback_file
    
    def collect_feedback(self, prompt: str, response: str, rating: float, feedback_type: str = "quality"):
        """Collect user feedback on a generated response."""
        from src.llm.dataset import save_feedback, create_feedback_entry
        
        entry = create_feedback_entry(prompt, response, rating, feedback_type)
        save_feedback([entry], self.feedback_file)
        logger.info(f"Collected feedback: rating={rating}/5")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model_path = "models/llm/sft_model/final"
    if os.path.exists(model_path):
        engine = create_inference_engine(model_path)
        engine.load_model()
        response = engine.generate("What causes Edge-Ring defects?")
        print("\n=== Response ===")
        print(response)
    else:
        logger.info(f"Model not found at {model_path}. Run training first!")
