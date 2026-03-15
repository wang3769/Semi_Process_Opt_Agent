"""
SFT Evaluation Script
====================

Evaluates the SFT model with test prompts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.inference import RCAInferenceEngine

# Test prompts
TEST_PROMPTS = [
    {
        "category": "Defect Classification",
        "prompt": "What is the WM-811K wafer map defect classification system? List the main defect types."
    },
    {
        "category": "Root Cause Analysis", 
        "prompt": "What causes EDGE_RING defects in semiconductor manufacturing and how can they be prevented?"
    },
    {
        "category": "Equipment",
        "prompt": "Explain the role of CVD (Chemical Vapor Deposition) chambers in semiconductor fabs."
    },
    {
        "category": "Process Control",
        "prompt": "What are the key sensor parameters monitored in plasma etch processes?"
    },
    {
        "category": "Yield Optimization",
        "prompt": "How do you perform root cause analysis for yield excursions in semiconductor manufacturing?"
    }
]

def main():
    print("=" * 60)
    print("SFT Model Evaluation")
    print("=" * 60)
    
    # Load model
    model_path = "models/llm/sft_model/final"
    print(f"\nLoading model from: {model_path}")
    
    try:
        engine = RCAInferenceEngine(model_path=model_path)
        engine.load_model()
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run test prompts
    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['category']}")
        print(f"{'='*60}")
        print(f"Prompt: {test['prompt']}")
        print(f"\nResponse:")
        print("-" * 40)
        
        response = engine.generate(
            test['prompt'],
            max_new_tokens=512,
            temperature=0.7
        )
        print(response)
        print()

if __name__ == "__main__":
    main()
