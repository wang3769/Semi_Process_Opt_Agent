"""
Generate Synthetic DPO Dataset
=============================

Creates synthetic preference pairs from existing training data for DPO training.
The script generates pairs by:
1. Taking the original good response
2. Creating a degraded "bad" response by adding noise

This is useful for POC before collecting real human feedback.
"""

import json
import random
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_degraded_response(response: str, noise_type: str = "random") -> str:
    """Create a degraded version of the response."""
    
    if noise_type == "truncate":
        # Truncate to 30%
        words = response.split()
        return " ".join(words[:len(words)//3]) + " [response truncated]"
    
    elif noise_type == "garble":
        # Replace key terms with wrong ones
        replacements = {
            "CVD": "TEST",
            "etch": "test",
            "chamber": "box",
            "defect": "error",
            "particle": "dust",
            "wafer": "sample",
            "process": "method",
            "temperature": "heat",
            "pressure": "force",
            "plasma": "gas",
            "chemical": "liquid",
            "semiconductor": "material",
            "yield": "output",
            "root cause": "source",
        }
        result = response
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result + " [Note: This response contains incorrect information.]"
    
    elif noise_type == "irrelevant":
        # Add irrelevant content
        return response + "\n\n[Additional note: The weather today is sunny with a chance of clouds.]"
    
    else:  # mixed
        words = response.split()
        if len(words) > 10:
            # Remove every 3rd word
            degraded = [w for i, w in enumerate(words) if i % 3 != 0]
            return " ".join(degraded) + " [response degraded]"
        return response

def generate_dpo_pairs(
    input_file: str = "data/processed/llm/combined_sft.jsonl",
    output_file: str = "data/processed/llm/dpo_train.jsonl",
    num_pairs: int = 100
):
    """Generate synthetic DPO pairs from training data."""
    
    print(f"Loading data from: {input_file}")
    
    # Read training data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} examples")
    
    # Generate pairs
    dpo_pairs = []
    noise_types = ["truncate", "garble", "irrelevant", "mixed"]
    
    # Shuffle and take samples
    random.seed(42)
    random.shuffle(data)
    samples = data[:num_pairs] if len(data) >= num_pairs else data
    
    for item in samples:
        # Get the good response - check multiple field names
        good_response = item.get("output", "") or item.get("response", "")
        instruction = item.get("instruction", "")
        context = item.get("input", "") or item.get("context", "")
        
        if not good_response:
            continue
        
        # Create prompt
        if context:
            prompt = f"Context: {context}\n\nInstruction: {instruction}"
        else:
            prompt = instruction
        
        # Create degraded response with random noise type
        noise_type = random.choice(noise_types)
        bad_response = create_degraded_response(good_response, noise_type)
        
        # Add pair
        dpo_pairs.append({
            "prompt": prompt,
            "chosen": good_response,
            "rejected": bad_response
        })
    
    # Save DPO dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\nGenerated {len(dpo_pairs)} DPO pairs")
    print(f"Saved to: {output_file}")
    
    # Show examples
    print("\n" + "="*60)
    print("Example DPO pair:")
    print("="*60)
    if dpo_pairs:
        example = dpo_pairs[0]
        print(f"\nPrompt: {example['prompt'][:200]}...")
        print(f"\n[CHOSEN]: {example['chosen'][:200]}...")
        print(f"\n[REJECTED]: {example['rejected'][:200]}...")
    
    return dpo_pairs

if __name__ == "__main__":
    generate_dpo_pairs()
