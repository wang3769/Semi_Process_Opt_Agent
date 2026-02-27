"""
Graph-Based LLM Training Pipeline
================================

This script orchestrates the full training pipeline:

1. Cold Start SFT: Train on graph walk trajectories
2. GRPO with Causal Rewards: Fine-tune with graph constraints

Usage:
    python scripts/train_graph_llm.py --phase sft
    python scripts/train_graph_llm.py --phase grpo
    python scripts/train_graph_llm.py --phase all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kg.graph_builder import create_simplified_fab_graph, KnowledgeGraph
from kg.trajectory import generate_training_examples, create_sft_prompt, create_grpo_reward_format
from llm.grpo_trainer import CausalRewardCalculator, RewardConfig, run_mock_grpo_training


def generate_training_data(output_dir: Path, num_examples: int = 100) -> List[Dict]:
    """Generate graph-based training data."""
    print("=" * 60)
    print("Step 1: Generating Graph-Based Training Data")
    print("=" * 60)
    
    # Create graph
    graph = create_simplified_fab_graph()
    
    # Save graph
    graph_file = output_dir / "fab_graph.json"
    graph.save(str(graph_file))
    print(f"Graph saved to: {graph_file}")
    print(f"  - Nodes: {len(graph.nodes)}")
    print(f"  - Edges: {len(graph.edges)}")
    
    # Generate training examples
    examples = generate_training_examples(graph, num_examples)
    print(f"\nGenerated {len(examples)} training examples")
    
    # Convert to SFT format
    sft_examples = [create_sft_prompt(ex) for ex in examples]
    
    # Convert to GRPO format
    grpo_examples = [create_grpo_reward_format(ex) for ex in examples]
    
    # Save
    sft_file = output_dir / "graph_sft_train.jsonl"
    grpo_file = output_dir / "graph_grpo_train.jsonl"
    
    with open(sft_file, "w", encoding="utf-8") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    with open(grpo_file, "w", encoding="utf-8") as f:
        for ex in grpo_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"\nSaved training files:")
    print(f"  - SFT: {sft_file}")
    print(f"  - GRPO: {grpo_file}")
    
    return examples


def run_cold_start_sft(output_dir: Path):
    """
    Step 2: Cold Start SFT on Graph Walks
    
    This teaches the model to:
    - Read fab graph structure
    - Navigate from defect → equipment → sensor → root cause
    - Form hypotheses based on valid paths
    """
    print("\n" + "=" * 60)
    print("Step 2: Cold Start SFT Training")
    print("=" * 60)
    
    sft_file = output_dir / "graph_sft_train.jsonl"
    
    if not sft_file.exists():
        print(f"Error: SFT file not found: {sft_file}")
        print("Run with --generate-data first")
        return
    
    # Load examples
    examples = []
    with open(sft_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} SFT examples")
    
    # In practice, this would use unsloth or trl for actual training
    # For now, we show the training format
    print("\nExample SFT prompt:")
    print("-" * 40)
    ex = examples[0]
    print(f"Instruction: {ex['instruction'][:200]}...")
    print(f"\nContext:\n{ex['context'][:300]}...")
    print(f"\nResponse (first 200 chars):\n{ex['response'][:200]}...")
    
    print("\n" + "=" * 60)
    print("COLD START SFT TRAINING")
    print("=" * 60)
    print("""
In practice, this would fine-tune a model like:
  - Llama-3.1-8B-Instruct
  - Qwen2.5-7B-Instruct
  
Training command (requires GPU):
```bash
trl sft \
  --model_name_orPath meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name sft_file \
  --output_dir models/graph_sft \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4
```

This teaches the "Map-Reader" - how to use the graph.
""")
    
    return examples


def run_grpo_training(output_dir: Path):
    """
    Step 3: GRPO with Causal Rewards
    
    This fine-tunes with:
    - Correctness Reward: Found true root cause?
    - Causal Link Reward: Valid graph paths? (penalize hallucinations!)
    - Efficiency Reward: Fewest steps?
    """
    print("\n" + "=" * 60)
    print("Step 3: GRPO Training with Causal Rewards")
    print("=" * 60)
    
    grpo_file = output_dir / "graph_grpo_train.jsonl"
    
    if not grpo_file.exists():
        print(f"Error: GRPO file not found: {grpo_file}")
        print("Run with --generate-data first")
        return
    
    # Load examples
    examples = []
    with open(grpo_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} GRPO examples")
    
    # Initialize reward calculator
    config = RewardConfig(
        correctness_weight=1.0,
        causal_link_weight=1.5,  # Higher weight!
        efficiency_weight=0.5,
        hallucination_penalty=-2.0  # Massive penalty!
    )
    
    calculator = CausalRewardCalculator(config)
    
    # Demo: Calculate rewards for sample responses
    print("\n" + "-" * 40)
    print("Demo: Reward Calculation")
    print("-" * 40)
    
    # Get first example
    ex = examples[0]
    available_nodes = list(ex.keys()) if isinstance(ex, dict) else []
    
    # Test good response
    good_response = f"""The defect is {ex.get('prompt', 'Edge-Ring')[:50]}.
Based on the graph, CVD_1 is related equipment.
The root cause is PRESSURE_VARIANCE."""
    
    # Test hallucinated response
    bad_response = """The laser_sensor in chamber_xyz caused this.
The quantum_effect from magnetic_coil is the issue."""
    
    # Calculate rewards
    reward, breakdown = calculator.calculate_total_reward(
        good_response,
        {
            "correct_root_cause": "PRESSURE_VARIANCE",
            "valid_graph_path": ["EDGE_RING", "CVD_1", "PRESSURE_1", "CHAMBER_PRESSURE", "PRESSURE_VARIANCE"],
            "available_nodes": ["CVD_1", "PRESSURE_1", "CHAMBER_PRESSURE", "EDGE_RING", "PRESSURE_VARIANCE"]
        }
    )
    
    print(f"\nGood Response (uses graph):")
    print(f"  Total: {reward:.3f}")
    print(f"  Breakdown: {breakdown}")
    
    reward, breakdown = calculator.calculate_total_reward(
        bad_response,
        {
            "correct_root_cause": "PRESSURE_VARIANCE",
            "valid_graph_path": ["EDGE_RING", "CVD_1", "PRESSURE_1", "CHAMBER_PRESSURE", "PRESSURE_VARIANCE"],
            "available_nodes": ["CVD_1", "PRESSURE_1", "CHAMBER_PRESSURE", "EDGE_RING", "PRESSURE_VARIANCE"]
        }
    )
    
    print(f"\nBad Response (hallucination!):")
    print(f"  Total: {reward:.3f}")
    print(f"  Breakdown: {breakdown}")
    
    print("\n" + "=" * 60)
    print("GRPO TRAINING")
    print("=" * 60)
    print("""
In practice, this would use trl's GRPOTrainer:

```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="models/graph_grpo",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    reward_weights={
        "correctness": 1.0,
        "causal_link": 1.5,
        "efficiency": 0.5
    }
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=grpo_dataset,
    reward_function=calculate_causal_reward
)

trainer.train()
```

Key Innovation:
- Hallucination penalty (-2.0) forces model to stay within graph
- Causal link reward encourages valid reasoning paths
- This creates a "faithful" RCA agent
""")


def main():
    parser = argparse.ArgumentParser(description="Graph-Based LLM Training Pipeline")
    parser.add_argument("--phase", choices=["generate", "sft", "grpo", "all"],
                       default="all", help="Training phase to run")
    parser.add_argument("--num_examples", type=int, default=100,
                       help="Number of examples to generate")
    parser.add_argument("--output_dir", type=str,
                       default="data/processed/llm",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.phase in ["generate", "all"]:
        generate_training_data(output_dir, args.num_examples)
    
    if args.phase in ["sft", "all"]:
        run_cold_start_sft(output_dir)
    
    if args.phase in ["grpo", "all"]:
        run_grpo_training(output_dir)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("""
Next Steps:
1. Run on GPU: python scripts/train_graph_llm.py --phase all
2. This generates training data for SFT + GRPO
3. Fine-tune Llama-3.1-8B with the graph-constrained data
4. The model will learn to do RCA by walking the graph

Files Generated:
- data/processed/llm/fab_graph.json (knowledge graph)
- data/processed/llm/graph_sft_train.jsonl (SFT training)
- data/processed/llm/graph_grpo_train.jsonl (GRPO training)
""")


if __name__ == "__main__":
    main()
