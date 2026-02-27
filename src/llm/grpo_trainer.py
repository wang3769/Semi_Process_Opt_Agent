"""
GRPO Trainer with Causal Rewards
================================

Implements Group Relative Policy Optimization for Graph-Based RCA.

Key innovations:
1. Correctness Reward: Did it find the true root cause?
2. Causal Link Reward: Did the reasoning follow valid graph paths?
   - PENALIZE hallucinations (tools/sensors not in graph)
3. Efficiency Reward: Did it find the cause in fewest steps?

This trains the model to be both accurate AND faithful to the graph.
"""

import json
import re
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# For training (would use trl in practice)
try:
    from trl import GRPOTrainer, GRPOConfig
    TR_AVAILABLE = True
except ImportError:
    TR_AVAILABLE = False
    print("Warning: trl not installed. Using mock implementation.")


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    correctness_weight: float = 1.0
    causal_link_weight: float = 1.5  # Higher weight to penalize hallucinations
    efficiency_weight: float = 0.5
    hallucination_penalty: float = -2.0  # Massive penalty for hallucinations
    
    # Thresholds
    min_steps_ideal: int = 3
    max_steps_acceptable: int = 5


@dataclass
class GRPOSample:
    """A single sample for GRPO training."""
    prompt: str
    response: str
    correct_root_cause: str
    valid_graph_path: List[str]
    available_nodes: List[str]  # Nodes in the provided subgraph
    num_steps: int = 0


class CausalRewardCalculator:
    """
    Calculates causal rewards for GRPO training.
    
    This is the key innovation - penalizing graph hallucinations
    while rewarding valid reasoning paths.
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.common_hallucinations = self._init_hallucination_set()
        
    def _init_hallucination_set(self) -> set:
        """Initialize set of common hallucinations in semiconductor RCA."""
        return {
            # Non-existent equipment types
            "laser", "optical_sensor", "beam_source", "radiation_detector",
            "magnetic_coil", "induction_heater", "capacitor_plate",
            # Fake sensor/equipment IDs
            "sensor_x", "sensor_99", "tool_abc", "chamber_xyz",
            "valve_mfc_99", "pump_primary_x",
            # Nonsense terms
            "quantum_effect", "nanobubble", "void formation",
            # Out-of-domain
            "cpu", "memory", "transistor_gate",
        }
    
    def calculate_total_reward(self, response: str, ground_truth: Dict) -> Tuple[float, Dict]:
        """
        Calculate total reward from all components.
        
        Args:
            response: Model's generated response
            ground_truth: Dict with keys:
                - correct_root_cause: str
                - valid_graph_path: List[str]
                - available_nodes: List[str]
                
        Returns:
            (total_reward, reward_breakdown)
        """
        correctness = self._calculate_correctness_reward(response, ground_truth)
        causal_link = self._calculate_causal_link_reward(response, ground_truth)
        efficiency = self._calculate_efficiency_reward(response, ground_truth)
        
        total = (
            self.config.correctness_weight * correctness +
            self.config.causal_link_weight * causal_link +
            self.config.efficiency_weight * efficiency
        )
        
        breakdown = {
            "correctness": correctness,
            "causal_link": causal_link,
            "efficiency": efficiency,
            "total": total
        }
        
        return total, breakdown
    
    def _calculate_correctness_reward(self, response: str, ground_truth: Dict) -> float:
        """
        Correctness Reward: Did it find the true root cause?
        
        Reward = 1.0 if correct, 0.0 if incorrect
        Partial credit for related causes.
        """
        correct_cause = ground_truth.get("correct_root_cause", "").lower()
        response_lower = response.lower()
        
        # Extract root cause from response
        # Look for explicit mentions
        root_cause_patterns = [
            r"root cause[:\s]+([a-z_]+)",
            r"primary cause[:\s]+([a-z_]+)",
            r"hypothesis[:\s]+([a-z_]+)",
            r"conclusion[:\s]+([a-z_]+)",
            r"identified cause[:\s]+([a-z_]+)",
        ]
        
        extracted_cause = None
        for pattern in root_cause_patterns:
            match = re.search(pattern, response_lower)
            if match:
                extracted_cause = match.group(1)
                break
        
        if extracted_cause is None:
            # Check if any keyword from correct cause is mentioned
            cause_keywords = correct_cause.split("_")
            if any(kw in response_lower for kw in cause_keywords if len(kw) > 3):
                extracted_cause = correct_cause
        
        # Calculate reward
        if extracted_cause is None:
            return 0.0
        
        # Exact match
        if correct_cause in extracted_cause or extracted_cause in correct_cause:
            return 1.0
        
        # Partial match (related causes get partial credit)
        correct_words = set(correct_cause.split("_"))
        extracted_words = set(extracted_cause.split("_"))
        overlap = correct_words.intersection(extracted_words)
        
        if overlap:
            return len(overlap) / max(len(correct_words), len(extracted_words))
        
        return 0.0
    
    def _calculate_causal_link_reward(self, response: str, ground_truth: Dict) -> float:
        """
        Causal Link Reward: Did reasoning follow valid graph paths?
        
        This is crucial - we penalize hallucinations heavily.
        
        Reward = 1.0 if all mentioned nodes are in graph
        Penalty = -2.0 if hallucinations detected
        """
        available_nodes = set(ground_truth.get("available_nodes", []))
        valid_path = ground_truth.get("valid_graph_path", [])
        
        # Extract mentioned nodes from response
        mentioned_nodes = self._extract_mentioned_nodes(response)
        
        if not mentioned_nodes:
            # No nodes mentioned - neutral
            return 0.0
        
        # Check for hallucinations
        hallucinations = mentioned_nodes - available_nodes
        
        if hallucinations:
            # MASSIVE PENALTY for hallucinations
            # This is the key innovation - we really punish making up tools
            return self.config.hallucination_penalty
        
        # Check if reasoning follows valid path
        valid_path_set = set(valid_path)
        correct_nodes = mentioned_nodes.intersection(valid_path_set)
        
        if len(correct_nodes) >= len(valid_path_set) * 0.5:
            return 1.0  # Good path following
        
        return 0.3  # Some correct but not optimal
    
    def _extract_mentioned_nodes(self, response: str) -> set:
        """Extract equipment/sensor/parameter names from response."""
        response_lower = response.lower()
        mentioned = set()
        
        # Equipment patterns
        equipment_patterns = [
            r"\b(cvd|etch|litho|cmp|clean|robot|chamber)[_\s]?(\d+)?\b",
            r"\b(chamber|station|port|tool|equipment)\b",
        ]
        
        # Sensor patterns  
        sensor_patterns = [
            r"\b(pressure|temperature|flow|particle)[_\s]?(sensor|monitor)?\b",
            r"\b(mfc|thermocouple|baratron)\b",
        ]
        
        # Parameter patterns
        param_patterns = [
            r"\b(pressure|temperature|power|time|flow)[_\s]?(setpoint|value)?\b",
        ]
        
        all_patterns = equipment_patterns + sensor_patterns + param_patterns
        
        for pattern in all_patterns:
            matches = re.findall(pattern, response_lower)
            for match in matches:
                if isinstance(match, tuple):
                    mentioned.add("".join(match).strip())
                else:
                    mentioned.add(match.strip())
        
        return mentioned
    
    def _calculate_efficiency_reward(self, response: str, ground_truth: Dict) -> float:
        """
        Efficiency Reward: Did it find cause in fewest steps?
        
        Reward = 1.0 if efficient, decreasing for more steps.
        """
        # Count reasoning steps in response
        step_patterns = [
            r"<step>",
            r"step\s*\d+",
            r"phase\s*\d+",
            r"\d+\.\s+(query|check|verify|hypothesize)",
        ]
        
        num_steps = 0
        for pattern in step_patterns:
            num_steps += len(re.findall(pattern, response.lower()))
        
        if num_steps == 0:
            # Try to count lines/actions
            action_verbs = ["query", "check", "observe", "hypothesize", "verify", "analyze"]
            for verb in action_verbs:
                if verb in response.lower():
                    num_steps += 1
        
        # Calculate reward based on steps
        ideal = self.config.min_steps_ideal
        max_acceptable = self.config.max_steps_acceptable
        
        if num_steps <= ideal:
            return 1.0
        elif num_steps <= max_acceptable:
            return 0.5
        else:
            # Too many steps - penalty
            return max(0.0, 0.5 - (num_steps - max_acceptable) * 0.1)


def create_grpo_dataset(
    graph_file: str = "data/processed/kg/fab_graph.json",
    num_samples: int = 100
) -> List[GRPOSample]:
    """
    Create GRPO dataset from graph trajectories.
    
    This combines:
    - The generated trajectories
    - Ground truth labels
    - Available graph nodes
    """
    from src.kg.graph_builder import KnowledgeGraph
    from src.kg.trajectory import generate_training_examples, create_grpo_reward_format
    
    # Load or create graph
    if os.path.exists(graph_file):
        graph = KnowledgeGraph.load(graph_file)
    else:
        from src.kg.graph_builder import create_simplified_fab_graph
        graph = create_simplified_fab_graph()
    
    # Generate examples
    examples = generate_training_examples(graph, num_samples)
    
    samples = []
    for ex in examples:
        # Get available nodes from subgraph
        available_nodes = list(ex["subgraph"]["nodes"].keys())
        
        sample = GRPOSample(
            prompt=ex["defect_pattern"] + "\n" + json.dumps(ex["sensor_readings"]),
            response="",  # Will be filled by model
            correct_root_cause=ex["true_root_cause"],
            valid_graph_path=ex["valid_graph_path"],
            available_nodes=available_nodes
        )
        samples.append(sample)
    
    return samples


def run_mock_grpo_training(
    train_data: List[Dict],
    num_epochs: int = 3,
    reward_config: RewardConfig = None
):
    """
    Mock GRPO training for demonstration.
    
    In practice, this would use the trl library's GRPOTrainer.
    """
    calculator = CausalRewardCalculator(reward_config)
    
    print("=" * 60)
    print("GRPO Training with Causal Rewards")
    print("=" * 60)
    print(f"Training samples: {len(train_data)}")
    print(f"Epochs: {num_epochs}")
    print(f"Reward weights:")
    print(f"  - Correctness: {reward_config.correctness_weight}")
    print(f"  - Causal Link: {reward_config.causal_link_weight}")
    print(f"  - Efficiency: {reward_config.efficiency_weight}")
    print(f"  - Hallucination Penalty: {reward_config.hallucination_penalty}")
    print()
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        epoch_rewards = []
        
        for i, sample in enumerate(train_data[:10]):  # Demo with 10 samples
            # Simulate model responses
            # In real training, this would be the model's actual output
            simulated_responses = _generate_simulated_responses(sample)
            
            best_reward = -float('inf')
            best_response = None
            
            for response in simulated_responses:
                reward, breakdown = calculator.calculate_total_reward(
                    response, 
                    {
                        "correct_root_cause": sample["true_root_cause"],
                        "valid_graph_path": sample["valid_graph_path"],
                        "available_nodes": list(sample["subgraph"]["nodes"].keys())
                    }
                )
                
                if reward > best_reward:
                    best_reward = reward
                    best_response = response
            
            epoch_rewards.append(best_reward)
            
            if i < 3:  # Show first 3 examples
                print(f"\nSample {i + 1}:")
                print(f"  Defect: {sample['defect_pattern']}")
                print(f"  Correct Cause: {sample['true_root_cause']}")
                print(f"  Best Reward: {best_reward:.3f}")
        
        avg_reward = np.mean(epoch_rewards)
        print(f"\nEpoch {epoch + 1} Average Reward: {avg_reward:.3f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


def _generate_simulated_responses(sample: Dict) -> List[str]:
    """Generate simulated model responses for training demo."""
    defect = sample["defect_pattern"]
    true_cause = sample["true_root_cause"]
    available = list(sample["subgraph"]["nodes"].keys())
    
    # Good response - uses graph correctly
    good_response = f"""<step>
Action: OBSERVE
Node: {defect} (defect)
Observation: Defect pattern identified
Reasoning: Based on wafer map analysis
</step>

<step>
Action: QUERY
Node: CVD_1 (equipment)  
Observation: Found related equipment in graph
Reasoning: Graph shows CVD_1 is associated with {defect}
</step>

<step>
Action: HYPOTHESIZE
Node: {true_cause} (root_cause)
Observation: Hypothesis formed
Reasoning: Followed valid graph path
</step>

<conclusion>
Root Cause: {true_cause}
Graph Path Validated: Yes
</conclusion>"""
    
    # Bad response - hallucinates
    bad_response = f"""Based on the wafer map, this is caused by quantum_effect in the laser_sensor.

The magnetic_coil in chamber_xyz is malfunctioning. 

Root cause: sensor_99 failure."""
    
    # Partial response - misses some steps
    partial_response = f"""The defect pattern is {defect}.

Root cause: {true_cause}"""
    
    return [good_response, bad_response, partial_response]


# Main execution
if __name__ == "__main__":
    from src.kg.graph_builder import create_simplified_fab_graph
    from src.kg.trajectory import generate_training_examples
    
    # Create graph and generate data
    graph = create_simplified_fab_graph()
    examples = generate_training_examples(graph, num_samples=50)
    
    print("=== GRPO Training Demo ===\n")
    
    # Initialize reward calculator
    config = RewardConfig(
        correctness_weight=1.0,
        causal_link_weight=1.5,
        efficiency_weight=0.5,
        hallucination_penalty=-2.0
    )
    
    calculator = CausalRewardCalculator(config)
    
    # Test on first few examples
    for ex in examples[:3]:
        available_nodes = list(ex["subgraph"]["nodes"].keys())
        
        # Test with good response
        good_response = f"""The defect pattern is {ex['defect_pattern']}.
Based on the graph, CVD_1 is the related equipment.
The root cause is {ex['true_root_cause']}."""
        
        # Test with hallucinated response
        bad_response = """The defect is caused by laser_sensor in chamber_xyz.
The quantum_effect in magnetic_coil is the root cause."""
        
        print(f"\n--- Example: {ex['case_id']} ---")
        print(f"Defect: {ex['defect_pattern']}")
        print(f"True Root Cause: {ex['true_root_cause']}")
        
        # Good response rewards
        reward, breakdown = calculator.calculate_total_reward(
            good_response,
            {
                "correct_root_cause": ex["true_root_cause"],
                "valid_graph_path": ex["valid_graph_path"],
                "available_nodes": available_nodes
            }
        )
        
        print(f"\nGood Response:")
        print(f"  Total Reward: {reward:.3f}")
        print(f"  Breakdown: {breakdown}")
        
        # Bad response rewards
        reward, breakdown = calculator.calculate_total_reward(
            bad_response,
            {
                "correct_root_cause": ex["true_root_cause"],
                "valid_graph_path": ex["valid_graph_path"],
                "available_nodes": available_nodes
            }
        )
        
        print(f"\nBad Response (Hallucination):")
        print(f"  Total Reward: {reward:.3f}")
        print(f"  Breakdown: {breakdown}")
    
    print("\n" + "=" * 60)
    print("Key Insight: Hallucinations get -2.0 penalty!")
    print("This teaches the model to ONLY use graph nodes.")
    print("=" * 60)
