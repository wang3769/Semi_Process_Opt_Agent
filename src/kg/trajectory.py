"""
Graph Trajectory Generator for Cold Start SFT
=============================================

Generates training data that teaches the LLM how to:
1. Read the fab graph structure
2. Navigate from defect → equipment → sensor → parameter → root cause
3. Form hypotheses based on graph paths
4. Verify with metrology tests

This is the "Map-Reader" training for GRPO.
"""

import json
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .graph_builder import KnowledgeGraph, Node, Edge, create_simplified_fab_graph


@dataclass
class TrajectoryStep:
    """A single step in a graph walk trajectory."""
    node_id: str
    node_type: str
    node_name: str
    action: str  # "query", "observe", "hypothesize", "verify"
    observation: str
    reasoning: str


@dataclass
class RCACase:
    """A complete RCA case for training."""
    case_id: str
    defect_pattern: str
    sensor_readings: Dict[str, Any]  # Simulated sensor data
    true_root_cause: str
    valid_graph_path: List[str]  # Nodes in valid path
    trajectory: List[TrajectoryStep]
    subgraph: Dict  # Relevant subgraph context


class TrajectoryGenerator:
    """
    Generates graph walk trajectories for RCA training.
    
    Each trajectory teaches the model to:
    - Start from a defect pattern
    - Query the graph for related equipment
    - Check sensor readings
    - Form a hypothesis
    - Verify with a metrology test
    """
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.defect_to_equipment = self._build_defect_equipment_map()
        self.equipment_to_sensors = self._build_equipment_sensor_map()
        self.sensor_to_parameters = self._build_sensor_parameter_map()
        
    def _build_defect_equipment_map(self) -> Dict[str, List[str]]:
        """Map defects to associated equipment."""
        mapping = {}
        for edge in self.graph.edges:
            if edge.relation == "associated_with":
                if edge.source not in mapping:
                    mapping[edge.source] = []
                mapping[edge.source].append(edge.target)
        return mapping
    
    def _build_equipment_sensor_map(self) -> Dict[str, List[str]]:
        """Map equipment to their sensors."""
        mapping = {}
        for edge in self.graph.edges:
            if edge.relation == "has_sensor":
                if edge.source not in mapping:
                    mapping[edge.source] = []
                mapping[edge.source].append(edge.target)
        return mapping
    
    def _build_sensor_parameter_map(self) -> Dict[str, List[str]]:
        """Map sensors to the parameters they measure."""
        mapping = {}
        for edge in self.graph.edges:
            if edge.relation == "measures":
                if edge.source not in mapping:
                    mapping[edge.source] = []
                mapping[edge.source].append(edge.target)
        return mapping
    
    def generate_case(self, defect_pattern: str, 
                     sensor_readings: Optional[Dict[str, Any]] = None,
                     true_root_cause: Optional[str] = None) -> RCACase:
        """
        Generate an RCA case with trajectory.
        
        Args:
            defect_pattern: The defect pattern observed (e.g., "EDGE_RING")
            sensor_readings: Optional simulated sensor data
            true_root_cause: The actual root cause (for training labels)
            
        Returns:
            RCACase with full trajectory
        """
        case_id = f"case_{defect_pattern}_{random.randint(1000, 9999)}"
        
        # Get associated equipment
        equipment = self.defect_to_equipment.get(defect_pattern, [])
        
        # If no explicit mapping, find via graph
        if not equipment:
            neighbors, _ = zip(*self.graph.get_neighbors(defect_pattern, "associated_with"))
            equipment = [n.id for n in neighbors] if neighbors else []
        
        # Get sensors for equipment
        all_sensors = []
        for eq in equipment:
            all_sensors.extend(self.equipment_to_sensors.get(eq, []))
        
        # Generate sensor readings (or use provided)
        if sensor_readings is None:
            sensor_readings = self._generate_sensor_readings(all_sensors, defect_pattern)
        
        # Find valid graph path
        valid_path = self._find_path(defect_pattern, true_root_cause or "PRESSURE_VARIANCE")
        
        # Build trajectory
        trajectory = self._build_trajectory(
            defect_pattern, equipment, sensor_readings, true_root_cause
        )
        
        # Get subgraph context
        subgraph = self.graph.get_subgraph([defect_pattern] + equipment[:2], depth=2)
        
        return RCACase(
            case_id=case_id,
            defect_pattern=defect_pattern,
            sensor_readings=sensor_readings,
            true_root_cause=true_root_cause or "PRESSURE_VARIANCE",
            valid_graph_path=valid_path,
            trajectory=trajectory,
            subgraph=subgraph
        )
    
    def _generate_sensor_readings(self, sensors: List[str], defect: str) -> Dict[str, Any]:
        """Generate realistic sensor readings based on defect pattern."""
        readings = {}
        
        # Default normal readings
        base_readings = {
            "pressure": {"value": 15.0, "unit": "Torr", "variance": 0.5},
            "temperature": {"value": 400, "unit": "C", "variance": 2.0},
            "flow": {"value": 50, "unit": "sccm", "variance": 1.0},
            "particle": {"value": 10, "unit": "count", "variance": 2},
        }
        
        # Add defect-specific anomalies
        if defect == "EDGE_RING":
            base_readings["pressure"]["variance"] = 5.0  # High variance
            base_readings["pressure"]["value"] = 18.0
        elif defect == "CENTER":
            base_readings["temperature"]["variance"] = 15.0  # High temp variance
            base_readings["temperature"]["value"] = 420
        elif defect == "LOCAL" or defect == "RANDOM":
            base_readings["particle"]["value"] = 150  # High particles
            base_readings["particle"]["variance"] = 30
        elif defect == "SCRATCH":
            base_readings["temperature"]["value"] = 25  # Room temp (handling)
        
        # Map sensors to readings
        for sensor in sensors:
            sensor_lower = sensor.lower()
            if "pressure" in sensor_lower:
                readings[sensor] = base_readings["pressure"]
            elif "temp" in sensor_lower:
                readings[sensor] = base_readings["temperature"]
            elif "flow" in sensor_lower:
                readings[sensor] = base_readings["flow"]
            elif "particle" in sensor_lower:
                readings[sensor] = base_readings["particle"]
            else:
                readings[sensor] = {"value": 0, "unit": "unknown"}
        
        return readings
    
    def _find_path(self, start: str, end: str) -> List[str]:
        """Find a path between two nodes."""
        paths = self.graph.find_path_between(start, end, max_length=5)
        if paths:
            return [n.id for n in paths[0]]
        return [start, end]
    
    def _build_trajectory(self, defect: str, equipment: List[str],
                         sensor_readings: Dict, true_cause: Optional[str]) -> List[TrajectoryStep]:
        """Build a trajectory of steps for this RCA case."""
        trajectory = []
        
        # Step 1: Identify defect pattern
        defect_node = self.graph.get_node(defect)
        trajectory.append(TrajectoryStep(
            node_id=defect,
            node_type="defect",
            node_name=defect_node.name if defect_node else defect,
            action="observe",
            observation=f"Defect pattern identified: {defect}",
            reasoning="The wafer map shows characteristic pattern. Based on WM-811K classification, this is a " + defect + " pattern."
        ))
        
        # Step 2: Query for related equipment
        equipment_names = [self.graph.get_node(e).name if self.graph.get_node(e) else e 
                         for e in equipment[:2]]
        trajectory.append(TrajectoryStep(
            node_id=equipment[0] if equipment else "UNKNOWN",
            node_type="equipment",
            node_name=equipment_names[0] if equipment_names else "Unknown Equipment",
            action="query",
            observation=f"Found {len(equipment)} related equipment: {', '.join(equipment_names)}",
            reasoning=f"Querying the knowledge graph for equipment associated with {defect} pattern. Graph shows {equipment_names[0] if equipment_names else 'multiple'} are commonly involved."
        ))
        
        # Step 3: Check sensors
        sensor_info = []
        for sensor_id, reading in list(sensor_readings.items())[:3]:
            sensor_node = self.graph.get_node(sensor_id)
            sensor_name = sensor_node.name if sensor_node else sensor_id
            val = reading.get("value", 0)
            var = reading.get("variance", 0)
            unit = reading.get("unit", "")
            sensor_info.append(f"{sensor_name}: {val}±{var} {unit}")
        
        trajectory.append(TrajectoryStep(
            node_id=list(sensor_readings.keys())[0] if sensor_readings else "SENSOR_1",
            node_type="sensor",
            node_name="Process Sensors",
            action="observe",
            observation="; ".join(sensor_info),
            reasoning="Checking sensor readings. Looking for anomalies that correlate with the defect pattern."
        ))
        
        # Step 4: Form hypothesis
        if true_cause:
            cause_node = self.graph.get_node(true_cause)
            cause_name = cause_node.name if cause_node else true_cause
            trajectory.append(TrajectoryStep(
                node_id=true_cause,
                node_type="root_cause",
                node_name=cause_name,
                action="hypothesize",
                observation=f"Hypothesis: {cause_name}",
                reasoning=f"Based on the graph path from {defect} → {equipment[0] if equipment else 'equipment'} → sensors → {cause_name}, this is a likely root cause. The sensor readings support this hypothesis."
            ))
        
        # Step 5: Verify with metrology
        trajectory.append(TrajectoryStep(
            node_id="METROLOGY",
            node_type="test",
            node_name="Verification Metrology",
            action="verify",
            observation="Recommended: Chamber pressure calibration check, Temperature uniformity map",
            reasoning="To confirm the hypothesis, run targeted metrology tests. This validates the root cause before taking corrective action."
        ))
        
        return trajectory


def generate_training_examples(graph: KnowledgeGraph, num_examples: int = 100) -> List[Dict]:
    """
    Generate training examples for Cold Start SFT.
    
    Each example teaches the model to:
    - Use the graph structure
    - Follow valid reasoning paths
    - Avoid hallucinations
    """
    generator = TrajectoryGenerator(graph)
    
    # Defect patterns to generate cases for
    defects = ["CENTER", "DONUT", "EDGE_RING", "EDGE_LOC", "LOCAL", "RANDOM", "SCRATCH"]
    
    # Root causes
    causes = ["TEMP_GRADIENT", "PRESSURE_VARIANCE", "PARTICLE_CONTAM", "EDGE_BEAD", 
              "CLAMP_MARKS", "HANDLING_DAMAGE", "CHEMICAL_PURITY", "UNIFORMITY"]
    
    examples = []
    
    for i in range(num_examples):
        defect = random.choice(defects)
        cause = random.choice(causes)
        
        rca_case = generator.generate_case(defect, true_root_cause=cause)
        
        # Build the training example
        example = {
            "case_id": rca_case.case_id,
            "defect_pattern": rca_case.defect_pattern,
            "sensor_readings": rca_case.sensor_readings,
            "true_root_cause": rca_case.true_root_cause,
            "valid_graph_path": rca_case.valid_graph_path,
            "subgraph": rca_case.subgraph,
            "trajectory": [
                {
                    "node": step.node_name,
                    "type": step.node_type,
                    "action": step.action,
                    "observation": step.observation,
                    "reasoning": step.reasoning
                }
                for step in rca_case.trajectory
            ]
        }
        
        examples.append(example)
    
    return examples


def create_sft_prompt(example: Dict) -> Dict[str, str]:
    """
    Convert an RCA case to SFT training format.
    
    Input: Defect pattern + subgraph context
    Output: Reasoning trajectory + final answer
    """
    # Build context from subgraph
    nodes = example["subgraph"]["nodes"]
    edges = example["subgraph"]["edges"]
    
    context_parts = [f"Defect Pattern: {example['defect_pattern']}"]
    context_parts.append("\nRelevant Equipment & Sensors:")
    
    equipment_in_graph = [n for n in nodes.values() if n["type"] == "equipment"]
    for eq in equipment_in_graph[:3]:
        context_parts.append(f"  - {eq['name']}: {eq['description']}")
    
    context_parts.append("\nSensor Readings:")
    for sensor_id, reading in list(example["sensor_readings"].items())[:4]:
        val = reading.get("value", "N/A")
        var = reading.get("variance", 0)
        unit = reading.get("unit", "")
        context_parts.append(f"  - {sensor_id}: {val}±{var} {unit}")
    
    context_parts.append("\nValid Graph Path:")
    context_parts.append(" → ".join(example["valid_graph_path"]))
    
    context = "\n".join(context_parts)
    
    # Build the response from trajectory
    trajectory = example["trajectory"]
    response_parts = []
    
    for step in trajectory:
        response_parts.append(f"<step>")
        response_parts.append(f"Action: {step['action'].upper()}")
        response_parts.append(f"Node: {step['node']} ({step['type']})")
        response_parts.append(f"Observation: {step['observation']}")
        response_parts.append(f"Reasoning: {step['reasoning']}")
        response_parts.append(f"</step>\n")
    
    response_parts.append(f"\n<conclusion>")
    response_parts.append(f"Root Cause: {example['true_root_cause']}")
    response_parts.append(f"Graph Path Validated: Yes")
    response_parts.append(f"</conclusion>")
    
    response = "\n".join(response_parts)
    
    instruction = f"""Given the defect pattern and fab graph context, perform root cause analysis by:
1. Querying the knowledge graph for related equipment
2. Checking sensor readings for anomalies
3. Forming a hypothesis based on valid graph paths
4. Suggesting verification metrology

Use ONLY the equipment and sensors shown in the graph context. Do NOT hallucinate tools or sensors that are not listed."""
    
    return {
        "instruction": instruction,
        "context": context,
        "response": response,
        "valid_path": example["valid_graph_path"],
        "true_root_cause": example["true_root_cause"]
    }


def create_grpo_reward_format(example: Dict) -> Dict[str, Any]:
    """
    Create GRPO reward format with causal rewards.
    
    This is used to train with:
    - Correctness Reward: Did it find the true root cause?
    - Causal Link Reward: Did it follow valid graph paths?
    - Efficiency Reward: Fewest steps?
    """
    return {
        "prompt": f"""Defect: {example['defect_pattern']}
Sensors: {json.dumps(example['sensor_readings'], indent=2)}
Graph: {json.dumps(list(example['subgraph']['nodes'].keys()))}""",
        "correct_answer": example["true_root_cause"],
        "valid_path": example["valid_graph_path"],
        "reward_keywords": _extract_keywords(example["true_root_cause"]),
        "hallucination_keywords": _get_common_hallucinations()
    }


def _extract_keywords(root_cause: str) -> List[str]:
    """Extract keywords for reward calculation."""
    keyword_map = {
        "TEMP_GRADIENT": ["temperature", "gradient", "uniformity", "heat"],
        "PRESSURE_VARIANCE": ["pressure", "variance", "instability", "fluctuation"],
        "PARTICLE_CONTAM": ["particle", "contamination", "clean", "purity"],
        "EDGE_BEAD": ["edge", "bead", "exposure", "ring"],
        "CLAMP_MARKS": ["clamp", "chuck", "wafer", "contact"],
        "HANDLING_DAMAGE": ["scratch", "handling", "robot", "transfer"],
        "CHEMICAL_PURITY": ["chemical", "purity", "impurity", "wet"],
        "UNIFORMITY": ["uniform", "uniformity", "profile"]
    }
    return keyword_map.get(root_cause, [])


def _get_common_hallucinations() -> List[str]:
    """Keywords that indicate hallucinations (not in the fab graph)."""
    return [
        "laser", "optical", "beam", "radiation",  # Not in semiconductor fabs
        "magnetic", "coil", "inductor",  # Not typical sensors
        "valve_x", "sensor_z", "tool_99"  # Non-existent IDs
    ]


# Main execution
if __name__ == "__main__":
    # Create the graph
    graph = create_simplified_fab_graph()
    
    print("=== Generating Graph Trajectory Training Data ===\n")
    
    # Generate examples
    examples = generate_training_examples(graph, num_examples=50)
    print(f"Generated {len(examples)} training examples")
    
    # Convert to SFT format
    sft_examples = [create_sft_prompt(ex) for ex in examples]
    print(f"Created {len(sft_examples)} SFT examples")
    
    # Convert to GRPO format
    grpo_examples = [create_grpo_reward_format(ex) for ex in examples]
    print(f"Created {len(grpo_examples)} GRPO reward examples")
    
    # Save
    output_dir = Path("data/processed/llm")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save SFT examples
    with open(output_dir / "graph_sft_train.jsonl", "w", encoding="utf-8") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # Save GRPO examples
    with open(output_dir / "graph_grpo_train.jsonl", "w", encoding="utf-8") as f:
        for ex in grpo_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"\nSaved to {output_dir}")
    
    # Print sample
    print("\n=== Sample SFT Example ===")
    print(f"Instruction: {sft_examples[0]['instruction'][:200]}...")
    print(f"\nContext: {sft_examples[0]['context'][:300]}...")
    print(f"\nResponse: {sft_examples[0]['response'][:300]}...")
