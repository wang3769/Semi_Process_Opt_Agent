"""
Knowledge Graph for Semiconductor Manufacturing
=============================================

Represents the fab as a graph of:
- Equipment/Tools (Chambers, Robots, Load Ports)
- Sensors (Pressure, Temperature, Flow)
- Parameters (Setpoints, Measurements)
- Defect Patterns
- Root Causes

This enables graph-based reasoning for RCA.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os


@dataclass
class Node:
    """A node in the knowledge graph."""
    id: str
    type: str  # "equipment", "sensor", "parameter", "defect", "root_cause"
    name: str
    description: str = ""
    properties: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "properties": self.properties
        }


@dataclass
class Edge:
    """An edge in the knowledge graph."""
    source: str
    target: str
    relation: str  # "has_sensor", "controls", "causes", "affects", "measured_by"
    weight: float = 1.0
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "description": self.description
        }


class KnowledgeGraph:
    """
    Knowledge Graph for Semiconductor Manufacturing RCA.
    
    Represents the causal relationships between:
    - Equipment → Sensors → Parameters → Root Causes
    - Defect Patterns → Equipment → Root Causes
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, List[Tuple[str, Edge]]] = defaultdict(list)
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        
    def add_node(self, node: Node):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.type_index[node.type].add(node.id)
        
    def add_edge(self, edge: Edge):
        """Add an edge to the graph."""
        self.edges.append(edge)
        self.adjacency[edge.source].append((edge.target, edge))
        
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type."""
        node_ids = self.type_index.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids]
    
    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[Node, Edge]]:
        """Get neighbors of a node, optionally filtered by relation."""
        neighbors = []
        for target_id, edge in self.adjacency.get(node_id, []):
            if relation is None or edge.relation == relation:
                neighbors.append((self.nodes[target_id], edge))
        return neighbors
    
    def traverse(self, start_id: str, max_depth: int = 3, 
                 relation_filter: Optional[str] = None) -> List[List[Node]]:
        """
        Traverse the graph from a starting node.
        
        Returns all paths up to max_depth.
        """
        paths = []
        
        def dfs(current: str, path: List[str], depth: int):
            if depth >= max_depth:
                paths.append([self.nodes[nid] for nid in path])
                return
            
            for target_id, edge in self.adjacency.get(current, []):
                if relation_filter and edge.relation != relation_filter:
                    continue
                path.append(target_id)
                dfs(target_id, path, depth + 1)
                path.pop()
        
        dfs(start_id, [start_id], 0)
        return paths
    
    def find_path_between(self, source_type: str, target_type: str, 
                         max_length: int = 4) -> List[List[Node]]:
        """Find all paths from a source type to a target type."""
        paths = []
        source_ids = self.type_index.get(source_type, set())
        target_ids = self.type_index.get(target_type, set())
        
        for source in source_ids:
            for target in target_ids:
                found = self._bfs_path(source, target, max_length)
                paths.extend(found)
        return paths
    
    def _bfs_path(self, start: str, end: str, max_length: int) -> List[List[Node]]:
        """BFS to find shortest paths."""
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        paths = []
        
        while queue and len(paths) < 100:  # Limit paths
            current, path = queue.popleft()
            
            if len(path) > max_length:
                continue
                
            if current == end:
                paths.append([self.nodes[nid] for nid in path])
                continue
            
            for neighbor, _ in self.adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def get_subgraph(self, center_nodes: List[str], depth: int = 2) -> Dict:
        """Extract a subgraph centered on given nodes."""
        included_nodes = set(center_nodes)
        included_edges = []
        
        # BFS to find all nodes within depth
        queue = list(center_nodes)
        current_depth = {n: 0 for n in center_nodes}
        
        while queue:
            node = queue.pop(0)
            if current_depth[node] >= depth:
                continue
            
            for neighbor, edge in self.adjacency.get(node, []):
                if neighbor not in included_nodes:
                    included_nodes.add(neighbor)
                    current_depth[neighbor] = current_depth[node] + 1
                    queue.append(neighbor)
                included_edges.append(edge)
        
        # Convert to dict format
        return {
            "nodes": {nid: self.nodes[nid].to_dict() for nid in included_nodes if nid in self.nodes},
            "edges": [e.to_dict() for e in included_edges]
        }
    
    def to_dict(self) -> Dict:
        """Export graph as dictionary."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges]
        }
    
    def save(self, filepath: str):
        """Save graph to JSON file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeGraph':
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        graph = cls()
        for nid, node_data in data.get("nodes", {}).items():
            graph.add_node(Node(**node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(Edge(**edge_data))
        return graph


def create_simplified_fab_graph() -> KnowledgeGraph:
    """
    Create a simplified fab knowledge graph for demonstration.
    
    Represents a typical semiconductor process line.
    """
    graph = KnowledgeGraph()
    
    # ===== EQUIPMENT =====
    equipment = [
        ("CVD_1", "CVD Chamber 1", "Chemical Vapor Deposition chamber"),
        ("CVD_2", "CVD Chamber 2", "Chemical Vapor Deposition chamber"),
        ("ETCH_1", "Etch Chamber 1", "Plasma etch chamber"),
        ("LITHO_1", "Lithography Station 1", "Coating and exposure"),
        ("CMP_1", "CMP Station 1", "Chemical Mechanical Polishing"),
        ("CLEAN_1", "Clean Station 1", "Wafer cleaning"),
        ("ROBOT_1", " wafer Robot", "Wafer handling robot"),
        ("LOAD_PORT_1", "Load Port 1", "Wafer input/output"),
    ]
    
    for eid, name, desc in equipment:
        graph.add_node(Node(
            id=eid,
            type="equipment",
            name=name,
            description=desc
        ))
    
    # ===== SENSORS =====
    sensors = [
        ("PRESSURE_1", "Pressure Sensor 1", "Chamber pressure", {"equipment": "CVD_1"}),
        ("TEMP_1", "Temperature Sensor 1", "Chuck temperature", {"equipment": "CVD_1"}),
        ("PRESSURE_2", "Pressure Sensor 2", "Chamber pressure", {"equipment": "CVD_2"}),
        ("TEMP_2", "Temperature Sensor 2", "Chuck temperature", {"equipment": "CVD_2"}),
        ("PRESSURE_E", "Pressure Sensor E", "Chamber pressure", {"equipment": "ETCH_1"}),
        ("TEMP_E", "Temperature Sensor E", "Plasma temperature", {"equipment": "ETCH_1"}),
        ("FLOW_1", "Gas Flow Sensor 1", "Mass flow controller", {"equipment": "CVD_1"}),
        ("FLOW_2", "Gas Flow Sensor 2", "Mass flow controller", {"equipment": "CVD_2"}),
        ("PARTICLE_1", "Particle Sensor 1", "In-situ particle monitor", {"equipment": "CVD_1"}),
        ("PARTICLE_CLEAN", "Particle Sensor C", "In-situ particle monitor", {"equipment": "CLEAN_1"}),
    ]
    
    for sid, name, desc, props in sensors:
        graph.add_node(Node(
            id=sid,
            type="sensor",
            name=name,
            description=desc,
            properties=props
        ))
    
    # ===== PARAMETERS =====
    parameters = [
        ("CHAMBER_PRESSURE", "Chamber Pressure", "Process pressure setpoint", {"unit": "Torr"}),
        ("CHUCK_TEMP", "Chuck Temperature", "Wafer chuck temperature", {"unit": "C"}),
        ("GAS_FLOW_RATE", "Gas Flow Rate", "Total gas flow", {"unit": "sccm"}),
        ("RF_POWER", "RF Power", "Plasma RF power", {"unit": "W"}),
        ("DEP_TIME", "Deposition Time", "Film deposition time", {"unit": "s"}),
        ("CLEAN_TIME", "Clean Time", "Cleaning duration", {"unit": "s"}),
    ]
    
    for pid, name, desc, props in parameters:
        graph.add_node(Node(
            id=pid,
            type="parameter",
            name=name,
            description=desc,
            properties=props
        ))
    
    # ===== DEFECT PATTERNS =====
    defects = [
        ("CENTER", "Center Defect", "Circular defects in wafer center"),
        ("DONUT", "Donut Defect", "Ring-shaped defects in middle region"),
        ("EDGE_RING", "Edge-Ring Defect", "Ring at wafer edge"),
        ("EDGE_LOC", "Edge-Loc Defect", "Localized defects at edge"),
        ("LOCAL", "Local Defect", "Scattered local defects"),
        ("RANDOM", "Random Defect", "Randomly distributed defects"),
        ("SCRATCH", "Scratch Defect", "Linear scratch patterns"),
        ("NEAR_FULL", "Near-Full Defect", "Almost entire wafer affected"),
    ]
    
    for did, name, desc in defects:
        graph.add_node(Node(
            id=did,
            type="defect",
            name=name,
            description=desc
        ))
    
    # ===== ROOT CAUSES =====
    root_causes = [
        ("TEMP_GRADIENT", "Temperature Gradient", "Non-uniform temperature across wafer"),
        ("PRESSURE_VARIANCE", "Pressure Variance", "Chamber pressure instability"),
        ("PARTICLE_CONTAM", "Particle Contamination", "Foreign particle introduction"),
        ("EDGE_BEAD", "Edge Bead Issue", "Problems at wafer edge"),
        ("CLAMP_MARKS", "Clamp Marks", "Wafer clamping issues"),
        ("HANDLING_DAMAGE", "Handling Damage", "Mechanical damage during transport"),
        ("CHEMICAL_PURITY", "Chemical Purity", "Impurities in process chemicals"),
        ("UNIFORMITY", "Process Uniformity", "Non-uniform process conditions"),
    ]
    
    for rid, name, desc in root_causes:
        graph.add_node(Node(
            id=rid,
            type="root_cause",
            name=name,
            description=desc
        ))
    
    # ===== EDGES: Equipment → Sensors =====
    graph.add_edge(Edge("CVD_1", "PRESSURE_1", "has_sensor"))
    graph.add_edge(Edge("CVD_1", "TEMP_1", "has_sensor"))
    graph.add_edge(Edge("CVD_1", "FLOW_1", "has_sensor"))
    graph.add_edge(Edge("CVD_1", "PARTICLE_1", "has_sensor"))
    graph.add_edge(Edge("CVD_2", "PRESSURE_2", "has_sensor"))
    graph.add_edge(Edge("CVD_2", "TEMP_2", "has_sensor"))
    graph.add_edge(Edge("CVD_2", "FLOW_2", "has_sensor"))
    graph.add_edge(Edge("ETCH_1", "PRESSURE_E", "has_sensor"))
    graph.add_edge(Edge("ETCH_1", "TEMP_E", "has_sensor"))
    graph.add_edge(Edge("CLEAN_1", "PARTICLE_CLEAN", "has_sensor"))
    
    # ===== EDGES: Sensors → Parameters =====
    graph.add_edge(Edge("PRESSURE_1", "CHAMBER_PRESSURE", "measures"))
    graph.add_edge(Edge("TEMP_1", "CHUCK_TEMP", "measures"))
    graph.add_edge(Edge("FLOW_1", "GAS_FLOW_RATE", "measures"))
    graph.add_edge(Edge("PRESSURE_2", "CHAMBER_PRESSURE", "measures"))
    graph.add_edge(Edge("TEMP_2", "CHUCK_TEMP", "measures"))
    graph.add_edge(Edge("FLOW_2", "GAS_FLOW_RATE", "measures"))
    graph.add_edge(Edge("PRESSURE_E", "CHAMBER_PRESSURE", "measures"))
    graph.add_edge(Edge("TEMP_E", "CHUCK_TEMP", "measures"))
    
    # ===== EDGES: Equipment → Parameters (controls) =====
    graph.add_edge(Edge("CVD_1", "CHAMBER_PRESSURE", "controls"))
    graph.add_edge(Edge("CVD_1", "CHUCK_TEMP", "controls"))
    graph.add_edge(Edge("CVD_1", "GAS_FLOW_RATE", "controls"))
    graph.add_edge(Edge("CVD_1", "DEP_TIME", "controls"))
    graph.add_edge(Edge("ETCH_1", "RF_POWER", "controls"))
    graph.add_edge(Edge("CLEAN_1", "CLEAN_TIME", "controls"))
    
    # ===== EDGES: Root Causes → Parameters (affects) =====
    graph.add_edge(Edge("TEMP_GRADIENT", "CHUCK_TEMP", "increases"))
    graph.add_edge(Edge("PRESSURE_VARIANCE", "CHAMBER_PRESSURE", "increases_variance"))
    graph.add_edge(Edge("PARTICLE_CONTAM", "PARTICLE_1", "increases"))
    graph.add_edge(Edge("PARTICLE_CONTAM", "PARTICLE_CLEAN", "increases"))
    graph.add_edge(Edge("CHEMICAL_PURITY", "PARTICLE_1", "increases"))
    graph.add_edge(Edge("UNIFORMITY", "CHAMBER_PRESSURE", "decreases"))
    graph.add_edge(Edge("UNIFORMITY", "CHUCK_TEMP", "decreases"))
    
    # ===== EDGES: Defects → Equipment (associated_with) =====
    graph.add_edge(Edge("CENTER", "CVD_1", "associated_with"))
    graph.add_edge(Edge("CENTER", "CVD_2", "associated_with"))
    graph.add_edge(Edge("DONUT", "LITHO_1", "associated_with"))
    graph.add_edge(Edge("EDGE_RING", "CVD_1", "associated_with"))
    graph.add_edge(Edge("EDGE_RING", "ETCH_1", "associated_with"))
    graph.add_edge(Edge("EDGE_LOC", "LOAD_PORT_1", "associated_with"))
    graph.add_edge(Edge("SCRATCH", "CMP_1", "associated_with"))
    graph.add_edge(Edge("SCRATCH", "ROBOT_1", "associated_with"))
    graph.add_edge(Edge("LOCAL", "CLEAN_1", "associated_with"))
    graph.add_edge(Edge("RANDOM", "CLEAN_1", "associated_with"))
    graph.add_edge(Edge("RANDOM", "CVD_1", "associated_with"))
    
    # ===== EDGES: Root Causes → Defects (causes) =====
    graph.add_edge(Edge("TEMP_GRADIENT", "CENTER", "causes"))
    graph.add_edge(Edge("PRESSURE_VARIANCE", "EDGE_RING", "causes"))
    graph.add_edge(Edge("PRESSURE_VARIANCE", "DONUT", "causes"))
    graph.add_edge(Edge("PARTICLE_CONTAM", "LOCAL", "causes"))
    graph.add_edge(Edge("PARTICLE_CONTAM", "RANDOM", "causes"))
    graph.add_edge(Edge("EDGE_BEAD", "EDGE_RING", "causes"))
    graph.add_edge(Edge("CLAMP_MARKS", "EDGE_LOC", "causes"))
    graph.add_edge(Edge("HANDLING_DAMAGE", "SCRATCH", "causes"))
    graph.add_edge(Edge("CHEMICAL_PURITY", "LOCAL", "causes"))
    
    return graph


# Example usage
if __name__ == "__main__":
    # Create simplified fab graph
    graph = create_simplified_fab_graph()
    
    print("=== Knowledge Graph Demo ===\n")
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")
    
    # List nodes by type
    print("\n=== Nodes by Type ===")
    for ntype in ["equipment", "sensor", "parameter", "defect", "root_cause"]:
        nodes = graph.get_nodes_by_type(ntype)
        print(f"\n{ntype.upper()} ({len(nodes)}):")
        for n in nodes[:5]:
            print(f"  - {n.id}: {n.name}")
    
    # Example: Get subgraph for Edge-Ring defect
    print("\n=== Subgraph for Edge-Ring Defect ===")
    subgraph = graph.get_subgraph(["EDGE_RING"], depth=2)
    print(f"Nodes in subgraph: {len(subgraph['nodes'])}")
    print(f"Edges in subgraph: {len(subgraph['edges'])}")
    
    # Find paths from defect to root cause
    print("\n=== Paths: Edge-Ring → Root Cause ===")
    paths = graph.find_path_between("defect", "root_cause", max_length=4)
    print(f"Found {len(paths)} paths")
    for path in paths[:3]:
        print(" → ".join([n.name for n in path]))
    
    # Save graph
    graph.save("data/processed/kg/fab_graph.json")
    print("\nGraph saved to data/processed/kg/fab_graph.json")
