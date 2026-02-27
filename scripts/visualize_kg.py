"""
Knowledge Graph Visualization
============================

This script generates visualizations of the semiconductor fab knowledge graph.

Usage:
    python scripts/visualize_kg.py
"""

import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.kg.graph_builder import create_simplified_fab_graph, KnowledgeGraph


def print_text_visualization(graph: KnowledgeGraph):
    """Print a text-based visualization of the knowledge graph."""
    
    print("=" * 80)
    print("SEMICONDUCTOR FAB KNOWLEDGE GRAPH - TEXT VISUALIZATION")
    print("=" * 80)
    
    # Group nodes by type
    node_types = {
        "equipment": [],
        "sensor": [],
        "parameter": [],
        "defect": [],
        "root_cause": []
    }
    
    for node_id, node in graph.nodes.items():
        if node.type in node_types:
            node_types[node.type].append(node)
    
    # Print nodes by type
    print("\n" + "=" * 80)
    print("NODES BY TYPE")
    print("=" * 80)
    
    for ntype, nodes in node_types.items():
        print(f"\n{ntype.upper()}S ({len(nodes)}):")
        print("-" * 40)
        for node in nodes:
            print(f"  [{node.id:20}] {node.name}")
            if node.description:
                print(f"       └─ {node.description}")
    
    # Print edges
    print("\n" + "=" * 80)
    print("EDGES (RELATIONSHIPS)")
    print("=" * 80)
    
    # Group edges by relation
    relations = {}
    for edge in graph.edges:
        if edge.relation not in relations:
            relations[edge.relation] = []
        relations[edge.relation].append(edge)
    
    for relation, edges in relations.items():
        print(f"\n{relation.upper()} ({len(edges)} edges):")
        print("-" * 40)
        for edge in edges[:10]:  # Show first 10
            source_name = graph.get_node(edge.source).name if graph.get_node(edge.source) else edge.source
            target_name = graph.get_node(edge.target).name if graph.get_node(edge.target) else edge.target
            print(f"  {source_name} → {target_name}")
        if len(edges) > 10:
            print(f"  ... and {len(edges) - 10} more")


def generate_mermaid_diagram(graph: KnowledgeGraph, output_file: str):
    """Generate a Mermaid diagram of the graph."""
    
    mermaid_lines = [
        "```mermaid",
        "flowchart TD",
        "    %% Semiconductor Fab Knowledge Graph",
        ""
    ]
    
    # Define node styles
    mermaid_lines.extend([
        "    %% Node Styles",
        "    classDef equipment fill:#e1f5fe,stroke:#0277bd,stroke-width:2px",
        "    classDef sensor fill:#fff3e0,stroke:#ef6c00,stroke-width:2px",
        "    classDef parameter fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px",
        "    classDef defect fill:#fce4ec,stroke:#c2185b,stroke-width:2px",
        "    classDef root_cause fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px",
        ""
    ])
    
    # Add nodes
    for node_id, node in graph.nodes.items():
        # Truncate long names
        name = node.name[:30] + "..." if len(node.name) > 30 else node.name
        # Escape special characters
        name = name.replace('"', "'").replace('(', '').replace(')', '')
        mermaid_lines.append(f'    {node_id}["{name}"]')
        mermaid_lines.append(f'    class {node_id} {node.type}')
    
    mermaid_lines.append("")
    
    # Add edges
    for edge in graph.edges:
        # Use different arrow styles for different relations
        if edge.relation == "causes":
            arrow = "-->"
        elif edge.relation == "associated_with":
            arrow == "-.->"
        else:
            arrow = "-->"
        
        mermaid_lines.append(f'    {edge.source} {arrow} {edge.target} %% {edge.relation}')
    
    mermaid_lines.append("```")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mermaid_lines))
    
    print(f"\nMermaid diagram saved to: {output_file}")


def generate_graphviz_code(graph: KnowledgeGraph, output_file: str):
    """Generate Graphviz DOT code for the graph."""
    
    dot_lines = [
        "/* Semiconductor Fab Knowledge Graph */",
        "digraph fab_graph {",
        "    rankdir=LR;",
        "    node [shape=box];",
        "",
        "    /* Node definitions */",
    ]
    
    # Define colors for each type
    colors = {
        "equipment": "lightblue",
        "sensor": "orange", 
        "parameter": "lightgreen",
        "defect": "pink",
        "root_cause": "lavender"
    }
    
    # Add nodes
    for node_id, node in graph.nodes.items():
        name = node.name.replace('"', '\\"')
        dot_lines.append(f'    {node_id} [label="{name}" fillcolor={colors.get(node.type, "white")} style=filled];')
    
    dot_lines.append("")
    dot_lines.append("    /* Edges */")
    
    # Add edges
    edge_relations = {}
    for edge in graph.edges:
        if edge.relation not in edge_relations:
            edge_relations[edge.relation] = []
        edge_relations[edge.relation].append(f"    {edge.source} -> {edge.target} [label={edge.relation}];")
    
    for relation, edges in edge_relations.items():
        dot_lines.append(f"    /* {relation} */")
        dot_lines.extend(edges)
        dot_lines.append("")
    
    dot_lines.append("}")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(dot_lines))
    
    print(f"Graphviz DOT code saved to: {output_file}")


def print_sample_paths(graph: KnowledgeGraph):
    """Print sample reasoning paths through the graph."""
    
    print("\n" + "=" * 80)
    print("SAMPLE RCA PATHS (Reasoning Chains)")
    print("=" * 80)
    
    # Find paths from defects to root causes
    paths = graph.find_path_between("defect", "root_cause", max_length=5)
    
    print("\nValid reasoning paths from Defect → Root Cause:")
    print("-" * 60)
    
    for i, path in enumerate(paths[:10], 1):
        path_str = " → ".join([n.name for n in path])
        print(f"{i:2}. {path_str}")
    
    print(f"\nTotal paths found: {len(paths)}")
    
    # Example reasoning for specific defect
    print("\n" + "=" * 80)
    print("EXAMPLE RCA REASONING: Edge-Ring Defect")
    print("=" * 80)
    
    subgraph = graph.get_subgraph(["EDGE_RING"], depth=3)
    
    print("\nRelevant subgraph nodes:")
    for node_id, node_data in subgraph["nodes"].items():
        print(f"  [{node_data['type']:12}] {node_data['name']}")
    
    print("\nReasoning chain:")
    print("  1. OBSERVE: Defect pattern identified as EDGE_RING")
    print("  2. QUERY: Graph shows EDGE_RING → CVD_1, ETCH_1")
    print("  3. CHECK: CVD_1 has PRESSURE_1 sensor")
    print("  4. HYPOTHESIZE: PRESSURE_VARIANCE causes EDGE_RING")
    print("  5. VERIFY: Check chamber pressure calibration")


def main():
    # Create or load graph
    graph_file = "data/processed/kg/fab_graph.json"
    
    if os.path.exists(graph_file):
        print(f"Loading graph from: {graph_file}")
        graph = KnowledgeGraph.load(graph_file)
    else:
        print("Creating new graph...")
        graph = create_simplified_fab_graph()
    
    print(f"\nGraph statistics:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Text visualization
    print_text_visualization(graph)
    
    # 2. Mermaid diagram
    generate_mermaid_diagram(graph, "docs/kg_diagram.md")
    
    # 3. Graphviz DOT
    generate_graphviz_code(graph, "docs/kg_diagram.dot")
    
    # 4. Sample paths
    print_sample_paths(graph)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print("""
Generated files:
  - docs/kg_diagram.md    (Mermaid flowchart)
  - docs/kg_diagram.dot   (Graphviz DOT code)

To view the Mermaid diagram:
  - Open docs/kg_diagram.md in a Markdown viewer
  - Or use https://mermaid.live/ to render

To view the Graphviz diagram:
  - Install Graphviz: https://graphviz.org/
  - Run: dot -Tpng docs/kg_diagram.dot -o kg_diagram.png
""")


if __name__ == "__main__":
    main()
