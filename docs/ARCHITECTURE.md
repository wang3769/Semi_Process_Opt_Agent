# Graph-Based RCA LLM Architecture

## Overview

This is an Agentic Graph-Retrieval system for Semiconductor Root Cause Analysis (RCA). It uses a novel "Cold Start SFT + GRPO on Paths" training strategy to create an LLM that performs RCA by walking a knowledge graph - never hallucinating equipment or sensors.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     AUTONOMOUS EXPERIMENT OPTIMIZATION COPILOT                       │
│                              (Graph-Based RCA LLM)                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                              INPUT LAYER                                     │
    │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐  │
    │  │ WM-811K         │  │ SECOM            │  │ User Query              │  │
    │  │ (Wafer Maps)    │  │ (Sensor Data)    │  │ "Why did yield drop?"   │  │
    │  │                 │  │                  │  │                         │  │
    │  │ CNN Defect      │  │ XGBoost Anomaly │  │ → RAG Retrieval         │  │
    │  │ Classification  │  │ Detection        │  │ → Knowledge Graph Query │  │
    │  └────────┬────────┘  └────────┬─────────┘  └───────────┬─────────────┘  │
    └───────────┼─────────────────────┼───────────────────────┼─────────────────┘
                │                     │                       │
                ▼                     ▼                       ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         DATA PROCESSING LAYER                               │
    │  ┌──────────────────────────────────────────────────────────────────────┐  │
    │  │                     RAG Knowledge Base                                │  │
    │  │  - Document chunks from fab manuals, yield reports                   │  │
    │  │  - Vector embeddings (BAAI/bge-small-en-v1.5)                        │  │
    │  │  - Retrieved context for user queries                                │  │
    │  └──────────────────────────────────────────────────────────────────────┘  │
    │                                    │                                         │
    │                                    ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────────┐  │
    │  │                   KNOWLEDGE GRAPH                                    │  │
    │  │                                                                       │  │
    │  │   Equipment    Sensors    Parameters    Defects    Root Causes       │  │
    │  │   ──────────  ────────  ──────────   ────────   ────────────     │  │
    │  │   CVD_1    ──►Pressure ──►Torr      CENTER  ──►TEMP_GRADIENT     │  │
    │  │   ETCH_1   ──►Temp     ──►°C        EDGE_RING►PRESSURE_VARIANCE  │  │
    │  │   CLEAN_1  ──►Particles►►count      SCRATCH  ►HANDLING_DAMAGE     │  │
    │  │                                                                       │  │
    │  │   Relations: has_sensor, controls, causes, associated_with          │  │
    │  └──────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         LLM TRAINING LAYER                                │
    │                                                                       │
    │  ┌─────────────────────┐    ┌─────────────────────────────────────┐   │
    │  │  COLD START SFT      │    │  GRPO WITH CAUSAL REWARDS          │   │
    │  │  "Map-Reader"        │    │  "Path Walker"                     │   │
    │  │                     │    │                                     │   │
    │  │  Input:              │    │  Rewards:                           │   │
    │  │  - Defect Pattern   │    │  ✓ Correctness: Found root cause?   │   │
    │  │  - Subgraph         │    │  ✓ Causal Link: Valid graph path?   │   │
    │  │                     │    │  ✓ Efficiency: Fewest steps?         │   │
    │  │  Output:             │    │  ✗ Hallucination: -2.0 penalty!     │   │
    │  │  - Trajectory walk  │    │                                     │   │
    │  │  - Reasoning chain  │    │  This ensures the model:             │   │
    │  │                     │    │  - NEVER hallucinates tools          │   │
    │  │  Teaches:           │    │  - STAYS faithful to graph          │   │
    │  │  "How to read map"  │    │  - Follows valid paths              │   │
    │  └─────────────────────┘    └─────────────────────────────────────┘   │
    │                                                                       │
    │  Training Data: graph_sft_train.jsonl + graph_grpo_train.jsonl         │
    │  Model: meta-llama/Llama-3.1-8B-Instruct (or Qwen2.5-7B)              │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         AGENT ORCHESTRATION LAYER                          │
    │                                                                       │
    │                    ┌───────────────────────────────┐                      │
    │                    │     LangGraph Workflow       │                      │
    │                    │                               │                      │
    │                    │  ┌─────────┐                 │                      │
    │                    │  │Analyze  │──────┐          │                      │
    │                    │  │Defect   │      │          │                      │
    │                    │  └─────────┘      ▼          │                      │
    │                    │  ┌─────────┐ ┌─────────┐     │                      │
    │                    │  │Query    │►│Retrieve │     │                      │
    │                    │  │Graph    │ │Context  │     │                      │
    │                    │  └─────────┘ └─────────┘     │                      │
    │                    │       │          │          │                      │
    │                    │       ▼          ▼          │                      │
    │                    │  ┌─────────────────────┐     │                      │
    │                    │  │Generate RCA Report │     │                      │
    │                    │  │with Graph Reasoning│     │                      │
    │                    │  └─────────────────────┘     │                      │
    │                    │              │                │                      │
    │                    │              ▼                │                      │
    │                    │  ┌─────────────────────┐     │                      │
    │                    │  │  Structured Output │     │                      │
    │                    │  │  - Root Cause      │     │                      │
    │                    │  │  - Evidence        │     │                      │
    │                    │  │  - Actions         │     │                      │
    │                    │  └─────────────────────┘     │                      │
    │                    └───────────────────────────────┘                      │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         VISION OUTPUT LAYER (Optional)                      │
    │                                                                       │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │              PROCESS ILLUSTRATION GENERATOR                       │  │
    │  │                                                                     │  │
    │  │   Text Prompt: "CVD chamber cross-section with gas flow"            │  │
    │  │                          │                                          │  │
    │  │                          ▼                                          │  │
    │  │   ┌─────────────────────────────────────────────────────────┐     │  │
    │  │   │  Stable Diffusion 2.1 + Optional LoRA Fine-tuning        │     │  │
    │  │   │  - Encoder: CLIP text encoder                           │     │  │
    │  │   │  - Decoder: Latent diffusion UNet                       │     │  │
    │  │   │  - Output: 512x512 or 768x512 for PPT                   │     │  │
    │  │   └─────────────────────────────────────────────────────────┘     │  │
    │  │                          │                                          │  │
    │  │                          ▼                                          │  │
    │  │   ┌─────────────────────────────────────────────────────────┐     │  │
    │  │   │  "Fake 3D" Cross-Section Illustration                  │     │  │
    │  - │   │  CVD chamber with gas flow arrows                     │     │  │
    │  │   │ with ion bombardment                     - Etch chamber │     │  │   │  │
    │  - Lithography exposure diagram                          │     │  │
    │  │   └─────────────────────────────────────────────────────────┘     │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Knowledge Graph (`src/kg/graph_builder.py`)

```
KnowledgeGraph
├── Nodes (40 total)
│   ├── Equipment (8): CVD_1, CVD_2, ETCH_1, LITHO_1, CMP_1, CLEAN_1, ROBOT_1, LOAD_PORT_1
│   ├── Sensors (10): Pressure_1, Temp_1, Pressure_2, Temp_2, Pressure_E, Temp_E, Flow_1, Flow_2, Particle_1, Particle_Clean
│   ├── Parameters (6): CHAMBER_PRESSURE, CHUCK_TEMP, GAS_FLOW_RATE, RF_POWER, DEP_TIME, CLEAN_TIME
│   ├── Defects (8): CENTER, DONUT, EDGE_RING, EDGE_LOC, LOCAL, RANDOM, SCRATCH, NEAR_FULL
│   └── Root Causes (8): TEMP_GRADIENT, PRESSURE_VARIANCE, PARTICLE_CONTAM, EDGE_BEAD, CLAMP_MARKS, HANDLING_DAMAGE, CHEMICAL_PURITY, UNIFORMITY
│
└── Edges (51 total)
    ├── Equipment → Sensors: has_sensor
    ├── Sensors → Parameters: measures
    ├── Equipment → Parameters: controls
    ├── Root Causes → Parameters: affects (increases/decreases)
    ├── Defects → Equipment: associated_with
    └── Root Causes → Defects: causes
```

### 2. Trajectory Generator (`src/kg/trajectory.py`)

```
TrajectoryGenerator
├── generate_case(defect_pattern, true_root_cause) → RCACase
│   ├── Generates sensor readings based on defect type
│   ├── Finds valid graph path from defect to root cause
│   ├── Builds step-by-step trajectory:
│   │   1. OBSERVE: Identify defect pattern
│   │   2. QUERY: Find related equipment
│   │   3. OBSERVE: Check sensor readings
│   │   4. HYPOTHESIZE: Form root cause hypothesis
│   │   5. VERIFY: Suggest metrology tests
│   └── Returns: RCACase with full reasoning chain
│
└── Output formats:
    ├── SFT format: instruction + context + response
    └── GRPO format: prompt + ground_truth + rewards
```

### 3. GRPO Causal Rewards (`src/llm/grpo_trainer.py`)

```
CausalRewardCalculator
├── calculate_total_reward(response, ground_truth) → (reward, breakdown)
│   │
│   ├── Correctness Reward: Did it find true root cause?
│   │   └── +1.0 if correct, 0.0 if wrong, partial for related
│   │
│   ├── Causal Link Reward: Did it follow valid graph paths?
│   │   ├── +1.0 if all mentioned nodes in graph
│   │   └── -2.0 if hallucinations detected (KEY INNOVATION!)
│   │
│   └── Efficiency Reward: Fewest reasoning steps?
│       └── +1.0 for 3-5 steps, decreases for more
│
└── Hallucination Detection:
    - Checks if mentioned equipment/sensors exist in graph
    - Penalizes: laser, magnetic_coil, sensor_99, etc.
    - This is the key safety feature for fab deployment
```

### 4. Process Image Generator (`src/vision/image_generator.py`)

```
ProcessImageGenerator
├── Model: Stable Diffusion 2.1-base
│   ├── Text Encoder: CLIP ViT-L/14
│   ├── UNet: 860M parameters
│   └── VAE: Latent space encoding
│
├── Features:
│   ├── Text-to-image generation
│   ├── Image-to-image variations
│   ├── Optional LoRA fine-tuning
│   └── VRAM optimized (attention slicing, VAE slicing)
│
├── Example Prompts:
│   ├── "cvd_chamber": CVD chamber cross-section with gas flow
│   ├── "etch_chamber": Plasma etching with ion bombardment
│   ├── "litho_exposure": Lithography UV exposure diagram
│   └── "cmp_process": Chemical mechanical polishing schematic
│
└── Output:
    ├── 512x512 or 768x512 for PPT
    ├── PNG format with transparency support
    └── Batch generation (4 variations)
```

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                    │
│                                                                             │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────────────┐    │
│  │  SYNTHETIC    │    │   COLD START   │    │   GRPO FINE-TUNING    │    │
│  │  DATA GEN     │───▶│   SFT          │───▶│   WITH CAUSAL REWARDS │    │
│  │               │    │               │    │                        │    │
│  │  Graph        │    │  "Map-Reader" │    │  "Path Walker"        │    │
│  │  Trajectories │    │  100 examples │    │  100 examples        │    │
│  │               │    │               │    │                        │    │
│  │  ↓            │    │  ↓            │    │  ↓                    │    │
│  │  SFT JSONL    │    │  Fine-tuned   │    │  Fine-tuned +         │    │
│  │  GRPO JSONL   │    │  Model v1     │    │  Graph-constrained    │    │
│  └────────────────┘    └───────────────┘    └────────────────────────┘    │
│                                                                             │
│  Data Sources (All Synthetic - No Real Fab Data!):                         │
│  - Defect patterns from WM-811K (public dataset)                          │
│  - Sensor anomalies from SECOM (public dataset)                            │
│  - Graph structure based on typical fab topology                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
Semi_Process_Opt_Agent/
├── src/
│   ├── kg/                      # Knowledge Graph
│   │   ├── __init__.py
│   │   ├── graph_builder.py     # KnowledgeGraph class
│   │   └── trajectory.py        # Trajectory generation
│   │
│   ├── llm/                     # LLM Training
│   │   ├── __init__.py
│   │   ├── config.py           # LLM configuration
│   │   ├── dataset.py          # Text-based SFT/DPO datasets
│   │   ├── sft_trainer.py      # SFT training
│   │   ├── dpo_trainer.py      # DPO training  
│   │   └── grpo_trainer.py     # GRPO with causal rewards
│   │
│   ├── rag/                     # RAG Knowledge Base
│   │   └── ...
│   │
│   ├── vision/                  # Vision/Image Generation
│   │   ├── __init__.py
│   │   └── image_generator.py   # Stable Diffusion for process illustrations
│   │
│   └── models/                  # ML Models
│       └── ...
│
├── scripts/
│   ├── train_graph_llm.py       # Main training pipeline
│   ├── train_llm.py            # Text-based LLM training
│   └── ...
│
├── data/
│   ├── processed/
│   │   ├── kg/
│   │   │   └── fab_graph.json  # Knowledge graph
│   │   ├── llm/
│   │   │   ├── graph_sft_train.jsonl    # SFT training data
│   │   │   └── graph_grpo_train.jsonl   # GRPO training data
│   │   └── vision/
│   │       └── lora_data/      # LoRA fine-tuning prompts
│   └── ...
│
├── docs/
│   ├── README.md               # Setup & Training Guide
│   └── ARCHITECTURE.md         # This file
│
└── requirements.txt
```

---

## Key Innovations

### 1. Graph-Constrained Reasoning

Unlike general-purpose LLMs that can hallucinate any equipment, this system:
- Only uses nodes from the knowledge graph
- Penalizes (-2.0) any mention of non-existent tools
- Rewards valid graph path following (+1.5)

### 2. Synthetic Data Generation

All training data is synthetic, making this safe for:
- IP-sensitive environments
- Testing architecture without real fab data
- Rapid prototyping

### 3. Causal Rewards

The GRPO reward structure explicitly rewards:
- Finding the correct root cause
- Following valid causal chains in the graph
- Efficiency (fewer steps = better)
- And penalizes hallucinations heavily

### 4. Process Illustration Generation

The vision module provides:
- Text-to-image for PPT presentations
- No precision requirement - "fake 3D" style
- Custom prompts for semiconductor processes
- Optional LoRA fine-tuning for custom styles

---

## Deployment

For production deployment:

```bash
# Train on GPU (requires CUDA)
python scripts/train_graph_llm.py --phase all --num_examples 1000

# Generate process illustrations
python -c "from src.vision.image_generator import ProcessImageGenerator; \
  gen = ProcessImageGenerator(); \
  img = gen.generate('CVD chamber cross-section'); \
  gen.save_image(img[0], 'output/cvd.png')"

# The fine-tuned model will be in:
# models/graph_grpo/
```

The model can then be deployed as a FastAPI service for real-time RCA queries.
