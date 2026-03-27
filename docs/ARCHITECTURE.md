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
    │  │  Output:             │    │  ✗ Hallucination: -2.0 penalty!    │   │
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
    │  │   Two-Stage Approach:                                               │  │
    │  │   ┌─────────────────────────┐  ┌────────────────────────────┐   │  │
    │  │   │  STAGE 1: CLIP          │  │  STAGE 2: Stable Diffusion  │   │  │
    │  │   │  Alignment Training    │  │  Diagram Generation        │   │  │
    │  │   │                         │  │                             │   │  │
    │  │   │  clip_diagram_model.py │  │  sd_diagram_model.py       │   │  │
    │  │   │  - Text encoder        │  │  - CLIP for semantic guide  │   │  │
    │  │   │  - Image encoder      │  │  - SD 2.1 pipeline         │   │  │
    │  │   │  - Contrastive loss   │  │  - Prompt enhancement      │   │  │
    │  │   └─────────────────────────┘  └────────────────────────────┘   │  │
    │  │                                                                     │  │
    │  │   Text Prompt: "CVD chamber cross-section with gas flow"            │  │
    │  │                          │                                          │  │
    │  │                          ▼                                          │  │
    │  │   ┌─────────────────────────────────────────────────────────┐     │  │
    │  │   │  Stable Diffusion 2.1 + Fine-tuned CLIP Guidance          │     │  │
    │  │   │  - Encoder: CLIP ViT-B/32 (frozen or fine-tuned)        │     │  │
    │  │   │  - Decoder: Latent diffusion UNet                        │     │  │
    │  │   │  - Output: 512x512 for PPT                               │     │  │
    │  │   └─────────────────────────────────────────────────────────┘     │  │
    │  │                          │                                          │  │
    │  │                          ▼                                          │  │
    │  │   ┌─────────────────────────────────────────────────────────┐     │  │
    │  │   │  "Fake 3D" Cross-Section Illustration                  │     │  │
    │  - │   │  CVD chamber with gas flow arrows                     │     │  │
    │  │   │  with ion bombardment                     - Etch chamber │     │  │
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

### 4. Vision Models (`src/vision/`)

#### 4.1 CLIP Alignment Model (`clip_diagram_model.py`)

```
CLIPDiagramModel (Stage 1: Training)
├── Model: openai/clip-vit-base-patch32 (512-dim embeddings)
│   ├── CLIPTextEncoder: Text → 512-dim embedding
│   └── CLIPImageEncoder: Image → 512-dim embedding
│
├── Training:
│   ├── Dataset: images + .txt captions
│   ├── Loss: Contrastive loss (symmetric)
│   └── Output: Fine-tuned CLIP for diagram alignment
│
└── Usage:
    python scripts/train_diagram_clip.py
    → saves to models/clip_diagram/best_model.pt
```

#### 4.2 Stable Diffusion Generator (`sd_diagram_model.py`)

```
SDDiagramGenerator (Stage 2: Generation)
├── Model: stabilityai/stable-diffusion-2-1-base
│   ├── CLIPTextEncoder: Semantic guidance
│   ├── UNet: Latent diffusion (860M params)
│   └── VAE: Image encoding/decoding
│
├── Features:
│   ├── Loads fine-tuned CLIP for better alignment
│   ├── Prompt enhancement for technical diagrams
│   ├── Image-to-image variations
│   └── Batch generation (4+ images)
│
└── Usage:
    from src.vision.sd_diagram_model import load_trained_generator
    generator = load_trained_generator("models/clip_diagram/best_model.pt")
    images = generator.generate("CMOS transistor cross-section")
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
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  VISION TRAINING PIPELINE                                            │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────┐    ┌────────────────────────────────┐   │  │
│  │  │  STAGE 1: CLIP          │    │  STAGE 2: Stable Diffusion    │   │  │
│  │  │  Alignment             │    │  Generation                    │   │  │
│  │  │                         │    │                                │   │  │
│  │  │  Train on diagram       │───▶│  Use trained CLIP as          │   │  │
│  │  │  image-caption pairs    │    │  semantic guide for SD        │   │  │
│  │  │                         │    │                                │   │  │
│  │  │  Output:                │    │  Output:                      │   │  │
│  │  │  clip_diagram_model.py │    │  sd_diagram_model.py          │   │  │
│  │  │  models/clip_diagram/ │    │  Generated diagrams            │   │  │
│  │  └─────────────────────────┘    └────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Data Sources (All Synthetic - No Real Fab Data!):                         │
│  - Defect patterns from WM-811K (public dataset)                           │
│  - Sensor anomalies from SECOM (public dataset)                           │
│  - Graph structure based on typical fab topology                          │
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
│   │   ├── clip_diagram_model.py    # CLIP alignment (Stage 1)
│   │   └── sd_diagram_model.py     # SD generation (Stage 2)
│   │
│   └── models/                  # ML Models
│       └── ...
│
├── scripts/
│   ├── train_graph_llm.py       # Main training pipeline
│   ├── train_llm.py             # Text-based LLM training
│   ├── train_diagram_clip.py    # CLIP training (Stage 1)
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
│   │       ├── diagrams/train/   # CLIP training images
│   │       └── diagrams/val/    # CLIP validation images
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

### 4. Two-Stage Process Illustration Generation

The vision module uses a two-stage approach:
- **Stage 1**: Fine-tune CLIP on diagram-caption pairs for better alignment
- **Stage 2**: Use fine-tuned CLIP as semantic guide for Stable Diffusion
- This ensures generated diagrams match technical descriptions accurately

---

## Deployment

For production deployment:

```bash
# Train LLM on GPU (requires CUDA)
python scripts/train_graph_llm.py --phase all --num_examples 1000

# Stage 1: Train CLIP for diagram alignment
python scripts/train_diagram_clip.py
# Output: models/clip_diagram/best_model.pt

# Stage 2: Generate process illustrations
python -c "from src.vision.sd_diagram_model import load_trained_generator; \
  gen = load_trained_generator('models/clip_diagram/best_model_pt'); \
  imgs = gen.generate('CVD chamber cross-section'); \
  gen.save_image(imgs[0], 'output/cvd.png')"

# The fine-tuned model will be in:
# models/graph_grpo/
```

The model can then be deployed as a FastAPI service for real-time RCA queries.
