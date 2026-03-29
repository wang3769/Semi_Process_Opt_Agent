# Graph-Based RCA LLM Architecture

## Overview

This is an Agentic Graph-Retrieval system for Semiconductor Root Cause Analysis (RCA). It uses a novel "Cold Start SFT + GRPO on Paths" training strategy to create an LLM that performs RCA by walking a knowledge graph.

## Vision Pipeline: Two-Stage Diagram Generation

The vision module uses a two-stage approach for generating process diagrams:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         VISION OUTPUT LAYER                                        │
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │              TWO-STAGE PROCESS ILLUSTRATION GENERATOR                        │ │
│  │                                                                            │ │
│  │  ┌───────────────────────────┐      ┌──────────────────────────────────┐  │ │
│  │  │  STAGE 1: CLIP            │      │  STAGE 2: Stable Diffusion       │  │ │
│  │  │  Alignment Training      │      │  Diagram Generation              │  │ │
│  │  │                          │      │                                  │  │ │
│  │  │  File:                  │      │  File:                           │  │ │
│  │  │  clip_diagram_model.py  │ ──▶  │  sd_diagram_model.py            │  │ │
│  │  │                          │      │  scripts/generate_diagrams.py    │  │ │
│  │  │                          │      │                                  │  │ │
│  │  │  Purpose:               │      │  Purpose:                        │  │ │
│  │  │  - Learn text-image     │      │  - Generate new diagrams        │  │ │
│  │  │    similarity           │      │    from text prompts            │  │ │
│  │  │  - Project embeddings   │      │  - Uses CLIP for semantic       │  │ │
│  │  │    to common space      │      │    guidance                     │  │ │
│  │  │                          │      │  - Stable Diffusion 2.1-base   │  │ │
│  │  │  Training:              │      │                                  │  │ │
│  │  │  train_diagram_clip.py  │      │  Usage:                         │  │ │
│  │  │                          │      │  generate_diagrams.py            │  │ │
│  │  └───────────────────────────┘      └──────────────────────────────────┘  │ │
│  │                                                                            │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: CLIP Alignment (`clip_diagram_model.py`)

```
Purpose: Learn to align text embeddings with image embeddings

Model: CLIP ViT-Base (openai/clip-vit-base-patch32)
- Text encoder: 512-dim → 512-dim
- Image encoder: 768-dim (vision) → 512-dim (projection)
- Loss: Contrastive loss (symmetric)

Training:
    python scripts/train_diagram_clip.py
    Output: models/clip_diagram/best_model.pt

What it learns:
- Which text descriptions match which diagrams
- Semantic similarity between text and images
```

### Stage 2: Stable Diffusion Generation (`sd_diagram_model.py`)

```
Purpose: Generate NEW diagrams from text descriptions

Model: Stable Diffusion 2.1-base
- CLIP text encoder for semantic guidance
- UNet for latent diffusion
- VAE for image encoding/decoding
- Can load fine-tuned CLIP for better alignment

Generation:
    # CLI
    python scripts/generate_diagrams.py --prompt "CMOS transistor cross-section"
    
    # Python
    from src.vision.sd_diagram_model import load_trained_generator
    generator = load_trained_generator("models/clip_diagram/best_model.pt")
    images = generator.generate("CVD chamber with gas flow", num_images=4)
```

## Complete Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     AUTONOMOUS EXPERIMENT OPTIMIZATION COPILOT                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                              INPUT LAYER                                     │
    │  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐  │
    │  │ WM-811K         │  │ SECOM            │  │ User Query              │  │
    │  │ (Wafer Maps)    │  │ (Sensor Data)    │  │ "Why did yield drop?"   │  │
    │  └────────┬────────┘  └────────┬─────────┘  └───────────┬─────────────┘  │
    └───────────┼─────────────────────┼───────────────────────┼─────────────────┘
                │                     │                       │
                ▼                     ▼                       ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         DATA PROCESSING LAYER                               │
    │  ┌──────────────────────────────────────────────────────────────────────┐  │
    │  │                     RAG Knowledge Base                                │  │
    │  │  - Document chunks from fab manuals, yield reports                   │  │
    │  │  - Vector embeddings (BAAI/bge-small-en-v1.5)                       │  │
    │  └──────────────────────────────────────────────────────────────────────┘  │
    │                                    │                                         │
    │                                    ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────────┐  │
    │  │                   KNOWLEDGE GRAPH                                    │  │
    │  │   Equipment ──► Sensors ──► Parameters ──► Defects ──► Root Causes  │  │
    │  └──────────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         LLM TRAINING LAYER                                │
    │  ┌─────────────────────┐    ┌─────────────────────────────────────┐   │
    │  │  COLD START SFT      │    │  GRPO WITH CAUSAL REWARDS          │   │
    │  └─────────────────────┘    └─────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         VISION PIPELINE                                    │
    │  ┌─────────────────────┐    ┌─────────────────────────────────────┐   │
    │  │  CLIP Alignment     │    │  Stable Diffusion Generation        │   │
    │  │  (Stage 1)         │───▶│  (Stage 2)                         │   │
    │  └─────────────────────┘    └─────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
Semi_Process_Opt_Agent/
├── src/
│   ├── vision/
│   │   ├── clip_diagram_model.py    # CLIP alignment (Stage 1)
│   │   └── sd_diagram_model.py     # SD generation (Stage 2)
│   │
│   ├── kg/                      # Knowledge Graph
│   ├── llm/                     # LLM Training
│   └── rag/                     # RAG Knowledge Base
│
├── scripts/
│   ├── train_diagram_clip.py    # Stage 1: Train CLIP
│   └── generate_diagrams.py     # Stage 2: Generate with SD
│
├── data/
│   └── diagrams/
│       ├── train/               # CLIP training images + captions
│       └── val/                 # CLIP validation images
│
└── docs/
    └── ARCHITECTURE.md
```

---

## Key Innovations

### 1. Graph-Constrained Reasoning (LLM)

Unlike general-purpose LLMs that can hallucinate any equipment, this system:
- Only uses nodes from the knowledge graph
- Penalizes (-2.0) any mention of non-existent tools
- Rewards valid graph path following (+1.5)

### 2. Two-Stage Vision Generation

**Stage 1 (CLIP)**: Learn semantic alignment between text and images
- Contrastive learning on diagram-caption pairs
- Projects both modalities to common embedding space

**Stage 2 (SD)**: Generate new diagrams conditioned on text
- Uses CLIP text encoder for semantic guidance
- Optionally loads fine-tuned CLIP for better alignment
- Prompt enhancement for technical diagrams

### 3. Synthetic Data Generation

All training data is synthetic, making this safe for:
- IP-sensitive environments
- Testing architecture without real fab data
- Rapid prototyping

---

## Deployment

### Vision Pipeline

```bash
# Stage 1: Train CLIP
python scripts/train_diagram_clip.py

# Stage 2: Generate diagrams
python scripts/generate_diagrams.py --prompt "CMOS transistor cross-section"
```

### LLM Pipeline (Coming Soon)

```bash
python scripts/train_graph_llm.py --phase all --num_examples 1000
```
