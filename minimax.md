# Semi_Process_Opt_Agent ⚠️ IMPORTANT: DO NOT RUN COMMANDS DIRECTLY

> **Always provide bash commands for the user to run themselves. Never execute commands directly.**

---

Autonomous Experiment Optimization Copilot for Semiconductor Manufacturing.

## Project Overview

This is an Agentic Graph-Retrieval system for Semiconductor Root Cause Analysis (RCA). It uses a novel "Cold Start SFT + GRPO on Paths" training strategy combined with vision generation for process illustrations.

## Core Components

### 1. Vision Pipeline (Two-Stage Generation)

**Stage 1: CLIP Alignment** (`src/vision/clip_diagram_model.py`)
- Text encoder: 512-dim output
- Image encoder: 768-dim → 512-dim projection
- Contrastive loss for text-image similarity

**Stage 2: Stable Diffusion** (`src/vision/sd_diagram_model.py`)
- `runwayml/stable-diffusion-v1-5` (public model)
- CLIP-guided semantic generation
- Prompt enhancement for technical diagrams

**Scripts:**
- `scripts/train_diagram_clip.py` - Train CLIP model
- `scripts/generate_diagrams.py` - Generate diagrams

### 2. Defect Detection Models (`src/models/`)

**Vision Model** (`vision_model.py`)
- Wafer map defect classification
- WM-811K dataset support
- Transfer learning from pretrained backbones

**Tabular Model** (`tabular_model.py`)
- SECOM sensor data anomaly detection
- Time-series sensor fusion
- Real-time monitoring support

**Training Scripts:**
- `scripts/train_wm811k_model.py`
- `scripts/train_secom_model.py`
- `scripts/download_wm811k.py`, `scripts/download_secom.py`

### 3. Knowledge Graph (`src/kg/`)

**Graph Builder** (`kg/graph_builder.py`)
- Equipment → Sensors → Parameters → Defects → Root Causes
- Node types: Equipment, Chamber, Parameter, Defect, RootCause
- Edge types: measures, causes, affects

**Trajectory Generator** (`kg/trajectory.py`)
- Generate RCA paths for training
- Synthetic graph traversal examples

**Visualization:** `scripts/visualize_kg.py`

### 4. LLM Training (`src/llm/`)

**Training Pipeline:**
1. **Cold Start SFT** (`llm/sft_trainer.py`)
   - Supervised fine-tuning on graph trajectories
   - Teacher forcing for response generation
   
2. **GRPO** (`llm/grpo_trainer.py`)
   - Group relative policy optimization
   - Causal rewards for valid graph paths
   - Penalizes hallucinated equipment names (-2.0)
   - Rewards valid graph traversal (+1.5)

3. **DPO** (`llm/dpo_trainer.py`)
   - Direct preference optimization
   - Contrastive learning on correct/incorrect paths

**Key Classes:**
- `llm/config.py` - Training configuration
- `llm/dataset.py` - Data loading and formatting
- `llm/inference.py` - Model inference and beam search

**Training Scripts:**
- `scripts/train_graph_llm.py` - Full pipeline
- `scripts/train_llm.py` - SFT only
- `scripts/eval_sft.py` - Evaluation

### 5. RAG Knowledge Base (`src/rag/`)

**Pipeline:**
1. Document Loader (`rag/document_loader.py`)
   - PDF, markdown, text file support
   - Fab manuals, yield reports, SOPs

2. Text Splitter (`rag/text_splitter.py`)
   - Chunk-based splitting
   - Overlap for context preservation

3. Embedding (`rag/embedding.py`)
   - `BAAI/bge-small-en-v1.5` for embeddings
   - Semantic search

4. Vector Store (`rag/vector_store.py`)
   - FAISS or ChromaDB backend
   - ANN indexing

5. Retriever (`rag/retriever.py`)
   - Dense retrieval
   - Hybrid search (optional)

**Scripts:**
- `scripts/build_rag.py` - Build knowledge base
- `scripts/scrape_book.py` - Scrape documentation

### 6. Data Pipeline (`src/data/`)

**Data Loader** (`data/loader.py`)
- Unified interface for WM-811K and SECOM
- Batch processing
- Data augmentation

## Project Structure

```
Semi_Process_Opt_Agent/
├── src/
│   ├── vision/
│   │   ├── clip_diagram_model.py   # CLIP alignment
│   │   ├── sd_diagram_model.py    # Stable Diffusion
│   │   └── image_generator.py     # Generation utilities
│   ├── models/
│   │   ├── vision_model.py        # Wafer map detection
│   │   └── tabular_model.py       # SECOM anomaly detection
│   ├── kg/
│   │   ├── graph_builder.py       # Build knowledge graph
│   │   └── trajectory.py          # RCA path generation
│   ├── llm/
│   │   ├── config.py              # Training config
│   │   ├── dataset.py             # Data loading
│   │   ├── sft_trainer.py        # SFT training
│   │   ├── grpo_trainer.py       # GRPO training
│   │   ├── dpo_trainer.py        # DPO training
│   │   └── inference.py          # Inference
│   ├── rag/
│   │   ├── document_loader.py    # Load docs
│   │   ├── text_splitter.py      # Split text
│   │   ├── embedding.py          # Embeddings
│   │   ├── vector_store.py       # Vector DB
│   │   └── retriever.py          # Retrieval
│   └── data/
│       └── loader.py              # Data loading
├── scripts/
│   ├── train_diagram_clip.py     # Train CLIP
│   ├── generate_diagrams.py      # Generate diagrams
│   ├── train_wm811k_model.py      # Train vision model
│   ├── train_secom_model.py       # Train tabular model
│   ├── train_graph_llm.py        # Train LLM (all phases)
│   ├── train_llm.py              # SFT only
│   ├── eval_sft.py               # Evaluate LLM
│   ├── generate_qa.py            # Generate QA pairs
│   ├── generate_dpo.py           # Generate DPO data
│   ├── build_rag.py              # Build RAG
│   ├── visualize_kg.py           # Visualize KG
│   └── collect_feedback.py       # Collect feedback
├── data/
│   ├── diagrams/                 # CLIP training images
│   ├── wm811k/                   # Wafer map data
│   └── secom/                    # SECOM sensor data
└── docs/
    └── ARCHITECTURE.md          # Detailed architecture
```

## Training Pipelines

### Vision: CLIP + SD
```bash
# Stage 1: Train CLIP
python scripts/train_diagram_clip.py

# Stage 2: Generate diagrams
python scripts/generate_diagrams.py --prompt "CMOS transistor"
```

### Detection Models
```bash
# Download data
python scripts/download_wm811k.py
python scripts/download_secom.py

# Train models
python scripts/train_wm811k_model.py
python scripts/train_secom_model.py
```

### LLM: SFT + GRPO + DPO
```bash
# Generate training data
python scripts/generate_qa.py
python scripts/generate_dpo.py

# Train LLM (full pipeline)
python scripts/train_graph_llm.py --phase all

# Or individual phases
python scripts/train_llm.py  # SFT only
python scripts/eval_sft.py   # Evaluate
```

### RAG
```bash
# Build knowledge base
python scripts/build_rag.py --documents docs/
```

## Key Classes Reference

### Vision
- `CLIPTextEncoder` - Text encoding (512-dim)
- `CLIPImageEncoder` - Image encoding (768→512 projection)
- `SemanticDiagramGenerator` - Combined CLIP model
- `SDDiagramGenerator` - Stable Diffusion generator

### Models
- `WaferMapClassifier` - Wafer defect classification
- `SECOMAnomalyDetector` - Sensor anomaly detection

### KG
- `KnowledgeGraph` - Graph structure
- `RCAGraphBuilder` - Build from data
- `TrajectorySampler` - Sample RCA paths

### LLM
- `GraphLLM` - Main LLM model
- `GRPOConfig` - GRPO configuration
- `TrajectoryDataset` - Training data format

### RAG
- `DocumentLoader` - Load documents
- `ChunkedRetriever` - Retrieve chunks
- `VectorStore` - FAISS/ChromaDB backend

## Development Guidelines

### When Adding Vision Features
1. Keep CLIP and SD in separate files
2. CLIP handles similarity/matching
3. SD handles generation
4. Use public models to avoid auth issues

### When Working on LLM/RCA
1. Use knowledge graph for grounded reasoning
2. Only reference nodes from the graph
3. Penalize hallucinated equipment names

### When Working on Detection
1. Use WM-811K for wafer maps
2. Use SECOM for sensor data
3. Both support real-time inference

### Data Conventions
- Images: PNG/JPG in `data/diagrams/`
- Captions: `.txt` files with same name
- Models: `.pt` checkpoints in `models/`
- Data: Standard dataset formats

## Common Commands

```bash
# Vision
python scripts/train_diagram_clip.py
python scripts/generate_diagrams.py --prompt "your prompt"

# Detection
python scripts/train_wm811k_model.py
python scripts/train_secom_model.py

# LLM
python scripts/train_graph_llm.py --phase all
python scripts/eval_sft.py

# RAG
python scripts/build_rag.py
```

## Notes

- Stable Diffusion: `runwayml/stable-diffusion-v1-5` (public)
- CLIP: `openai/clip-vit-base-patch32` (512-dim)
- Vision: 768-dim → 512-dim projection
- RAG embedding: `BAAI/bge-small-en-v1.5`
- GRPO reward: +1.5 for valid path, -2.0 for hallucination
- **IMPORTANT: Always give commands to user, never run directly**