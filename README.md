# Semi_Process_Opt_Agent

Autonomous Experiment Optimization Copilot for Semiconductor Manufacturing

## Features

- **Knowledge Graph RCA**: Graph-based root cause analysis with causal reasoning
- **LLM Training**: Cold Start SFT + GRPO for faithful graph traversal
- **Vision Generation**: Two-stage diagram generation (CLIP + Stable Diffusion)
- **RAG Knowledge Base**: Document retrieval for context-aware responses

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VISION PIPELINE                          │
│                                                             │
│  ┌─────────────────┐      ┌─────────────────────────────┐   │
│  │  Stage 1       │      │  Stage 2                   │   │
│  │  CLIP Training │ ──▶  │  Stable Diffusion          │   │
│  │                │      │  Generation                │   │
│  │  Text-Image    │      │                           │   │
│  │  Alignment     │      │  Uses fine-tuned CLIP      │   │
│  │                │      │  for semantic guidance    │   │
│  └─────────────────┘      └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install diffusers for image generation
pip install diffusers
```

## Training

### Vision: Two-Stage Approach

**Stage 1: Train CLIP for Text-Image Alignment**
```bash
python scripts/train_diagram_clip.py
# Output: models/clip_diagram/best_model.pt
```

**Stage 2: Generate Diagrams with Stable Diffusion**
```bash
# Single prompt
python scripts/generate_diagrams.py --prompt "CMOS transistor cross-section"

# Batch prompts
python scripts/generate_diagrams.py --batch --prompts_file prompts.txt

# Or in Python:
from src.vision.sd_diagram_model import load_trained_generator

generator = load_trained_generator("models/clip_diagram/best_model.pt")
images = generator.generate("CVD chamber with gas flow", num_images=4)
generator.save_image(images[0], "output.png")
```

### LLM Training (Coming Soon)
```bash
python scripts/train_graph_llm.py --phase all --num_examples 100
```

## Data

- `data/diagrams/train/` - CLIP training images (images + .txt captions)
- `data/diagrams/val/` - CLIP validation images

## Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture.
