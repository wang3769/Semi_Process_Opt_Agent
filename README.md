# Semi_Process_Opt_Agent

Autonomous Experiment Optimization Copilot for Semiconductor Manufacturing

## Features

- **Knowledge Graph RCA**: Graph-based root cause analysis with causal reasoning
- **LLM Training**: Cold Start SFT + GRPO for faithful graph traversal
- **Vision Generation**: Two-stage process illustration generation
- **RAG Knowledge Base**: Document retrieval for context-aware responses

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets (optional)
python scripts/download_all.py
```

## Training

### LLM Training
```bash
# Train graph-constrained LLM
python scripts/train_graph_llm.py --phase all --num_examples 100
```

### Vision Training (Two-Stage)

**Stage 1: CLIP Alignment**
```bash
# Train CLIP on diagram-caption pairs
python scripts/train_diagram_clip.py
# Output: models/clip_diagram/best_model.pt
```

**Stage 2: Diagram Generation**
```python
from src.vision.sd_diagram_model import load_trained_generator

generator = load_trained_generator("models/clip_diagram/best_model.pt")
images = generator.generate("CMOS transistor cross-section diagram")
```

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture.

## Data

- `data/diagrams/train/` - CLIP training images
- `data/diagrams/val/` - CLIP validation images

Each image should have a corresponding `.txt` file with the caption.
