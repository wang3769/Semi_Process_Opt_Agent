# Graph-Based RCA LLM - Setup & Training Guide

This document explains each step of the Graph-Based RCA (Root Cause Analysis) LLM system.

---

## Step 1: Generate Knowledge Graph

Creates a synthetic semiconductor fab knowledge graph with equipment, sensors, parameters, and root causes.

### Command:
```powershell
python -c "from src.kg.graph_builder import create_simplified_fab_graph; g = create_simplified_fab_graph(); g.save('data/processed/kg/fab_graph.json')"
```

### What it does:
- Creates 40 nodes (8 equipment, 10 sensors, 6 parameters, 8 defect patterns, 8 root causes)
- Creates 51 edges (relationships between nodes)
- Saves to JSON for later use

---

## Step 2: Generate Training Data

Generates synthetic RCA cases and converts them to SFT and GRPO training formats.

### Command:
```powershell
python scripts/train_graph_llm.py --phase generate --num_examples 100
```

### What it does:
1. Loads the knowledge graph
2. Generates 100 synthetic RCA cases:
   - Random defect pattern (CENTER, EDGE_RING, SCRATCH, etc.)
   - Random root cause (TEMP_GRADIENT, PRESSURE_VARIANCE, etc.)
   - Simulated sensor readings with defect-specific anomalies
   - Valid graph path from defect to root cause
3. Converts to SFT format (instruction + context + response)
4. Converts to GRPO format (prompt + rewards)
5. Saves:
   - `data/processed/llm/graph_sft_train.jsonl`
   - `data/processed/llm/graph_grpo_train.jsonl`

---

## Step 3: Cold Start SFT Training

Trains the LLM on graph walk trajectories to teach "how to read the map."

### Command:
```powershell
python scripts/train_graph_llm.py --phase sft
```

### What it does:
- Loads SFT training data
- Shows example format (for actual training, use trl or unsloth)
- Teaches the model to:
  - Navigate from defect → equipment → sensor → parameter → root cause
  - Use only nodes from the provided graph
  - Follow valid reasoning paths

### For actual GPU training:
```bash
trl sft \
  --model_name_orPath meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name data/processed/llm/graph_sft_train.jsonl \
  --output_dir models/graph_sft \
  --learning_rate 1e-5 \
  --num_train_epochs 3
```

---

## Step 4: GRPO Training with Causal Rewards

Fine-tunes with rewards that penalize hallucinations and reward valid graph reasoning.

### Command:
```powershell
python scripts/train_graph_llm.py --phase grpo
```

### What it does:
1. Loads GRPO training data
2. Demonstrates reward calculation:
   - **Correctness Reward**: Did it find the true root cause? (+1.0)
   - **Causal Link Reward**: Did it follow valid graph paths? (+1.5)
   - **Hallucination Penalty**: Did it mention non-existent tools? (-2.0!)
   - **Efficiency Reward**: Did it find the cause in few steps? (+0.5)

### For actual GPU training:
```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="models/graph_grpo",
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=grpo_dataset,
    reward_function=calculate_causal_reward
)

trainer.train()
```

---

## Quick Start (All in One)

```powershell
# Generate everything
python scripts/train_graph_llm.py --phase all --num_examples 100
```

---

## Files Generated

| File | Description |
|------|-------------|
| `data/processed/kg/fab_graph.json` | Knowledge graph |
| `data/processed/llm/graph_sft_train.jsonl` | SFT training data |
| `data/processed/llm/graph_grpo_train.jsonl` | GRPO training data |

---

## Key Innovation: Hallucination Penalty

The key difference from standard RCA LLMs:

```
Standard LLM: Can hallucinate any tool/sensor → UNSAFE for fabs!
Graph-RCA LLM: Must use only nodes in graph → FAITHFUL to fab reality

Example:
  ❌ "The laser_sensor in chamber_xyz caused this" → -2.0 penalty!
  ✅ "The CVD_1 chamber pressure variance caused this" → +1.0 reward
```

This architecture ensures the model NEVER makes up equipment or sensors that don't exist in your fab.

---

## Step 5: Process Image Generation (Optional)

Generate "fake 3D" illustrations for semiconductor processes using Stable Diffusion.

### Install Dependencies:
```bash
pip install diffusers peft transformers accelerate safetensors
```

### Quick Usage:
```python
from src.vision.image_generator import ProcessImageGenerator, generate_example_prompts

# Initialize (requires ~4GB VRAM)
gen = ProcessImageGenerator()

# Generate from prompt
images = gen.generate(
    prompt="CVD chamber cross-section showing gas flow pattern",
    num_inference_steps=25,
    guidance_scale=7.5
)

# Save
gen.save_image(images[0], "output/cvd_chamber.png")
```

### Example Prompts:
```python
# Get built-in prompts
prompts = generate_example_prompts()

# CVD Process
prompts["cvd_chamber"]
# "CVD chamber cross-section, gas inlet, wafer susceptor, rf coil,..."

# Etch Process
prompts["etch_chamber"]
# "Plasma etching chamber, ion bombardment visualization..."

# Lithography
prompts["litho_exposure"]
# "Lithography exposure, UV light through mask..."

# CMP Process
prompts["cmp_process"]
# "Chemical mechanical polishing, wafer polishing..."
```

### For PPT Illustrations:
```python
# Generate multiple variations
images = gen.generate(
    prompt="Semiconductor wafer cross-section, multi-layer stack",
    num_images=4,  # Generate 4 variations
    width=768,
    height=512,  # Wide format for slides
    seed=42  # Reproducible
)

# Save all
gen.save_images(images, "output/wafer_sections")
```

### With LoRA Fine-tuning (Optional):
```python
# After fine-tuning with your process images
gen = ProcessImageGenerator(lora_path="models/semi_process_lora.safetensors")

# Now generates in your custom style
images = gen.generate(prompt="Your custom process description")
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (8GB+ VRAM recommended)
- For image generation: 4GB+ VRAM
