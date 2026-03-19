"""
Training Script for CLIP-Guided Diagram Generation Model
========================================================

This script trains the CLIP-guided semantic diagram generation model
for semiconductor process diagrams.

Usage:
    python scripts/train_diagram_clip.py --data_dir data/diagrams --output_dir models/clip_diagram

User's responsibility:
1. Collect diagram images (500-1000 recommended)
2. Write text captions for each image in .txt files
3. Place in data/diagrams/train/ and data/diagrams/val/
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.clip_diagram_model import (
    DiagramConfig,
    SemanticDiagramGenerator,
    DiagramDataset,
    CLIPTextEncoder,
    CLIPImageEncoder,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    data_dir: str = "data/diagrams"
    image_size: int = 512
    
    # Model
    clip_model: str = "openai/clip-vit-large-patch14"
    sd_model: str = "stabilityai/stable-diffusion-2-1-base"
    
    # Training
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    
    # Loss weights
    semantic_weight: float = 0.5
    reconstruction_weight: float = 1.0
    diversity_weight: float = 0.1
    
    # LoRA
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    
    # Output
    output_dir: str = "models/clip_diagram"
    log_interval: int = 10
    save_interval: int = 1
    
    # Device
    device: str = "cuda"


class DiagramTrainer:
    """
    Trainer for CLIP-guided diagram generation.
    
    Training strategy:
    1. Use frozen CLIP for semantic embedding extraction
    2. Train text encoder to better align with diagrams
    3. Use prompt-to-image generation with CLIP guidance
    4. Apply LoRA for efficient fine-tuning
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize model
        self._init_model()
        
        # Initialize datasets
        self._init_datasets()
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_semantic_loss": [],
            "train_recon_loss": [],
            "val_loss": [],
            "val_similarity": [],
        }
    
    def _init_model(self):
        """Initialize the model."""
        print("Initializing CLIP-guided diagram generator...")
        
        # Create model config
        model_config = DiagramConfig(
            clip_model_name=self.config.clip_model,
            sd_model_name=self.config.sd_model,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            semantic_weight=self.config.semantic_weight,
            reconstruction_weight=self.config.reconstruction_weight,
            use_lora=self.config.use_lora,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
        )
        
        self.model = SemanticDiagramGenerator(model_config)
        self.model.to(self.device)
        
        print(f"Model initialized on: {self.device}")
    
    def _init_datasets(self):
        """Initialize datasets."""
        # Load SD pipeline for generation
        self.model.load_stable_diffusion()
        
        # Create datasets
        self.train_dataset = DiagramDataset(
            data_dir=self.config.data_dir,
            image_size=self.config.image_size,
            split="train"
        )
        
        self.val_dataset = None
        val_path = Path(self.config.data_dir) / "val"
        if val_path.exists() and any(val_path.glob("*.png")):
            self.val_dataset = DiagramDataset(
                data_dir=self.config.data_dir,
                image_size=self.config.image_size,
                split="val"
            )
        
        # Create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        
        if self.val_dataset:
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        
        print(f"Train samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Val samples: {len(self.val_dataset)}")
    
    def compute_clip_loss(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP contrastive loss.
        
        Maximizes similarity between matching text-image pairs
        while minimizing for non-matching pairs.
        """
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(text_embeddings, image_embeddings.T)
        
        # Symmetric contrastive loss
        batch_size = text_embeddings.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        images = batch["image"].to(self.device)
        captions = batch["caption"]
        
        # Get text embeddings
        text_embeddings = self.model.get_text_embeddings(captions)
        
        # Get image embeddings from target images (for semantic loss)
        with torch.no_grad():
            target_embeddings = self.model.image_encoder.extract_features(images)
        
        # Compute semantic loss (CLIP alignment)
        semantic_loss = self.compute_clip_loss(text_embeddings, target_embeddings)
        
        # Total loss
        loss = self.config.semantic_weight * semantic_loss
        
        # Backward
        loss.backward()
        
        return {
            "loss": loss.item(),
            "semantic_loss": semantic_loss.item(),
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        if not self.val_loader:
            return {"loss": 0.0, "similarity": 0.0}
        
        self.model.eval()
        
        total_loss = 0.0
        total_similarity = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            captions = batch["caption"]
            
            # Get embeddings
            text_embeddings = self.model.get_text_embeddings(captions)
            image_embeddings = self.model.image_encoder.extract_features(images)
            
            # Compute similarity
            similarity = F.cosine_similarity(
                text_embeddings, image_embeddings, dim=-1
            ).mean()
            
            total_similarity += similarity.item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "similarity": total_similarity / num_batches,
        }
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.text_encoder.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        
        # Create output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(output_path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        best_loss = float("inf")
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_semantic_loss = 0.0
            
            pbar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
            )
            
            for batch_idx, batch in enumerate(pbar):
                # Forward
                losses = self.train_step(batch)
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Logging
                epoch_loss += losses["loss"]
                epoch_semantic_loss += losses["semantic_loss"]
                
                pbar.set_postfix({
                    "loss": f"{losses['loss']:.4f}",
                    "sem": f"{losses['semantic_loss']:.4f}",
                })
                
                global_step += 1
            
            # Average losses
            avg_loss = epoch_loss / len(self.train_loader)
            avg_semantic = epoch_semantic_loss / len(self.train_loader)
            
            self.history["train_loss"].append(avg_loss)
            self.history["train_semantic_loss"].append(avg_semantic)
            
            # Validation
            if self.val_loader:
                val_metrics = self.validate()
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_similarity"].append(val_metrics["similarity"])
                
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                      f"Semantic={avg_semantic:.4f}, "
                      f"Val Similarity={val_metrics['similarity']:.4f}")
            else:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Semantic={avg_semantic:.4f}")
            
            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(output_path / "best_model.pt", epoch, avg_loss)
            
            # Periodic save
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(
                    output_path / f"checkpoint-epoch-{epoch+1}.pt", 
                    epoch, 
                    avg_loss
                )
            
            scheduler.step()
        
        # Save final model
        self.save_checkpoint(output_path / "final_model.pt", epoch, avg_loss)
        
        # Save history
        with open(output_path / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Model saved to: {output_path}")
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "history": self.history,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")


def create_sample_data(data_dir: str = "data/diagrams"):
    """
    Create sample data structure for the user to fill in.
    
    This creates the directory structure and a README explaining
    what the user needs to provide.
    """
    data_path = Path(data_dir)
    
    # Create directories
    for split in ["train", "val"]:
        (data_path / split).mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme = """# Diagram Dataset

This directory should contain your semiconductor process diagram images
for training the CLIP-guided diagram generation model.

## Directory Structure

```
data/diagrams/
├── train/
│   ├── image_001.png    # Diagram image
│   ├── image_001.txt    # Caption for the image
│   ├── image_002.png
│   ├── image_002.txt
│   └── ...
└── val/
    ├── image_001.png
    ├── image_001.txt
    └── ...
```

## Requirements

### Images
- Format: PNG or JPG
- Size: 512x512 or larger
- Content: Semiconductor process diagrams, flowcharts, cross-sections, etc.

### Captions (text files)
- Each image should have a corresponding .txt file with the same name
- Caption should describe what's shown in the diagram
- 1-3 sentences recommended

## Example Captions

Here are some example captions for semiconductor diagrams:

1. "CMOS process flow showing oxidation, lithography, etching, and metallization steps"
2. "Cross-section of MOSFET transistor after gate formation showing source, drain, and channel regions"
3. "CVD chamber schematic with gas flow pattern showing reactant gas inlet and wafer susceptor"
4. "Wafer processing steps in semiconductor fabrication: cleaning, oxidation, diffusion, and testing"
5. "Lithography exposure process diagram showing UV light passing through mask onto photoresist"
6. "Plasma etching chamber cross-section with ion bombardment direction and wafer surface"
7. "Chemical mechanical polishing process showing wafer, polishing pad, and slurry interaction"
8. "Ion implantation system schematic with ion beam direction and wafer positioning"

## Data Collection Suggestions

To build your dataset, you can:

1. **From textbooks**: Scan or screenshot diagrams from semiconductor manufacturing textbooks
2. **From papers**: Download figures from IEEE papers on semiconductor process technology
3. **From websites**: Collect diagrams from semiconductor company websites and educational resources
4. **Create yourself**: Use tools like Draw.io, Lucidchart, or Graphviz to create process flow diagrams

## Recommended Dataset Size

- Minimum: 100 images
- Recommended: 500-1000 images
- More data = better model performance

## Next Steps

1. Collect and organize your diagram images
2. Write captions for each image
3. Run training:
   ```
   python scripts/train_diagram_clip.py --data_dir data/diagrams --output_dir models/clip_diagram
   ```
"""
    
    readme_path = data_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    
    print(f"Sample data structure created at: {data_path}")
    print(f"Please add your diagram images and captions to the train/ and val/ directories")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train CLIP-guided diagram generation model"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/diagrams",
        help="Path to diagram dataset directory"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=512,
        help="Image size for training"
    )
    
    # Model arguments
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model name"
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Stable Diffusion model name"
    )
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Loss weights
    parser.add_argument("--semantic_weight", type=float, default=0.5)
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    
    # Output
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models/clip_diagram"
    )
    parser.add_argument("--create_sample_data", action="store_true",
                       help="Create sample data structure")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        create_sample_data(args.data_dir)
        return
    
    # Check if data exists
    train_path = Path(args.data_dir) / "train"
    if not train_path.exists() or not any(train_path.glob("*.png")):
        print(f"No training data found in {train_path}")
        print("Use --create_sample_data to create the directory structure")
        create_sample_data(args.data_dir)
        return
    
    # Create training config
    config = TrainingConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        clip_model=args.clip_model,
        sd_model=args.sd_model,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        semantic_weight=args.semantic_weight,
        reconstruction_weight=args.reconstruction_weight,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        output_dir=args.output_dir,
    )
    
    # Train
    trainer = DiagramTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
