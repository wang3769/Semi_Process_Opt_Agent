"""
Training Script for CLIP-Guided Diagram Generation Model
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
# here the PROJECT_ROOT has been assigned to a variable. You can tell that the inconsistency of variable assignment is the signature of LLM generated code, it might learned from somewhere.
import sys
sys.path.insert(0, str(PROJECT_ROOT))
# The above code is to ensure that the script can import modules from the project root directory, which is necessary for importing the CLIP diagram model and dataset classes; highest priority
from src.vision.clip_diagram_model import DiagramConfig, SemanticDiagramGenerator, DiagramDataset
# trainer in scripts, model in src, and image in model is a common pattern in LLM repo.

@dataclass # quick definition of class, no need to write __init__ method
class TrainingConfig:
    data_dir: str = "data/diagrams"
    image_size: int = 224
    clip_model: str = "openai/clip-vit-base-patch32"
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 4
    semantic_weight: float = 0.5
    output_dir: str = "models/clip_diagram"
    device: str = "cuda"


class DiagramTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        self._init_model()
        self._init_datasets()
        
        self.history = {"train_loss": [], "train_semantic_loss": [], "val_similarity": []}
    
    def _init_model(self):
        print("Initializing model...")
        
        model_config = DiagramConfig(
            clip_model_name=self.config.clip_model,
            device=self.config.device,
        )
        
        self.model = SemanticDiagramGenerator(model_config)
        print("Model initialized")
    
    def _init_datasets(self):
        self.train_dataset = DiagramDataset(
            data_dir=self.config.data_dir,
            image_size=self.config.image_size,
            split="train"
        )
        
        # Always initialize val_loader to None first
        self.val_loader = None
        self.val_dataset = None
        
        val_path = Path(self.config.data_dir) / "val"
        if val_path.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                if any(val_path.glob(ext)):
                    self.val_dataset = DiagramDataset(self.config.data_dir, self.config.image_size, "val")
                    self.val_loader = torch.utils.data.DataLoader(
                        self.val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0
                    )
                    break
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0
        )
        
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset) if self.val_dataset else 0}")
    
    def compute_clip_loss(self, text_emb, image_emb):
        text_emb = text_emb.to(self.device)
        image_emb = image_emb.to(self.device)
        
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        logits = torch.matmul(text_emb, image_emb.T)
        
        batch_size = text_emb.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        # Forces the model to maximize the diagonal of this matrix (where the correct text matches the correct image) and minimize everything else.
        return (loss_i2t + loss_t2i) / 2
    
    def train_step(self, batch, training: bool = True):
        images = batch["image"]
        captions = batch["caption"]
        
        text_emb = self.model.get_text_embeddings(captions, training=training)
        image_emb = self.model.image_encoder.extract_features(images, training=training)
        
        loss = self.compute_clip_loss(text_emb, image_emb) * self.config.semantic_weight
        loss.backward()
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_sim = 0.0
        num = 0
        
        for batch in self.val_loader:
            images = batch["image"]
            captions = batch["caption"]
            
            text_emb = self.model.get_text_embeddings(captions, training=False)
            image_emb = self.model.image_encoder.extract_features(images, training=False)
            
            sim = F.cosine_similarity(text_emb, image_emb, dim=-1).mean()
            total_sim += sim.item()
            num += 1
        
        return total_sim / num
    
    def train(self):
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)
        
        trainable_params = (
            list(self.model.text_encoder.parameters()) +
            list(self.model.image_encoder.projection.parameters())
        )
        
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
        
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        best_loss = float("inf")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                losses = self.train_step(batch, training=True)
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += losses["loss"]
                pbar.set_postfix({"loss": f"{losses['loss']:.4f}"})
            
            avg_loss = epoch_loss / len(self.train_loader)
            self.history["train_loss"].append(avg_loss)
            
            val_sim = 0.0
            if self.val_loader is not None:
                val_sim = self.validate()
                self.history["val_similarity"].append(val_sim)
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Sim={val_sim:.4f}")
            else:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "config": asdict(self.config),
                }, output_path / "best_model.pt")
            
            scheduler.step()
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": asdict(self.config),
        }, output_path / "final_model.pt")
        
        with open(output_path / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nDone! Best loss: {best_loss:.4f}")
        return self.history


def main():
    config = TrainingConfig()
    
    train_path = Path(config.data_dir) / "train"
    root_path = Path(config.data_dir)
    
    def has_images(path):
        if not path.exists():
            return False
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            if any(path.glob(ext)):
                return True
        return False
    
    found = False
    if train_path.exists() and has_images(train_path):
        print(f"Using: {train_path}")
        found = True
    elif root_path.exists() and has_images(root_path):
        print(f"Using: {root_path}")
        found = True
    
    if not found:
        print(f"No images in {config.data_dir}")
        return
    
    trainer = DiagramTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()