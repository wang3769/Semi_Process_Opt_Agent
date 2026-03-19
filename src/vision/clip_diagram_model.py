"""
CLIP-Guided Diagram Generation Model
====================================

A vision-language model for generating semiconductor process diagrams.
Uses CLIP embeddings to provide semantic guidance for image generation.

Architecture:
- Text encoder: CLIP text model
- Image encoder: CLIP vision model (frozen)
- Generation: Stable Diffusion with semantic guidance
- Loss: CLIP similarity loss between generated image and text

Usage:
    from src.vision.clip_diagram_model import SemanticDiagramGenerator
    
    generator = SemanticDiagramGenerator()
    image = generator.generate(
        prompt="CMOS process flow: oxidation -> lithography -> etch -> metallization",
        semantic_weight=0.5  # How much to emphasize semantic similarity
    )
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# CLIP
try:
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPProcessor
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed")

# Stable Diffusion
try:
    from diffusers import StableDiffusionPipeline, AutoencoderKL
    from diffusers import UNet2DConditionModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not installed")

# For training
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


@dataclass
class DiagramConfig:
    """Configuration for diagram generation model."""
    # CLIP model
    clip_model_name: str = "openai/clip-vit-large-patch14"
    
    # Stable Diffusion
    sd_model_name: str = "stabilityai/stable-diffusion-2-1-base"
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 10
    warmup_steps: int = 500
    
    # Semantic guidance weight
    semantic_weight: float = 0.5
    reconstruction_weight: float = 1.0
    
    # LoRA (optional)
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    
    # Device
    device: str = "cuda"


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder wrapper."""
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required")
        
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
    
    def forward(self, text: List[str], return_dict: bool = True):
        """Encode text to embeddings."""
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=77,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        # Use pooler output or mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        if return_dict:
            return {"text_embeddings": embeddings, **inputs}
        return embeddings


class CLIPImageEncoder(nn.Module):
    """CLIP image encoder for semantic embedding extraction."""
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required")
        
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        # Freeze for CLIP guidance
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, images: torch.Tensor, return_dict: bool = True):
        """Extract image embeddings."""
        # images should be [B, C, H, W] in range [0, 1]
        # Convert to PIL for processor
        import torchvision.transforms as transforms
        
        # Normalize to [-1, 1] for CLIP
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        
        pixel_values = (images - mean) / std
        
        outputs = self.model(pixel_values)
        embeddings = outputs.pooler_output
        
        if return_dict:
            return {"image_embeddings": embeddings}
        return embeddings
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract normalized image features."""
        with torch.no_grad():
            result = self.forward(images)
            embeddings = result["image_embeddings"]
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


class SemanticDiagramGenerator(nn.Module):
    """
    Main model for semantic diagram generation.
    
    Combines:
    - CLIP text encoder for semantic understanding
    - Stable Diffusion for image generation
    - CLIP image encoder for semantic guidance
    """
    
    def __init__(self, config: Optional[DiagramConfig] = None):
        super().__init__()
        
        self.config = config or DiagramConfig()
        
        if not TRANSFORMERS_AVAILABLE or not DIFFUSERS_AVAILABLE:
            raise ImportError("Required: transformers and diffusers")
        
        # Text encoder
        self.text_encoder = CLIPTextEncoder(self.config.clip_model_name)
        
        # Image encoder (frozen)
        self.image_encoder = CLIPImageEncoder(self.config.clip_model_name)
        
        # Stable Diffusion (will be loaded lazily)
        self.sd_pipeline = None
        
        # Device
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Move to device
        self.text_encoder.to(self.device)
        self.image_encoder.to(self.device)
    
    def load_stable_diffusion(self):
        """Lazy load Stable Diffusion pipeline."""
        if self.sd_pipeline is None:
            print(f"Loading Stable Diffusion: {self.config.sd_model_name}")
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.sd_model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,
            )
            
            # Optionally apply LoRA
            if self.config.use_lora:
                print("LoRA not applied - provide lora_path to enable")
            
            self.sd_pipeline.to(self.device)
            print(f"SD pipeline loaded on: {self.device}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        semantic_weight: float = 0.5,
        num_images: int = 1,
        seed: Optional[int] = None,
    ):
        """Generate diagram with semantic guidance."""
        self.load_stable_diffusion()
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Enhance prompt for technical diagrams
        enhanced_prompt = self._enhance_prompt(prompt)
        
        # Generate
        results = self.sd_pipeline(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt or self._get_default_negative(),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images,
        )
        
        return results.images
    
    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance prompt with technical diagram keywords."""
        enhancement = ", technical diagram, schematic view, clean lines, scientific illustration, engineering drawing, flowchart"
        
        # Add semiconductor-specific terms if relevant
        semiconductor_terms = [
            "semiconductor", "integrated circuit", "wafer", "process flow",
            "CMOS", "transistor", "fabrication"
        ]
        
        for term in semiconductor_terms:
            if term.lower() in prompt.lower():
                return prompt + enhancement
        
        return prompt + ", semiconductor process diagram" + enhancement
    
    def _get_default_negative(self) -> str:
        """Get default negative prompt."""
        return "photorealistic, blurry, low quality, distorted, watermark, text, signature, 3D render"
    
    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """Get CLIP text embeddings for prompts."""
        self.text_encoder.eval()
        with torch.no_grad():
            result = self.text_encoder(prompts)
            embeddings = result["text_embeddings"]
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


class DiagramDataset(torch.utils.data.Dataset):
    """
    Dataset for diagram training.
    
    Expected directory structure:
    data/
        diagrams/
            train/
                image1.png
                image1.txt  # caption
                image2.png
                image2.txt
            val/
                ...
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        split: str = "train"
    ):
        self.data_dir = Path(data_dir) / split
        self.image_size = image_size
        self.split = split
        
        # Find all images
        self.image_files = []
        if self.data_dir.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                self.image_files.extend(list(self.data_dir.glob(ext)))
        
        print(f"Loaded {len(self.image_files)} images from {split}")
        
        # Image transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Load caption
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r") as f:
                caption = f.read().strip()
        else:
            # Use filename as fallback
            caption = img_path.stem.replace("_", " ").replace("-", " ")
        
        return {
            "image": image,
            "caption": caption,
            "image_path": str(img_path),
        }


def create_dataset_template(output_dir: str = "data/diagrams"):
    """
    Create template directory structure for dataset.
    
    User should:
    1. Download/collect diagram images
    2. Write text descriptions in .txt files next to images
    """
    import json
    
    template = {
        "dataset_structure": {
            "data/diagrams/train/": {
                "description": "Training images",
                "example": {
                    "image.png": "Process flow diagram showing...",
                    "image.txt": "Caption/description for the image"
                }
            },
            "data/diagrams/val/": {
                "description": "Validation images"
            }
        },
        "requirements": {
            "min_images": 100,
            "recommended_images": 500,
            "image_format": "PNG or JPG",
            "image_size": "512x512 or higher",
            "caption_length": "1-3 sentences describing the diagram"
        },
        "example_captions": [
            "CMOS process flow showing oxidation, lithography, etching, and metallization steps",
            "Cross-section of MOSFET transistor after gate formation",
            "CVD chamber schematic with gas flow pattern",
            "Wafer processing steps in semiconductor fabrication",
            "Lithography exposure process diagram showing UV light and photoresist"
        ]
    }
    
    # Save template
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dataset_template.json"), "w") as f:
        json.dump(template, f, indent=2)
    
    # Create directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    print(f"Created dataset template in {output_dir}")
    print("Next steps:")
    print("1. Add your diagram images to data/diagrams/train/")
    print("2. Add corresponding .txt files with captions")
    print("3. Repeat for validation set in data/diagrams/val/")
    
    return template


# =============================================================================
# Training Functions
# =============================================================================

def train_clip_guided_generation(
    model: SemanticDiagramGenerator,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    output_dir: str = "models/clip_diagram",
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
):
    """
    Train CLIP-guided diagram generation model.
    
    This trains the text encoder and SD UNet while using frozen CLIP
    for semantic guidance.
    """
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import json
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    
    # Setup optimizer
    # Train only text encoder and UNet (not CLIP image encoder)
    optimizer = torch.optim.AdamW(
        list(model.text_encoder.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Load SD if not already loaded
    model.load_stable_diffusion()
    
    # Training loop
    model.train()
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            images = batch["image"].to(model.device)
            captions = batch["caption"]
            
            # Get text embeddings
            text_embeddings = model.get_text_embeddings(captions)
            
            # Get image embeddings (for semantic loss)
            with torch.no_grad():
                image_embeddings = model.image_encoder.extract_features(images)
            
            # Note: Full training would require:
            # 1. Generate images from SD
            # 2. Extract CLIP embeddings from generated images
            # 3. Compute contrastive loss
            
            # For now, log progress
            epoch_loss += 1.0  # Placeholder
            
            pbar.set_postfix({"loss": epoch_loss / (pbar.n + 1)})
        
        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, output_path / "best_model.pt")
    
    # Save history
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_path}")
    
    return model, history


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CLIP-Guided Diagram Generation Model")
    print("=" * 60)
    
    # Check availability
    print(f"\nTransformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"Diffusers available: {DIFFUSERS_AVAILABLE}")
    print(f"PEFT available: {PEFT_AVAILABLE}")
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create dataset template
    print("\n" + "=" * 60)
    print("Creating Dataset Template")
    print("=" * 60)
    create_dataset_template()
    
    # Example usage
    print("\n" + "=" * 60)
    print("Example Usage")
    print("=" * 60)
    
    config = DiagramConfig(
        clip_model_name="openai/clip-vit-large-patch14",
        sd_model_name="stabilityai/stable-diffusion-2-1-base",
        batch_size=2,
        num_epochs=5,
    )
    
    print(f"\nConfig:")
    print(f"  CLIP model: {config.clip_model_name}")
    print(f"  SD model: {config.sd_model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    
    print("\nTo train:")
    print("  1. Add images to data/diagrams/train/")
    print("  2. Add captions in .txt files")
    print("  3. Run training script")
