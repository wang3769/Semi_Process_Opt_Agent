"""
CLIP-Guided Diagram Generation Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

@dataclass
class DiagramConfig:
    clip_model_name: str = DEFAULT_CLIP_MODEL
    sd_model_name: str = "stabilityai/stable-diffusion-2-1-base"
    hf_token: Optional[str] = None
    device: str = "cuda"


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL):
        super().__init__()
        print(f"Loading CLIP text encoder: {model_name}")
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        print(f"Text encoder hidden size: {self.hidden_size}")
    
    def forward(self, text: List[str], return_dict: bool = True):
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embeddings = outputs.pooler_output
        
        if return_dict:
            return {"text_embeddings": embeddings, **inputs}
        return embeddings
    
    def get_embeddings(self, text: List[str], training: bool = False):
        """Get text embeddings with proper gradient flow."""
        if not training:
            with torch.no_grad():
                return self.forward(text)
        return self.forward(text)


class CLIPImageEncoder(nn.Module):
    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL):
        super().__init__()
        print(f"Loading CLIP image encoder: {model_name}")
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.vision_hidden_size = self.model.config.hidden_size
        print(f"Image encoder hidden size: {self.vision_hidden_size}")
        
        # Projection layer (trainable)
        self.projection = nn.Linear(self.vision_hidden_size, 512)
        
        # Freeze vision backbone
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, images: torch.Tensor, return_dict: bool = True):
        images = images.to(self.model.device)
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        pixel_values = (images - mean) / std
        
        outputs = self.model(pixel_values)
        vision_emb = outputs.pooler_output
        embeddings = self.projection(vision_emb)
        
        if return_dict:
            return {"image_embeddings": embeddings}
        return embeddings
    
    def extract_features(self, images: torch.Tensor, training: bool = False) -> torch.Tensor:
        """Extract image features with optional gradients."""
        if not training:
            with torch.no_grad():
                result = self.forward(images)
                embeddings = result["image_embeddings"]
                return F.normalize(embeddings, p=2, dim=-1)
        
        result = self.forward(images)
        embeddings = result["image_embeddings"]
        return F.normalize(embeddings, p=2, dim=-1)


class SemanticDiagramGenerator(nn.Module):
    def __init__(self, config: Optional[DiagramConfig] = None):
        super().__init__()
        self.config = config or DiagramConfig()
        
        model_name = self.config.clip_model_name
        
        self.text_encoder = CLIPTextEncoder(model_name)
        self.image_encoder = CLIPImageEncoder(model_name)
        
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
    def get_text_embeddings(self, prompts: List[str], training: bool = False) -> torch.Tensor:
        """Get text embeddings - set training=True during training."""
        result = self.text_encoder.get_embeddings(prompts, training=training)
        embeddings = result["text_embeddings"]
        embeddings = F.normalize(embeddings, p=2, dim=-1) # L2 normalization p=2
        return embeddings


class DiagramDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, image_size: int = 224, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        
        split_dir = self.data_dir / split
        if split_dir.exists() and (any(split_dir.glob("*.png")) or any(split_dir.glob("*.jpg"))):
            self.data_dir = split_dir
        
        self.image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            self.image_files.extend(list(self.data_dir.glob(ext)))
        
        print(f"Loaded {len(self.image_files)} images from {self.data_dir}")
        
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            except UnicodeDecodeError:
                caption = img_path.stem.replace("_", " ").replace("-", " ")
        else:
            caption = img_path.stem.replace("_", " ").replace("-", " ")
        
        return {"image": image, "caption": caption, "image_path": str(img_path)}