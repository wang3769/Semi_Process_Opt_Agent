"""
Stable Diffusion Diagram Generation Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel


@dataclass
class SDDiagramConfig:
    """Configuration for SD diagram generation."""
    sd_model_name: str = "runwayml/stable-diffusion-v1-5"  # Public model
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_model_path: Optional[str] = None
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    image_size: int = 512


class CLIPEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.projection = nn.Linear(768, 512)
        
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(self.text_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.pooler_output
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def load_finetuned(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded fine-tuned CLIP from {checkpoint_path}")


class SDDiagramGenerator:
    def __init__(self, config: Optional[SDDiagramConfig] = None):
        self.config = config or SDDiagramConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CLIP encoder
        self.clip_encoder = CLIPEncoder(self.config.clip_model_name)
        self.clip_encoder.to(self.device)
        
        if self.config.clip_model_path:
            self.clip_encoder.load_finetuned(self.config.clip_model_path)
        
        # Stable Diffusion
        self._load_stable_diffusion()
        
        self._tech_enhancement = (
            ", technical diagram, schematic view, engineering drawing, "
            "clean lines, scientific illustration, flowchart, blueprint style"
        )
        self._negative_prompt = (
            "photorealistic, blurry, low quality, distorted, watermark, "
            "text, signature, 3D render, photo, realistic"
        )
    
    def _load_stable_diffusion(self):
        print(f"Loading Stable Diffusion: {self.config.sd_model_name}")
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.sd_model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,
            )
            self.pipeline.to(self.device)
            print(f"SD pipeline loaded on: {self.device}")
        except Exception as e:
            print(f"Error loading SD: {e}")
            self.pipeline = None
    
    def _enhance_prompt(self, prompt: str) -> str:
        enhanced = prompt + self._tech_enhancement
        semiconductor_terms = ["semiconductor", "CMOS", "transistor", "fabrication", 
                             "etch", "deposition", "lithography", "chamber", "plasma"]
        if any(term in prompt.lower() for term in semiconductor_terms):
            enhanced += ", semiconductor process diagram, cross-section view"
        return enhanced
    
    def generate(self, prompt: str, num_images: int = 1, num_inference_steps: int = 25,
                 guidance_scale: float = 7.5, seed: Optional[int] = None) -> List[Image.Image]:
        if self.pipeline is None:
            raise RuntimeError("Stable Diffusion pipeline not loaded")
        
        enhanced = self._enhance_prompt(prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
        
        with torch.inference_mode():
            results = self.pipeline(
                prompt=enhanced,
                negative_prompt=self._negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_images,
            )
        return results.images
    
    def save_image(self, image: Image.Image, path: str):
        image.save(path)
        print(f"Saved: {path}")


def load_trained_generator(clip_checkpoint: str = "models/clip_diagram/best_model.pt",
                          sd_model: str = "runwayml/stable-diffusion-v1-5") -> SDDiagramGenerator:
    config = SDDiagramConfig(sd_model_name=sd_model, clip_model_path=clip_checkpoint)
    return SDDiagramGenerator(config)


if __name__ == "__main__":
    print("SD Diagram Generator ready!")
    print("Usage: from src.vision.sd_diagram_model import load_trained_generator")
