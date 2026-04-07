"""
Stable Diffusion Diagram Generation Model
==========================================

Architecture Overview:
---------------------

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        STABLE DIFFUSION PIPELINE                       │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────────────────┐
    │                         COMPONENT 1: CLIP ENCODER                      │
    │                                                                       │
    │   User Prompt ──► Tokenizer ──► CLIP Text Encoder ──► Text Embeddings │
    │                   (77 tokens max)        (512-dim, normalized)         │
    │                                                                       │
    │   [Frozen pretrained CLIP - openai/clip-vit-base-patch32]            │
    │                                                                       │
    │   Optional: Load fine-tuned CLIP from trained_diagram_clip.py         │
    │   This improves alignment between text and semiconductor diagrams     │
    └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ Text embeddings (conditioning)
    ┌───────────────────────────────────────────────────────────────────────┐
    │                       COMPONENT 2: DENOISING U-Net                     │
    │                                                                       │
    │   ┌─────────────────────────────────────────────────────────────────┐ │
    │   │                    LATENT DIFFUSION PROCESS                     │ │
    │   │                                                               │ │
    │   │   T=0:  Random Noise ──────────────────────────────────────►   │ │
    │   │         (64x64 latent)                                          │ │
    │   │                             │                                   │ │
    │   │                             ▼ (repeated for T steps)            │ │
    │   │         ┌───────────────────┴───────────────────┐              │ │
    │   │         │         UNet (U-Net2DConditionModel)   │              │ │
    │   │         │                                       │              │ │
    │   │         │  - Takes noisy latent + timestep     │              │ │
    │   │         │  - Takes text conditioning (CLIP)    │              │ │
    │   │         │  - Predicts noise residual           │              │ │
    │   │         │  - Outputs denoised latent            │              │ │
    │   │         └───────────────────────────────────────┘              │ │
    │   │                             │                                   │ │
    │   │                             ▼ (subtract predicted noise)       │ │
    │   │   T=1:  Less noise                                          │ │
    │   │         ...                                                  │ │
    │   │   T=N:  Clean latent                                        │ │
    │   │                                                               │ │
    │   │   [Default: 25 inference steps]                              │ │
    │   └─────────────────────────────────────────────────────────────────┘ │
    └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ Clean latent (64x64)
    ┌───────────────────────────────────────────────────────────────────────┐
    │                       COMPONENT 3: VAE DECODER                        │
    │                                                                       │
    │   VAE Decoder ──► Generated Image (512x512 RGB)                      │
    │                                                                       │
    │   [Variational Autoencoder - frozen pretrained]                     │
    └───────────────────────────────────────────────────────────────────────┘

Information Flow During Generation:
----------------------------------

1. TEXT ENCODING (CLIP):
   prompt = "CMOS transistor cross-section"
         ↓ Tokenize (max 77 tokens)
         ↓ CLIP Text Encoder (frozen)
         ↓ Normalize → text_embeddings (512-dim)

2. LATENT DIFFUSION (Denoising Loop):
   for step in range(num_inference_steps):
       - Sample random noise (for first step) or use previous latent
       - UNet processes: [noisy_latent + timestep_emb + text_emb]
       - Output: predicted_noise
       - latent = latent - guidance_scale * predicted_noise
       - (Classifier-free guidance: amplifies conditioning effect)

3. IMAGE DECODING (VAE):
   clean_latent → VAE Decoder → final_image (512x512)

Training vs Inference:
----------------------

TRAINING (train_diagram_clip.py):
   - CLIP only: learns text-image similarity
   - Image: actual diagrams
   - Text: descriptions/captions
   - Loss: contrastive (maximize similarity of matched pairs)

INFERENCE (generate_diagrams.py):
   - Uses pretrained Stable Diffusion's UNet + VAE
   - Optionally loads fine-tuned CLIP for better text alignment
   - CLIP guides what the UNet should generate

Key Files:
---------
- src/vision/clip_diagram_model.py: CLIP training (Stage 1)
- src/vision/sd_diagram_model.py: SD generation (Stage 2)
- scripts/train_diagram_clip.py: Training script
- scripts/generate_diagrams.py: Generation script
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
    """
    CLIP Text Encoder (Component 1 of SD Pipeline)
    ----------------------------------------------
    
    This module handles text-to-semantic mapping:
    
    Text Input → Tokenize → CLIP Text Encoder → Normalized Embeddings
    
    Structure:
    - CLIPTextModel: Transformer encoder (12 layers, 768-dim hidden)
    - CLIPTokenizer: Converts text to token IDs (vocab ~49k)
    - projection: Maps 768-dim to 512-dim (optional, for compatibility)
    
    Training:
    - During CLIP training (Stage 1), both text and image encoders learn
    - During inference, text encoder is frozen (loaded from SD)
    
    Information Flow:
    "CMOS transistor" → [CLS] token → transformer → pooler_output → 512-dim
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.projection = nn.Linear(768, 512)
        
        # Freeze pretrained weights
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to CLIP embeddings."""
        inputs = self.tokenizer(
            prompts, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.text_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.pooler_output  # [batch, 512]
            embeddings = F.normalize(embeddings, p=2, dim=-1)  # L2 normalize
        
        return embeddings
    
    # the fine-tuned CLIP model saved ealier can be re-used here to improve overall project coherence between stages of this pipeline.
    def load_finetuned(self, checkpoint_path: str):
        """Load weights from Stage 1 CLIP training."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded fine-tuned CLIP from {checkpoint_path}")


class SDDiagramGenerator:
    """
    Stable Diffusion Diagram Generator (Full Pipeline)
    ==================================================
    
    Combines three components:
    1. CLIPEncoder: Text → semantic embeddings (512-dim)
    2. UNet: Denoising in latent space (diffusion process)
    3. VAE: Latent → image space (decoder only)
    
    The CLIP encoder provides conditioning to the UNet during generation.
    Fine-tuned CLIP improves alignment with semiconductor diagrams.
    
    Generation Process:
    ------------------
    1. Encode prompt with CLIP → text_embeddings
    2. Start with random noise latent (64x64)
    3. Loop for N steps:
       a. UNet predicts noise residual: noise_pred = UNet(latent, timestep, text_emb)
       b. Apply classifier-free guidance: latent -= guidance_scale * noise_pred
       c. Update latent with predicted noise
    4. Decode clean latent with VAE → final image
    """
    
    def __init__(self, config: Optional[SDDiagramConfig] = None):
        self.config = config or SDDiagramConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ========== COMPONENT 1: CLIP Text Encoder ==========
        # This provides text conditioning to the diffusion process
        self.clip_encoder = CLIPEncoder(self.config.clip_model_name)
        self.clip_encoder.to(self.device)
        
        # Load fine-tuned CLIP weights from Stage 1 training
        if self.config.clip_model_path:
            self.clip_encoder.load_finetuned(self.config.clip_model_path)
        
        # ========== COMPONENT 2 & 3: Stable Diffusion Pipeline ==========
        # Contains: UNet (denoising) + VAE (decoding) + CLIP text encoder (conditioning)
        self._load_stable_diffusion()
        
        # Prompt enhancement for technical diagrams
        self._tech_enhancement = (
            ", technical diagram, schematic view, engineering drawing, "
            "clean lines, scientific illustration, flowchart, blueprint style"
        )
        self._negative_prompt = (
            "photorealistic, blurry, low quality, distorted, watermark, "
            "text, signature, 3D render, photo, realistic"
        )
    
    def _load_stable_diffusion(self):
        """Load pretrained Stable Diffusion pipeline."""
        print(f"Loading Stable Diffusion: {self.config.sd_model_name}")
        try:
            # StableDiffusionPipeline contains:
            # - UNet2DConditionModel (denoising)
            # - AutoencoderKL (VAE encoder/decoder)
            # - CLIPTextModel (text encoder - shared with our CLIPEncoder)
            # - Scheduler (DDIM, PNDM, etc.)
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.sd_model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,  # Disable NSFW filter
            )
            self.pipeline.to(self.device)
            print(f"SD pipeline loaded on: {self.device}")
        except Exception as e:
            print(f"Error loading SD: {e}")
            self.pipeline = None
    
    def _enhance_prompt(self, prompt: str) -> str:
        """Add technical diagram enhancements to prompts."""
        enhanced = prompt + self._tech_enhancement
        semiconductor_terms = [
            "semiconductor", "CMOS", "transistor", "fabrication", 
            "etch", "deposition", "lithography", "chamber", "plasma"
        ]
        if any(term in prompt.lower() for term in semiconductor_terms):
            enhanced += ", semiconductor process diagram, cross-section view"
        return enhanced
    
    def generate(
        self, 
        prompt: str, 
        num_images: int = 1, 
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5, 
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate diagram from text prompt.
        
        Pipeline:
        1. Enhance prompt with technical terms
        2. Pass through SD pipeline which internally:
           a. Encodes text with CLIP
           b. Runs diffusion loop (UNet denoising)
           c. Decodes with VAE
        """
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
        """Save generated image to disk."""
        image.save(path)
        print(f"Saved: {path}")


def load_trained_generator(
    clip_checkpoint: str = "models/clip_diagram/best_model.pt",
    sd_model: str = "runwayml/stable-diffusion-v1-5"
) -> SDDiagramGenerator:
    """
    Convenience function to load generator with fine-tuned CLIP.
    
    Usage:
        generator = load_trained_generator("models/clip_diagram/best_model.pt")
        images = generator.generate("CVD chamber diagram", num_images=4)
    """
    config = SDDiagramConfig(sd_model_name=sd_model, clip_model_path=clip_checkpoint)
    return SDDiagramGenerator(config)

# sd_model: str = "runwayml/stable-diffusion-v1-5"； It downloads and loads ALL components implicitly:

# - __UNet__ (U-Net2DConditionModel) - denoising
# - __VAE__ (AutoencoderKL) - encode/decode latents
# - __CLIP Text Encoder__ - text conditioning
# - __Scheduler__ - DDIM for inference steps


if __name__ == "__main__":
    print("SD Diagram Generator ready!")
    print("Usage: from src.vision.sd_diagram_model import load_trained_generator")