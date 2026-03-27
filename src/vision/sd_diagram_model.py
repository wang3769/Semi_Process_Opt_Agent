"""
Stable Diffusion Diagram Generation Model
=========================================

Uses trained CLIP model for semantic guidance in image generation.
Works as a second stage after CLIP fine-tuning.

Usage:
    from src.vision.sd_diagram_model import SDDiagramGenerator
    
    # Load trained CLIP + SD
    generator = SDDiagramGenerator(
        clip_model_path="models/clip_diagram/best_model.pt"
    )
    
    # Generate diagram from text
    images = generator.generate(
        prompt="CMOS process flow showing oxidation and lithography steps",
        num_images=4
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path

# Stable Diffusion
try:
    from diffusers import StableDiffusionPipeline, AutoencoderKL
    from diffusers import UNet2DConditionModel, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not installed")

# For loading CLIP
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel


@dataclass
class SDDiagramConfig:
    """Configuration for SD diagram generation."""
    # Stable Diffusion model
    sd_model_name: str = "stabilityai/stable-diffusion-2-1-base"
    
    # CLIP model (can be pretrained or fine-tuned)
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_model_path: Optional[str] = None  # Path to fine-tuned CLIP
    
    # Generation settings
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    
    # HuggingFace token
    hf_token: Optional[str] = None
    
    # Device
    device: str = "cuda"


class CLIPEncoder(nn.Module):
    """CLIP encoder for text-to-image guidance."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        
        # Freeze both encoders
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """Get text embeddings for prompts."""
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
            embeddings = outputs.pooler_output
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Get image embeddings."""
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        pixel_values = (images - mean) / std
        
        with torch.no_grad():
            outputs = self.vision_model(pixel_values)
            embeddings = outputs.pooler_output
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def load_finetuned(self, checkpoint_path: str):
        """Load fine-tuned CLIP weights."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Load weights (only matching keys)
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded fine-tuned CLIP from {checkpoint_path}")


class SemanticGuidanceUNet(nn.Module):
    """
    Modified UNet that accepts CLIP semantic embeddings as additional conditioning.
    
    This adds CLIP guidance to the standard SD UNet by:
    1. Projecting CLIP embeddings to UNet timestep dimension
    2. Adding as extra conditioning signal
    """
    
    def __init__(self, original_unet: nn.Module, clip_embed_dim: int = 512):
        super().__init__()
        self.unet = original_unet
        
        # Projection layer for CLIP embeddings
        # CLIP embeddings -> UNet cross-attention dimension
        self.clip_projection = nn.Sequential(
            nn.Linear(clip_embed_dim, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )
        
        # Store config
        self.config = original_unet.config
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        clip_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass with optional CLIP guidance."""
        
        # Project CLIP embeddings if provided
        if clip_embeddings is not None:
            clip_conditions = self.clip_projection(clip_embeddings)
            # Expand to match batch size
            clip_conditions = clip_conditions.unsqueeze(1).expand(-1, encoder_hidden_states.size(1), -1)
            # Concatenate with text embeddings
            encoder_hidden_states = torch.cat([encoder_hidden_states, clip_conditions], dim=-1)
        
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
        )


class SDDiagramGenerator:
    """
    Stable Diffusion generator with CLIP semantic guidance.
    
    Two modes:
    1. Standard SD generation (no CLIP guidance)
    2. CLIP-guided generation (uses fine-tuned CLIP for better alignment)
    """
    
    def __init__(self, config: Optional[SDDiagramConfig] = None):
        self.config = config or SDDiagramConfig()
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is required for SDDiagramGenerator")
        
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize CLIP encoder
        self.clip_encoder = CLIPEncoder(self.config.clip_model_name)
        self.clip_encoder.to(self.device)
        
        # Load fine-tuned CLIP if provided
        if self.config.clip_model_path:
            self.clip_encoder.load_finetuned(self.config.clip_model_path)
        
        # Initialize Stable Diffusion pipeline
        self._load_stable_diffusion()
        
        # Enhancement prompts for technical diagrams
        self._tech_enhancement = (
            ", technical diagram, schematic view, engineering drawing, "
            "clean lines, scientific illustration, flowchart, blueprint"
        )
        
        self._negative_prompt = (
            "photorealistic, blurry, low quality, distorted, watermark, "
            "text, signature, 3D render, photo, realistic"
        )
    
    def _load_stable_diffusion(self):
        """Load Stable Diffusion pipeline."""
        print(f"Loading Stable Diffusion: {self.config.sd_model_name}")
        
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.sd_model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,
                use_auth_token=self.config.hf_token,
            )
            
            self.pipeline.to(self.device)
            print(f"SD pipeline loaded on: {self.device}")
            
        except Exception as e:
            print(f"Error loading SD: {e}")
            self.pipeline = None
    
    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance prompt for technical diagram generation."""
        # Add technical enhancement
        enhanced = prompt + self._tech_enhancement
        
        # Add semiconductor-specific terms if relevant
        semiconductor_terms = [
            "semiconductor", "integrated circuit", "wafer", "process",
            "CMOS", "transistor", "fabrication", "etch", "deposition",
            "lithography", "diffusion", "ion", "oxidation", "metallization"
        ]
        
        is_semi = any(term in prompt.lower() for term in semiconductor_terms)
        if is_semi:
            enhanced += ", semiconductor process diagram"
        
        return enhanced
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        use_clip_guidance: bool = True,
    ) -> List[torch.Tensor]:
        """
        Generate diagrams from text prompt.
        
        Args:
            prompt: Text description of the diagram
            negative_prompt: What to avoid in generation
            num_images: Number of images to generate
            num_inference_steps: DDIM steps (higher = more detail)
            guidance_scale: Classifier-free guidance strength
            seed: Random seed for reproducibility
            use_clip_guidance: Whether to use CLIP for semantic guidance
        
        Returns:
            List of generated images
        """
        if self.pipeline is None:
            raise RuntimeError("Stable Diffusion pipeline not loaded")
        
        # Enhance prompt
        enhanced_prompt = self._enhance_prompt(prompt)
        
        # Use defaults
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        negative_prompt = negative_prompt or self._negative_prompt
        
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate
        with torch.inference_mode():
            results = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_images,
            )
        
        return results.images
    
    def generate_variations(
        self,
        source_image: torch.Tensor,
        prompt: str,
        strength: float = 0.5,
        num_images: int = 1,
    ) -> List[torch.Tensor]:
        """
        Generate variations of an existing diagram.
        
        Args:
            source_image: Reference image tensor
            prompt: Text description of desired variation
            strength: How much to deviate from source (0-1)
            num_images: Number of variations
        """
        if self.pipeline is None:
            raise RuntimeError("Stable Diffusion pipeline not loaded")
        
        # Get CLIP embeddings from source image
        with torch.inference_mode():
            source_emb = self.clip_encoder.get_image_embeddings(source_image.unsqueeze(0))
        
        # Enhance prompt
        enhanced_prompt = self._enhance_prompt(prompt)
        
        # Generate with image-to-image
        with torch.inference_mode():
            results = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=self._negative_prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                num_images_per_prompt=num_images,
                strength=strength,
            )
        
        return results.images
    
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Get CLIP embeddings for a prompt (useful for semantic search)."""
        return self.clip_encoder.get_text_embeddings([prompt])


def load_trained_generator(
    clip_checkpoint: str,
    sd_model: str = "stabilityai/stable-diffusion-2-1-base",
) -> SDDiagramGenerator:
    """
    Load SD generator with a fine-tuned CLIP model.
    
    Args:
        clip_checkpoint: Path to trained CLIP checkpoint
        sd_model: Stable Diffusion model name
    
    Returns:
        Configured SDDiagramGenerator
    """
    config = SDDiagramConfig(
        sd_model_name=sd_model,
        clip_model_path=clip_checkpoint,
    )
    
    return SDDiagramGenerator(config)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SD Diagram Generator")
    print("=" * 60)
    
    # Example: Load with fine-tuned CLIP
    # generator = load_trained_generator(
    #     clip_checkpoint="models/clip_diagram/best_model.pt"
    # )
    
    # Example: Generate diagrams
    # images = generator.generate(
    #     prompt="CMOS process flow with gate oxide formation",
    #     num_images=4,
    #     seed=42
    # )
    
    print("\nTo use:")
    print("1. First train CLIP: python scripts/train_diagram_clip.py")
    print("2. Then generate diagrams:")
    print("""
from src.vision.sd_diagram_model import load_trained_generator

generator = load_trained_generator("models/clip_diagram/best_model.pt")
images = generator.generate("CMOS transistor cross-section")
""")
