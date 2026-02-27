"""
Process Image Generator
======================

Text-to-image generation for semiconductor manufacturing process illustrations.
Uses Stable Diffusion with optional LoRA fine-tuning for custom process styles.

Usage:
    from src.vision.image_generator import ProcessImageGenerator
    
    gen = ProcessImageGenerator()
    image = gen.generate(
        prompt="CVD chamber cross-section showing gas flow pattern",
        negative_prompt="photorealistic, blurry, low quality",
        num_inference_steps=25,
        guidance_scale=7.5
    )
    gen.save_image(image, "output.png")
"""

import os
from typing import Optional, List, Dict
from pathlib import Path
import torch
from PIL import Image
import json

# Diffusers imports
try:
    from diffusers import (
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        AutoencoderKL
    )
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not installed. Install with: pip install diffusers")

# LoRA support
try:
    from peft import PeftModel, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. Install with: pip install peft")

# transformers for CLIP
try:
    from transformers import CLIPTextModel, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ProcessImageGenerator:
    """
    Text-to-image generator for semiconductor process illustrations.
    
    Features:
    - Pre-trained Stable Diffusion 2.1 base
    - Optional LoRA fine-tuning for custom styles
    - Text-to-image and image-to-image pipelines
    - Optimized for RTX 5090
    """
    
    # Default prompts for semiconductor processes
    DEFAULT_PROCESS_PROMPTS = {
        "cvd": "Cross-section diagram of CVD chamber, showing gas inlet, wafer susceptor, rf coil, chemical vapor deposition process, clean technical illustration, schematic view",
        "etch": "Plasma etching chamber cross-section, ion bombardment visualization, wafer surface, technical diagram, clean lines, scientific illustration",
        "lithography": "Lithography process cross-section, photoresist coating, UV light exposure, mask alignment, semiconductor wafer, technical illustration",
        "cmp": "Chemical mechanical polishing cross-section, wafer surface, polishing pad, slurry distribution, CMP process, technical diagram",
        "deposition": "Thin film deposition cross-section, layer-by-layer growth, substrate, deposited material, clean technical illustration",
        "cleaning": "Wafer cleaning process, scrubber, chemical bath, particle removal, technical diagram, clean lines",
        "ion_implantation": "Ion implantation chamber cross-section, ion beam, wafer, dopant atoms, technical illustration",
        "diffusion": "Furnace diffusion process, high temperature, wafer loading, dopant diffusion, technical diagram",
    }
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        device: str = "cuda",
        lora_path: Optional[str] = None,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
    ):
        """
        Initialize the image generator.
        
        Args:
            model_id: HuggingFace model ID for Stable Diffusion
            device: Device to use ('cuda' or 'cpu')
            lora_path: Optional path to LoRA weights
            enable_attention_slicing: Reduce VRAM usage
            enable_vae_slicing: Reduce VRAM usage
        """
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lora_path = lora_path
        self.pipeline = None
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is required. Install: pip install diffusers")
        
        # Load model
        self._load_pipeline(
            enable_attention_slicing=enable_attention_slicing,
            enable_vae_slicing=enable_vae_slicing
        )
        
        # Load LoRA if provided
        if lora_path and os.path.exists(lora_path):
            self._load_lora(lora_path)
    
    def _load_pipeline(
        self,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True
    ):
        """Load Stable Diffusion pipeline."""
        print(f"Loading Stable Diffusion model: {self.model_id}")
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,  # Disable for generation
        )
        
        # Optimize for VRAM
        if self.device == "cuda":
            if enable_attention_slicing:
                self.pipeline.enable_attention_slicing()
            if enable_vae_slicing:
                self.pipeline.enable_vae_slicing()
            
            # Enable CPU offloading if VRAM is limited
            # self.pipeline.enable_model_cpu_offload()
        
        # Use faster scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        self.pipeline = self.pipeline.to(self.device)
        print(f"Model loaded on: {self.device}")
    
    def _load_lora(self, lora_path: str):
        """Load LoRA weights onto the pipeline."""
        if not PEFT_AVAILABLE:
            print("Warning: peft not available, skipping LoRA")
            return
        
        print(f"Loading LoRA from: {lora_path}")
        
        # For full Stable Diffusion LoRA (not PEFT)
        from safetensors.torch import load_file
        
        # Load LoRA weights
        state_dict = load_file(lora_path)
        
        # Apply LoRA to UNet
        self.pipeline.unet.load_state_dict(state_dict, strict=False)
        print("LoRA loaded successfully")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> List[Image.Image]:
        """
        Generate images from text prompt.
        
        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid in the image
            width: Image width
            height: Image height
            num_inference_steps: More steps = better quality but slower
            guidance_scale: How closely to follow prompt (7-8 recommended)
            seed: Random seed for reproducibility
            num_images: Number of images to generate
            
        Returns:
            List of PIL Images
        """
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Enhance prompt for technical illustrations
        enhanced_prompt = self._enhance_prompt(prompt)
        
        # Generate
        results = self.pipeline(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt or self._get_default_negative(),
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images,
        )
        
        return results.images
    
    def generate_variation(
        self,
        source_image: Image.Image,
        prompt: str,
        strength: float = 0.5,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> List[Image.Image]:
        """
        Generate variations of an existing image.
        
        Args:
            source_image: Source PIL Image
            prompt: Text description to guide variation
            strength: How much to deviate from source (0-1)
            num_inference_steps: More steps = better quality
            guidance_scale: How closely to follow prompt
            
        Returns:
            List of PIL Images
        """
        # Create img2img pipeline if not exists
        if not hasattr(self, 'img2img_pipeline'):
            self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pipe(self.pipeline)
            self.img2img_pipeline = self.img2img_pipeline.to(self.device)
        
        results = self.img2img_pipeline(
            prompt=prompt,
            image=source_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        return results.images
    
    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance prompt with technical illustration keywords."""
        enhancement = ", technical diagram, schematic view, clean lines, scientific illustration, engineering drawing"
        
        # Check if prompt matches known process
        prompt_lower = prompt.lower()
        for key, default_prompt in self.DEFAULT_PROCESS_PROMPTS.items():
            if key in prompt_lower:
                return f"{prompt}, {default_prompt}"
        
        return prompt + enhancement
    
    def _get_default_negative(self) -> str:
        """Get default negative prompt."""
        return "photorealistic, blurry, low quality, distorted, watermark, text, signature"
    
    def save_image(self, image: Image.Image, filepath: str):
        """Save image to file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        image.save(filepath)
        print(f"Image saved to: {filepath}")
    
    def save_images(self, images: List[Image.Image], output_dir: str, prefix: str = "gen"):
        """Save multiple images to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            filepath = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
            img.save(filepath)
            print(f"Saved: {filepath}")
    
    def get_memory_usage(self) -> Dict:
        """Get current VRAM usage."""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "reserved": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            }
        return {}


# =============================================================================
# LoRA Training
# =============================================================================

def create_lora_dataset(
    output_dir: str = "data/processed/vision/lora_data",
    num_samples: int = 100
):
    """
    Create synthetic dataset for LoRA fine-tuning.
    
    Generates text prompts for semiconductor processes that can be
    used with a pretrained model (for DreamBooth-style training).
    """
    prompts = []
    
    # Process types
    processes = [
        "CVD", "etch", "lithography", "CMP", "deposition",
        "cleaning", "ion implantation", "diffusion", "oxidation"
    ]
    
    # Views
    views = [
        "cross-section", "top-down view", "side view", 
        "3D schematic", "technical diagram", "process flow"
    ]
    
    # Details
    details = [
        "gas flow pattern", "ion bombardment", "film growth",
        "particle distribution", "temperature gradient", "wafer surface",
        "chamber components", "reaction mechanism"
    ]
    
    # Generate combinations
    import random
    for _ in range(num_samples):
        process = random.choice(processes)
        view = random.choice(views)
        detail = random.choice(details)
        
        prompt = f"{process} {view}, {detail}, technical illustration, engineering drawing"
        prompts.append({
            "prompt": prompt,
            "process": process,
            "view": view,
            "detail": detail
        })
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Created {len(prompts)} prompts in {output_dir}/prompts.json")
    return prompts


def generate_example_prompts() -> Dict[str, str]:
    """
    Generate example prompts for semiconductor process illustrations.
    These can be used directly with the image generator.
    """
    return {
        # CVD Processes
        "cvd_chamber": "CVD chamber cross-section, gas inlet, wafer susceptor, rf coil, chemical vapor deposition, technical diagram",
        "cvd_layers": "Thin film deposition cross-section, multi-layer stack, Si substrate, oxide/nitride layers, technical illustration",
        
        # Etch Processes
        "etch_chamber": "Plasma etching chamber, ion bombardment visualization, wafer surface, reactive ions, technical diagram",
        "etch_profile": "anisotropic etch profile, vertical sidewall, high aspect ratio, technical illustration",
        
        # Lithography
        "litho_exposure": "Lithography exposure, UV light through mask, photoresist, pattern transfer, technical diagram",
        "litho_stack": "Lithography stack cross-section, substrate, PR, anti-reflective coating, technical illustration",
        
        # CMP
        "cmp_process": "Chemical mechanical polishing, wafer polishing, slurry distribution, pad contact, technical diagram",
        
        # General
        "wafer_map": "Wafer map, defect density plot, heat map visualization, engineering drawing",
        "process_flow": "Semiconductor process flow diagram, step-by-step, clean technical illustration",
        
        # Chamber diagrams
        "chamber_components": "Semiconductor process chamber internal components, detailed technical diagram, labeled",
        "equipment_schematic": "Processing equipment schematic, front view, side view, engineering drawing",
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Process Image Generator")
    print("=" * 60)
    
    # Check availability
    print(f"\nDiffusers available: {DIFFUSERS_AVAILABLE}")
    print(f"PEFT available: {PEFT_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Generate example prompts
    print("\n" + "=" * 60)
    print("Example Prompts")
    print("=" * 60)
    
    prompts = generate_example_prompts()
    for name, prompt in prompts.items():
        print(f"\n{name}:")
        print(f"  {prompt}")
    
    # Create dataset
    print("\n" + "=" * 60)
    print("Creating LoRA Dataset")
    print("=" * 60)
    create_lora_dataset()
