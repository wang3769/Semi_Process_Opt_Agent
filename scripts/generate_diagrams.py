"""
Generate Diagrams from Text using Stable Diffusion
"""

import argparse # this is what needed for dynamic CLI-based configuration; such that you can use bash command with --args to specify different parameters without changing the code
# this is in sharp constract with hardcoding a training configuration class as in CLIP where flexibility is a bit less needed but reproducibility is critical
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.sd_diagram_model import SDDiagramGenerator, SDDiagramConfig


def generate_single(generator, prompt, args):
    print(f"\nGenerating: '{prompt}'")
    
    images = generator.generate(
        prompt=prompt,
        num_images=args.num_images,
        num_inference_steps=args.steps, #steps: How many iterations the AI takes to "denoise" the image.
        guidance_scale=args.guidance, # guidance: How strictly the AI follows your text prompt.
        seed=args.seed,
    )
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    safe_name = prompt.replace(" ", "_")[:50]
    for i, img in enumerate(images):
        filepath = output_path / f"{safe_name}_{i}.png"
        generator.save_image(img, str(filepath))
    
    return images


def generate_batch(generator, prompts, args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Generating: '{prompt}'")
        images = generator.generate(prompt, num_images=args.num_images, 
                                   num_inference_steps=args.steps,
                                   guidance_scale=args.guidance, seed=args.seed)
        
        safe_name = prompt.replace(" ", "_")[:50]
        for i, img in enumerate(images):
            generator.save_image(img, str(output_path / f"{safe_name}_{i}.png"))
    
    print(f"\nGenerated {len(prompts)} diagrams in {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate process diagrams")
    parser.add_argument("--prompt", type=str, help="Single prompt")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--prompts_file", type=str, help="File with prompts")
    parser.add_argument("--clip_checkpoint", type=str, 
                       default="models/clip_diagram/best_model.pt")
    parser.add_argument("--sd_model", type=str,
                       default="runwayml/stable-diffusion-v1-5")  # Public model
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output/diagrams")
    
    args = parser.parse_args()
    
    print("Loading generator...")
    config = SDDiagramConfig(
        sd_model_name=args.sd_model,
        clip_model_path=args.clip_checkpoint if Path(args.clip_checkpoint).exists() else None,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    )
    
    generator = SDDiagramGenerator(config)
    
    if args.prompt:
        generate_single(generator, args.prompt, args)
    else:
        print("\nDemo mode - generating sample diagrams...")
        demos = [
            "CMOS transistor cross-section with gate oxide",
            "CVD chamber with gas flow pattern",
            "Plasma etching process diagram",
        ]
        generate_batch(generator, demos, args)


if __name__ == "__main__":
    main()