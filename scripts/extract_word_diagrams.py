"""
Extract Images and Captions from Word Document
==============================================

This script extracts images from a Word document where each image is 
followed by its explanation text. It creates image-caption pairs
for training the CLIP diagram model.

Uses POSITION-BASED matching: each image is paired with the text that 
appears after it in the document (top-down order).

Usage:
    python scripts/extract_word_diagrams.py --input docs/process_flow.docx --output data/diagrams/train

Requirements:
    pip install python-docx pillow requests
"""

import os
import argparse
import re
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

# For Word document processing
try:
    from docx import Document
    from docx.document import Document as DocDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("Warning: python-docx not installed. Install with: pip install python-docx")

# For image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# For extracting images from docx
try:
    import zipfile
    ZIP_AVAILABLE = True
except ImportError:
    ZIP_AVAILABLE = False

# For API calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# MiniMax Translation Module
# =============================================================================

class MiniMaxTranslator:
    """MiniMax API translator with semiconductor domain expertise."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.minimax.io/v1", model: str = "minimax-m2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
    def translate(self, text: str, source_lang: str = "zh", target_lang: str = "en", context: str = "") -> str:
        """Translate text using MiniMax API."""
        if text is None or not str(text).strip():
            return text if text else ""
        
        prompt = self._build_translation_prompt(text, source_lang, target_lang, context)
        
        try:
            result = self._call_api(prompt)
            return result if result else text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def _build_translation_prompt(self, text: str, source_lang: str, target_lang: str, context: str) -> str:
        """Build translation prompt with semiconductor context."""
        
        source_name = "Chinese" if source_lang == "zh" else source_lang
        target_name = "English" if target_lang == "en" else target_lang
        
        system_prompt = f"""You are an expert translator specializing in semiconductor manufacturing and integrated circuit fabrication. 
Translate the following text from {source_name} to {target_name}.
Maintain technical accuracy and use standard semiconductor industry terminology.

Technical terms:
- wafer, process, oxidation, lithography, etching, deposition
- ion implantation, CMP, metallization, diffusion, CVD, PVD
- plasma, epitaxy, MOSFET, CMOS, transistor, gate, source, drain
- channel, doping, threshold voltage, yield, defect, photoresist
- furnace, monocrystalline silicon, polysilicon, silicon wafer

Translate accurately: {text}"""
        
        return system_prompt
    
    def _call_api(self, prompt: str) -> str:
        """Call MiniMax API."""
        url = f"{self.base_url}/text/chatcompletion_v2"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "minimax-m2.5",
            "messages": [
                {"role": "system", "content": "You are a professional technical translator specializing in semiconductor manufacturing. Translate accurately."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Debug: print response structure
        if "choices" in result and result["choices"]:
            content = result["choices"][0].get("message", {}).get("content")
            if content:
                return content.strip()
        
        if result.get("reply"):
            return str(result["reply"]).strip()
        
        # Log the full response for debugging
        print(f"  Warning: Unexpected response: {result}")
        return None
    
    def translate_batch(self, texts: List[str], source_lang: str = "zh", target_lang: str = "en", 
                      context: str = "", delay: float = 0.5) -> List[str]:
        """Translate multiple texts with rate limiting."""
        translations = []
        
        for i, text in enumerate(texts):
            print(f"  Translating {i+1}/{len(texts)}...")
            
            if text and str(text).strip():
                translation = self.translate(text, source_lang, target_lang, context)
                translations.append(translation)
            else:
                translations.append(str(text) if text else "")
            
            if i < len(texts) - 1:
                time.sleep(delay)
        
        return translations


# =============================================================================
# Position-Based Image Extraction Module
# =============================================================================

@dataclass
class DocumentElement:
    """Represents a document element (image or text) with position."""
    element_type: str  # 'image' or 'text'
    position: int  # Order in document
    data: any  # Image bytes or text string
    image_format: str = ""


def extract_elements_with_positions(docx_path: str) -> Tuple[List[DocumentElement], List[DocumentElement]]:
    """
    Extract images and text from document with their positions.
    
    Strategy:
    1. Each image has a position (0, 1, 2, ...)
    2. Each text paragraph has a position
    3. For each image, find the text that comes AFTER it
    4. Use that text as the caption
    
    Returns:
        (images, texts) - both sorted by position
    """
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx is required")
    
    # Extract images with positions
    images = []
    with zipfile.ZipFile(docx_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        media_files = [f for f in file_list if f.startswith('word/media/')]
        
        for pos, img_path in enumerate(media_files):
            try:
                image_data = zip_ref.read(img_path)
                ext = os.path.splitext(img_path)[1].lower()
                
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                    img_format = ext[1:]
                    if img_format == 'jpeg':
                        img_format = 'jpeg'
                    
                    images.append(DocumentElement(
                        element_type='image',
                        position=pos,
                        data=image_data,
                        image_format=img_format
                    ))
            except Exception as e:
                print(f"Warning: Could not extract {img_path}: {e}")
    
    # Extract text with positions
    # Word doesn't give exact positions, so we use paragraph order
    doc = Document(docx_path)
    texts = []
    
    for pos, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:  # Only non-empty paragraphs
            texts.append(DocumentElement(
                element_type='text',
                position=pos,
                data=text,
                image_format=""
            ))
    
    return images, texts


def match_images_with_texts(images: List[DocumentElement], texts: List[DocumentElement], 
                           max_text_before: int = 2,
                           max_text_after: int = 3) -> List[Tuple[DocumentElement, str]]:
    """
    Match images with text based on position.
    
    For each image, find text from BOTH before AND after it.
    Combines: [text before] + [text after] = full caption
    
    Args:
        images: List of image elements
        texts: List of text elements  
        max_text_before: Max paragraphs to include BEFORE image
        max_text_after: Max paragraphs to include AFTER image
        
    Returns:
        List of (image, caption) tuples
    """
    pairs = []
    
    # Sort by position
    texts_sorted = sorted(texts, key=lambda x: x.position)
    
    for img_idx, img in enumerate(images):
        text_before = []
        text_after = []
        
        # Find texts BEFORE the image
        for text in texts_sorted:
            if text.position < img.position:
                if len(text.data) > 5:  # Skip very short texts
                    text_before.append(text.data)
        
        # Find texts AFTER the image
        for text in texts_sorted:
            if text.position > img.position:
                if len(text.data) > 5:
                    text_after.append(text.data)
        
        # Take last N texts before (most relevant - closest to image)
        text_before = text_before[-max_text_before:] if text_before else []
        
        # Take first N texts after (most relevant - closest to image)  
        text_after = text_after[:max_text_after] if text_after else []
        
        # Combine: text_before + text_after
        # Usually the explanation is AFTER the image, so weight that more
        combined_parts = []
        
        if text_before:
            combined_parts.append(" ".join(text_before))
        
        if text_after:
            combined_parts.append(" ".join(text_after))
        
        if combined_parts:
            best_caption = " | ".join(combined_parts)
        else:
            best_caption = f"Semiconductor process diagram {img_idx + 1}"
        
        pairs.append((img, best_caption))
    
    return pairs


def clean_caption(text: str, max_length: int = 500) -> str:
    """Clean and truncate caption text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common non-content prefixes
    text = re.sub(r'^(图\d+|Figure \d+|图\d+[:\.\s]+)', '', text, flags=re.IGNORECASE)
    # Limit length
    text = text[:max_length]
    return text.strip()


# =============================================================================
# Save Functions
# =============================================================================

def save_pairs(pairs: List[Tuple[DocumentElement, str]], output_dir: str, 
               save_original: bool = False) -> int:
    """Save image-caption pairs to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    for i, (img, caption) in enumerate(pairs):
        # Save image
        img_filename = f"image_{i+1:04d}.{img.image_format}"
        img_path = output_path / img_filename
        with open(img_path, 'wb') as f:
            f.write(img.data)
        
        # Save caption
        caption_filename = f"image_{i+1:04d}.txt"
        caption_path = output_path / caption_filename
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption)
        
        saved_count += 1
        print(f"Saved: {img_filename}")
        print(f"  Caption: {caption[:80]}...")
    
    return saved_count


# =============================================================================
# Main
# =============================================================================

def main():
    # Get default paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_docx_dir = project_root / "data" / "docx_library"
    default_output_dir = project_root / "data" / "diagrams"
    
    parser = argparse.ArgumentParser(
        description="Extract images and captions from Word document (position-based matching)"
    )
    
    # Input/Output
    parser.add_argument("--input", type=str, default=str(default_docx_dir), 
                       help=f"Input Word document or directory (default: {default_docx_dir})")
    parser.add_argument("--output", type=str, default=str(default_output_dir), 
                       help=f"Output directory (default: {default_output_dir})")
    
    # MiniMax API
    # TODO: Replace with your MiniMax API key before running
    # Get your API key from: https://platform.minimax.io/
    parser.add_argument("--api_key", type=str, default="API KEY HERE", 
                       help="MiniMax API key")
    parser.add_argument("--api_base", type=str, default="https://api.minimax.io/v1")
    parser.add_argument("--model", type=str, default="minimax-m2.5")
    
    # Options
    parser.add_argument("--no_translate", action="store_true", help="Skip translation")
    parser.add_argument("--source_lang", type=str, default="zh")
    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--save_original", action="store_true", help="Save original captions")
    parser.add_argument("--max_text_before", type=int, default=2, 
                       help="Max paragraphs BEFORE image to include")
    parser.add_argument("--max_text_after", type=int, default=3, 
                       help="Max paragraphs AFTER image to include")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not DOCX_AVAILABLE:
        print("Error: python-docx required")
        print("Install: pip install python-docx")
        return
    
    if not REQUESTS_AVAILABLE:
        print("Error: requests required")
        print("Install: pip install requests")
        return
    
    # Check input - handle both file and directory
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path not found: {args.input}")
        return
    
    # If input is a directory, look for .docx files
    docx_files = []
    if input_path.is_dir():
        docx_files = list(input_path.glob("*.docx"))
        if not docx_files:
            print(f"Error: No .docx files found in {args.input}")
            return
        print(f"Found {len(docx_files)} .docx files in directory")
    else:
        # Single file
        if input_path.suffix.lower() != '.docx':
            print(f"Error: Input file must be a .docx file")
            return
        docx_files = [input_path]
    
    # Extract elements from all docx files
    try:
        all_images = []
        all_texts = []
        
        for docx_file in docx_files:
            print(f"\nProcessing: {docx_file}")
            
            images, texts = extract_elements_with_positions(str(docx_file))
            all_images.extend(images)
            all_texts.extend(texts)
            print(f"  Found {len(images)} images, {len(texts)} text paragraphs")
        
        images = all_images
        texts = all_texts
        
        print(f"\nTotal: Found {len(images)} images")
        print(f"Total: Found {len(texts)} text paragraphs")
        
        if not images:
            print("No images found!")
            return
        
        # Match images with text
        print("\nMatching images with text (position-based)...")
        pairs = match_images_with_texts(images, texts, args.max_text_before, args.max_text_after)
        
        # Clean captions
        cleaned_pairs = []
        for img, caption in pairs:
            cleaned = clean_caption(caption)
            cleaned_pairs.append((img, cleaned))
            print(f"  Image {img.position}: {cleaned[:60]}...")
        
        # Translate if API key provided
        if not args.no_translate and args.api_key:
            print(f"\n{'='*60}")
            print("Translating captions to English using MiniMax...")
            print(f"{'='*60}")
            
            translator = MiniMaxTranslator(args.api_key, args.api_base, args.model)
            
            captions = [caption for _, caption in cleaned_pairs]
            translations = translator.translate_batch(
                captions,
                source_lang=args.source_lang,
                target_lang=args.target_lang
            )
            
            cleaned_pairs = [(img, trans) for (img, _), trans in zip(cleaned_pairs, translations)]
            print("\nTranslation complete!")
        
        elif args.no_translate:
            print("\nSkipping translation")
        else:
            print("\nNo API key - using original captions")
        
        # Save pairs
        print(f"\n{'='*60}")
        print(f"Saving to: {args.output}")
        print(f"{'='*60}")
        
        saved = save_pairs(cleaned_pairs, args.output, args.save_original)
        
        print(f"\nDone! Saved {saved} image-caption pairs")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
