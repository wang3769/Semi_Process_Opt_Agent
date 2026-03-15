"""
Scrape Book: Extract Text from PDF
==================================

Extract text from "Fundamentals of Semiconductor Manufacturing and Process Control"
with structural metadata (chapter, section, paragraph).

Usage:
    python scripts/scrape_book.py
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

# PDF extraction
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not installed. Install: pip install PyPDF2")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not installed. Install: pip install pdfplumber")


# =============================================================================
# Configuration
# =============================================================================

PDF_PATH = "data/pdf_library/Fundamentals of Semiconductor Manufacturing and Process Control G. May C. Spanos.pdf"
OUTPUT_DIR = "data/processed/book"
MIN_PARAGRAPH_LENGTH = 100  # Minimum characters for a valid paragraph
MAX_PARAGRAPH_LENGTH = 2000  # Maximum characters per chunk


# =============================================================================
# PDF Extraction Functions
# =============================================================================

def extract_with_pdfplumber(pdf_path: str) -> List[Dict]:
    """Extract text using pdfplumber (better for layout)."""
    chunks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Processing {len(pdf.pages)} pages...")
        
        current_chapter = "Unknown"
        current_section = "Unknown"
        
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text:
                continue
            
            # Try to detect chapter headers (usually all caps or numbered)
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect chapter headings (e.g., "CHAPTER 1" or "1. Introduction")
                if re.match(r'^(CHAPTER\s+\d+|^\d+\.\s+[A-Z])', line, re.IGNORECASE):
                    current_chapter = line
                    current_section = "Unknown"
                # Detect section headings
                elif re.match(r'^\d+\.\d+\s+', line) and len(line) < 100:
                    current_section = line
            
            # Extract paragraphs
            paragraphs = text.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                # Clean up the paragraph
                para = re.sub(r'\s+', ' ', para)
                para = para.replace('\n', ' ')
                
                if len(para) >= MIN_PARAGRAPH_LENGTH:
                    chunks.append({
                        "page": page_num,
                        "chapter": current_chapter,
                        "section": current_section,
                        "text": para[:MAX_PARAGRAPH_LENGTH],
                        "char_count": len(para)
                    })
    
    return chunks


def extract_with_pypdf2(pdf_path: str) -> List[Dict]:
    """Extract text using PyPDF2 (simpler)."""
    chunks = []
    
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        print(f"Processing {len(reader.pages)} pages...")
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if not text:
                continue
            
            # Split into paragraphs
            paragraphs = text.split('\n\n')
            
            current_chapter = "Unknown"
            current_section = "Unknown"
            
            for para in paragraphs:
                para = para.strip()
                para = re.sub(r'\s+', ' ', para)
                
                if len(para) >= MIN_PARAGRAPH_LENGTH:
                    chunks.append({
                        "page": page_num,
                        "chapter": current_chapter,
                        "section": current_section,
                        "text": para[:MAX_PARAGRAPH_LENGTH],
                        "char_count": len(para)
                    })
    
    return chunks


def extract_book(pdf_path: str) -> List[Dict]:
    """Extract text from PDF using best available method."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    print(f"Extracting from: {pdf_path}")
    
    if PDFPLUMBER_AVAILABLE:
        print("Using pdfplumber (better layout extraction)")
        return extract_with_pdfplumber(pdf_path)
    elif PYPDF2_AVAILABLE:
        print("Using PyPDF2")
        return extract_with_pypdf2(pdf_path)
    else:
        raise ImportError("Please install pdfplumber or PyPDF2")


# =============================================================================
# Post-processing
# =============================================================================

def filter_rca_content(chunks: List[Dict]) -> List[Dict]:
    """Filter chunks to focus on RCA-related content."""
    
    # Keywords that indicate RCA/process control content
    rca_keywords = [
        'yield', 'defect', 'fault', 'root cause', 'rca', 'troubleshooting',
        'process control', 'spc', 'statistical', 'variation', 'control chart',
        'capability', 'cpk', 'six sigma', 'quality', 'contamination',
        'equipment', 'chamber', 'maintenance', 'diagnostics', 'monitoring',
        'sensor', 'parameter', 'setpoint', 'recipe', 'process window',
        ' wafer', 'fabrication', 'semiconductor', 'integrated circuit'
    ]
    
    filtered = []
    for chunk in chunks:
        text_lower = chunk['text'].lower()
        
        # Count keyword matches
        matches = sum(1 for kw in rca_keywords if kw in text_lower)
        
        # Keep if has at least 2 keyword matches
        if matches >= 2:
            chunk['keyword_matches'] = matches
            filtered.append(chunk)
    
    print(f"Filtered {len(chunks)} chunks → {len(filtered)} RCA-related chunks")
    return filtered


def save_chunks(chunks: List[Dict], output_dir: str):
    """Save extracted chunks to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full extraction
    output_file = os.path.join(output_dir, "book_chunks.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to: {output_file}")
    
    # Save summary statistics
    stats = {
        "total_chunks": len(chunks),
        "total_chars": sum(c['char_count'] for c in chunks),
        "pages_covered": len(set(c['page'] for c in chunks)),
        "chapters": list(set(c['chapter'] for c in chunks)),
    }
    
    stats_file = os.path.join(output_dir, "extraction_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Stats: {stats}")
    return output_file


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Book Scraping: Semiconductor Process Control")
    print("=" * 60)
    
    # Check for PDF
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Please ensure the PDF is in the data/pdf_library/ directory")
        return
    
    # Extract
    print("\n[1/3] Extracting text from PDF...")
    chunks = extract_book(PDF_PATH)
    print(f"Extracted {len(chunks)} text chunks")
    
    # Filter for RCA content
    print("\n[2/3] Filtering for RCA-related content...")
    filtered_chunks = filter_rca_content(chunks)
    
    # Save
    print("\n[3/3] Saving chunks...")
    save_chunks(filtered_chunks, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)
    print(f"""
Next steps:
1. Review the extracted chunks in: {OUTPUT_DIR}/book_chunks.json
2. Run generate_qa.py to create QA pairs:
   python scripts/generate_qa.py
3. The QA pairs will be saved to: data/processed/llm/book_sft_train.jsonl
""")


if __name__ == "__main__":
    main()
