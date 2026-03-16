"""
Generate QA: Create SFT Training Data from Book Chunks
====================================================

Uses MiniMax API (via OpenAI SDK) to generate Q&A pairs from extracted book content.

Usage:
    # First run scrape_book.py to extract text chunks
    python scripts/scrape_book.py
    
    # Then generate QA pairs
    python scripts/generate_qa.py
    
    # Or run both together
    python scripts/generate_qa.py --run-scrape
"""

import os
import json
import time
import argparse
from typing import List, Dict, Optional

# =============================================================================
# Configuration
# =============================================================================

# MiniMax API Configuration
# Your API key from MiniMax dashboard
MINIMAX_API_KEY = "xxx"

# MiniMax API base URL (via OpenAI SDK)
#MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
MINIMAX_BASE_URL = "https://api.minimax.io/v1"

# Model - MiniMax models
MODEL = "MiniMax-M2.5"

# Input/Output paths
BOOK_CHUNKS_FILE = "data/processed/book/book_chunks.json"
OUTPUT_FILE = "data/processed/llm/book_sft_train.jsonl"

# QA Generation settings
QA_PER_CHUNK = 2  # Number of QA pairs per chunk
MAX_RETRIES = 3  # Retry failed requests
RETRY_DELAY = 5  # Seconds between retries


# =============================================================================
# OpenAI SDK for MiniMax
# =============================================================================

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    print("Warning: openai package not installed. Install: pip install openai")


def get_openai_client():
    """Create OpenAI client configured for MiniMax."""
    if not OPENAI_SDK_AVAILABLE:
        raise ImportError("Please install openai: pip install openai")
    
    return OpenAI(
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL
    )


# =============================================================================
# QA Prompt Templates
# =============================================================================

QA_SYSTEM_PROMPT = """You are an expert in semiconductor manufacturing and process control.
Generate high-quality question-answer pairs from the given text about:
- Root cause analysis methodologies
- Statistical process control (SPC)
- Yield improvement techniques
- Equipment troubleshooting
- Defect detection and classification
- Process capability and variation
- Fault detection and diagnosis

The Q&A pairs should be clear, accurate, and suitable for training a domain-specific LLM."""


QA_USER_PROMPT_TEMPLATE = """Based on the following text from "Fundamentals of Semiconductor Manufacturing and Process Control":

---
{context}
---

Generate {num_qa} question-answer pairs. Each pair should:
1. Be based on the information in the text
2. Test understanding of semiconductor process control concepts
3. Include both the question and a detailed answer

Respond in JSON format:
```json
[
  {{
    "instruction": "Question text here?",
    "input": "",
    "output": "Detailed answer here."
  }},
  ...
]
```"""


# =============================================================================
# MiniMax API Functions (OpenAI SDK)
# =============================================================================

def call_minimax_api(prompt: str, system_prompt: str = QA_SYSTEM_PROMPT) -> Optional[str]:
    """Call MiniMax API to generate text using OpenAI SDK."""
    
    client = get_openai_client()
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    return None


def parse_qa_response(response: str) -> List[Dict]:
    """Parse QA pairs from API response."""
    
    # Try to extract JSON from response
    # Handle markdown code blocks
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    elif "```" in response:
        response = response.split("```")[1].split("```")[0]
    
    try:
        qa_pairs = json.loads(response.strip())
        return qa_pairs
    except json.JSONDecodeError:
        # Try to find JSON array in response
        import re
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            try:
                qa_pairs = json.loads(match.group())
                return qa_pairs
            except:
                pass
        
        print(f"Failed to parse QA response: {response[:200]}...")
        return []


# =============================================================================
# QA Generation
# =============================================================================

def load_chunks() -> List[Dict]:
    """Load extracted book chunks."""
    
    if not os.path.exists(BOOK_CHUNKS_FILE):
        print(f"ERROR: Chunks file not found: {BOOK_CHUNKS_FILE}")
        print("Please run scrape_book.py first!")
        return []
    
    with open(BOOK_CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def generate_qa_for_chunk(chunk: Dict) -> List[Dict]:
    """Generate QA pairs for a single chunk."""
    
    prompt = QA_USER_PROMPT_TEMPLATE.format(
        context=chunk['text'],
        num_qa=QA_PER_CHUNK
    )
    
    response = call_minimax_api(prompt)
    
    if response:
        qa_pairs = parse_qa_response(response)
        
        # Add metadata to each QA pair
        for qa in qa_pairs:
            qa['source_page'] = chunk.get('page', 0)
            qa['source_chapter'] = chunk.get('chapter', 'Unknown')
            qa['source_section'] = chunk.get('section', 'Unknown')
        
        return qa_pairs
    
    return []


def generate_all_qa(chunks: List[Dict], start_idx: int = 0, max_chunks: Optional[int] = None) -> List[Dict]:
    """Generate QA pairs for all chunks."""
    
    all_qa = []
    
    # Limit chunks if specified
    if max_chunks:
        chunks = chunks[start_idx:start_idx + max_chunks]
    else:
        chunks = chunks[start_idx:]
    
    total = len(chunks)
    print(f"Generating QA for {total} chunks...")
    
    for i, chunk in enumerate(chunks):
        idx = start_idx + i
        print(f"[{idx+1}/{total}] Processing chunk {idx} (page {chunk.get('page', '?')})...")
        
        qa_pairs = generate_qa_for_chunk(chunk)
        
        if qa_pairs:
            all_qa.extend(qa_pairs)
            print(f"  Generated {len(qa_pairs)} QA pairs")
        else:
            print(f"  Failed to generate QA")
        
        # Rate limiting - be nice to the API
        time.sleep(1)
    
    return all_qa


def save_qa(qa_pairs: List[Dict], output_file: str):
    """Save QA pairs to JSONL file."""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            # Only save the SFT fields
            sft_record = {
                "instruction": qa.get("instruction", ""),
                "input": qa.get("input", ""),
                "output": qa.get("output", "")
            }
            f.write(json.dumps(sft_record, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(qa_pairs)} QA pairs to: {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate QA from book chunks")
    parser.add_argument("--run-scrape", action="store_true", 
                        help="Run scrape_book.py first")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Maximum number of chunks to process (for testing)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting chunk index")
    args = parser.parse_args()
    
    print("=" * 60)
    print("QA Generation: MiniMax API (OpenAI SDK)")
    print("=" * 60)
    print(f"API Key: {MINIMAX_API_KEY[:15]}...")
    print(f"Base URL: {MINIMAX_BASE_URL}")
    print(f"Model: {MODEL}")
    
    # Run scrape first if requested
    if args.run_scrape:
        print("\n[1/3] Running scrape_book.py first...")
        import subprocess
        result = subprocess.run(["python", "scripts/scrape_book.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return
    
    # Load chunks
    print("\n[2/3] Loading book chunks...")
    chunks = load_chunks()
    if not chunks:
        return
    
    # Generate QA
    print("\n[3/3] Generating QA pairs...")
    qa_pairs = generate_all_qa(chunks, start_idx=args.start_idx, 
                               max_chunks=args.max_chunks)
    
    if qa_pairs:
        save_qa(qa_pairs, OUTPUT_FILE)
        
        print("\n" + "=" * 60)
        print("QA Generation Complete!")
        print("=" * 60)
        print(f"""
Generated {len(qa_pairs)} QA pairs.

Output file: {OUTPUT_FILE}

Next steps:
1. Combine with graph-based QA:
   cat {OUTPUT_FILE} data/processed/llm/graph_sft_train.jsonl > data/processed/llm/combined_sft.jsonl
   
2. Train the LLM:
   python scripts/train_llm.py --data data/processed/llm/combined_sft.jsonl
        """)
    else:
        print("No QA pairs generated. Check API key and try again.")


if __name__ == "__main__":
    main()
