"""
Text Splitter Module
===================

Splits documents into semantic chunks for embedding.
"""

from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


class TextSplitter:
    """Split documents into semantic chunks."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            separators: List of separator patterns (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n\n",  # Large section breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence boundaries
            ", ",      # Clause boundaries
            " ",       # Word boundaries
        ]
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            
        Returns:
            List of document chunks with metadata
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self.split_text(doc['content'], doc.get('metadata', {}))
            chunks.extend(doc_chunks)
        
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def split_text(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """
        Split a single text into chunks.
        
        Args:
            text: Text content to split
            metadata: Document metadata to propagate
            
        Returns:
            List of text chunks with metadata
        """
        if not text:
            return []
        
        metadata = metadata or {}
        
        # First, try to split by larger sections
        sections = self._split_by_separators(text, self.separators[:2])
        
        chunks = []
        for section in sections:
            # If section is small enough, keep it
            if len(section) <= self.chunk_size:
                chunks.append({
                    'content': section.strip(),
                    'metadata': metadata.copy()
                })
            else:
                # Split by smaller separators
                sub_chunks = self._split_by_separators(
                    section, 
                    self.separators[2:]
                )
                
                current_chunk = ""
                for sub_chunk in sub_chunks:
                    if len(current_chunk) + len(sub_chunk) <= self.chunk_size:
                        current_chunk += sub_chunk
                    else:
                        if current_chunk:
                            chunks.append({
                                'content': current_chunk.strip(),
                                'metadata': metadata.copy()
                            })
                        
                        # Start new chunk with overlap
                        if self.chunk_overlap > 0 and current_chunk:
                            overlap_text = current_chunk[-self.chunk_overlap:]
                            current_chunk = overlap_text + sub_chunk
                        else:
                            current_chunk = sub_chunk
                
                # Don't forget the last chunk
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': metadata.copy()
                    })
        
        return chunks
    
    def _split_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """Split text by the first available separator."""
        if not separators:
            return [text]
        
        separator = separators[0]
        
        if separator not in text:
            # Try next separator
            if len(separators) > 1:
                return self._split_by_separators(text, separators[1:])
            return [text]
        
        # Split by current separator
        parts = text.split(separator)
        
        # If we got too many small parts, try next separator
        if len(parts) > len(text) / 10 and len(separators) > 1:
            return self._split_by_separators(text, separators[1:])
        
        # Combine parts that are too small
        result = []
        current = ""
        
        for part in parts:
            if len(current) + len(separator) + len(part) <= self.chunk_size:
                current += separator + part
            else:
                if current:
                    result.append(current)
                current = part
        
        if current:
            result.append(current)
        
        # If result has only one item, try next separator
        if len(result) == 1 and len(separators) > 1:
            return self._split_by_separators(text, separators[1:])
        
        return result


def split_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Convenience function to split documents.
    
    Args:
        documents: List of documents
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    # Test the splitter
    logging.basicConfig(level=logging.INFO)
    
    test_doc = {
        'content': """Introduction to Semiconductor Manufacturing
            
Semiconductor manufacturing is a complex process that involves multiple stages.
Each stage is critical to the final product quality.

## Wafer Processing

The wafer processing stage includes:
- Cleaning
- Diffusion
- Lithography
- Etching
- Ion Implantation

Each process step can introduce defects that affect yield.

## Defect Analysis

Defects can be classified into several categories:
1. Particle defects
2. Pattern defects
3. Contamination defects
4. Mechanical defects

Root cause analysis requires understanding the relationship between 
defect patterns and process parameters.""",
        'metadata': {'source': 'test.md', 'filename': 'test.md'}
    }
    
    splitter = TextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_text(test_doc['content'], test_doc['metadata'])
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk['content'])} chars) ---")
        print(chunk['content'][:100] + "...")
