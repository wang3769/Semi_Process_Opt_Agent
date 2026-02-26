"""
Document Loader Module
=====================

Handles loading documents from various sources:
- PDF files
- Text files
- Markdown files
"""

from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from various file formats."""
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.txt', '.md', '.markdown'}
    
    def load_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of documents with content and metadata
        """
        docs = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return docs
        
        for file_path in dir_path.rglob('*'):
            if file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self.load_file(str(file_path))
                    if doc:
                        docs.append(doc)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        return docs
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document with content and metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        ext = path.suffix.lower()
        
        if ext == '.pdf':
            return self._load_pdf(path)
        elif ext in {'.txt', '.md', '.markdown'}:
            return self._load_text(path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return None
    
    def _load_pdf(self, path: Path) -> Dict[str, Any]:
        """Load PDF file using pdfplumber."""
        try:
            import pdfplumber
            
            text_content = []
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"--- Page {page_num} ---\n{page_text}")
            
            return {
                'content': '\n\n'.join(text_content),
                'metadata': {
                    'source': str(path),
                    'filename': path.name,
                    'type': 'pdf',
                }
            }
        except ImportError:
            logger.error("pdfplumber not installed. Run: pip install pdfplumber")
            return None
    
    def _load_text(self, path: Path) -> Dict[str, Any]:
        """Load text/markdown file."""
        try:
            content = path.read_text(encoding='utf-8')
            return {
                'content': content,
                'metadata': {
                    'source': str(path),
                    'filename': path.name,
                    'type': path.suffix[1:],
                }
            }
        except Exception as e:
            logger.error(f"Failed to read text file {path}: {e}")
            return None


def load_documents(data_dir: str = "data/pdf_library") -> List[Dict[str, Any]]:
    """
    Convenience function to load all documents from the PDF library.
    
    Args:
        data_dir: Directory containing documents
        
    Returns:
        List of loaded documents
    """
    loader = DocumentLoader()
    docs = loader.load_directory(data_dir)
    logger.info(f"Loaded {len(docs)} documents from {data_dir}")
    return docs


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)
    docs = load_documents()
    for doc in docs:
        print(f"Loaded: {doc['metadata']['filename']} ({len(doc['content'])} chars)")
