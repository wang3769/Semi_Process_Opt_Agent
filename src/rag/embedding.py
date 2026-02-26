"""
Embedding Model Module
=====================

Handles text embeddings using sentence-transformers.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Generate embeddings for text chunks using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
            
            if device is None:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"Loaded embedding model '{model_name}' on {device}")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    
    def embed_documents(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()


def create_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> EmbeddingModel:
    """
    Convenience function to create an embedding model.
    
    Args:
        model_name: Name of the sentence-transformer model
        device: Device to use
        
    Returns:
        EmbeddingModel instance
    """
    return EmbeddingModel(model_name=model_name, device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the embedding model
    model = create_embedding_model()
    
    print(f"Embedding dimension: {model.get_embedding_dimension()}")
    
    # Test embedding
    test_texts = [
        "Semiconductor yield prediction using machine learning",
        "Wafer defect classification with deep learning",
        "Root cause analysis for process variation",
    ]
    
    embeddings = model.embed_documents(test_texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
    
    # Test query
    query = "defect detection in semiconductor manufacturing"
    query_embedding = model.embed_query(query)
    print(f"\nQuery embedding: {len(query_embedding)} dimensions")
