"""
Vector Store Module
=================

ChromaDB-based vector store for document retrieval.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store for semantic search."""
    
    def __init__(
        self,
        collection_name: str = "semiconductor_docs",
        persist_directory: str = "data/vector_store",
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create client
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Semiconductor process knowledge base"}
            )
            
            logger.info(f"Initialized vector store at: {persist_directory}")
            
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
    ):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document chunks
            embeddings: List of embedding vectors
            ids: Optional list of IDs (auto-generated if not provided)
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Extract texts and metadata
        texts = [doc['content'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(documents)} documents to collection")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Search results with documents, distances, and metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "distances", "metadatas", "ids"],
        )
        
        # Format results
        formatted_results = {
            'ids': results['ids'][0] if results['ids'] else [],
            'documents': results['documents'][0] if results['documents'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
        }
        
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory,
        }
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def reset(self):
        """Reset the entire database."""
        self.client.reset()
        logger.warning("Reset entire ChromaDB database")


def create_vector_store(
    collection_name: str = "semiconductor_docs",
    persist_directory: str = "data/vector_store",
) -> VectorStore:
    """
    Convenience function to create a vector store.
    
    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist the database
        
    Returns:
        VectorStore instance
    """
    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the vector store
    store = create_vector_store()
    
    # Get info
    info = store.get_collection_info()
    print(f"Collection: {info['name']}")
    print(f"Document count: {info['count']}")
    
    # If there are documents, test search
    if info['count'] > 0:
        print("\nTesting search...")
        
        # Search for something
        from .embedding import create_embedding_model
        model = create_embedding_model()
        query_emb = model.embed_query("wafer defect root cause analysis")
        
        results = store.search(query_emb, n_results=3)
        
        print(f"\nFound {len(results['documents'])} results:")
        for i, (doc, dist) in enumerate(zip(results['documents'], results['distances'])):
            print(f"\n--- Result {i+1} (distance: {dist:.4f}) ---")
            print(doc[:200] + "...")
    else:
        print("\nNo documents in collection. Run build_rag.py first.")
