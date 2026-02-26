"""
Retriever Module
==============

Retrieves relevant documents for queries.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant documents from vector store."""
    
    def __init__(
        self,
        embedding_model,
        vector_store,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize retriever.
        
        Args:
            embedding_model: EmbeddingModel instance
            vector_store: VectorStore instance
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            n_results: Number of results to retrieve
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=filter_metadata,
        )
        
        # Format and filter results
        formatted_results = []
        
        for i in range(len(results['documents'])):
            # Convert distance to similarity score (1 - distance)
            distance = results['distances'][i]
            similarity = 1 - distance
            
            # Filter by threshold
            if similarity >= self.similarity_threshold:
                formatted_results.append({
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'id': results['ids'][i],
                    'similarity': similarity,
                    'distance': distance,
                })
        
        logger.info(f"Retrieved {len(formatted_results)} relevant documents for query: '{query[:50]}...'")
        
        return formatted_results
    
    def retrieve_with_context(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> str:
        """
        Retrieve documents and format as context string.
        
        Args:
            query: Query string
            n_results: Number of results to retrieve
            filter_metadata: Optional metadata filter
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, n_results, filter_metadata)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('filename', 'Unknown')
            context_parts.append(
                f"[{i}] Source: {source}\n"
                f"Relevance: {result['similarity']:.2%}\n"
                f"{result['content']}\n"
            )
        
        return "\n---\n".join(context_parts)


def create_retriever(
    embedding_model,
    vector_store,
    similarity_threshold: float = 0.7,
) -> Retriever:
    """
    Convenience function to create a retriever.
    
    Args:
        embedding_model: EmbeddingModel instance
        vector_store: VectorStore instance
        similarity_threshold: Minimum similarity score
        
    Returns:
        Retriever instance
    """
    return Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        similarity_threshold=similarity_threshold,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from .embedding import create_embedding_model
    from .vector_store import create_vector_store
    
    # Create components
    embedding_model = create_embedding_model()
    vector_store = create_vector_store()
    
    # Create retriever
    retriever = create_retriever(embedding_model, vector_store)
    
    # Test retrieval
    query = "What causes yield loss in semiconductor manufacturing?"
    results = retriever.retrieve(query, n_results=3)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(results)} documents:")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Similarity: {result['similarity']:.2%}")
        print(f"Source: {result['metadata'].get('filename', 'Unknown')}")
        print(f"Content: {result['content'][:150]}...")
