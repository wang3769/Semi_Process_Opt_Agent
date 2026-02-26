"""
Build RAG Knowledge Base Script
============================

Loads documents, generates embeddings, and builds the vector store.

Usage:
    python scripts/build_rag.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.document_loader import DocumentLoader
from src.rag.text_splitter import TextSplitter
from src.rag.embedding import EmbeddingModel
from src.rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_knowledge_base():
    """Create sample knowledge base documents."""
    docs = []
    
    # Document 1: Defect Patterns
    docs.append({
        'content': """Wafer Defect Pattern Classification Guide
============================================

The WM-811K dataset contains 9 major defect patterns:

1. Center (C): Circular defects in the center region
   - Root cause: Film deposition issues, center-only process variations
   - Common in: PVD, CVD processes
   - Action: Check deposition uniformity, temperature distribution

2. Donut (D): Ring-shaped defect in middle region
   - Root cause: Edge bead removal issues, uneven coating
   - Common in: Spin coating, lithography
   - Action: Verify spin speed uniformity, edge bead removal

3. Edge-Ring (ER): Ring at wafer edge
   - Root cause: Edge exposure, clamp marks
   - Common in: Lithography, ion implantation
   - Action: Check chuck condition, exposure alignment

4. Edge-Loc (EL): Localized defects at edge
   - Root cause: Particle contamination at edge
   - Common in: Transport, loading
   - Action: Clean load ports, check wafer handling

5. Local (L): Scattered local defects
   - Root cause: Random particle contamination
   - Common in: All processes
   - Action: Cleanroom optimization, equipment maintenance

6. Random (R): Randomly distributed defects
   - Root cause: Systemic contamination, process drift
   - Common in: Wet processes, cleaning
   - Action: Process audit, chemical quality

7. Scratch (S): Linear scratch patterns
   - Root cause: Wafer handling, mechanical damage
   - Common in: CMP, wet etch
   - Action: Check robot arms, cassette handling

8. Near-Full (NF): Almost entire wafer affected
   - Root cause: Major process failure, complete contamination
   - Common in: Equipment failure
   - Action: Emergency shutdown, equipment repair

9. None (OK): No defects detected
   - Wafer passed quality control
""",
        'metadata': {'source': 'defect_patterns.md', 'filename': 'defect_patterns.md', 'type': 'markdown'}
    })
    
    # Document 2: SECOM Feature Guide
    docs.append({
        'content': """SECOM Semiconductor Data Feature Guide
===========================================

The SECOM dataset contains 590 sensor measurements:

Key Feature Categories:
- Optical measurements (spectroscopy)
- Thickness measurements (ellipsometry)  
- Surface roughness (AFM/STM)
- Electrical properties (IV, CV)
- Particle counts (laser inspection)

Important Features for Yield Prediction:

1. Process Parameter Features (Features 1-100)
   - Temperature readings
   - Pressure readings
   - Gas flow rates
   - Power levels

2. Sensor Array Features (Features 101-300)
   - Multi-wavelength reflectance
   - Phase measurements
   - Intensity variations

3. Quality Metrics (Features 301-400)
   - Film thickness uniformity
   - Refractive index
   - Stress measurements

4. Defect Density Features (Features 401-500)
   - Particle counts
   - Defect sizes
   - Defect locations

5. Electrical Features (Features 501-590)
   - Sheet resistance
   - Carrier mobility
   - Doping concentration

Root Cause Indicators:
- High feature variance → Process instability
- Specific feature outliers → Equipment issues
- Feature drift over time → Process degradation
- Correlated features → Shared root cause
""",
        'metadata': {'source': 'secom_features.md', 'filename': 'secom_features.md', 'type': 'markdown'}
    })
    
    # Document 3: Root Cause Analysis Process
    docs.append({
        'content': """Semiconductor Manufacturing Root Cause Analysis Framework
======================================================

Step 1: Data Collection
- Gather metrology data (SECOM features)
- Collect wafer maps (defect images)
- Review equipment logs
- Note process parameters

Step 2: Pattern Identification
- Use WM-811K model to classify defect patterns
- Analyze spatial distribution of defects
- Identify temporal patterns (time of occurrence)

Step 3: Feature Correlation
- Use SECOM model to identify important features
- Correlate defect patterns with sensor readings
- Look for anomalies in key process parameters

Step 4: Hypothesis Generation
- Based on defect pattern → likely process issues
- Based on feature importance → probable root causes
- Consider equipment history and maintenance

Step 5: Verification
- Review RAG knowledge base for similar cases
- Design confirmation experiments
- Implement corrective actions

Common Root Causes by Defect Pattern:

Center Donut → Temperature gradient, deposition uniformity
Edge Ring → Edge process issues, clamp marks
Local Random → Particle contamination, cleaning issues
Scratches → Handling damage, mechanical issues
Full Wafer → Major equipment failure, chemical contamination

Tools Used:
- Vision Model: WM-811K CNN classifier
- Tabular Model: XGBoost on SECOM data
- Knowledge Base: RAG retrieval system
- Analysis: Statistical process control
""",
        'metadata': {'source': 'rca_process.md', 'filename': 'rca_process.md', 'type': 'markdown'}
    })
    
    return docs


def build_rag(pdf_dir: str = "data/pdf_library"):
    """Build the RAG knowledge base."""
    
    logger.info("=" * 60)
    logger.info("Building RAG Knowledge Base")
    logger.info("=" * 60)
    
    # Step 1: Load sample knowledge base
    logger.info("\n[1/5] Creating sample knowledge base...")
    docs = create_sample_knowledge_base()
    logger.info(f"Created {len(docs)} sample documents")
    
    # Step 2: Load PDFs from library
    logger.info(f"\n[2/5] Loading PDFs from {pdf_dir}...")
    loader = DocumentLoader()
    pdf_docs = loader.load_directory(pdf_dir)
    
    if pdf_docs:
        docs.extend(pdf_docs)
        logger.info(f"Loaded {len(pdf_docs)} PDF documents")
    else:
        logger.info("No PDF documents found in library")
    
    # Step 3: Split documents into chunks
    logger.info("\n[3/5] Splitting documents into chunks...")
    splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Step 4: Generate embeddings
    logger.info("\n[4/5] Generating embeddings...")
    embedding_model = EmbeddingModel()
    
    texts = [chunk['content'] for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts, show_progress=True)
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Step 5: Store in vector database
    logger.info("\n[5/5] Storing in vector database...")
    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)
    
    # Print summary
    info = vector_store.get_collection_info()
    logger.info("\n" + "=" * 60)
    logger.info("RAG Knowledge Base Built Successfully!")
    logger.info("=" * 60)
    logger.info(f"Collection: {info['name']}")
    logger.info(f"Documents: {info['count']}")
    logger.info(f"Location: {info['persist_directory']}")
    
    return vector_store


if __name__ == "__main__":
    build_rag()
