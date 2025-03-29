# RAG Pipeline and Evaluation Interface

This document describes the Retrieval Augmented Generation (RAG) pipeline and evaluation interface for the Journal system.

## RAG Pipeline Overview

The RAG pipeline consists of the following components:

### 1. Configuration
- Uses `config_rag.yaml` for all RAG-specific parameters:
  - Embedding model settings
  - Text splitting parameters
  - Retrieval settings
  - Vector store configuration
  - Metadata matching fields
  - Logging configuration

### 2. Document Processing
- Journal entries are split into chunks using `RecursiveCharacterTextSplitter`
  - Chunk size: Configurable (default: 500 characters)
  - Chunk overlap: Configurable (default: 100 characters)
  - Separators: Configurable (default: ["\n\n", "\n", " ", ""])

### 3. Vector Store
- Uses ChromaDB as the vector store
- Embeddings: Configurable (default: sentence-transformers/all-MiniLM-L6-v2)
- Documents are stored with metadata including:
  - Date
  - Time
  - Title
  - Emotion
  - Topic
  - Tags

### 4. Retrieval Process
- Hybrid retrieval combining:
  - Semantic search using embeddings
  - Exact matching on metadata fields
- Configurable parameters:
  - Initial retrieval size (k)
  - Final number of documents
  - Maximum chunks per document
  - Exact match score threshold
- Filters out duplicate content
- Includes match scores and match types for each document

## Evaluation Interface

The RAG evaluation interface (`rag_eval.py`) provides a dashboard for testing and analyzing the RAG system.

### Features

1. **Query Interface**
   - Query input with search button
   - Real-time results display
   - Support for both Korean and English queries

2. **Debug Panel**
   - RAG Information
     - Query
     - Total documents in index
     - Number of retrieved documents
   - RAG Timing Metrics
     - Retrieval time
     - Processing time
     - Total RAG time
     - Number of initial documents retrieved
     - Number of unique documents after filtering
   - Retrieved Documents
     - Document details with match scores
     - Match type (exact/semantic)
     - Formatted display of content

### Performance Metrics

The interface provides detailed timing information for each component:

1. **RAG Metrics**
   - Retrieval time: Time spent getting documents from vector store
   - Processing time: Time spent filtering and formatting documents
   - Total RAG time: Combined retrieval and processing time

2. **Document Statistics**
   - Total documents in index
   - Number of documents initially retrieved
   - Number of unique documents after filtering
   - Number of exact matches vs semantic matches

3. **Chunk Statistics**
   - Average chunks per document
   - Documents with multiple chunks
   - Chunks filtered due to limits

## Usage

1. Configure RAG parameters in `config_rag.yaml`:
   ```yaml
   # Example configuration
   embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
   chunk_size: 500
   chunk_overlap: 100
   initial_k: 100
   final_k: 15
   max_chunks_per_doc: 3
   exact_match_score: 0.9
   ```

2. Start the RAG evaluation interface:
   ```bash
   python rag_eval.py
   ```

3. Access the interface at `http://localhost:8051`

4. Use the interface to:
   - Test queries in Korean or English
   - Monitor retrieval performance
   - Analyze document matching quality

## Debugging Tips

1. **Slow Retrieval**
   - Check the number of documents in the index
   - Monitor retrieval time in the debug panel
   - Consider adjusting chunk size/overlap in config

2. **Duplicate Documents**
   - Check the number of unique documents vs. retrieved documents
   - Adjust chunk overlap if too many duplicates
   - Review max_chunks_per_doc setting

3. **Match Quality**
   - Monitor exact vs semantic match ratio
   - Adjust exact_match_score if needed
   - Review exact_match_fields configuration

## Future Improvements

1. **Performance Optimization**
   - Implement batch processing for document splitting
   - Add caching for vector store operations
   - Optimize document filtering

2. **Enhanced Debugging**
   - Add visualization of document relationships
   - Implement detailed timing breakdowns
   - Add export functionality for debug data

3. **RAG Enhancements**
   - Support for multiple embedding models
   - Add document reranking
   - Implement advanced metadata filtering 