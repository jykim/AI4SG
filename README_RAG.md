# RAG Pipeline and Chat Evaluation Interface

This document describes the Retrieval Augmented Generation (RAG) pipeline and chat evaluation interface for the Journal Chat system.

## RAG Pipeline Overview

The RAG pipeline consists of the following components:

### 1. Document Processing
- Journal entries are split into chunks using `RecursiveCharacterTextSplitter`
  - Chunk size: 1000 characters
  - Chunk overlap: 200 characters
  - Separators: ["\n\n", "\n", " ", ""]

### 2. Vector Store
- Uses ChromaDB as the vector store
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Documents are stored with metadata including:
  - Date
  - Time
  - Title
  - Emotion
  - Topic
  - Tags

### 3. Retrieval Process
- Retrieves up to 7 most relevant documents
- Uses similarity score threshold for retrieval
- Filters out duplicate content
- Includes match scores for each document

### 4. Response Generation
Two modes of operation:

#### Normal Mode
- Uses both recent entries (past 7 days) and RAG-retrieved entries
- Provides full context for comprehensive responses

#### RAG-Only Mode
- Uses only RAG-retrieved entries
- Focuses on semantic similarity for responses
- Useful for testing RAG effectiveness

## Chat Evaluation Interface

The chat evaluation interface (`chat_eval.py`) provides a dashboard for testing and analyzing the RAG system.

### Features

1. **Chat Interface**
   - Real-time chat with the AI assistant
   - Message history display
   - Input area with send button

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
     - Formatted display of content
   - Response Generation Info
     - Model used
     - Tokens used
     - Processing time
   - Cache Information
     - Cache hit status
     - Cache key

3. **Mode Toggles**
   - Debug Info Toggle: Shows/hides detailed debug information
   - RAG-Only Mode Toggle: Switches between normal and RAG-only modes

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

3. **Response Metrics**
   - Total processing time
   - Token usage
   - Cache hit rate

## Usage

1. Start the chat evaluation interface:
   ```bash
   python chat_eval.py
   ```

2. Access the interface at `http://localhost:8051`

3. Use the toggles to:
   - Show/hide debug information
   - Switch between normal and RAG-only modes

4. Monitor the debug panel for:
   - RAG performance metrics
   - Retrieved document details
   - Response generation statistics

## Debugging Tips

1. **Slow Retrieval**
   - Check the number of documents in the index
   - Monitor retrieval time in the debug panel
   - Consider adjusting chunk size/overlap

2. **Duplicate Documents**
   - Check the number of unique documents vs. retrieved documents
   - Adjust chunk overlap if too many duplicates

3. **Response Quality**
   - Compare responses between normal and RAG-only modes
   - Check match scores of retrieved documents
   - Monitor token usage and processing time

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
   - Implement hybrid search (keyword + semantic)
   - Add document reranking
   - Support for multiple embedding models 