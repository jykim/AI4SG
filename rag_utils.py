#!/usr/bin/env python3
"""
RAG Utilities for Journal Chat

This module implements Retrieval Augmented Generation (RAG) functionality
for retrieving relevant past journal entries during chat interactions.
"""

import os
# Set tokenizer parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import time
import yaml

class JournalRAG:
    """RAG system for retrieving relevant journal entries"""
    
    def __init__(self, config=None):
        """Initialize the RAG system with configuration"""
        self.config = config
        if config is None:
            config_path = Path("config_rag.yaml")
            if not config_path.exists():
                raise FileNotFoundError("config_rag.yaml not found")
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif hasattr(config, 'rag_config'):  # If config is a Config object with rag_config attribute
            self.config = config.rag_config
        elif isinstance(config, dict):  # If config is already a dictionary
            self.config = config
        else:
            raise ValueError("Config must be a dictionary or a Config object with rag_config attribute")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model']
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            length_function=len,
            separators=self.config['separators']
        )
        self.vector_store = None
        self.retriever = None
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format=self.config['log_format']
        )
        
    def _create_documents(self, entries: List[Dict[str, Any]]) -> List[Document]:
        """Convert journal entries to LangChain documents"""
        documents = []
        for entry in entries:
            # Create a formatted text representation of the entry
            date = entry.get('Date', '')
            time = entry.get('Time', '')
            title = entry.get('Title', 'Untitled')
            content = entry.get('Content', '')
            emotion = entry.get('emotion', '')
            topic = entry.get('topic', '')
            tags = entry.get('Tags', '')
            
            text = f"""
            Date: {date} {time}
            Title: {title}
            Emotion: {emotion}
            Topic: {topic}
            Tags: {tags}
            
            Content:
            {content}
            """
            
            # Create metadata with string values
            metadata = {
                'date': str(date),  # Convert to string
                'time': str(time),  # Convert to string
                'title': str(title) if pd.notna(title) else 'Untitled',
                'emotion': str(emotion) if pd.notna(emotion) else '',
                'topic': str(topic) if pd.notna(topic) else '',
                'tags': str(tags) if pd.notna(tags) else ''
            }
            
            # Create document
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        return self.text_splitter.split_documents(documents)
    
    def initialize_vector_store(self, entries: List[Dict[str, Any]], persist_directory: str = "vector_store"):
        """Initialize or update the vector store with journal entries"""
        # Create documents from entries
        documents = self._create_documents(entries)
        logging.info(f"Created {len(documents)} documents from entries")
        
        # Split documents into chunks
        split_docs = self._split_documents(documents)
        logging.info(f"Split into {len(split_docs)} chunks")
        
        # Create or update vector store
        persist_directory = Path(persist_directory)
        persist_directory.mkdir(exist_ok=True)
        
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=str(persist_directory)
        )
        
        # Create simple retriever without compression
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",  # Changed from similarity_score_threshold
            search_kwargs={
                "k": 40  # Increased to get more initial results
            }
        )
        logging.info("Vector store and retriever initialized")
    
    def get_relevant_entries(self, query: str, k: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Retrieve relevant journal entries for a given query with document-level diversity.
        Combines embedding-based retrieval with exact metadata matching.
        """
        if k is None:
            k = self.config['final_k']
            
        if not self.retriever:
            logging.warning("Vector store not initialized. Returning empty list.")
            return [], {}
        
        try:
            timing_metrics = {}
            
            # Measure retrieval time
            start_time = time.time()
            logging.info(f"Invoking retriever with query: {query}")
            
            # 1. Get embedding-based results
            results = self.vector_store.similarity_search_with_score(
                query,
                k=self.config['initial_k']  # Use config value
            )
            relevant_docs = [doc for doc, _ in results]
            scores = [score for _, score in results]
            
            logging.info(f"Retrieved {len(relevant_docs)} documents from embedding search")
            logging.info(f"Score range: {min(scores):.3f} to {max(scores):.3f}")
            
            # 2. Get exact match results from metadata
            exact_match_docs = []
            exact_match_scores = []
            
            # Get all documents from vector store
            collection = self.vector_store.get()
            documents = collection['documents']
            metadatas = collection['metadatas']
            
            # Search in metadata fields
            for i, metadata in enumerate(metadatas):
                # Check configured metadata fields
                if any(query.lower() in metadata.get(field, '').lower() 
                      for field in self.config['exact_match_fields']):
                    # Create a Document object with the content and metadata
                    doc = Document(
                        page_content=documents[i],
                        metadata=metadata
                    )
                    exact_match_docs.append(doc)
                    # Use configured score for exact matches
                    exact_match_scores.append(self.config['exact_match_score'])
            
            logging.info(f"Found {len(exact_match_docs)} exact matches in metadata")
            
            # 3. Combine results
            # Create a set of document IDs to track unique documents
            seen_doc_ids = set()
            combined_docs = []
            combined_scores = []
            
            # First add exact matches (they get priority)
            for doc, score in zip(exact_match_docs, exact_match_scores):
                doc_id = f"{doc.metadata.get('date', '')}_{doc.metadata.get('title', '')}"
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    combined_docs.append(doc)
                    combined_scores.append(score)
            
            # Then add embedding-based results
            for doc, score in zip(relevant_docs, scores):
                doc_id = f"{doc.metadata.get('date', '')}_{doc.metadata.get('title', '')}"
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    combined_docs.append(doc)
                    combined_scores.append(score)
            
            logging.info(f"Combined results: {len(combined_docs)} unique documents")
            timing_metrics['retrieval_time'] = time.time() - start_time
            
            # Measure document processing time
            start_time = time.time()
            seen_content = set()
            seen_documents = set()  # Track unique documents
            chunks_per_doc = {}  # Track number of chunks per document
            relevant_entries = []
            
            # Sort documents by score
            docs_with_scores = list(zip(combined_docs, combined_scores))
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            logging.info(f"Sorted {len(docs_with_scores)} documents by score")
            
            # First pass: Get chunks from different documents
            for doc, score in docs_with_scores:
                # Get document identifier (date + title)
                doc_id = f"{doc.metadata.get('date', '')}_{doc.metadata.get('title', '')}"
                
                # Skip if we've seen this content before
                if doc.page_content in seen_content:
                    logging.info(f"Skipping duplicate content from document: {doc_id} (score: {score:.3f})")
                    continue
                
                # Skip if we already have max chunks from this document
                if doc_id in chunks_per_doc and chunks_per_doc[doc_id] >= self.config['max_chunks_per_doc']:
                    logging.info(f"Skipping additional chunks from document: {doc_id} (already have {self.config['max_chunks_per_doc']} chunks)")
                    continue
                
                seen_content.add(doc.page_content)
                chunks_per_doc[doc_id] = chunks_per_doc.get(doc_id, 0) + 1
                seen_documents.add(doc_id)
                
                # Log content preview for debugging
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logging.info(f"Adding entry: {doc_id} (chunk {chunks_per_doc[doc_id]}/3)")
                logging.info(f"Score: {score:.3f}")
                logging.info(f"Content preview: {content_preview}")
                
                entry = {
                    'Date': doc.metadata.get('date', ''),
                    'Time': doc.metadata.get('time', ''),
                    'Title': doc.metadata.get('title', ''),
                    'Content': doc.page_content,
                    'emotion': doc.metadata.get('emotion', ''),
                    'topic': doc.metadata.get('topic', ''),
                    'Tags': doc.metadata.get('tags', ''),
                    'match_score': score,
                    'doc_id': doc_id,
                    'chunk_index': chunks_per_doc[doc_id] - 1,  # 0-based index
                    'match_type': 'exact' if score == self.config['exact_match_score'] else 'semantic'  # Add match type for debugging
                }
                relevant_entries.append(entry)
                
                # Break if we have enough entries
                if len(relevant_entries) >= k * 2:
                    logging.info(f"Reached target number of entries ({k*2})")
                    break
            
            timing_metrics['processing_time'] = time.time() - start_time
            timing_metrics['total_time'] = timing_metrics['retrieval_time'] + timing_metrics['processing_time']
            timing_metrics['num_docs_retrieved'] = len(combined_docs)
            timing_metrics['num_unique_docs'] = len(seen_documents)
            timing_metrics['num_chunks'] = len(relevant_entries)
            timing_metrics['num_exact_matches'] = len(exact_match_docs)
            timing_metrics['num_semantic_matches'] = len(relevant_docs)
            
            logging.info(f"Final results: {len(relevant_entries)} entries from {len(seen_documents)} unique documents")
            logging.info(f"Filtering stats:")
            logging.info(f"- Initial documents: {len(combined_docs)}")
            logging.info(f"- Exact matches: {len(exact_match_docs)}")
            logging.info(f"- Semantic matches: {len(relevant_docs)}")
            logging.info(f"- Duplicate content filtered: {len(combined_docs) - len(seen_content)}")
            logging.info(f"- Max chunks per doc filtered: {sum(max(0, count - self.config['max_chunks_per_doc']) for count in chunks_per_doc.values())}")
            logging.info(f"- Final unique documents: {len(seen_documents)}")
            logging.info(f"- Final entries: {len(relevant_entries)}")
            logging.info(f"- Chunks per document: {chunks_per_doc}")
            return relevant_entries, timing_metrics
            
        except Exception as e:
            logging.error(f"Error retrieving relevant entries: {e}")
            return [], {}
    
    def update_vector_store(self, new_entries: List[Dict[str, Any]]):
        """Update the vector store with new journal entries"""
        if not self.vector_store:
            self.initialize_vector_store(new_entries)
        else:
            # Create and split new documents
            new_documents = self._create_documents(new_entries)
            split_docs = self._split_documents(new_documents)
            
            # Add new documents to existing vector store
            self.vector_store.add_documents(split_docs)
            self.vector_store.persist() 