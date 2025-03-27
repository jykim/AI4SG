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

class JournalRAG:
    """RAG system for retrieving relevant journal entries"""
    
    def __init__(self, config=None):
        """Initialize the RAG system with configuration"""
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size
            chunk_overlap=100,  # Reduced overlap
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = None
        self.retriever = None
        
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
        
        # Split documents into chunks
        split_docs = self._split_documents(documents)
        
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
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 10,  # Get more docs initially to filter duplicates
                "score_threshold": 0.0  # Get all scores
            }
        )
    
    def get_relevant_entries(self, query: str, k: int = 15) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:  # Increased k
        """
        Retrieve relevant journal entries for a given query
        
        Returns:
            Tuple containing:
            - List of relevant entries
            - Dict of timing metrics for each step
        """
        if not self.retriever:
            logging.warning("Vector store not initialized. Returning empty list.")
            return [], {}
        
        try:
            timing_metrics = {}
            
            # Measure retrieval time
            start_time = time.time()
            relevant_docs = self.retriever.invoke(query)
            timing_metrics['retrieval_time'] = time.time() - start_time
            
            # Measure document processing time
            start_time = time.time()
            seen_content = set()
            relevant_entries = []
            
            # Sort documents by score if available
            if hasattr(relevant_docs, 'scores'):
                docs_with_scores = list(zip(relevant_docs, relevant_docs.scores))
                docs_with_scores.sort(key=lambda x: x[1], reverse=True)
                relevant_docs = [doc for doc, _ in docs_with_scores]
            
            for doc in relevant_docs:
                # Skip if we've seen this content before
                if doc.page_content in seen_content:
                    continue
                    
                seen_content.add(doc.page_content)
                
                # Extract score from metadata if available
                score = doc.metadata.get('score', 0.0)
                
                entry = {
                    'Date': doc.metadata.get('date', ''),
                    'Time': doc.metadata.get('time', ''),
                    'Title': doc.metadata.get('title', ''),
                    'Content': doc.page_content,
                    'emotion': doc.metadata.get('emotion', ''),
                    'topic': doc.metadata.get('topic', ''),
                    'Tags': doc.metadata.get('tags', ''),
                    'match_score': score
                }
                relevant_entries.append(entry)
                
                # Break if we have enough unique entries
                if len(relevant_entries) >= k:
                    break
            
            timing_metrics['processing_time'] = time.time() - start_time
            timing_metrics['total_time'] = timing_metrics['retrieval_time'] + timing_metrics['processing_time']
            timing_metrics['num_docs_retrieved'] = len(relevant_docs)
            timing_metrics['num_unique_docs'] = len(relevant_entries)
            
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