#!/usr/bin/env python3
"""
Information Retrieval Utilities using BM25S

This module implements document retrieval using BM25S, a fast implementation
of BM25 ranking algorithm using sparse matrices.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from datetime import datetime
import logging
import time
import yaml
import numpy as np
from bm25s import BM25, tokenize
import re
import Stemmer  # Optional: for stemming
import json

class BM25Retriever:
    """BM25-based document retriever using BM25S"""
    
    def __init__(self, config=None):
        """Initialize the BM25 retriever with configuration"""
        self.config = config
        if config is None:
            config_path = Path("config_rag.yaml")
            if not config_path.exists():
                raise FileNotFoundError("config_rag.yaml not found")
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif hasattr(config, 'rag_config'):
            self.config = config.rag_config
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError("Config must be a dictionary or a Config object with rag_config attribute")
        
        # Get BM25 configuration
        self.bm25_config = self.config.get('bm25', {})
        
        # Initialize BM25 with configuration
        self.bm25 = BM25(
            k1=self.bm25_config.get('k1', 1.5),
            b=self.bm25_config.get('b', 0.75)
        )
        
        # Initialize stemmer if enabled
        self.stemmer = None
        stemming_config = self.bm25_config.get('stemming', {})
        if stemming_config.get('enabled', False):
            try:
                self.stemmer = Stemmer.Stemmer(stemming_config.get('language', 'english'))
            except Exception as e:
                logging.warning(f"Failed to initialize stemmer: {e}")
        
        self.documents = []  # Document texts for BM25
        self.metadata = []   # Document metadata (includes doc_type)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format=self.config['log_format']
        )
        
        # Load existing index if persistence is enabled
        persistence_config = self.bm25_config.get('persistence', {})
        if persistence_config.get('enabled', False):
            self._load_index(persistence_config.get('directory', 'bm25_index'))
    
    def _create_document(self, entry: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Convert a journal entry to a document with metadata"""
        # Create a formatted text representation of the entry
        date = entry.get('Date', '')
        time = entry.get('Time', '')
        title = entry.get('Title', 'Untitled')
        content = entry.get('Content', '')
        emotion = entry.get('emotion', '')
        topic = entry.get('topic', '')
        tags = entry.get('Tags', '')
        doc_type = entry.get('doc_type', '')  # Get doc_type from entry
        
        text = f"""
        Date: {date} {time}
        Title: {title}
        Emotion: {emotion}
        Topic: {topic}
        Tags: {tags}
        Type: {doc_type}
        
        Content:
        {content}
        """
        
        # Create metadata
        metadata = {
            'date': str(date),
            'time': str(time),
            'title': str(title) if pd.notna(title) else 'Untitled',
            'emotion': str(emotion) if pd.notna(emotion) else '',
            'topic': str(topic) if pd.notna(topic) else '',
            'tags': str(tags) if pd.notna(tags) else '',
            'doc_type': str(doc_type) if pd.notna(doc_type) else ''  # Add doc_type to metadata
        }
        
        return text, metadata
    
    def index_documents(self, entries: List[Dict[str, Any]]):
        """Index journal entries using BM25S"""
        logging.info(f"Indexing {len(entries)} documents")
        
        # Clear existing documents
        self.documents = []
        self.metadata = []
        
        # Process each entry
        for entry in entries:
            text, metadata = self._create_document(entry)
            self.documents.append(text)
            self.metadata.append(metadata)
        
        # Tokenize documents using bm25s tokenize function
        tokenized_docs = tokenize(
            self.documents,
            stopwords=self.bm25_config.get('tokenizer', {}).get('stopwords', 'en'),
            stemmer=self.stemmer
        )
        
        # Index documents
        self.bm25.index(tokenized_docs)
        logging.info("Document indexing completed")
        
        # Save index if persistence is enabled
        persistence_config = self.bm25_config.get('persistence', {})
        if persistence_config.get('enabled', False):
            self._save_index(persistence_config.get('directory', 'bm25_index'))
    
    def _save_index(self, directory: str):
        """Save the BM25 index and documents to disk"""
        try:
            # Create directory if it doesn't exist
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Save BM25 index and documents
            self.bm25.save(directory, corpus=self.documents)
            
            # Save metadata to a separate file
            metadata_file = Path(directory) / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f)
            
            logging.info(f"Saved index to {directory}")
        except Exception as e:
            logging.error(f"Failed to save index: {e}")
    
    def _load_index(self, directory: str):
        """Load the BM25 index and documents from disk"""
        try:
            if Path(directory).exists():
                # Load BM25 index and documents
                self.bm25 = BM25.load(directory, load_corpus=True)
                self.documents = self.bm25.corpus
                
                # Load metadata from separate file
                metadata_file = Path(directory) / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.metadata = json.load(f)
                    logging.info(f"Loaded metadata from {metadata_file}")
                else:
                    logging.warning(f"No metadata file found at {metadata_file}")
                    self.metadata = []
                
                logging.info(f"Loaded index from {directory}")
            else:
                logging.info(f"No existing index found at {directory}")
        except Exception as e:
            logging.error(f"Failed to load index: {e}")
    
    def get_relevant_entries(self, query: str, k: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Retrieve relevant journal entries for a given query using BM25 ranking.
        """
        if k is None:
            k = self.config['final_k']
            
        if not self.documents:
            logging.warning("No documents indexed. Returning empty list.")
            return [], {}
        
        try:
            timing_metrics = {}
            start_time = time.time()
            
            # Tokenize query using bm25s tokenize function
            query_tokens = tokenize(
                [query],
                stopwords=self.bm25_config.get('tokenizer', {}).get('stopwords', 'en'),
                stemmer=self.stemmer
            )
            
            # Get BM25 scores and results
            results, scores = self.bm25.retrieve(query_tokens, k=k, corpus=self.documents)
            
            # Prepare results
            relevant_entries = []
            for i in range(results.shape[1]):
                doc = results[0, i]  # This is the document text
                score = scores[0, i]
                
                # Find the index of the document in our documents list
                idx = self.documents.index(doc)
                metadata = self.metadata[idx]  # Get corresponding metadata
                
                entry = {
                    'Date': metadata['date'],
                    'Time': metadata['time'],
                    'Title': metadata['title'],
                    'Content': doc,  # Use the document text directly
                    'emotion': metadata['emotion'],
                    'topic': metadata['topic'],
                    'Tags': metadata['tags'],
                    'match_score': float(score),
                    'doc_id': f"{metadata['date']}_{metadata['title']}",
                    'match_type': 'bm25',
                    'doc_type': metadata['doc_type']  # Get doc_type from metadata
                }
                relevant_entries.append(entry)
            
            timing_metrics['total_time'] = time.time() - start_time
            timing_metrics['num_docs_retrieved'] = len(relevant_entries)
            
            if relevant_entries:
                logging.info(f"Retrieved {len(relevant_entries)} documents")
                logging.info(f"Score range: {min(scores[0])} to {max(scores[0])}")
            
            return relevant_entries, timing_metrics
            
        except Exception as e:
            logging.error(f"Error retrieving relevant entries: {e}")
            return [], {}
    
    def update_index(self, new_entries: List[Dict[str, Any]]):
        """Update the index with new journal entries"""
        if not self.documents:
            self.index_documents(new_entries)
        else:
            # Process new entries
            new_documents = []
            new_metadata = []
            for entry in new_entries:
                text, metadata = self._create_document(entry)
                new_documents.append(text)
                new_metadata.append(metadata)
            
            # Tokenize new documents using bm25s tokenize function
            tokenized_docs = tokenize(
                new_documents,
                stopwords=self.bm25_config.get('tokenizer', {}).get('stopwords', 'en'),
                stemmer=self.stemmer
            )
            
            # Add new documents to index
            self.bm25.index(tokenized_docs)
            
            # Update document and metadata lists
            self.documents.extend(new_documents)
            self.metadata.extend(new_metadata)
            
            logging.info(f"Added {len(new_entries)} new documents to index")
            
            # Save updated index if persistence is enabled
            persistence_config = self.bm25_config.get('persistence', {})
            if persistence_config.get('enabled', False):
                self._save_index(persistence_config.get('directory', 'bm25_index')) 