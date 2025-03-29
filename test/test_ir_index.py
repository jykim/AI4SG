#!/usr/bin/env python3
"""
Test script for evaluating IR index with sample queries
"""

import os
from pathlib import Path
import logging
import yaml
import pandas as pd
from ir_utils import BM25Retriever

def load_config():
    """Load configuration from config_rag.yaml"""
    config_path = Path("config_rag.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config_rag.yaml not found")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_journal_entries():
    """Load journal entries from the annotated CSV file"""
    try:
        # Look for the annotated CSV file in the output directory
        output_dir = Path("output")
        annotated_file = output_dir / 'journal_entries_annotated.csv'
        if not annotated_file.exists():
            logging.warning(f"No annotated journal entries found in {output_dir}")
            return pd.DataFrame()
        
        logging.info(f"Loading journal entries from {annotated_file}")
        
        df = pd.read_csv(annotated_file)
        if df.empty:
            logging.warning("Loaded CSV file is empty")
            return pd.DataFrame()
            
        # Clean up and parse dates, dropping any rows with invalid dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
        
        # Replace blank/null emotions with empty string
        df['emotion'] = df['emotion'].fillna('')
        df['emotion'] = df['emotion'].apply(lambda x: '' if pd.isna(x) or str(x).strip() == '' else x)
        
        # Clean up title formatting - remove quotes and asterisks
        df['Title'] = df['Title'].apply(lambda x: str(x).strip('"*') if pd.notna(x) else x)
        
        # Create Tags column with emojis, filtering out NaN values
        df['Tags'] = df.apply(lambda row: ' '.join(filter(None, [
            row['topic_visual'] if pd.notna(row['topic_visual']) else '',
            row['etc_visual'] if pd.notna(row['etc_visual']) else ''
        ])), axis=1)
        
        return df
    except Exception as e:
        logging.error(f"Error loading journal entries: {e}")
        return pd.DataFrame()

def print_results(entries, query):
    """Print search results in a formatted way"""
    print(f"\n=== Results for query: '{query}' ===")
    if not entries:
        print("No results found.")
        return
    
    for i, entry in enumerate(entries, 1):
        print(f"\nResult {i} (Score: {entry['match_score']:.3f}):")
        print(f"Date: {entry['Date']}")
        print(f"Title: {entry['Title']}")
        print(f"Emotion: {entry['emotion']}")
        print(f"Topic: {entry['topic']}")
        print(f"Tags: {entry['Tags']}")
        print(f"Content: {entry['Content'][:200]}...")  # Show first 200 chars

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config = load_config()
    
    # Initialize BM25 retriever
    retriever = BM25Retriever(config)
    
    # Check if index exists
    index_dir = Path("bm25_index")
    if index_dir.exists():
        logging.info(f"Found existing index at {index_dir}")
        # The index will be automatically loaded by BM25Retriever
        logging.info(f"Loaded {len(retriever.documents)} documents from existing index")
    else:
        # Load journal entries and create new index
        logging.info("No existing index found. Creating new index...")
        journal_entries = load_journal_entries()
        if journal_entries.empty:
            logging.error("No journal entries found. Please ensure there are CSV files in the input directory.")
            return
        
        # Index documents
        logging.info("Indexing documents...")
        retriever.index_documents(journal_entries.to_dict('records'))
        logging.info(f"Indexed {len(retriever.documents)} documents")
    
    # Test queries
    test_queries = [
        "golf",
        "ai",
        "practice"
    ]
    
    # Run queries
    for query in test_queries:
        logging.info(f"\nProcessing query: {query}")
        entries, metrics = retriever.get_relevant_entries(query)
        print_results(entries, query)
        
        # Print metrics
        if metrics:
            print("\nMetrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")

if __name__ == "__main__":
    main() 