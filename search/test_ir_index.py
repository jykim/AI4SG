#!/usr/bin/env python3
"""
Test script for evaluating IR index with sample queries
"""

import sys
from pathlib import Path
# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import os
import logging
import time
import yaml
import pandas as pd
import shutil
import tempfile
from datetime import datetime
from search.ir_utils import BM25Retriever
from search.index_documents import index_documents, save_to_csv, save_to_markdown, Config

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

def test_tokenization():
    """Test tokenization directly"""
    from bm25s import tokenize
    from ir_utils import KoreanEnglishTokenizer
    import Stemmer
    
    # Initialize tokenizer
    stemmer = Stemmer.Stemmer('english')
    tokenizer = KoreanEnglishTokenizer(stemmer=stemmer)
    
    # Test queries
    test_queries = [
        "golf",
        "ai",
        "practice"
    ]
    
    print("\n=== Testing Tokenization ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        # Test bm25s tokenize directly
        bm25s_tokens = tokenize([query], stemmer=stemmer)[0]
        print(f"BM25S tokens: {bm25s_tokens}")
        # Test our custom tokenizer
        custom_tokens = tokenizer(query)
        print(f"Custom tokens: {custom_tokens}")

def test_incremental_indexing():
    """Test incremental indexing functionality"""
    print("\n=== Testing Incremental Indexing ===")
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_docs_dir = temp_path / "test_docs"
        test_output_dir = temp_path / "output"
        test_index_dir = temp_path / "test_docs"
        
        # Create test directories
        test_docs_dir.mkdir(exist_ok=True)
        test_output_dir.mkdir(exist_ok=True)
        
        # Create test markdown files
        file1_path = test_docs_dir / "test1.md"
        file2_path = test_docs_dir / "test2.md"
        file3_path = test_docs_dir / "test3.md"
        
        with open(file1_path, "w") as f:
            f.write("# Test Document 1\n\nThis is a test document for incremental indexing.")
        with open(file2_path, "w") as f:
            f.write("# Test Document 2\n\nThis is another test document for incremental indexing.")
        with open(file3_path, "w") as f:
            f.write("# Test Document 3\n\nThis is a third test document for incremental indexing.")
        
        # Create a temporary config
        config = Config()
        config.output_dir = test_output_dir
        config.index_dir = test_index_dir
        
        # First indexing - full index
        print("Performing initial full indexing...")
        entries_info, timing_stats = index_documents(str(test_index_dir), debug=True, incremental=False)
        save_to_csv(entries_info, test_output_dir / "repo_index.csv")
        save_to_markdown(entries_info, test_output_dir / "repo_index.md")
        
        # Verify initial index
        initial_csv = pd.read_csv(test_output_dir / "repo_index.csv")
        print(f"Initial index contains {len(initial_csv)} entries")
        
        # Wait a moment to ensure timestamps are different
        time.sleep(1)
        
        # Modify a file
        print("\nModifying test2.md...")
        with open(file2_path, "a") as f:
            f.write("\n\nThis is an update to test incremental indexing.")
        
        # Run incremental indexing
        print("\nRunning incremental indexing...")
        updated_entries, timing_stats = index_documents(str(test_index_dir), debug=True, incremental=True)
        save_to_csv(updated_entries, test_output_dir / "repo_index.csv")
        save_to_markdown(updated_entries, test_output_dir / "repo_index.md")
        
        # Verify updated index
        updated_csv = pd.read_csv(test_output_dir / "repo_index.csv")
        print(f"Updated index contains {len(updated_csv)} entries")
        
        # Check that the modified file was updated
        modified_entry = updated_csv[updated_csv['path'].str.contains('test2.md')].iloc[0]
        assert "update to test incremental" in modified_entry['content'], \
            "Modified content not found in updated index"
        
        print("Incremental indexing test passed!")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Test tokenization first
    test_tokenization()
    
    # Test incremental indexing
    test_incremental_indexing()
    
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