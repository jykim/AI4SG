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
from search.bm25_utils import BM25Retriever
from search.index_documents import index_documents, save_to_csv, save_to_markdown, Config

def load_config():
    """Load configuration from config.yaml"""
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    # Get the parent directory (project root)
    project_root = script_dir.parent
    # Use the project root for config path
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_journal_entries():
    """Load journal entries from the annotated CSV file"""
    try:
        # Get the directory of the current script
        script_dir = Path(__file__).parent
        # Get the parent directory (project root)
        project_root = script_dir.parent
        # Look for the annotated CSV file in the output directory
        output_dir = project_root / "output"
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
    """Test tokenization functionality"""
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    try:
        # Get the directory of the current script
        script_dir = Path(__file__).parent
        # Get the parent directory (project root)
        project_root = script_dir.parent
        
        # Create test files
        temp_path = Path(temp_dir)
        file1 = temp_path / "test1.md"
        file2 = temp_path / "test2.md"
        file3 = temp_path / "test3.md"
        
        # Create test entries
        entries_info = [
            {
                "Date": "2024-03-20",
                "Time": "10:00",
                "Title": "AI와 Machine Learning",
                "Content": """인공지능(AI)은 컴퓨터 과학의 한 분야로, 기계가 학습하고 문제를 해결할 수 있도록 하는 기술입니다.
Machine learning is a subset of AI that focuses on developing systems that can learn from data.

딥러닝은 neural networks를 사용하여 복잡한 패턴을 학습합니다.
Deep learning has revolutionized many fields including computer vision.""",
                "emotion": "neutral",
                "topic": "technology",
                "path": "test1.md"
            },
            {
                "Date": "2024-03-20",
                "Time": "11:00",
                "Title": "Data Science and Analytics",
                "Content": """데이터 과학은 데이터에서 의미 있는 인사이트를 도출하는 학문입니다.
Data scientists use various tools and techniques to analyze large datasets.

Python is one of the most popular programming languages for data analysis.
통계적 방법과 머신러닝 알고리즘을 활용하여 데이터를 분석합니다.""",
                "emotion": "neutral",
                "topic": "technology",
                "path": "test2.md"
            },
            {
                "Date": "2024-03-20",
                "Time": "12:00",
                "Title": "Natural Language Processing",
                "Content": """자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.
Natural Language Processing helps computers understand and process human language.

텍스트 분석과 기계 번역은 NLP의 주요 응용 분야입니다.
Text analysis and machine translation are key applications of NLP.""",
                "emotion": "neutral",
                "topic": "technology",
                "path": "test3.md"
            }
        ]
        
        # Write test files
        for entry in entries_info:
            with open(temp_path / entry["path"], "w") as f:
                f.write(f"# {entry['Title']}\n\n{entry['Content']}")
        
        # Create a custom Config object for testing
        test_config = Config()
        test_config.index_dir = temp_path
        test_config.output_dir = temp_path
        
        # Save the original config and replace it with our test config
        import search.index_documents as index_docs
        original_config = index_docs.config
        index_docs.config = test_config
        
        try:
            # Test initial search functionality
            retriever = BM25Retriever(test_config)
            retriever.index_documents(entries_info)
            
            # Test Korean search
            results, _ = retriever.get_relevant_entries("인공지능", k=1)
            assert len(results) > 0, "Korean search failed"
            assert "인공지능" in results[0]["Content"], "Korean term not found in results"
            
            # Test English search
            results, _ = retriever.get_relevant_entries("machine learning", k=1)
            assert len(results) > 0, "English search failed"
            assert "machine learning" in results[0]["Content"].lower(), "English term not found in results"
            
            # Test mixed language search
            results, _ = retriever.get_relevant_entries("딥러닝 neural", k=1)
            assert len(results) > 0, "Mixed language search failed"
            assert "딥러닝" in results[0]["Content"] and "neural" in results[0]["Content"].lower(), "Mixed language terms not found in results"
            
            # Test cross-document search
            results, _ = retriever.get_relevant_entries("AI data", k=3)
            assert len(results) > 1, "Cross-document search failed"
            found_ai = False
            found_data = False
            for result in results:
                if "AI" in result["Content"]:
                    found_ai = True
                if "data" in result["Content"].lower():
                    found_data = True
            assert found_ai and found_data, "Cross-document terms not found"
            
            print("Search tests passed!")
            
        finally:
            # Restore the original config
            index_docs.config = original_config
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Test tokenization first
    test_tokenization()
    
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
        # Use k=3 since we have a small test corpus
        entries, metrics = retriever.get_relevant_entries(query, k=3)
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