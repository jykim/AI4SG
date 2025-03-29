#!/usr/bin/env python3
"""
Test script for evaluating BM25Retriever implementation
"""

import os
from pathlib import Path
import logging
import time
from typing import List, Dict, Any
import yaml
import bm25s
import Stemmer
from ir_utils import BM25Retriever

def create_test_corpus() -> List[Dict[str, Any]]:
    """Create a test corpus with journal entries"""
    return [
        {
            "Date": "2024-03-20",
            "Time": "10:00",
            "Title": "Cat Behavior",
            "Content": "a cat is a feline and likes to purr",
            "emotion": "happy",
            "topic": "pets",
            "Tags": "cat, behavior"
        },
        {
            "Date": "2024-03-20",
            "Time": "11:00",
            "Title": "Dog Friendship",
            "Content": "a dog is the human's best friend and loves to play",
            "emotion": "excited",
            "topic": "pets",
            "Tags": "dog, friendship"
        },
        {
            "Date": "2024-03-20",
            "Time": "12:00",
            "Title": "Bird Flight",
            "Content": "a bird is a beautiful animal that can fly",
            "emotion": "amazed",
            "topic": "nature",
            "Tags": "bird, flight"
        },
        {
            "Date": "2024-03-20",
            "Time": "13:00",
            "Title": "Fish Life",
            "Content": "a fish is a creature that lives in water and swims",
            "emotion": "calm",
            "topic": "nature",
            "Tags": "fish, water"
        }
    ]

def test_direct_bm25s():
    """Test direct usage of bm25s library"""
    print("\n=== Testing Direct BM25S Usage ===")
    
    # Create corpus
    corpus = [entry["Content"] for entry in create_test_corpus()]
    
    # Create stemmer
    stemmer = Stemmer.Stemmer("english")
    
    # Tokenize corpus
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    
    # Create and index BM25 model
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    # Test queries
    queries = [
        "does the fish purr like a cat?",
        "what animals can fly?",
        "which pet is a human's best friend?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=2, corpus=corpus)
        
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            print(f"Rank {i+1} (score: {score:.2f}): {doc}")

def test_bm25_retriever():
    """Test our BM25Retriever implementation"""
    print("\n=== Testing BM25Retriever Implementation ===")
    
    # Create test config
    config = {
        "bm25": {
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
            "tokenizer": {
                "lowercase": True,
                "remove_punctuation": True,
                "remove_numbers": False,
                "min_length": 2,
                "max_length": 15,
                "stopwords": "en"
            },
            "stemming": {
                "enabled": True,
                "language": "english"
            },
            "persistence": {
                "enabled": True,
                "directory": "test_bm25_index"
            }
        },
        "final_k": 2,
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(levelname)s - %(message)s"
    }
    
    # Initialize retriever
    retriever = BM25Retriever(config)
    
    # Create and index test corpus
    test_corpus = create_test_corpus()
    retriever.index_documents(test_corpus)
    
    # Test queries
    queries = [
        "does the fish purr like a cat?",
        "what animals can fly?",
        "which pet is a human's best friend?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results, metrics = retriever.get_relevant_entries(query)
        
        for i, entry in enumerate(results):
            print(f"Rank {i+1} (score: {entry['match_score']:.2f}): {entry['Content']}")
            print(f"Metadata: {entry['Title']} ({entry['topic']})")

def main():
    """Run all tests"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Test direct bm25s usage
    test_direct_bm25s()
    
    # Test our implementation
    test_bm25_retriever()
    
    print("\nTests completed!")

if __name__ == "__main__":
    main() 