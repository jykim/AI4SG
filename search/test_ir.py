#!/usr/bin/env python3
"""
Test script for evaluating BM25Retriever implementation
"""

import sys
from pathlib import Path
# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import os
import logging
import time
from typing import List, Dict, Any
import yaml
import bm25s
import Stemmer
from konlpy.tag import Kkma
from ir_utils import BM25Retriever, KoreanEnglishTokenizer

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

def create_korean_test_corpus() -> List[Dict[str, Any]]:
    """Create a test corpus with Korean journal entries"""
    return [
        {
            "Date": "2024-03-20",
            "Time": "10:00",
            "Title": "고양이 행동",
            "Content": "고양이는 야옹하고 좋아하며 우유를 마시는 것을 좋아합니다",
            "emotion": "행복",
            "topic": "반려동물",
            "Tags": "고양이, 행동"
        },
        {
            "Date": "2024-03-20",
            "Time": "11:00",
            "Title": "강아지 우정",
            "Content": "강아지는 사람의 가장 친한 친구이며 놀기를 좋아합니다",
            "emotion": "신나",
            "topic": "반려동물",
            "Tags": "강아지, 우정"
        },
        {
            "Date": "2024-03-20",
            "Time": "12:00",
            "Title": "새의 비행",
            "Content": "새는 아름다운 동물이며 하늘을 날 수 있습니다",
            "emotion": "감동",
            "topic": "자연",
            "Tags": "새, 비행"
        },
        {
            "Date": "2024-03-20",
            "Time": "13:00",
            "Title": "물고기 생활",
            "Content": "물고기는 물 속에서 살며 수영하는 생물입니다",
            "emotion": "평온",
            "topic": "자연",
            "Tags": "물고기, 물"
        }
    ]

def create_mixed_test_corpus() -> List[Dict[str, Any]]:
    """Create a test corpus with mixed Korean and English entries"""
    return [
        {
            "Date": "2024-03-20",
            "Time": "10:00",
            "Title": "Cat and 고양이",
            "Content": "a cat is a feline and likes to purr. 고양이는 야옹하고 좋아하며 우유를 마시는 것을 좋아합니다",
            "emotion": "happy",
            "topic": "pets",
            "Tags": "cat, 고양이, behavior"
        },
        {
            "Date": "2024-03-20",
            "Time": "11:00",
            "Title": "Dog and 강아지",
            "Content": "a dog is the human's best friend and loves to play. 강아지는 사람의 가장 친한 친구이며 놀기를 좋아합니다",
            "emotion": "excited",
            "topic": "pets",
            "Tags": "dog, 강아지, friendship"
        },
        {
            "Date": "2024-03-20",
            "Time": "12:00",
            "Title": "Bird and 새",
            "Content": "a bird is a beautiful animal that can fly. 새는 아름다운 동물이며 하늘을 날 수 있습니다",
            "emotion": "amazed",
            "topic": "nature",
            "Tags": "bird, 새, flight"
        },
        {
            "Date": "2024-03-20",
            "Time": "13:00",
            "Title": "Fish and 물고기",
            "Content": "a fish is a creature that lives in water and swims. 물고기는 물 속에서 살며 수영하는 생물입니다",
            "emotion": "calm",
            "topic": "nature",
            "Tags": "fish, 물고기, water"
        }
    ]

def test_korean_bm25():
    """Test BM25 with Korean text using Kkma"""
    print("\n=== Testing BM25 with Korean Text ===")
    
    # Create corpus
    corpus = [entry["Content"] for entry in create_korean_test_corpus()]
    
    # Initialize Kkma
    kkma = Kkma()
    
    # Create custom tokenizer function for Korean
    def korean_tokenizer(text):
        # Convert text to list of tokens (nouns)
        tokens = kkma.nouns(text)
        # Ensure we return a list of tokens
        return [str(token) for token in tokens]
    
    # Create and index BM25 model with Korean tokenizer
    retriever = bm25s.BM25()
    
    # Tokenize corpus using Korean tokenizer
    print("\nTokenizing corpus...")
    corpus_tokens = []
    for doc in corpus:
        tokens = korean_tokenizer(doc)
        print(f"Document tokens: {tokens}")
        corpus_tokens.append(tokens)
    
    print("\nIndexing documents...")
    retriever.index(corpus_tokens)
    
    # Test queries
    queries = [
        "고양이가 우유를 마시나요?",
        "어떤 동물이 날 수 있나요?",
        "사람의 가장 친한 친구는 누구인가요?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        query_tokens = korean_tokenizer(query)
        print(f"Query tokens: {query_tokens}")
        results, scores = retriever.retrieve([query_tokens], k=2, corpus=corpus)
        
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            print(f"Rank {i+1} (score: {score:.2f}): {doc}")

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

def test_mixed_language_tokenizer():
    """Test how KoreanEnglishTokenizer handles mixed language text"""
    print("\n=== Testing Mixed Language Tokenizer ===")
    
    # Initialize tokenizer
    kkma = Kkma()
    stemmer = Stemmer.Stemmer("english")
    tokenizer = KoreanEnglishTokenizer(kkma=kkma, stemmer=stemmer)
    
    # Test cases with mixed language text
    test_cases = [
        "고양이가 우유를 마시나요? does the cat drink milk?",
        "강아지는 사람의 best friend입니다. a dog is man's best friend.",
        "새는 beautiful bird입니다. a bird is beautiful.",
        "물고기는 water에서 swim합니다. a fish swims in water."
    ]
    
    for text in test_cases:
        print(f"\nInput text: {text}")
        tokens = tokenizer(text)
        print(f"Tokens: {tokens}")
        
        # Show Korean and English parts separately
        korean_text = ' '.join(tokenizer.korean_pattern.findall(text))
        english_text = ' '.join(word for word in text.split() if not tokenizer.korean_pattern.search(word))
        print(f"Korean part: {korean_text}")
        print(f"English part: {english_text}")
        
        # Show how each part is tokenized
        korean_tokens = kkma.nouns(korean_text) if korean_text else []
        print(f"Korean nouns: {korean_tokens}")
        
        # Show the combined text that gets tokenized
        combined_text = f"{text} {' '.join(korean_tokens)}"
        print(f"Combined text: {combined_text}")

def test_mixed_language_bm25():
    """Test BM25 with mixed Korean and English text"""
    print("\n=== Testing BM25 with Mixed Language Text ===")
    
    # Create corpus
    corpus = [entry["Content"] for entry in create_mixed_test_corpus()]
    
    # Create and index BM25 model
    retriever = bm25s.BM25()
    
    # Create custom tokenizer
    kkma = Kkma()
    stemmer = Stemmer.Stemmer("english")
    tokenizer = KoreanEnglishTokenizer(kkma=kkma, stemmer=stemmer)
    
    # Tokenize corpus using custom tokenizer
    print("\nTokenizing corpus...")
    corpus_tokens = []
    for doc in corpus:
        tokens = tokenizer(doc)
        print(f"Document tokens: {tokens}")
        corpus_tokens.append(tokens)
    
    print("\nIndexing documents...")
    retriever.index(corpus_tokens)
    
    # Test queries in both languages
    queries = [
        "고양이가 우유를 마시나요? does the cat drink milk?",
        "어떤 동물이 날 수 있나요? which animals can fly?",
        "사람의 가장 친한 친구는 누구인가요? who is man's best friend?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        query_tokens = tokenizer(query)
        print(f"Query tokens: {query_tokens}")
        results, scores = retriever.retrieve([query_tokens], k=2, corpus=corpus)
        
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            print(f"Rank {i+1} (score: {score:.2f}): {doc}")

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
    
    # Test Korean text
    test_korean_bm25()
    
    # Test mixed language tokenizer
    test_mixed_language_tokenizer()
    
    # Test mixed language
    test_mixed_language_bm25()
    
    print("\nTests completed!")

if __name__ == "__main__":
    main() 