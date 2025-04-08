#!/usr/bin/env python3
"""
Tests for the Graph Visualization Dashboard
"""

import sys
from pathlib import Path
# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from dash_graph import DocumentManager, load_config

# Test configuration
@pytest.fixture
def config():
    """Load test configuration"""
    return {
        'output_dir': 'test/test_data/output',
        'bm25': {
            'k1': 1.5,
            'b': 0.75,
            'final_k': 10
        },
        'log_level': 'INFO',
        'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }

# Test DocumentManager class
def test_document_manager_initialization():
    """Test DocumentManager initialization"""
    kg = DocumentManager(config)
    assert kg.config == config
    assert kg.documents == []
    assert kg.indexed == False

def test_document_manager_has_index():
    """Test DocumentManager has_index method"""
    kg = DocumentManager(config)
    kg.index_documents()
    assert len(kg.documents) > 0
    assert kg.indexed == True

def test_document_manager_load_journal_entries():
    """Test DocumentManager load_journal_entries method"""
    kg = DocumentManager(config)
    kg.index_documents()
    assert len(kg.documents) > 0
    assert kg.indexed == True

# Test search functionality
def test_search(config):
    """Test search functionality"""
    kg = DocumentManager(config)
    kg.index_documents()
    
    # Test empty query
    results = kg.search("")
    assert len(results) == 0
    
    # Test valid query
    results = kg.search("test query")
    assert isinstance(results, list)
    
    # Test document type filtering
    journal_results = kg.search("test query", doc_type="journal")
    reading_results = kg.search("test query", doc_type="reading")
    all_results = kg.search("test query", doc_type="all")
    
    # Check that filtered results only contain the specified type
    assert all(doc.get('doc_type') == 'journal' for doc in journal_results)
    assert all(doc.get('doc_type') == 'reading' for doc in reading_results)
    assert len(all_results) >= len(journal_results)
    assert len(all_results) >= len(reading_results) 