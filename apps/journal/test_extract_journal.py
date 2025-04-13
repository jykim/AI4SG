#!/usr/bin/env python3
"""
Tests for journal extraction functionality
"""

import logging
from pathlib import Path
import sys
import os

# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .extract_journal import (
    Config as BaseConfig,
    process_markdown_file,
    main as base_main,
    config as extract_config
)
import pytest
import tempfile
import shutil

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

@pytest.fixture
def test_config():
    return BaseConfig()

@pytest.fixture
def temp_journal_dir():
    """Create a temporary directory for test journal files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

def create_test_journal_file(directory: Path, filename: str, content: str) -> Path:
    """Create a test journal file with given content"""
    file_path = directory / filename
    file_path.write_text(content)
    return file_path

class TestConfig:
    """Test configuration loading and validation"""
    
    def test_load_config(self, test_config):
        """Test loading configuration from file"""
        test_config.load_config()
        assert test_config.input_dir.exists()
        assert test_config.output_dir.exists()
        assert test_config.api_cache_dir.exists()
        
    def test_single_entry_mode(self, temp_journal_dir):
        """Test processing a single entry journal file"""
        # Create a test journal file
        test_content = """# My Journal Entry

This is a test journal entry without any sections.
It should be treated as a single entry when single_entry_mode is enabled.
"""
        test_file = create_test_journal_file(temp_journal_dir, "test-entry.md", test_content)
        
        # Create config with single_entry_mode enabled
        config = BaseConfig()
        config.journal_dir = temp_journal_dir
        config.single_entry_mode = True
        
        # Set single_entry_mode in extract_journal's config
        extract_config.single_entry_mode = True
        
        # Process the file
        entries = process_markdown_file(test_file)
        
        # Verify results
        assert len(entries) == 1
        date, title, section, content, time = entries[0]
        assert title == "test-entry"
        assert section == "Entry"
        assert "This is a test journal entry" in content
        assert date is not None  # Should get date from file metadata

def main():
    # Initialize test configuration
    config = TestConfig()
    
    # Process all markdown files in the directory
    logging.info(f"Looking for markdown files in: {config.journal_dir}")
    markdown_files = list(config.journal_dir.glob('*.md'))
    logging.info(f"Found {len(markdown_files)} markdown files: {[f.name for f in markdown_files]}")
    
    # Use the base main function with our test config
    base_main(config)

if __name__ == '__main__':
    main() 