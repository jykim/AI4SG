import logging
from pathlib import Path
from apps.journal.extract_journal import (
    Config as BaseConfig,
    process_markdown_file,
    main as base_main
)
import pytest

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

class TestConfig:
    """Test configuration loading and validation"""
    
    def test_load_config(self, test_config):
        """Test loading configuration from file"""
        test_config.load_config()
        assert test_config.input_dir.exists()
        assert test_config.output_dir.exists()
        assert test_config.api_cache_dir.exists()

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