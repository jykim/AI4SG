import logging
import sys
import os
from pathlib import Path
import openai
import pytest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.journal.annotate_journal import (
    Config as BaseConfig,
    update_csv_with_tags,
    get_tags_for_entry
)

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test/test_data/output/test_annotate.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

@pytest.fixture
def test_config():
    config = BaseConfig()
    # Load API key from environment
    config.openai_api_key = os.getenv('OPENAI_API_KEY', '')
    if not config.openai_api_key:
        pytest.skip("OpenAI API key not found in environment")
    openai.api_key = config.openai_api_key
    return config

class TestConfig:
    """Test configuration loading and validation"""
    
    def test_load_config(self, test_config):
        """Test loading configuration from file"""
        test_config.load_config()
        assert test_config.input_dir.exists()
        assert test_config.output_dir.exists()
        assert test_config.api_cache_dir.exists()

def test_api_connection(test_config):
    """Test the API connection with a simple entry"""
    logger.info("Testing API connection...")
    test_content = "This is a test entry."
    
    tags, title = get_tags_for_entry(test_content, test_config)
    logger.info(f"API test result - Tags: {tags}, Title: {title}")
    
    assert tags is not None, "Tags should not be None"
    assert isinstance(tags, dict), "Tags should be a dictionary"
    assert 'emotion' in tags or tags == {}, "Tags should contain emotion or be empty"
    assert 'topic' in tags or tags == {}, "Tags should contain topic or be empty"
    assert 'etc' in tags or tags == {}, "Tags should contain etc or be empty"

def main():
    # Initialize test configuration
    logger.info("Initializing test configuration...")
    config = TestConfig()
    
    # Create test data directories if they don't exist
    logger.debug("Creating test directories...")
    config.input_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.api_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Test API connection first
    if not test_api_connection(config):
        logger.error("API connection test failed. Exiting...")
        return
    
    # Create a sample journal entry CSV file for testing
    sample_csv_path = config.output_dir / 'journal_entries.csv'
    logger.info(f"Creating sample CSV at {sample_csv_path}")
    if not sample_csv_path.exists():
        with open(sample_csv_path, 'w', encoding='utf-8') as f:
            f.write('Date,Title,Section,Content,Time\n')
            f.write('2025-01-01,Test Entry,Test,This is a test journal entry for testing the annotation process.,12:00\n')
    
    # Process entries with test configuration
    logger.info(f"Processing entries from: {sample_csv_path}")
    try:
        update_csv_with_tags(
            input_csv_file=str(sample_csv_path),
            retag_all=True,  # Process all entries for testing
            target_date=None,  # Process all dates
            dry_run=False,  # Write output files
            config=config  # Pass the test config
        )
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main() 