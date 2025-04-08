import logging
import sys
import os
from pathlib import Path
import openai

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotate_journal import (
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

class TestConfig(BaseConfig):
    """Test configuration class that overrides paths for testing"""
    def load_config(self) -> None:
        """Override config to use test data directory"""
        # Get the absolute path of the test directory
        test_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        test_data_dir = test_dir / 'test_data'
        
        config = {
            'input_dir': str(test_data_dir / 'input'),
            'output_dir': str(test_data_dir / 'output'),
            'api_cache_dir': str(test_data_dir / 'api_cache'),
            'min_process_interval': 0  # No delay for testing
        }
        
        # Set configuration values using absolute paths
        self.input_dir = Path(config['input_dir'])
        self.output_dir = Path(config['output_dir'])
        self.api_cache_dir = Path(config['api_cache_dir'])
        self.min_process_interval = config['min_process_interval']
        
        # Use actual API key from environment and clean it
        api_key = os.getenv('OPENAI_API_KEY', '')
        if api_key:
            # Remove newlines and extra whitespace
            api_key = ''.join(api_key.split())
            logger.debug(f"API Key found with length: {len(api_key)}")
            
            # Set the API key for both the config and the openai client
            self.openai_api_key = api_key
            openai.api_key = api_key
            logger.info("OpenAI API key set successfully")
        else:
            logger.warning("No OpenAI API key found in environment variables")
            self.openai_api_key = ''

def test_api_connection(config):
    """Test the API connection with a simple entry"""
    logger.info("Testing API connection...")
    test_content = "This is a test entry."
    try:
        tags, title = get_tags_for_entry(test_content, config)
        logger.info(f"API test result - Tags: {tags}, Title: {title}")
        return True
    except Exception as e:
        logger.error(f"API connection test failed: {str(e)}", exc_info=True)
        return False

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