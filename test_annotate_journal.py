import logging
from pathlib import Path
from annotate_journal import (
    Config as BaseConfig,
    update_csv_with_tags
)

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class TestConfig(BaseConfig):
    """Test configuration class that overrides paths for testing"""
    def load_config(self) -> None:
        """Override config to use test data directory"""
        config = {
            'input_dir': 'input',
            'output_dir': 'test_data/output',
            'api_cache_dir': 'test_data/api_cache',
            'min_process_interval': 0  # No delay for testing
        }
        
        # Set configuration values
        self.input_dir = Path(config.get('input_dir', 'input'))
        self.output_dir = Path(config.get('output_dir', 'test_data/output'))
        self.api_cache_dir = Path(config.get('api_cache_dir', 'test_data/api_cache'))
        self.min_process_interval = config.get('min_process_interval', 0)
        
        # For testing, we'll use a mock API key
        self.openai_api_key = 'test-api-key'
        
        # Create all necessary directories
        self.setup_directories()

def main():
    # Initialize test configuration
    config = TestConfig()
    
    # Input CSV file from test data output
    input_csv = config.output_dir / 'journal_entries.csv'
    
    # Process entries with test configuration
    logging.info(f"Processing entries from: {input_csv}")
    update_csv_with_tags(
        input_csv_file=str(input_csv),
        retag_all=False,  # Only process untagged entries
        target_date=None,  # Process all dates
        dry_run=False,  # Write output files
        config=config  # Pass the test config
    )

if __name__ == '__main__':
    main() 