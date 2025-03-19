import logging
from pathlib import Path
from extract_journal import (
    Config as BaseConfig,
    process_markdown_file,
    main as base_main
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
            'api_cache_dir': 'api_cache',
            'journal_dir': 'test_data'
        }
        
        # Set configuration values
        self.input_dir = Path(config.get('input_dir', 'input'))
        self.output_dir = Path(config.get('output_dir', 'test_data/output'))
        self.api_cache_dir = Path(config.get('api_cache_dir', 'api_cache'))
        self.journal_dir = Path(config.get('journal_dir', 'test_data'))

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