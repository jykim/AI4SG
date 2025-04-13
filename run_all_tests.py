#!/usr/bin/env python3
import os
import sys
import logging
import pytest
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def main():
    """Run all tests in the project."""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Add the project root to Python path
    sys.path.insert(0, str(project_root))
    
    logging.info("Starting test suite...")
    
    # Run pytest with the following configuration:
    # -v: verbose output
    # -s: show print statements
    # --tb=short: shorter traceback format
    # --color=yes: colored output
    # --rootdir: specify project root
    pytest_args = [
        "-v",
        "-s",
        "--tb=short",
        "--color=yes",
        f"--rootdir={project_root}",
        # Add test directories
        "test",
        "apps",
        "search"
    ]
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        logging.info("All tests passed successfully!")
    else:
        logging.error(f"Tests failed with exit code: {exit_code}")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 