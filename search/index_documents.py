"""
Document Indexer
Scans directories for document files and generates a catalog with metadata.
Extracts title, content, date, properties, tags, path, size, and links.
"""

import os
import pathlib
import argparse
import csv
import unicodedata
from typing import List, Dict, Optional, Tuple
import re
from pathlib import Path
from datetime import datetime, date
import yaml
import logging
import json
import sys
import time

# Import BM25Retriever for indexing
from .ir_utils import BM25Retriever

class Config:
    """Configuration class to manage application settings"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.load_config()
        self.setup_directories()

    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found. Using default values.")
            config = {
                'index_dir': '.',
                'output_dir': 'output'
            }
        
        # Set configuration values
        self.index_dir = Path(os.path.expanduser(config.get('index_dir', '.')))
        self.output_dir = Path(config.get('output_dir', 'output'))
        
        # Load RAG config if available
        self.rag_config = {}
        rag_config_path = Path("config_rag.yaml")
        if rag_config_path.exists():
            try:
                with open(rag_config_path, 'r') as f:
                    self.rag_config = yaml.safe_load(f)
            except Exception as e:
                logging.warning(f"Could not load RAG config: {e}")

    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        self.output_dir.mkdir(exist_ok=True)

# Initialize configuration
config = Config()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.output_dir / 'index.log'),
        logging.StreamHandler()
    ]
)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle date objects and other non-serializable types"""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return str(obj)

def normalize_text(text: str) -> str:
    """
    Normalize text by standardizing Unicode characters and whitespace.
    
    This function:
    1. Converts text to NFC form for consistent Unicode representation
    2. Replaces multiple spaces with single space
    3. Removes leading/trailing whitespace
    
    Args:
        text: Raw input text string
        
    Returns:
        Normalized text string with consistent spacing and Unicode form
    """
    text = unicodedata.normalize('NFC', text)
    return ' '.join(text.split())

def read_file_content(file_path: str) -> str:
    """
    Read the entire content of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        File content as a string, empty string if file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Could not read {file_path}: {str(e)}")
        return ""

def remove_code_blocks(content: str) -> str:
    """
    Remove content surrounded by triple backticks (```) from the text.
    
    Args:
        content: Document content
        
    Returns:
        Content with code blocks removed
    """
    # Pattern to match content between triple backticks
    code_block_pattern = r'```[\s\S]*?```'
    return re.sub(code_block_pattern, '', content)

def extract_title_from_content(content: str, file_path: str) -> str:
    """
    Extract title from filename.
    
    Args:
        content: Document content (not used)
        file_path: Path to the document
        
    Returns:
        Document title (filename without extension)
    """
    # Use filename without extension
    filename = pathlib.Path(file_path).stem
    return filename.replace('_', ' ').replace('-', ' ').strip()

def extract_tags_from_content(content: str) -> List[str]:
    """
    Extract hashtags from document content.
    
    Args:
        content: Document content
        
    Returns:
        List of hashtags found in the content
    """
    # Find all hashtags in the content
    hashtag_pattern = r'#([a-zA-Z0-9_]+)'
    matches = re.findall(hashtag_pattern, content)
    return list(set(matches))  # Remove duplicates

def extract_links_from_content(content: str) -> Tuple[List[str], List[str]]:
    """
    Extract internal and external links from document content.
    
    Args:
        content: Document content
        
    Returns:
        Tuple of (internal_links, external_links)
    """
    # Find all markdown links
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(link_pattern, content)
    
    internal_links = []
    external_links = []
    
    for _, url in matches:
        # Check if it's an internal link (no http/https and not an anchor)
        if not url.startswith(('http://', 'https://', 'ftp://', 'mailto:')) and not url.startswith('#'):
            internal_links.append(url)
        else:
            external_links.append(url)
    
    return internal_links, external_links

def get_relative_path(file_path: str, root_dir: str) -> str:
    """
    Get the relative path of a file within the root directory.
    
    Args:
        file_path: Absolute path to the file
        root_dir: Root directory path
        
    Returns:
        Relative path from root directory to the file
    """
    try:
        return os.path.relpath(file_path, root_dir)
    except ValueError:
        # Handle case where paths are on different drives
        return file_path

def extract_metadata(file_path: str, debug: bool = False) -> Dict[str, any]:
    """
    Extract metadata from document file.
    
    Extraction process:
    1. Get file information (size, last modified date)
    2. Extract title from content or filename
    3. Extract content
    4. Extract properties from YAML frontmatter
    5. Extract tags from content
    6. Extract internal and external links
    7. Get relative path
    
    Args:
        file_path: Path to the document file
        debug: Enable detailed debug output
        
    Returns:
        Dictionary containing document metadata
    """
    try:
        path = pathlib.Path(file_path)
        
        # Get file information
        file_size = path.stat().st_size / 1024  # Convert bytes to KB
        size = f"{file_size:.1f}"  # Keep one decimal place
        
        # Get last modified date
        last_modified = datetime.fromtimestamp(path.stat().st_mtime)
        date = last_modified.strftime('%Y-%m-%d')
        
        # Get modification time for incremental indexing
        mtime = path.stat().st_mtime
        
        # Read file content
        content = read_file_content(file_path)
        
        # Extract title
        title = extract_title_from_content(content, file_path)
        
        # Initialize metadata
        metadata = {
            'date': date,
            'title': title,
            'content': content,
            'properties': {},
            'tags': [],
            'path': get_relative_path(file_path, str(config.index_dir)),
            'size': size,
            'internal_links': [],
            'external_links': [],
            'mtime': mtime  # Add modification time for incremental indexing
        }
        
        # Extract YAML frontmatter if present
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            try:
                # Parse YAML frontmatter
                frontmatter_data = yaml.safe_load(frontmatter)
                
                # Extract properties from frontmatter
                if frontmatter_data:
                    metadata['properties'] = frontmatter_data
                    
                    # Remove frontmatter from content
                    content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
                    metadata['content'] = content.strip()
                
            except yaml.YAMLError as e:
                if debug:
                    logging.warning(f"Failed to parse YAML frontmatter: {str(e)}")
        
        # Extract tags
        metadata['tags'] = extract_tags_from_content(content)
        
        # Extract links
        internal_links, external_links = extract_links_from_content(content)
        metadata['internal_links'] = internal_links
        metadata['external_links'] = external_links
        
        if debug:
            logging.info(f"Extracted metadata: {metadata}")
            
        return metadata
        
    except Exception as e:
        logging.error(f"Failed to extract metadata for {file_path}: {str(e)}")
        return {
            'date': '',
            'title': '',
            'content': '',
            'properties': {},
            'tags': [],
            'path': '',
            'size': '',
            'internal_links': [],
            'external_links': [],
            'mtime': 0  # Default mtime for error cases
        }

def index_documents(root_dir: str, debug: bool = False, max_files: Optional[int] = None, incremental: bool = False) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Main indexing function that processes all document files in a directory.
    
    Process:
    1. Recursively find all document files
    2. Extract metadata for each file
    3. Apply file limit if specified
    4. If incremental=True, only process files that have changed since last indexing
    
    Args:
        root_dir: Root directory to scan for document files
        debug: Enable detailed debug output
        max_files: Maximum number of files to process
        incremental: Whether to perform incremental indexing
        
    Returns:
        Tuple of (entries_info, timing_stats)
        - entries_info: List of dictionaries containing file information and metadata
        - timing_stats: Dictionary containing timing information for different steps
    """
    start_time = time.time()
    timing_stats = {
        'load_existing_index': 0.0,
        'file_scanning': 0.0,
        'metadata_extraction': 0.0,
        'merge_entries': 0.0,
        'total_time': 0.0
    }
    
    entries_info = []
    root_path = pathlib.Path(root_dir)
    files_processed = 0
    files_skipped = 0
    
    # Load existing index for incremental indexing
    existing_index = {}
    if incremental:
        load_start = time.time()
        try:
            index_file = config.output_dir / 'repo_index.csv'
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Store full path and mtime for comparison
                        if 'full_path' in row and 'mtime' in row:
                            try:
                                existing_index[row['full_path']] = float(row['mtime'])
                            except (ValueError, TypeError):
                                # Handle cases where mtime might not be a valid float
                                existing_index[row['full_path']] = 0
                if debug:
                    logging.info(f"Loaded {len(existing_index)} entries from existing index")
        except Exception as e:
            logging.warning(f"Failed to load existing index for incremental indexing: {e}")
        timing_stats['load_existing_index'] = time.time() - load_start
    
    # Walk through all directories recursively
    scan_start = time.time()
    md_files = list(root_path.rglob("*.md"))  # Get list of all markdown files
    timing_stats['file_scanning'] = time.time() - scan_start
    
    metadata_start = time.time()
    for path in md_files:  # Currently only supporting markdown files
        # Skip files and folders starting with "_" or "."
        if any(part.startswith(('_', '.')) for part in path.parts):
            if debug:
                logging.info(f"Skipping excluded file/folder: {path}")
            continue
            
        if max_files is not None and len(entries_info) >= max_files:
            break
            
        full_path = str(path.absolute())
        
        # Check if file has changed for incremental indexing
        if incremental and full_path in existing_index:
            try:
                current_mtime = path.stat().st_mtime
                if current_mtime <= existing_index[full_path]:
                    if debug:
                        logging.info(f"Skipping unchanged file: {full_path}")
                    files_skipped += 1
                    continue
            except Exception as e:
                logging.warning(f"Failed to check modification time for {full_path}: {e}")
        
        # Extract metadata from file path
        metadata = extract_metadata(full_path, debug)
        
        # Remove code blocks from content
        metadata['content'] = remove_code_blocks(metadata['content'])
        
        if debug:
            logging.info(f"\nProcessed {len(entries_info) + 1} files")
            if max_files:
                logging.info(f"Progress: {len(entries_info) + 1} of {max_files}")
            logging.info(f"Title: '{metadata['title']}'")
        
        entries_info.append({
            'full_path': full_path,
            'date': metadata['date'],
            'title': metadata['title'],
            'content': metadata['content'],
            'properties': json.dumps(metadata['properties'], cls=CustomJSONEncoder),
            'tags': json.dumps(metadata['tags']),
            'path': metadata['path'],
            'size': metadata['size'],
            'internal_links': json.dumps(metadata['internal_links']),
            'external_links': json.dumps(metadata['external_links']),
            'mtime': metadata['mtime']  # Add modification time to the index
        })
        files_processed += 1
    
    timing_stats['metadata_extraction'] = time.time() - metadata_start
    
    # For incremental indexing, merge with existing entries
    merge_start = time.time()
    if incremental and existing_index:
        try:
            # Load all existing entries
            existing_entries = []
            index_file = config.output_dir / 'repo_index.csv'
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Skip entries that we've already updated
                        if row['full_path'] not in [entry['full_path'] for entry in entries_info]:
                            existing_entries.append(row)
                            if debug:
                                logging.info(f"Keeping existing entry: {row['full_path']}")
            
            # Combine existing entries with new/updated entries
            entries_info = existing_entries + entries_info
            
            if debug:
                logging.info(f"Merged {len(existing_entries)} existing entries with {len(entries_info) - len(existing_entries)} new/updated entries")
                logging.info("New/updated entries:")
                for entry in entries_info[len(existing_entries):]:
                    logging.info(f"- {entry['full_path']} (mtime: {entry['mtime']})")
        except Exception as e:
            logging.warning(f"Failed to merge with existing entries: {e}")
    
    timing_stats['merge_entries'] = time.time() - merge_start
    timing_stats['total_time'] = time.time() - start_time
    
    # Log timing information
    logging.info("\nIndexing Performance:")
    logging.info(f"Total files found: {len(md_files)}")
    logging.info(f"Files processed: {files_processed}")
    logging.info(f"Files skipped: {files_skipped}")
    logging.info("\nTiming breakdown:")
    for step, duration in timing_stats.items():
        logging.info(f"{step}: {duration:.2f} seconds")
    
    return entries_info, timing_stats

def save_to_csv(entries_info: List[Dict], output_file: Path) -> None:
    """
    Save indexed documents to a CSV file.
    
    Process:
    1. Read existing entries from CSV file if it exists
    2. Process new/updated entries
    3. Merge existing and updated entries
    4. Write all entries to CSV file
    
    Args:
        entries_info: List of dictionaries containing file information and metadata
        output_file: Path to the output CSV file
    """
    # Read existing entries from CSV file if it exists
    existing_entries = {}
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_entries[row['full_path']] = row
            logging.info(f"Read {len(existing_entries)} existing entries from {output_file}")
        except Exception as e:
            logging.warning(f"Failed to read existing CSV file: {e}")
    
    # Process new/updated entries
    updated_entries = {}
    for entry in entries_info:
        # Clean up the values - remove newlines and extra whitespace
        clean_entry = {
            'date': str(entry.get('date', '')).strip().replace('\n', ' '),
            'title': str(entry.get('title', '')).strip().replace('\n', ' '),
            'content': str(entry.get('content', '')).strip(),
            'properties': str(entry.get('properties', '{}')),
            'tags': str(entry.get('tags', '[]')),
            'path': str(entry.get('path', '')).strip().replace('\n', ' '),
            'size': str(entry.get('size', '')).strip().replace('\n', ' '),
            'internal_links': str(entry.get('internal_links', '[]')),
            'external_links': str(entry.get('external_links', '[]')),
            'full_path': str(entry.get('full_path', '')).strip().replace('\n', ' '),
            'mtime': str(entry.get('mtime', '0'))  # Add mtime field
        }
        updated_entries[clean_entry['full_path']] = clean_entry
    
    # Merge existing and updated entries
    final_entries = {**existing_entries, **updated_entries}
    logging.info(f"Writing {len(final_entries)} total entries to {output_file}")
    logging.info(f"- {len(existing_entries)} existing entries")
    logging.info(f"- {len(updated_entries)} new/updated entries")
    
    # Write all entries to CSV file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'date', 'title', 'content', 'properties', 'tags', 'path', 'size',
            'internal_links', 'external_links', 'full_path', 'mtime'
        ])
        writer.writeheader()
        for entry in final_entries.values():
            writer.writerow(entry)

def save_to_markdown(entries_info: List[Dict], output_file: str = None):
    """
    Save the indexed documents to a Markdown file.
    
    Process:
    1. Sort entries by date
    2. Format each entry in markdown with metadata
    3. Write to markdown file
    
    Args:
        entries_info: List of dictionaries containing document metadata
        output_file: Name of the output markdown file (default: from config)
    """
    if output_file is None:
        output_file = config.output_dir / 'repo_index.md'
    
    # Sort entries by date
    entries_info.sort(key=lambda x: x.get('date', ''))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries_info:
            # Write metadata
            f.write(f"**Date:** {entry['date']}\n")
            f.write(f"**Title:** {entry['title']}\n")
            f.write(f"**Path:** {entry['path']}\n")
            f.write(f"**Size:** {entry['size']} KB\n")
            
            # Write modification time if available
            if 'mtime' in entry:
                try:
                    mtime_float = float(entry['mtime'])
                    mtime_str = datetime.fromtimestamp(mtime_float).strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"**Last Modified:** {mtime_str}\n")
                except (ValueError, TypeError):
                    # Skip if mtime is not a valid float
                    pass
            
            # Write properties if available
            properties = json.loads(entry['properties'])
            if properties:
                f.write("**Properties:**\n")
                for key, value in properties.items():
                    f.write(f"- {key}: {value}\n")
            
            # Write tags if available
            tags = json.loads(entry['tags'])
            if tags:
                f.write("**Tags:**\n")
                for tag in tags:
                    f.write(f"- #{tag}\n")
            
            # Write links if available
            internal_links = json.loads(entry['internal_links'])
            external_links = json.loads(entry['external_links'])
            
            if internal_links:
                f.write("**Internal Links:**\n")
                for link in internal_links:
                    f.write(f"- {link}\n")
            
            if external_links:
                f.write("**External Links:**\n")
                for link in external_links:
                    f.write(f"- {link}\n")
            
            f.write("\n")
            
            # Write content preview (first 500 characters)
            if entry['content']:
                f.write("**Content Preview:**\n")
                preview = entry['content'][:500] + ("..." if len(entry['content']) > 500 else "")
                f.write(preview)
                f.write("\n\n")
            
            f.write("---\n\n")  # Add separator between entries

def build_bm25_index(entries_info: List[Dict]) -> None:
    """
    Build a BM25 index from the document entries.
    
    Args:
        entries_info: List of document entries with metadata
    """
    try:
        # Initialize BM25 retriever with config
        bm25_retriever = BM25Retriever(config.rag_config)
        
        # Convert entries to the format expected by BM25Retriever
        documents = []
        for entry in entries_info:
            # Create document text that includes both content and metadata
            text = f"{entry['title']} {entry['content']} {entry['tags']}"
            
            # Add properties if available
            if 'properties' in entry and entry['properties']:
                try:
                    properties = json.loads(entry['properties'])
                    for key, value in properties.items():
                        text += f" {key} {value}"
                except:
                    pass
            
            # Create document object
            doc = {
                'doc_id': f"{entry['date']}_{entry['title']}",
                'Date': entry['date'],
                'Title': entry['title'],
                'Content': entry['content'],
                'Tags': entry['tags'],
                'path': entry['path'],
                'size': entry['size'],
                'doc_type': entry['path'].split('/')[0] if '/' in entry['path'] else 'other'
            }
            
            # Add properties if available
            if 'properties' in entry and entry['properties']:
                try:
                    properties = json.loads(entry['properties'])
                    doc.update(properties)
                except:
                    pass
            
            documents.append(doc)
        
        # Index documents using BM25Retriever
        bm25_retriever.index_documents(documents)
        logging.info(f"Built BM25 index for {len(documents)} documents")
        
        # Save the index
        bm25_retriever._save_index(str(config.output_dir / 'bm25_index'))
        logging.info(f"Saved BM25 index to {config.output_dir / 'bm25_index'}")
        
    except Exception as e:
        logging.error(f"Error building BM25 index: {e}")

def main():
    """
    Main entry point for the indexing script.
    
    Features:
    1. Command line argument parsing
    2. Directory validation
    3. Progress reporting
    4. Summary output
    """
    parser = argparse.ArgumentParser(description='Index documents from markdown files')
    parser.add_argument('--debug',
                       action='store_true',
                       help='Enable debug output')
    parser.add_argument('--max-files',
                       type=int,
                       default=None,
                       help='Maximum number of files to process (default: unlimited)')
    parser.add_argument('--config',
                       type=str,
                       default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--force',
                       action='store_true',
                       help='Force a complete rebuild of the index')
    parser.add_argument('--build-bm25',
                       action='store_true',
                       help='Build BM25 index for search')
    parser.add_argument('--incremental',
                       action='store_true',
                       help='Perform incremental indexing (only process changed files)')
    args = parser.parse_args()
    
    # Update config if specified
    global config
    config = Config(args.config)
    
    # Validate directory
    if not config.index_dir.is_dir():
        logging.error(f"Error: {config.index_dir} is not a valid directory")
        return
    
    # If force option is used, remove existing index files
    if args.force:
        output_dir = config.output_dir
        index_files = ['repo_index.csv', 'repo_index.md']
        for file in index_files:
            file_path = output_dir / file
            if file_path.exists():
                logging.info(f"Removing existing index file: {file_path}")
                file_path.unlink()
    
    # Start timing for the entire process
    total_start_time = time.time()
    
    # Find and process all document files
    entries_info, indexing_stats = index_documents(str(config.index_dir), args.debug, args.max_files, args.incremental)
    
    # Save results in both formats
    save_start_time = time.time()
    save_to_csv(entries_info, config.output_dir / 'repo_index.csv')
    save_to_markdown(entries_info)
    save_time = time.time() - save_start_time
    
    # Build BM25 index if requested
    bm25_time = 0
    if args.build_bm25:
        bm25_start_time = time.time()
        build_bm25_index(entries_info)
        bm25_time = time.time() - bm25_start_time
    
    total_time = time.time() - total_start_time
    
    # Print summary
    logging.info(f"\nProcessed {len(entries_info)} document files" + 
          (f" (max: {args.max_files})" if args.max_files else "") + ".")
    
    if not args.debug:  # Only show entries if not in debug mode
        logging.info(f"Generated {config.output_dir}/repo_index.csv and {config.output_dir}/repo_index.md with the following entries:")
        for entry in entries_info:
            logging.info(f"- {entry['date']} - {entry['title']}")
    
    # Print overall timing summary
    logging.info("\nOverall Performance Summary:")
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Index saving time: {save_time:.2f} seconds")
    if args.build_bm25:
        logging.info(f"BM25 index build time: {bm25_time:.2f} seconds")
    
    # Print detailed indexing stats if available
    if indexing_stats:
        logging.info("\nDetailed Indexing Breakdown:")
        for step, duration in indexing_stats.items():
            if step != 'total_time':  # Already shown in overall summary
                logging.info(f"  {step}: {duration:.2f} seconds")
        
        # Calculate and show percentage breakdown
        logging.info("\nTime Distribution:")
        for step, duration in indexing_stats.items():
            if step != 'total_time':
                percentage = (duration / indexing_stats['total_time']) * 100
                logging.info(f"  {step}: {percentage:.1f}%")

if __name__ == "__main__":
    main() 