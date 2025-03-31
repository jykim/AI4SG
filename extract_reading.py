"""
Reading List Indexer
Scans directories for text files and generates a catalog with metadata.
Extracts date, title, author, and source from reading list entries.
"""

import os
import pathlib
import argparse
import csv
import unicodedata
from typing import List, Dict, Optional
import re
from pathlib import Path
from datetime import datetime
import yaml
import logging
import colorsys

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
                'reading_dir': '~/Library/Mobile Documents/iCloud~md~obsidian/Documents/OV2024/Reading',
                'output_dir': 'output'
            }
        
        # Set configuration values
        self.reading_dir = Path(os.path.expanduser(config.get('reading_dir', '~/Library/Mobile Documents/iCloud~md~obsidian/Documents/OV2024/Reading')))
        self.output_dir = Path(config.get('output_dir', 'output'))

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
        logging.FileHandler(config.output_dir / 'extract.log'),
        logging.StreamHandler()
    ]
)

def generate_color_from_string(text: str) -> str:
    """
    Generate a consistent color code from a string using a hash function.
    The color will be visually pleasing and maintain good contrast.
    
    Args:
        text: Input string to generate color from
        
    Returns:
        Hex color code in the format #RRGGBB
    """
    if not text:
        return '#95A5A6'  # Default gray for empty text
    
    # Use hash of the string to generate consistent colors
    hash_value = hash(text.lower())
    
    # Generate HSL values for a visually pleasing color
    # Use golden ratio to spread colors evenly
    golden_ratio = 0.618033988749895
    hue = (hash_value * golden_ratio) % 1.0  # Spread hues evenly
    saturation = 0.7  # Keep colors vibrant but not too intense
    lightness = 0.6  # Keep colors visible but not too dark
    
    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    
    # Convert to hex color code
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )
    
    return hex_color

def get_source_color(source: str) -> str:
    """
    Get color code for a source type.
    Generates a consistent color based on the source name.
    
    Args:
        source: Source string to get color for
        
    Returns:
        Hex color code for the source type
    """
    if not source:
        return '#95A5A6'  # Default gray for empty source
    
    # Clean up the source name
    clean_source = source.strip()
    
    # Remove language suffix if present
    if clean_source.endswith(' (kr)'):
        clean_source = clean_source[:-5].strip()
    
    # Generate color from the cleaned source name
    return generate_color_from_string(clean_source)

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

def read_first_lines(file_path: str, n_lines: int = 10) -> List[str]:
    """
    Read and normalize the first N lines from a text file.
    
    Handles potential encoding issues and file errors gracefully.
    Useful for extracting metadata from file content.
    
    Args:
        file_path: Path to the text file to read
        n_lines: Number of lines to read (default: 10)
        
    Returns:
        List of normalized text lines, empty list if file cannot be read
    """
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(n_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(normalize_text(line))
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {str(e)}")
        return []
    return lines

def extract_non_alphanumeric_strings(text: str) -> List[str]:
    """
    Extract meaningful strings enclosed in brackets or parentheses.
    
    Useful for finding metadata in filenames and folder names.
    Examples: [Author Name], (Series Title), {Volume Number}
    
    Args:
        text: Input text to process
        
    Returns:
        List of extracted strings with their enclosing characters
    """
    pattern = r'[\[\(\{].*?[\]\)\}]'
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches if m.strip() and len(m.strip()) > 2]

def is_meaningful_title(title: str) -> bool:
    """
    Check if a title string contains meaningful content.
    
    Validates that the title:
    1. Has at least 2 characters
    2. Is not just repetition of the same character
    3. Contains actual alphanumeric content
    
    Args:
        title: Title string to validate
        
    Returns:
        Boolean indicating if title is meaningful
    """
    clean_title = ''.join(c for c in title if c.isalnum() or c.isspace()).strip()
    if len(clean_title) < 2:
        return False
    if len(set(clean_title)) == 1:
        return False
    return True

def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract date from filename in YYYY-MM-DD format.
    
    Args:
        filename: Name of the file
        
    Returns:
        Date string in YYYY-MM-DD format if found, None otherwise
    """
    # Remove extension
    filename = pathlib.Path(filename).stem
    
    # Extract date pattern YYYY-MM-DD
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if date_match:
        try:
            # Validate date format
            date_str = date_match.group(1)
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            return None
    return None

def extract_title_from_filename(filename: str) -> str:
    """
    Extract title from filename after the date.
    
    Args:
        filename: Name of the file
        
    Returns:
        Title string without date prefix
    """
    # Remove extension
    filename = pathlib.Path(filename).stem
    
    # Remove date if present
    date_match = re.search(r'\d{4}-\d{2}-\d{2}-(.*)', filename)
    if date_match:
        return date_match.group(1).strip()
    
    # If no date pattern found, remove any date-like prefix
    title = re.sub(r'^\d{4}-\d{2}-\d{2}\s*', '', filename)
    return title.strip()

def extract_title(file_path: str, first_lines: List[str], debug: bool = False) -> str:
    """
    Extract the most likely title using multiple strategies.
    
    Priority order:
    1. Parent folder name if meaningful
    2. First non-blank line of content
    3. Filename without extension
    4. Special strings from filename
    
    Args:
        file_path: Path to the text file
        first_lines: First few lines of file content
        debug: Enable detailed debug output
        
    Returns:
        Best candidate for book title, or "Unknown Title" if none found
    """
    if debug:
        print("\nDEBUG - Processing:", file_path)
    
    # Get filename special strings first
    filename = pathlib.Path(file_path).stem
    filename_special = extract_non_alphanumeric_strings(filename)
    if debug and filename_special:
        print(f"DEBUG - Found special strings in filename: {filename_special}")
    
    # Get parent folder name
    parent_folder = pathlib.Path(file_path).parent.name
    parent_folder_title = normalize_text(parent_folder.replace('_', ' ').replace('-', ' ')) if parent_folder != '.' else ''
    
    # Get base title using priority order
    base_title = None
    
    # Try parent folder first if it's meaningful
    if parent_folder_title and is_meaningful_title(parent_folder_title):
        base_title = parent_folder_title.strip()
        if debug:
            print(f"DEBUG - Using parent folder title: '{base_title}'")
    
    # If no good parent folder title, try first non-blank line
    if not base_title:
        if debug:
            print("DEBUG - Checking first lines:")
        for line in first_lines:
            line = normalize_text(line)
            if line and is_meaningful_title(line):
                base_title = line.strip()
                if debug:
                    print(f"DEBUG - Using first non-blank line: '{base_title}'")
                break
    
    # If still no title, try filename
    if not base_title:
        clean_filename = normalize_text(filename.replace('_', ' ').replace('-', ' ')).strip()
        if is_meaningful_title(clean_filename):
            base_title = clean_filename
            if debug:
                print(f"DEBUG - Using filename: '{base_title}'")
    
    # If we still don't have a meaningful title, try using the special strings
    if not base_title and filename_special:
        base_title = filename_special[0].strip('[](){} ')
        if debug:
            print(f"DEBUG - Using first special string as title: '{base_title}'")
    
    # If we still don't have a title, use Unknown Title
    if not base_title:
        return "Unknown Title"
    
    # Append special strings from filename if they're meaningful and not already in title
    if filename_special:
        # Only append strings that aren't already in the base title
        new_special = [s for s in filename_special 
                      if s.lower() not in base_title.lower() 
                      and len(s) > 2  # Must be longer than 2 chars
                      and not s.strip('[](){} ').isdigit()]  # Not just numbers
        if new_special:
            base_title = f"{base_title} ({' '.join(new_special)})"
            if debug:
                print(f"DEBUG - Added special strings to title: '{base_title}'")
    
    return base_title

def load_prompt_template() -> str:
    """Load the metadata extraction prompt template."""
    template_path = Path(__file__).parent / 'extract_metadata.md'
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_metadata(file_path: str, debug: bool = False) -> Dict[str, str]:
    """
    Extract metadata from file path and content.
    
    Extraction process:
    1. Extract date from filename (YYYY-MM-DD format)
    2. Extract title from filename
    3. Extract metadata from YAML frontmatter
    4. Get file size
    5. Extract URL from frontmatter or markdown content
    6. Extract source from folder name if not in frontmatter
    7. Get first 2000 bytes of content
    
    Args:
        file_path: Path to the text file
        debug: Enable detailed debug output
        
    Returns:
        Dictionary containing:
        - date: Reading date in YYYY-MM-DD format
        - title: Reading title
        - author: Author name
        - source: Source of the reading (from frontmatter or folder name)
        - source_color: Color code for the source type
        - size: File size in human readable format
        - url: URL of the reading material
        - content: First 2000 bytes of content
    """
    try:
        path = pathlib.Path(file_path)
        filename = path.name
        
        # Extract date and title from filename
        date = extract_date_from_filename(filename)
        title = extract_title_from_filename(filename)
        
        # Get file size in KB
        file_size = path.stat().st_size / 1024  # Convert bytes to KB
        size = f"{file_size:.1f}"  # Keep one decimal place
        
        # Initialize metadata
        metadata = {
            'date': date or '',
            'title': title.strip(),
            'author': '',
            'source': '',
            'source_color': '',
            'size': size,
            'url': '',
            'content': ''
        }
        
        # Read file content to extract YAML frontmatter and URL
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract YAML frontmatter if present
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            try:
                # Parse YAML frontmatter
                frontmatter_data = yaml.safe_load(frontmatter)
                
                # Extract metadata from frontmatter
                if frontmatter_data:
                    # Handle author field (remove "By " prefix if present)
                    if 'author' in frontmatter_data:
                        author = frontmatter_data['author']
                        if author.startswith('By '):
                            author = author[3:].strip()
                        metadata['author'] = author
                    
                    # Handle source field (combine multiple sources if present)
                    sources = []
                    if 'source' in frontmatter_data:
                        if isinstance(frontmatter_data['source'], list):
                            sources.extend(frontmatter_data['source'])
                        else:
                            sources.append(frontmatter_data['source'])
                    if sources:
                        metadata['source'] = ', '.join(sources)
                    
                    # Extract URL from frontmatter if present
                    if 'url' in frontmatter_data:
                        metadata['url'] = frontmatter_data['url']
                    
                    # Use frontmatter date if available
                    if 'date' in frontmatter_data:
                        try:
                            # Convert date to string if it's a datetime object
                            date_str = str(frontmatter_data['date'])
                            # Validate date format
                            datetime.strptime(date_str, '%Y-%m-%d')
                            metadata['date'] = date_str
                        except (ValueError, TypeError):
                            pass  # Keep existing date if frontmatter date is invalid
                
            except yaml.YAMLError as e:
                if debug:
                    print(f"Warning: Failed to parse YAML frontmatter: {str(e)}")
        
        # If URL not found in frontmatter, try to extract from markdown content
        if not metadata['url']:
            # Look for URL in markdown link after "By"
            url_match = re.search(r'By\s+\[([^\]]+)\]\(([^)]+)\)', content)
            if url_match:
                metadata['url'] = url_match.group(2)
        
        # If source is empty, try to get it from the folder name
        if not metadata['source']:
            # Get the parent folder name and normalize it
            parent_folder = path.parent.name
            if parent_folder and parent_folder != '.':
                # Clean up the folder name by removing common separators and normalizing spaces
                clean_folder = re.sub(r'[_-]', ' ', parent_folder)
                clean_folder = ' '.join(clean_folder.split())  # Normalize spaces
                if clean_folder and is_meaningful_title(clean_folder):
                    metadata['source'] = clean_folder
                    if debug:
                        print(f"DEBUG - Using folder name as source: '{clean_folder}'")
        
        # Set source color based on the source type
        metadata['source_color'] = get_source_color(metadata['source'])
        
        # Get entire content after frontmatter
        content_without_frontmatter = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
        metadata['content'] = content_without_frontmatter.strip()
        
        if debug:
            print(f"DEBUG - Extracted metadata: {metadata}")
            
        return metadata
        
    except Exception as e:
        print(f"Warning: Failed to extract metadata for {file_path}: {str(e)}")
        return {
            'date': '',
            'title': '',
            'author': '',
            'source': '',
            'source_color': '#95A5A6',  # Default gray color
            'size': '',
            'url': '',
            'content': ''
        }

def index_txt_files(root_dir: str, debug: bool = False, max_files: Optional[int] = None) -> List[Dict]:
    """
    Main indexing function that processes all markdown files in a directory.
    
    Process:
    1. Recursively find all .md files
    2. Extract metadata for each file
    3. Apply file limit if specified
    
    Args:
        root_dir: Root directory to scan for markdown files
        debug: Enable detailed debug output
        max_files: Maximum number of files to process
        
    Returns:
        List of dictionaries containing file information and metadata
    """
    entries_info = []
    root_path = pathlib.Path(root_dir)
    
    # Walk through all directories recursively
    for path in root_path.rglob("*.md"):
        if max_files is not None and len(entries_info) >= max_files:
            break
            
        full_path = str(path.absolute())
        
        # Extract metadata from file path
        metadata = extract_metadata(full_path, debug)
        
        if debug:
            print(f"\nProcessed {len(entries_info) + 1} files")
            if max_files:
                print(f"Progress: {len(entries_info) + 1} of {max_files}")
            print(f"Title: '{metadata['title']}'")
        
        entries_info.append({
            'full_path': full_path,
            'date': metadata['date'],
            'title': metadata['title'],
            'author': metadata['author'],
            'source': metadata['source'],
            'source_color': metadata['source_color'],
            'size': metadata['size'],
            'url': metadata['url'],
            'content': metadata['content']
        })
    
    return entries_info

def save_to_csv(entries_info: List[Dict], output_file: str = None):
    """
    Save the indexed reading entries to a CSV file.
    
    Process:
    1. Clean and normalize all text fields
    2. Remove newlines and extra whitespace
    3. Write to CSV with proper quoting
    
    Args:
        entries_info: List of dictionaries containing entry metadata
        output_file: Name of the output CSV file (default: from config)
    """
    if output_file is None:
        output_file = config.output_dir / 'reading_entries.csv'
    
    fieldnames = ['date', 'title', 'author', 'source', 'source_color', 'size', 'url', 'content', 'full_path']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for entry in entries_info:
            # Clean up the values - remove newlines and extra whitespace
            clean_entry = {
                'date': str(entry.get('date', '')).strip().replace('\n', ' '),
                'title': str(entry.get('title', '')).strip().replace('\n', ' '),
                'author': str(entry.get('author', '')).strip().replace('\n', ' '),
                'source': str(entry.get('source', '')).strip().replace('\n', ' '),
                'source_color': str(entry.get('source_color', '#95A5A6')).strip(),
                'size': str(entry.get('size', '')).strip().replace('\n', ' '),
                'url': str(entry.get('url', '')).strip().replace('\n', ' '),
                'content': str(entry.get('content', '')).strip(),
                'full_path': str(entry.get('full_path', '')).strip().replace('\n', ' ')
            }
            writer.writerow(clean_entry)

def save_to_markdown(entries_info: List[Dict], output_file: str = None):
    """
    Save the indexed reading entries to a Markdown file.
    
    Process:
    1. Sort entries by date
    2. Format each entry in markdown with full content
    3. Write to markdown file
    
    Args:
        entries_info: List of dictionaries containing entry metadata
        output_file: Name of the output markdown file (default: from config)
    """
    if output_file is None:
        output_file = config.output_dir / 'reading_entries.md'
    
    # Sort entries by date
    entries_info.sort(key=lambda x: x.get('date', ''))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries_info:
            # Write metadata
            f.write(f"**Date:** {entry['date']}\n")
            if entry['author']:
                f.write(f"**Author:** {entry['author']}\n")
            if entry['source']:
                f.write(f"**Source:** <span style='color: {entry['source_color']}'>{entry['source']}</span>\n")
            if entry['url']:
                f.write(f"**URL:** {entry['url']}\n")
            f.write("\n")
            
            # Write content preview
            if entry['content']:
                f.write("**Content:**\n")
                f.write(entry['content'])
                f.write("\n\n")
            
            # Read and include full content
            try:
                with open(entry['full_path'], 'r', encoding='utf-8') as content_file:
                    content = content_file.read()
                    # Remove YAML frontmatter if present
                    content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
                    f.write(content.strip())
                    f.write("\n\n---\n\n")  # Add separator between entries
            except Exception as e:
                f.write(f"*Error reading content: {str(e)}*\n\n---\n\n")

def main():
    """
    Main entry point for the indexing script.
    
    Features:
    1. Command line argument parsing
    2. Directory validation
    3. Progress reporting
    4. Summary output
    """
    parser = argparse.ArgumentParser(description='Index reading list entries from markdown files')
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
    args = parser.parse_args()
    
    # Update config if specified
    global config
    config = Config(args.config)
    
    # Validate directory
    if not config.reading_dir.is_dir():
        print(f"Error: {config.reading_dir} is not a valid directory")
        return
    
    # Find and process all markdown files
    entries_info = index_txt_files(str(config.reading_dir), args.debug, args.max_files)
    
    # Save results in both formats
    save_to_csv(entries_info)
    save_to_markdown(entries_info)
    
    # Print summary
    print(f"\nProcessed {len(entries_info)} markdown files" + 
          (f" (max: {args.max_files})" if args.max_files else "") + ".")
    if not args.debug:  # Only show entries if not in debug mode
        print(f"Generated {config.output_dir}/reading_entries.csv and {config.output_dir}/reading_entries.md with the following entries:")
        for entry in entries_info:
            print(f"- {entry['date']} - {entry['title']}")

if __name__ == "__main__":
    main()
