import os
import re
import csv
from datetime import datetime
from pathlib import Path

def extract_date_and_title_from_filename(filename):
    """Extract date and title from filename (assuming format like YYYY-MM-DD-title.md)"""
    # Remove .md extension
    filename = filename.replace('.md', '')
    
    # Extract date
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if not date_match:
        return None, None
        
    date = date_match.group(1)
    
    # Extract title (everything after the date)
    title = filename[filename.find(date) + len(date):].strip('-')
    
    return date, title

def split_into_parts(content):
    """Split content into parts using h3 headers or horizontal lines"""
    # Split by horizontal lines or h3 headers
    parts = re.split(r'(?:^|\n)(?:---|### .*)\n', content)
    # Remove empty parts and strip whitespace
    return [part.strip() for part in parts if part.strip()]

def is_only_links_or_code(content):
    """Check if content contains only links or code blocks"""
    # Remove all whitespace
    content = re.sub(r'\s+', '', content)
    
    # Check if content is only links
    if re.match(r'^\[\[.*\]\]$', content):
        return True
        
    # Check if content is only code blocks
    if re.match(r'^```.*```$', content):
        return True
        
    return False

def process_markdown_file(file_path):
    """Process a single markdown file and extract sections"""
    # Get date and title from filename and skip if no date found
    date, title = extract_date_and_title_from_filename(file_path.name)
    if not date:
        return []
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by h2 headers
    sections = re.split(r'(?:^|\n)## ', content)
    
    entries = []
    for section in sections:
        if not section.strip():
            continue
            
        # Split section into title and content
        lines = section.split('\n', 1)
        section_title = lines[0].strip()
        
        # Skip if title is empty or just a horizontal line
        if not section_title or section_title == '---':
            continue
            
        # Only process Reflections section
        if section_title != 'Reflections':
            continue
            
        content = lines[1].strip() if len(lines) > 1 else ""
        
        # Skip if content is empty or just a horizontal line
        if not content or content == '---':
            continue
        
        # Split content into parts
        parts = split_into_parts(content)
        
        # Add each part as a separate entry
        for part in parts:
            # Skip if part is empty, just a horizontal line, or only contains links/code
            if not part or part == '---' or is_only_links_or_code(part):
                continue
            entries.append((date, title, section_title, part))
    
    return entries

def main():
    # Get output directory from environment variable or use default
    output_dir = Path(os.environ.get('OUTPUT_DIR', 'output'))
    output_dir.mkdir(exist_ok=True)
    
    # Journal directory path
    journal_dir = Path('/Users/lifidea/Library/Mobile Documents/iCloud~md~obsidian/Documents/OV2024/Journal')
    
    # Output files
    output_md = output_dir / 'reflections.md'
    output_csv = output_dir / 'reflections.csv'
    
    all_entries = []
    
    # Process all markdown files in the directory
    for file_path in journal_dir.glob('*.md'):
        entries = process_markdown_file(file_path)
        all_entries.extend(entries)
    
    # Sort entries by date
    all_entries.sort(key=lambda x: x[0] if x[0] else '')
    
    # Write to markdown file
    with open(output_md, 'w', encoding='utf-8') as f:
        for date, title, section_title, content in all_entries:
            f.write(f'## {date} {title}\n')
            f.write(f'{content}\n\n')
    
    # Write to CSV file
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Title', 'Section', 'Content'])
        for date, title, section_title, content in all_entries:
            writer.writerow([date, title, section_title, content])
    
    print(f"Processed {len(all_entries)} reflection entries from journal files")
    print(f"Results saved to {output_md} and {output_csv}")

if __name__ == '__main__':
    main()
