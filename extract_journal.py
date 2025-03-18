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

def extract_time_from_title(title):
    """Extract time from title if present (format: 10am / 7pm)"""
    time_match = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))', title.lower())
    if time_match:
        return time_match.group(1)
    return None

def extract_time_and_title_from_header(header):
    """Extract time and title from a header line (format: ### 10am Title or ### Title)"""
    if not header.startswith('### '):
        return None, None
        
    # Remove the ### prefix
    header = header[4:].strip()
    
    # Extract time if present
    time = extract_time_from_title(header)
    if time:
        # Extract title (everything after the time)
        title = header[header.find(time) + len(time):].strip()
        return time, title
    else:
        # If no time found, use the entire header as title
        return None, header

def split_into_parts(content):
    """Split content into parts using h3 headers or horizontal lines"""
    # First, handle the first header if it exists
    parts = []
    current_part = []
    
    for line in content.split('\n'):
        if line.startswith('### '):
            # If we have collected content, add it as a part
            if current_part:
                parts.append('\n'.join(current_part).strip())
                current_part = []
            # Add the header line
            current_part.append(line)
        elif line.strip() == '---':
            # If we have collected content, add it as a part
            if current_part:
                parts.append('\n'.join(current_part).strip())
                current_part = []
        else:
            current_part.append(line)
    
    # Add the last part if it exists
    if current_part:
        parts.append('\n'.join(current_part).strip())
    
    # Remove empty parts
    return [part for part in parts if part.strip()]

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

def extract_title_from_heading(content):
    """Extract title from the first heading in content if present"""
    lines = content.split('\n')
    for line in lines:
        if line.startswith('### '):
            return line[4:].strip()
    return None

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
            
        # Only process Reflections and Schedule sections
        if section_title not in ['Reflections', 'Schedule']:
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
                
            # Extract time and title from header for Schedule sections
            time = None
            entry_title = title
            if section_title == 'Schedule':
                # Extract time and title from the first line if it's a header
                first_line = part.split('\n')[0]
                if first_line.startswith('### '):
                    time, schedule_title = extract_time_and_title_from_header(first_line)
                    # Use schedule title as the main title
                    entry_title = schedule_title
                    # Remove the header line from content
                    part = '\n'.join(part.split('\n')[1:]).strip()
            else:
                # For Reflections, try to extract title from heading
                heading_title = extract_title_from_heading(part)
                if heading_title:
                    entry_title = heading_title
                    # Remove the heading line from content
                    lines = part.split('\n')
                    part = '\n'.join([l for l in lines if not l.startswith('### ')]).strip()
                
            entries.append((date, entry_title, section_title, part, time))
    
    return entries

def main():
    # Get output directory from environment variable or use default
    output_dir = Path(os.environ.get('OUTPUT_DIR', 'output'))
    output_dir.mkdir(exist_ok=True)
    
    # Journal directory path
    journal_dir = Path('/Users/lifidea/Library/Mobile Documents/iCloud~md~obsidian/Documents/OV2024/Journal')
    
    # Output files
    output_md = output_dir / 'journal_entries.md'
    output_csv = output_dir / 'journal_entries.csv'
    
    all_entries = []
    
    # Process all markdown files in the directory
    for file_path in journal_dir.glob('*.md'):
        entries = process_markdown_file(file_path)
        all_entries.extend(entries)
    
    # Sort entries by date and time
    all_entries.sort(key=lambda x: (x[0] if x[0] else '', x[4] if x[4] else ''))
    
    # Write to markdown file
    with open(output_md, 'w', encoding='utf-8') as f:
        for date, title, section_title, content, time in all_entries:
            f.write(f'## {date} {title}\n')
            if time:
                f.write(f'**Time:** {time}\n')
            f.write(f'**Section:** {section_title}\n')
            f.write(f'{content}\n\n')
    
    # Write to CSV file
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Title', 'Section', 'Content', 'Time'])
        for date, title, section_title, content, time in all_entries:
            writer.writerow([date, title, section_title, content, time])
    
    print(f"Processed {len(all_entries)} entries from journal files")
    print(f"Results saved to {output_md} and {output_csv}")

if __name__ == '__main__':
    main()
