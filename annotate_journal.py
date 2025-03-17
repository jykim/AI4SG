#!/usr/bin/env python3
"""
Journal Entry Tagger

This script processes journal entries from a CSV file and adds semantic tags using OpenAI's GPT-4.
It can process new entries, retag existing entries, and maintains a separate annotated file.

Features:
- Processes only new or untagged entries by default
- Option to retag all entries
- Maintains original entries in input file
- Creates/updates an annotated version with tags
- Supports incremental updates

Tags generated:
- emotion: The emotional state expressed in the entry
- topic: Main topics or themes discussed
- etc: Additional contextual information
"""

import os
import csv
import openai
import argparse
import re
from pathlib import Path
import pandas as pd
import hashlib
import json

# OpenAI API configuration
openai.api_key = "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"

# Cache configuration
CACHE_DIR = Path('api_cache')
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(content: str) -> str:
    """
    Generate a cache key for the given content using SHA-256 hash.
    
    Args:
        content (str): The content to generate a key for
        
    Returns:
        str: The hex digest of the SHA-256 hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_cached_response(cache_key: str) -> dict:
    """
    Try to get a cached response for the given key.
    
    Args:
        cache_key (str): The cache key to look up
        
    Returns:
        dict: The cached response if found, None otherwise
    """
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading cache file: {e}")
    return None

def save_to_cache(cache_key: str, response: dict) -> None:
    """
    Save an API response to the cache.
    
    Args:
        cache_key (str): The cache key to save under
        response (dict): The response to cache including prompt and content
    """
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving to cache: {e}")

def load_prompt_template():
    """
    Load the prompt template for GPT-4 from file.
    
    Returns:
        str: The prompt template content
    
    Raises:
        FileNotFoundError: If tagging_prompt.md is not found
    """
    with open('tagging_prompt.md', 'r', encoding='utf-8') as f:
        return f.read()

def get_tags_for_entry(content):
    """
    Get semantic tags for a journal entry using OpenAI's GPT-4 API.
    Uses caching to avoid duplicate API calls.
    
    Args:
        content (str): The journal entry content to analyze
        
    Returns:
        tuple: (tags dict, suggested_title)
        
    Note:
        - Content is truncated if it exceeds token limits
        - Returns empty dict if API call fails
        - Uses cached response if available
    """
    # Truncate content if too long (to avoid token limit)
    max_chars = 2000
    if len(content) > max_chars:
        content = content[:max_chars] + "..."

    try:
        # Check cache first
        cache_key = get_cache_key(content)
        cached_response = get_cached_response(cache_key)
        
        if cached_response:
            print("Using cached response...")
            tags_text = cached_response['content']
            if 'prompt' in cached_response:
                print("(Prompt used: Journal entry analysis for tags [emotion/topic/etc])")
        else:
            # Load and format prompt template
            prompt_template = load_prompt_template()
            prompt = prompt_template.format(content=content)

            print("\nSending request to OpenAI API...")
            print("(Using prompt: Journal entry analysis for tags [emotion/topic/etc])")
            
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes journal entries and provides semantic tags in English, even for Korean text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract tags from response
            tags_text = response.choices[0].message.content
            
            # Cache both prompt and response
            save_to_cache(cache_key, {
                'content': tags_text,
                'prompt': prompt,
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        print(f"\nAPI Response:\n{tags_text}")
        
        # Parse tags and title from response
        tags = {}
        suggested_title = None
        
        for line in tags_text.split('\n'):
            if ':' in line:
                dimension, value = line.split(':', 1)
                dimension = dimension.strip().lower()  # Convert dimension to lowercase
                
                # Extract title if found
                if dimension == 'title':
                    suggested_title = value.strip()
                # Extract other tags
                elif dimension in ['emotion', 'topic', 'etc']:
                    # Convert tag value to lowercase and normalize
                    tag_value = normalize_tags(value.strip().lower())
                    tags[dimension] = tag_value
        
        return tags, suggested_title
    except Exception as e:
        print(f"Error getting tags: {e}")
        return {}, None

def normalize_tags(tag):
    """
    Normalize tag format by cleaning up separators and whitespace.
    
    Args:
        tag (str): Raw tag string
        
    Returns:
        str: Normalized tag string
        
    Example:
        >>> normalize_tags("[happy / excited]")
        "happy,excited"
    """
    if not tag:
        return ''
    # Remove square brackets
    tag = tag.replace('[', '').replace(']', '')
    # Replace any separator (/, |) with comma
    tag = re.sub(r'[/|]', ',', tag)
    # Remove extra spaces around commas
    tag = re.sub(r'\s*,\s*', ',', tag)
    # Remove leading/trailing spaces
    return tag.strip()

def is_entry_untagged(entry):
    """
    Check if an entry is missing any tags.
    
    Args:
        entry (dict): Journal entry with potential tag fields
        
    Returns:
        bool: True if any tag field is empty, False if all tags exist
    """
    # Get tag values, defaulting to empty string if missing
    emotion = entry.get('emotion', '').strip()
    topic = entry.get('topic', '').strip()
    etc = entry.get('etc', '').strip()
    
    # Entry is untagged if any tag field is empty
    is_untagged = not emotion or not topic or not etc
    
    # Debug logging
    if is_untagged:
        print(f"Entry is untagged: emotion='{emotion}', topic='{topic}', etc='{etc}'")
    else:
        print("Entry has all tags")
    
    return is_untagged

def update_csv_with_tags(input_csv_file, retag_all=False, target_date=None, dry_run=False):
    """
    Process journal entries and add semantic tags.
    
    Args:
        input_csv_file (str): Path to input CSV file
        retag_all (bool): Whether to retag all entries or just new/untagged ones
        target_date (str): Optional date to process (format: YYYY-MM-DD)
        dry_run (bool): If True, prints results without writing to file
    """
    # Get output directory from environment variable or use default
    output_dir = Path(os.environ.get('OUTPUT_DIR', 'output'))
    output_dir.mkdir(exist_ok=True)
    
    # Convert input path to Path object
    input_path = Path(input_csv_file)
    if not input_path.is_absolute():
        # If relative path doesn't exist in current directory, try in output directory
        if not input_path.exists() and (output_dir / input_path.name).exists():
            input_path = output_dir / input_path.name
    
    # Read input CSV first
    input_entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Add original_index to preserve order
        for idx, entry in enumerate(reader):
            entry['original_index'] = idx
            input_entries.append(entry)
    
    print(f"\nFound {len(input_entries)} total entries in input file")
    
    # Filter entries by date if specified
    if target_date:
        input_entries = [e for e in input_entries if e['Date'] == target_date]
        print(f"After date filter: {len(input_entries)} entries for date {target_date}")
    
    # Create a set of unique identifiers from input entries
    input_keys = {(e['Date'], e['Content']) for e in input_entries}
    print(f"Number of unique entries in input file: {len(input_keys)}")
    
    # Handle existing annotated file - use input filename with _annotated suffix
    output_filename = input_path.stem
    if target_date:
        output_filename += f'_{target_date}'
    output_filename += '_annotated.csv'
    output_csv_file = output_dir / output_filename
    
    # Create lookup of existing entries
    existing_lookup = {}
    if output_csv_file.exists() and not retag_all:  # Only use existing entries if not retagging all
        with open(output_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_entries = list(reader)
            print(f"Found {len(existing_entries)} entries in annotated file")
            
            # Check for duplicates in existing annotated file
            existing_keys = set()
            for e in existing_entries:
                key = (e['Date'], e['Content'])
                if key in existing_keys:
                    print(f"Warning: Duplicate entry found in annotated file - Date: {e['Date']}, Title: {e['Title']}")
                existing_keys.add(key)
            print(f"Number of unique entries in annotated file: {len(existing_keys)}")
            
            # Create lookup using Date and Content as key
            for e in existing_entries:
                key = (e['Date'], e['Content'])
                if key in input_keys:  # Only keep entries that exist in input file
                    existing_lookup[key] = e
    
    # Identify entries needing processing
    entries_to_process = []
    for entry in input_entries:
        key = (entry['Date'], entry['Content'])
            
        if key not in existing_lookup:
            print(f"\nNew entry found: {entry['Date']} - {entry['Title'] or 'Untitled'}")
            entries_to_process.append(entry)
        elif retag_all:
            print(f"\nRetagging: {entry['Date']} - {entry['Title'] or 'Untitled'}")
            entries_to_process.append(entry)
        elif not all(existing_lookup[key].get(field) for field in ['emotion', 'topic']):
            print(f"\nUntagged entry found: {entry['Date']} - {entry['Title'] or 'Untitled'}")
            entries_to_process.append(entry)
    
    print(f"\nFound {len(entries_to_process)} entries to process\n")
    
    # Process entries
    for i, entry in enumerate(entries_to_process, 1):
        print(f"\nProcessing entry {i}/{len(entries_to_process)}")
        
        content = entry['Content']
        if not content or pd.isna(content):
            print(f"Skipping empty content for entry: {entry['Date']} - {entry['Title'] or 'Untitled'}")
            continue
            
        print(f"Content length: {len(content)} characters")
        print(f"First 100 chars of content: {content[:100]}...")
        
        try:
            # Get tags from GPT
            tags, suggested_title = get_tags_for_entry(content)
            if not tags:
                print(f"Warning: No tags returned for entry: {entry['Date']} - {entry['Title'] or 'Untitled'}")
                continue
            
            # Update entry with tags and title if needed
            if suggested_title and not entry['Title']:
                entry['Title'] = suggested_title
                print(f"Using suggested title: {entry['Title']}")
            
            entry.update(tags)
            print(f"Adding tags: emotion={tags.get('emotion', '')}, topic={tags.get('topic', '')}, etc={tags.get('etc', '')}")
            
            # Update lookup using Date and Content as key
            key = (entry['Date'], entry['Content'])
            existing_lookup[key] = entry
            
        except Exception as e:
            print(f"Error processing entry: {str(e)}")
            continue
    
    # Create final entries list from input entries to maintain order
    final_entries = []
    for entry in input_entries:
        key = (entry['Date'], entry['Content'])
        if key in existing_lookup:
            final_entries.append(existing_lookup[key])
        else:
            final_entries.append(entry)
    
    print(f"\nFinal number of entries: {len(final_entries)}")
    
    # Sort entries by date and then by original_index
    final_entries.sort(key=lambda x: (x['Date'], x.get('original_index', 999999)))
    
    if dry_run:
        print("\nDRY RUN - Would write the following entries:")
        for entry in final_entries:
            if target_date and entry['Date'] == target_date:
                print(f"\nDate: {entry['Date']}")
                print(f"Title: {entry['Title']}")
                print(f"Original Index: {entry.get('original_index', 'N/A')}")
                print(f"Content preview: {entry['Content'][:50]}...")
                print("-" * 50)
        return
    
    # Write updated CSV with all fields
    fieldnames = ['Date', 'Title', 'Section', 'Content', 'Time', 'emotion', 'topic', 'etc']
    with open(output_csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Remove original_index before writing
        for entry in final_entries:
            entry_copy = entry.copy()
            if 'original_index' in entry_copy:
                del entry_copy['original_index']
            writer.writerow(entry_copy)
    print(f"\nCreated/Updated annotated CSV file: {output_csv_file}")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--retag-all', action='store_true',
                       help='Re-tag all entries, not just untagged ones')
    parser.add_argument('--input', default='journal_entries.csv',
                       help='Input CSV file path (default: journal_entries.csv)')
    parser.add_argument('--date', help='Process entries for specific date (format: YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print results without writing to file')
    
    args = parser.parse_args()
    
    print("Starting CSV processing...")
    update_csv_with_tags(args.input, args.retag_all, args.date, args.dry_run)
    print("\nAnnotation complete!")

if __name__ == '__main__':
    main()
