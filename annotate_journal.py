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

# OpenAI API configuration
openai.api_key = "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"

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
    
    Args:
        content (str): The journal entry content to analyze
        
    Returns:
        dict: A dictionary containing emotion, topic, and etc tags
        
    Note:
        - Content is truncated if it exceeds token limits
        - Returns empty dict if API call fails
    """
    # Truncate content if too long (to avoid token limit)
    max_chars = 2000
    if len(content) > max_chars:
        content = content[:max_chars] + "..."

    try:
        # Load and format prompt template
        prompt_template = load_prompt_template()
        prompt = prompt_template.format(content=content)

        print("\nSending request to OpenAI API...")
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes journal entries and provides semantic tags in English, even for Korean text."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract tags from response
        tags_text = response.choices[0].message.content
        print(f"\nAPI Response:\n{tags_text}")
        
        # Parse tags from response
        tags = {}
        for line in tags_text.split('\n'):
            if ':' in line:
                dimension, tag = line.split(':', 1)
                tags[dimension.strip()] = tag.strip()
        
        print(f"Extracted tags: {tags}")
        return tags
    except Exception as e:
        print(f"Error getting tags: {e}")
        return {}

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

def update_csv_with_tags(input_csv_file, retag_all=False):
    """
    Process journal entries and add semantic tags.
    
    Args:
        input_csv_file (str): Path to input CSV file
        retag_all (bool): Whether to retag all entries or just new/untagged ones
        
    The function:
    1. Reads entries from input file
    2. Checks for existing annotated file
    3. Identifies new or untagged entries
    4. Gets tags using GPT-4
    5. Saves updated entries to annotated file
    """
    # Get output directory from environment variable or use default
    output_dir = Path(os.environ.get('OUTPUT_DIR', 'output'))
    output_dir.mkdir(exist_ok=True)
    
    # Convert input path to Path object and ensure it's in output directory
    input_path = Path(input_csv_file)
    if not input_path.is_absolute():
        input_path = output_dir / input_path
    
    # Read input CSV first
    input_entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        input_entries = list(reader)
    
    print(f"\nFound {len(input_entries)} entries in input file")
    
    # Handle existing annotated file
    output_csv_file = output_dir / 'reflections_annotated.csv'
    if output_csv_file.exists():
        # Read existing annotations
        with open(output_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_entries = list(reader)
        print(f"Found {len(existing_entries)} entries in annotated file")
        
        # Find new entries by comparing dates and content
        existing_keys = {(e['Date'], e['Content']) for e in existing_entries}
        new_entries = [e for e in input_entries if (e['Date'], e['Content']) not in existing_keys]
        
        if new_entries:
            print(f"\nFound {len(new_entries)} new entries to process")
            # Initialize tag fields for new entries
            for entry in new_entries:
                entry['emotion'] = ''
                entry['topic'] = ''
                entry['etc'] = ''
            entries = existing_entries + new_entries
        else:
            print("\nNo new entries found")
            entries = existing_entries
    else:
        print("\nNo existing annotated file found, processing all entries")
        entries = input_entries
        # Initialize tag fields
        for entry in entries:
            entry['emotion'] = ''
            entry['topic'] = ''
            entry['etc'] = ''
    
    # Filter entries to process
    if not retag_all:
        entries_to_process = [e for e in entries if is_entry_untagged(e)]
        print(f"\nFound {len(entries_to_process)} entries to process (new or untagged)")
    else:
        entries_to_process = entries
        print(f"\nRe-tagging all {len(entries_to_process)} entries")
    
    if not entries_to_process:
        print("\nNo entries to process. Exiting.")
        return
    
    # Process entries
    for i, entry in enumerate(entries_to_process, 1):
        print(f"\nProcessing entry {i}/{len(entries_to_process)}")
        print(f"Content length: {len(entry['Content'])} characters")
        print(f"First 100 chars of content: {entry['Content'][:100]}...")
        
        # Get and normalize tags
        tags = get_tags_for_entry(entry['Content'])
        entry['emotion'] = normalize_tags(tags.get('emotion', ''))
        entry['topic'] = normalize_tags(tags.get('topic', ''))
        entry['etc'] = normalize_tags(tags.get('etc', ''))
        
        print(f"Adding tags: emotion={entry['emotion']}, topic={entry['topic']}, etc={entry['etc']}")
    
    # Save results
    with open(output_csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Date', 'Title', 'Section', 'Content', 'emotion', 'topic', 'etc'])
        writer.writeheader()
        writer.writerows(entries)
    print(f"\nCreated/Updated annotated CSV file: {output_csv_file}")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--retag-all', action='store_true',
                       help='Re-tag all entries, not just untagged ones')
    parser.add_argument('--input', default='reflections.csv',
                       help='Input CSV file path (default: reflections.csv)')
    
    args = parser.parse_args()
    
    print("Starting CSV processing...")
    update_csv_with_tags(args.input, args.retag_all)
    print("\nAnnotation complete!")

if __name__ == '__main__':
    main()
