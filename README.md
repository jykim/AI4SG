# Journal Entry Tagger

A Python tool that processes journal entries and adds semantic tags using OpenAI's GPT-4. The tool analyzes journal entries and adds structured tags for emotions, topics, and additional context, along with visual elements (colors and emojis) for better visualization.

## Features

- **Semantic Tagging**: Automatically adds three types of tags to each entry:
  - `emotion`: The emotional state expressed in the entry (with color)
  - `topic`: Main topics or themes discussed (with emoji)
  - `etc`: Additional contextual information (with emoji)

- **Visual Elements**: Each tag includes a visual element:
  - Emotions have associated colors (e.g., "happy / #FFD700")
  - Topics have associated emojis (e.g., "travel / ✈️")
  - Additional tags have associated emojis (e.g., "family / 👨‍👩‍👧‍👦")

- **Smart Processing**:
  - Processes only new or untagged entries by default
  - Option to retag all entries
  - Maintains original entries in input file
  - Creates/updates an annotated version with tags
  - Supports incremental updates

- **Caching**: Uses a local cache to avoid duplicate API calls for the same content

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/journal-entry-tagger.git
cd journal-entry-tagger
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Basic Usage

Process new or untagged entries:
```bash
python annotate_journal.py --input journal_entries.csv
```

### Command Line Options

- `--input`: Input CSV file path (default: journal_entries.csv)
- `--retag-all`: Re-tag all entries, not just untagged ones
- `--date`: Process entries for specific date (format: YYYY-MM-DD)
- `--dry-run`: Print results without writing to file

### Input CSV Format

The input CSV should have the following columns:
- `Date`: Entry date (YYYY-MM-DD)
- `Title`: Entry title (optional)
- `Section`: Section name (optional)
- `Content`: Entry content
- `Time`: Entry time (optional)

### Output Format

The script creates an annotated CSV file with the following additional columns:
- `emotion`: Emotional state (e.g., "happy")
- `emotion_visual`: Color code (e.g., "#FFD700")
- `topic`: Main topic (e.g., "travel")
- `topic_emoji`: Associated emoji (e.g., "✈️")
- `etc`: Additional context (e.g., "family")
- `etc_emoji`: Associated emoji (e.g., "👨‍👩‍👧‍👦")

## Project Structure

```
journal-entry-tagger/
├── annotate_journal.py    # Main script for processing entries
├── tagging_prompt.md      # GPT-4 prompt template
├── requirements.txt       # Python dependencies
├── api_cache/            # Cache directory for API responses
└── output/               # Directory for annotated CSV files
```

## Dependencies

- Python 3.8+
- OpenAI API
- pandas
- dash (for visualization)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 