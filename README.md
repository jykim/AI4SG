# AI4G - AI-Powered Journal Analysis

A dashboard application that processes and visualizes journal entries with AI-powered semantic analysis. The system extracts entries from markdown files, adds semantic tags using GPT-4, and provides an interactive dashboard for exploration.

## Features

- **Journal Entry Extraction**: Processes markdown files from Obsidian or similar note-taking apps
- **Semantic Analysis**: Uses GPT-4 to add emotion, topic, and contextual tags to entries
- **Interactive Dashboard**: Visualizes entries with timeline and detailed table views
- **Real-time Updates**: Automatically detects and processes new entries
- **Search & Filter**: Find entries by date, content, or tags
- **API Caching**: Optimizes API usage by caching GPT-4 responses

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Obsidian or similar markdown-based note-taking app

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI4G.git
cd AI4G
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

4. Configure the application:
   - Copy `config.yaml.example` to `config.yaml`
   - Update the paths in `config.yaml` to match your system:
     ```yaml
     input_dir: "input"  # Directory for raw journal files
     output_dir: "output"  # Directory for processed files
     api_cache_dir: "api_cache"  # Directory for API response cache
     journal_dir: "~/path/to/your/journal"  # Your journal directory
     ```

## Usage

### Running the Dashboard

Start the dashboard application:
```bash
python dash_journal.py
```

The dashboard will be available at http://127.0.0.1:8050/

### Manual Processing

1. Extract journal entries:
```bash
python extract_journal.py
```

2. Annotate entries with semantic tags:
```bash
python annotate_journal.py --input journal_entries.csv
```

### Dashboard Features

- **Timeline View**: Visual representation of entries over time
- **Data Table**: Detailed view of all entries with tags
- **Search**: Find entries by content or tags
- **Date Range**: Filter entries by date
- **Auto-Update**: Dashboard refreshes automatically when new entries are found
- **Manual Refresh**: Press Ctrl+Z to force a refresh of entries

## Configuration

The application is configured through `config.yaml`:

```yaml
# Directory Settings
input_dir: "input"  # Directory containing raw journal files
output_dir: "output"  # Directory for processed files
api_cache_dir: "api_cache"  # Directory for caching API responses
journal_dir: "~/path/to/your/journal"  # Your journal directory

# API Settings
openai_api_key: ""  # Set via environment variable

# Processing Settings
min_process_interval: 600  # Minimum seconds between processing runs
```

## Project Structure

```
AI4G/
├── config.yaml           # Configuration file
├── dash_journal.py       # Dashboard application
├── extract_journal.py    # Journal entry extraction
├── annotate_journal.py   # Semantic analysis
├── tagging_prompt.md     # GPT-4 prompt template
├── input/               # Raw journal files
├── output/              # Processed files
└── api_cache/           # Cached API responses
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 