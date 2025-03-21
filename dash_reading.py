"""
Reading with AI Dashboard
A Dash web application for managing and reading text files with encoding support.
Specifically designed for handling Korean text files with various encodings.
"""

import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.express as px
from flask import send_file
import os
import urllib.parse
import re
import webbrowser
import csv

# Initialize the Dash app
app = dash.Dash(__name__)

# Configuration
app.server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Constants for encodings
DEFAULT_ENCODINGS = ['utf-8', 'cp949', 'euc-kr']  # List of default encodings to try

# Load and prepare data
def load_library_data():
    """Load and sort the library catalog with ratings."""
    try:
        df = pd.read_csv('output/reading_entries.csv')
        if df.empty:
            return pd.DataFrame()
            
        # Load ratings
        ratings = load_ratings()
        
        # Add ratings column
        def get_rating_stars(row):
            rating = ratings.get(row['full_path'], 0)
            return '★' * rating if rating > 0 else ''
        
        df['rating'] = df.apply(get_rating_stars, axis=1)
        return df.sort_values('date', ascending=False, na_position='last')
    except Exception as e:
        print(f"Error loading library data: {e}")
        return pd.DataFrame()

def load_ratings():
    """
    Load book ratings from CSV file.
    Creates the file if it doesn't exist.
    
    Returns:
        Dictionary mapping file paths to ratings
    """
    ratings = {}
    try:
        with open('output/reading_ratings.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ratings[row['file_path']] = int(row['rating'])
    except FileNotFoundError:
        # Create the file with headers if it doesn't exist
        with open('output/reading_ratings.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_path', 'rating'])
            writer.writeheader()
    return ratings

def save_rating(file_path, rating):
    """
    Save or update a book's rating in the CSV file.
    
    Args:
        file_path: Path to the book file
        rating: Integer rating (1-3)
    """
    ratings = load_ratings()
    ratings[file_path] = rating
    
    with open('output/reading_ratings.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file_path', 'rating'])
        writer.writeheader()
        for path, r in ratings.items():
            writer.writerow({'file_path': path, 'rating': r})

# Initialize data
df = load_library_data()

def read_book_content(file_path, fallback_encoding='cp949'):
    """
    Read book content with multiple encoding fallbacks.
    
    Process:
    1. Try default encodings in sequence (UTF-8, CP949, EUC-KR)
    2. Fall back to specified encoding if defaults fail
    3. Handle encoding errors gracefully
    
    Args:
        file_path: Path to the text file
        fallback_encoding: Last-resort encoding to try
        
    Returns:
        Tuple of (content, error_message, used_encoding)
        - content: File contents if successful, None if failed
        - error_message: None if successful, error description if failed
        - used_encoding: The encoding that successfully read the file
    """
    # Try default encodings first
    for encoding in DEFAULT_ENCODINGS:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return content, None, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return None, f"Error reading file: {str(e)}", None

    # If default encodings fail, try the fallback encoding
    if fallback_encoding not in DEFAULT_ENCODINGS:
        try:
            with open(file_path, 'r', encoding=fallback_encoding) as f:
                content = f.read()
            return content, None, fallback_encoding
        except Exception as e:
            return None, f"Error reading file with {fallback_encoding}: {str(e)}", None

    return None, "Could not read file with any supported encoding", None

# Store the current file path globally
current_file = {'path': None}

def filter_metadata(content):
    """
    Remove YAML-style metadata block from content.
    
    Args:
        content: Text content to filter
        
    Returns:
        Content without metadata block
    """
    if not content:
        return ""
    
    # Pattern to match YAML-style metadata block
    # Matches content between --- markers, including the markers
    pattern = r'^---\s*\n(.*?)\n---\s*\n'
    
    # Remove metadata block if present
    filtered_content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    return filtered_content.strip()

def format_markdown(content):
    """
    Basic markdown formatting for titles and links.
    
    Handles:
    1. Headers (# Title)
    2. Links [text](url)
    
    Args:
        content: Text content to format
        
    Returns:
        Formatted markdown content
    """
    if not content:
        return ""
    
    # First filter out metadata
    content = filter_metadata(content)
    
    # Split into lines
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Handle headers
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            if level <= 3:  # Only handle h1-h3
                formatted_lines.append(f'{"#" * level} {text}')
            continue
        
        # Handle links
        if '[' in line and '](' in line:
            # Keep markdown links as is
            formatted_lines.append(line)
            continue
        
        # Regular text
        if line.strip():
            formatted_lines.append(line)
        else:
            formatted_lines.append('')
    
    return '\n'.join(formatted_lines)

# Create the layout
app.layout = html.Div([
    html.Div([
        # Left side panel
        html.Div([
            # Title and Source Filter
            html.Div([
                html.H2('Reading with AI', 
                    style={
                        'margin': '0',
                        'padding': '10px',
                        'color': '#2c3e50',
                        'fontSize': '20px',
                        'fontWeight': 'bold',
                        'borderBottom': '1px solid #eee'
                    }
                ),
                html.Div([
                    html.Div([
                        html.Label('Source:', style={'marginRight': '5px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='genre-filter',
                            options=[{'label': source, 'value': source} 
                                   for source in df['source'].dropna().unique().tolist()],
                            value='All Sources',
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1', 'marginRight': '10px'}),
                    html.Div([
                        html.Label('Rating:', style={'marginRight': '5px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='rating-filter',
                            options=[
                                {'label': '★', 'value': 1},
                                {'label': '★★', 'value': 2},
                                {'label': '★★★', 'value': 3}
                            ],
                            value=[],
                            multi=True,
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1'}),
                    html.Div([
                        html.Label('Search:', style={'marginRight': '5px', 'fontWeight': 'bold'}),
                        dcc.Input(
                            id='text-filter',
                            type='text',
                            placeholder='Search title or author...',
                            style={'width': '100%', 'padding': '5px', 'border': '1px solid #ddd', 'borderRadius': '4px'}
                        ),
                    ], style={'flex': '1'})
                ], style={
                    'display': 'flex',
                    'margin': '10px',
                    'gap': '10px'
                }),
            ], style={
                'backgroundColor': 'white',
                'borderBottom': '1px solid #ddd'
            }),
            # DataTable
            html.Div([
                dash_table.DataTable(
                    id='book-table',
                    columns=[
                        {'name': 'Date', 'id': 'date', 'type': 'datetime', 'format': {'specifier': '%Y-%m-%d'}},
                        {'name': 'Title', 'id': 'title', 'type': 'text'},
                        {'name': 'Author', 'id': 'author', 'type': 'text'},
                        {'name': 'Source', 'id': 'source', 'type': 'text'},
                        {'name': 'Size (KB)', 'id': 'size', 'type': 'numeric',
                         'format': {'specifier': '.1f'}},
                        {'name': 'Rating', 'id': 'rating', 'type': 'text'},
                    ],
                    data=df.to_dict('records'),
                    style_table={
                        'height': 'calc(100vh - 200px)',  # Adjusted to make room for info panel
                        'overflowY': 'auto',
                        'overflowX': 'auto',
                    },
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'fontFamily': 'Arial',
                        'fontSize': '14px',
                        'minWidth': '100px',
                        'maxWidth': '400px',
                    },
                    style_header={
                        'backgroundColor': '#34495e',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'position': 'sticky',
                        'top': 0,
                        'zIndex': 1000
                    },
                    style_data={
                        'backgroundColor': 'white',
                        'color': '#2c3e50',
                        'cursor': 'pointer'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#f9f9f9'
                        }
                    ],
                    virtualization=True,
                    page_action='none',
                    fixed_rows={'headers': True},
                    sort_action="native",
                    sort_mode="multi",
                    filter_action="native",
                    row_selectable=False,
                    cell_selectable=True,
                    style_filter={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'position': 'sticky',
                        'top': '40px',
                        'zIndex': 1000
                    }
                ),
                # Book Info Panel
                html.Div([
                    html.Div([
                        html.Strong('Selected Book Info:', style={'marginBottom': '5px'}),
                        html.Div([
                            html.Div(id='selected-book-info'),
                            html.A(
                                'Search on Google',
                                id='google-search-link',
                                target='_blank',
                                rel="noopener noreferrer",
                                style={
                                    'color': '#2196F3',
                                    'textDecoration': 'none',
                                    'marginTop': '5px',
                                    'display': 'inline-block'
                                }
                            )
                        ], style={'marginTop': '5px'})
                    ], style={
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'border': '1px solid #ddd',
                        'borderRadius': '4px'
                    })
                ], style={
                    'marginTop': '10px',
                    'height': '80px',  # Fixed height for info panel
                    'backgroundColor': 'white'
                })
            ], style={'flex': '1', 'overflow': 'hidden', 'display': 'flex', 'flexDirection': 'column'})
        ], style={
            'flex': '0.45',  # Changed to 0.45 for 45%
            'height': '100vh',
            'backgroundColor': 'white',
            'overflow': 'hidden',
            'display': 'flex',
            'flexDirection': 'column'
        }),
        
        # Right side - Book Content
        html.Div([
            # Header with title, download link, and encoding selector
            html.Div([
                html.Div([
                    html.Span(id='selected-book-title', style={
                        'marginLeft': '5px',
                        'color': '#666',  # Slightly muted color for better UI
                        'fontStyle': 'italic',  # Italic for the placeholder text
                        'fontWeight': 'bold'  # Make the title bold
                    }),
                    html.A(
                        html.Button(
                            '⬇ Download',
                            style={
                                'marginLeft': '10px',
                                'padding': '2px 8px',
                                'fontSize': '12px',
                                'backgroundColor': '#4CAF50',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '4px',
                                'cursor': 'pointer'
                            }
                        ),
                        id='download-link',
                        style={'textDecoration': 'none'},
                        target='_blank'
                    ),
                    html.A(
                        html.Button(
                            '↗ New Tab',
                            style={
                                'marginLeft': '10px',
                                'padding': '2px 8px',
                                'fontSize': '12px',
                                'backgroundColor': '#FF9800',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '4px',
                                'cursor': 'pointer'
                            }
                        ),
                        id='newtab-link',
                        style={'textDecoration': 'none'},
                        target='_blank'
                    ),
                    html.Button(
                        'Formst',
                        id='format-sentences-button',
                        style={
                            'marginLeft': '10px',
                            'padding': '2px 8px',
                            'fontSize': '12px',
                            'backgroundColor': '#2196F3',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px',
                            'cursor': 'pointer'
                        }
                    ),
                    html.Button(
                        '−',  # Minus sign for smaller text
                        id='decrease-font-button',
                        style={
                            'marginLeft': '10px',
                            'padding': '2px 8px',
                            'fontSize': '12px',
                            'backgroundColor': '#9E9E9E',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px 0 0 4px',
                            'cursor': 'pointer',
                            'width': '30px'
                        }
                    ),
                    html.Button(
                        '+',  # Plus sign for larger text
                        id='increase-font-button',
                        style={
                            'padding': '2px 8px',
                            'fontSize': '12px',
                            'backgroundColor': '#9E9E9E',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '0 4px 4px 0',
                            'cursor': 'pointer',
                            'width': '30px',
                            'borderLeft': '1px solid rgba(255,255,255,0.2)'
                        }
                    ),
                    html.Button(
                        '←',  # Decrease margin
                        id='decrease-margin-button',
                        style={
                            'marginLeft': '10px',
                            'padding': '2px 8px',
                            'fontSize': '12px',
                            'backgroundColor': '#9E9E9E',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px 0 0 4px',
                            'cursor': 'pointer',
                            'width': '30px'
                        }
                    ),
                    html.Button(
                        '→',  # Increase margin
                        id='increase-margin-button',
                        style={
                            'padding': '2px 8px',
                            'fontSize': '12px',
                            'backgroundColor': '#9E9E9E',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '0 4px 4px 0',
                            'cursor': 'pointer',
                            'width': '30px',
                            'borderLeft': '1px solid rgba(255,255,255,0.2)'
                        }
                    ),
                    # Store components for font size and margin
                    dcc.Store(id='font-size', data=14),
                    dcc.Store(id='margin-size', data=10),
                    dcc.Store(id='rating-update-trigger', data=0),
                    # Store to trigger scroll reset
                    dcc.Store(id='scroll-reset-trigger', data=0)
                ], style={'flex': '1', 'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    html.Button(
                        '★',
                        id='rate-1-button',
                        style={
                            'padding': '2px 8px',
                            'fontSize': '16px',
                            'backgroundColor': '#FFD700',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '4px 0 0 4px',
                            'cursor': 'pointer',
                            'width': '40px'
                        }
                    ),
                    html.Button(
                        '★★',
                        id='rate-2-button',
                        style={
                            'padding': '2px 8px',
                            'fontSize': '16px',
                            'backgroundColor': '#FFD700',
                            'color': 'white',
                            'border': 'none',
                            'borderLeft': '1px solid rgba(255,255,255,0.2)',
                            'cursor': 'pointer',
                            'width': '40px'
                        }
                    ),
                    html.Button(
                        '★★★',
                        id='rate-3-button',
                        style={
                            'padding': '2px 8px',
                            'fontSize': '16px',
                            'backgroundColor': '#FFD700',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '0 4px 4px 0',
                            'borderLeft': '1px solid rgba(255,255,255,0.2)',
                            'cursor': 'pointer',
                            'width': '40px'
                        }
                    ),
                    html.Div(id='current-rating', style={'marginLeft': '10px', 'display': 'inline-block'})
                ], style={'marginLeft': '10px', 'display': 'flex', 'alignItems': 'center'})
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'padding': '5px',
                'marginBottom': '5px',
                'fontSize': '14px',
                'backgroundColor': '#f8f9fa',
                'border': '1px solid #ddd',
                'borderRadius': '4px'
            }),
            # Content area
            dcc.Markdown(
                id='book-content',
                style={
                    'width': 'calc(100% - 20px)',
                    'height': 'calc(100vh - 60px)',  # Adjusted for header
                    'fontFamily': 'monospace',
                    'fontSize': '14px',
                    'padding': '10px',
                    'border': '1px solid #ddd',
                    'borderRadius': '4px',
                    'backgroundColor': '#f8f9fa',
                    'overflowY': 'auto',
                    'whiteSpace': 'pre-wrap'
                }
            )
        ], style={
            'flex': '0.55',  # Changed to 0.55 for 55%
            'height': '100vh',
            'backgroundColor': 'white',
            'display': 'flex',
            'flexDirection': 'column',
            'marginLeft': '10px'
        })
    ], style={
        'display': 'flex',
        'height': '100vh',
        'width': '100vw',
        'margin': 0,
        'padding': 0,
        'backgroundColor': '#f0f2f5',
        'overflow': 'hidden'
    })
])

# Callback Definitions
@app.callback(
    Output('book-table', 'data'),
    [Input('genre-filter', 'value'),
     Input('rating-filter', 'value'),
     Input('rating-update-trigger', 'data'),
     Input('text-filter', 'value')]
)
def update_table(selected_source, selected_ratings, rating_trigger, text_filter):
    """
    Filter the book table based on source, rating selection, and text search.
    
    Features:
    1. Show all books if 'All Sources' selected
    2. Filter by specific source if selected
    3. Filter by selected ratings (if any)
    4. Filter by text search in title and author fields
    5. Preserve sort order within filtered results
    6. Refresh data when ratings change
    
    Args:
        selected_source: Source to filter by
        selected_ratings: List of ratings to filter by
        rating_trigger: Trigger to refresh table data
        text_filter: Text to search in title and author fields
        
    Returns:
        Dictionary of filtered book records
    """
    # Reload the data to get fresh ratings
    global df
    df = load_library_data()
    
    if df is None or df.empty:
        return []
    
    filtered_df = df.copy()
    
    # Apply source filter
    if selected_source and selected_source != 'All Sources':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    # Apply rating filter
    if selected_ratings:
        # Load current ratings
        ratings = load_ratings()
        # Create a mask for books with selected ratings
        mask = filtered_df['full_path'].apply(lambda x: ratings.get(x, 0) in selected_ratings)
        filtered_df = filtered_df[mask]
    
    # Apply text filter
    if text_filter:
        # Convert text filter to lowercase for case-insensitive search
        text_filter = text_filter.lower()
        # Create mask for title and author matches
        title_mask = filtered_df['title'].str.lower().str.contains(text_filter, na=False)
        author_mask = filtered_df['author'].str.lower().str.contains(text_filter, na=False)
        # Combine masks
        filtered_df = filtered_df[title_mask | author_mask]
    
    return filtered_df.to_dict('records')

@app.callback(
    [Output('book-content', 'children'),
     Output('selected-book-title', 'children'),
     Output('scroll-reset-trigger', 'data')],
    [Input('book-table', 'active_cell'),
     Input('format-sentences-button', 'n_clicks')],
    [State('book-table', 'data'),
     State('book-content', 'children'),
     State('scroll-reset-trigger', 'data')]
)
def update_book_content_and_format(active_cell, n_clicks, data, current_content, scroll_trigger):
    """
    Handle book content loading and text formatting.
    
    Features:
    1. Load new book content when selection changes
    2. Apply sentence formatting when requested
    3. Reset scroll position on new selection
    4. Apply basic markdown formatting
    
    Args:
        active_cell: Currently selected table cell
        n_clicks: Format button click count
        data: Current table data
        current_content: Current text content
        scroll_trigger: Scroll position reset trigger
        
    Returns:
        Tuple of (content, title, scroll_trigger)
    """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Handle sentence formatting
    if trigger_id == 'format-sentences-button' and current_content:
        sentences = re.split(r'([.!?][\s\n]+)', current_content)
        formatted_text = ''
        for i in range(0, len(sentences)-1, 2):
            formatted_text += sentences[i] + sentences[i+1] + '\n'
        if len(sentences) % 2:
            formatted_text += sentences[-1]
        return formatted_text, dash.no_update, scroll_trigger

    # Handle book content loading
    if active_cell:  # New book selected
        row = data[active_cell['row']]
        current_file['path'] = df[df['title'] == row['title']]['full_path'].iloc[0]
        title = row['title']
        scroll_trigger += 1
    elif current_file['path']:  # Encoding changed for current book
        title = df[df['full_path'] == current_file['path']]['title'].iloc[0]
    else:
        return get_welcome_message(), "Choose a book!", scroll_trigger

    content, error, used_encoding = read_book_content(current_file['path'], 'cp949')
    if error:
        return error, title, scroll_trigger
    
    # Apply markdown formatting
    formatted_content = format_markdown(content) if content else "Failed to read file content"
    return formatted_content, title, scroll_trigger

def get_welcome_message():
    """Return the ASCII art welcome message."""
    return """





                                ,___________________________,
                               |  _____________________  ,'|
                               | |                     | | |
                               | |                     | | |
                               | |    📚 Welcome!      | | |
                               | |                     | | |
                               | |   Select a book     | | |
                               | |   from the list     | | |
                               | |   to start reading  | | |
                               | |                     | | |
                               | |                     | |,'
                               | |_____________________|/
                               |_________________________|





        """

# Add callback for updating download and new tab links when a book is selected
@app.callback(
    [Output('download-link', 'href'),
     Output('newtab-link', 'href')],
    [Input('book-table', 'active_cell')],
    [State('book-table', 'data')]
)
def update_links(active_cell, data):
    """
    Update download and new tab links for selected book.
    
    Features:
    1. Generate download URL for current file
    2. Create system file open URL
    3. Handle no selection case
    
    Args:
        active_cell: Currently selected table cell
        data: Current table data
        
    Returns:
        Tuple of (download_url, open_url)
    """
    if active_cell:
        row = data[active_cell['row']]
        file_path = df[df['title'] == row['title']]['full_path'].iloc[0]
        # Convert the file path to URL-friendly format for download
        download_url = f"/download/{os.path.basename(file_path)}"
        # Create open-file URL
        open_url = f"/open/{os.path.basename(file_path)}"
        return download_url, open_url
    return "", ""

# Add route to open file in system default application
@app.server.route('/open/<path:filename>')
def open_file(filename):
    """
    Open file in system default application.
    
    Process:
    1. Locate file in catalog
    2. Generate file URL
    3. Open in default system application
    
    Args:
        filename: Name of file to open
        
    Returns:
        Tuple of (status_message, http_status_code)
    """
    try:
        # Find the full path of the file
        file_path = df[df['full_path'].str.endswith(filename)]['full_path'].iloc[0]
        # Open the file using the default system application
        webbrowser.open(f'file://{file_path}')
        return "File opened in default application", 200
    except Exception as e:
        return str(e), 500

# Add download route to Flask server
@app.server.route('/download/<path:filename>')
def download(filename):
    """
    Handle file download requests.
    
    Process:
    1. Locate file in catalog
    2. Send file as attachment
    3. Handle file not found errors
    
    Args:
        filename: Name of file to download
        
    Returns:
        Flask send_file response
    """
    # Find the full path of the file
    file_path = df[df['full_path'].str.endswith(filename)]['full_path'].iloc[0]
    return send_file(file_path, as_attachment=True)

# Add callback for updating book info and Google search link
@app.callback(
    [Output('selected-book-info', 'children'),
     Output('google-search-link', 'href')],
    [Input('book-table', 'active_cell')],
    [State('book-table', 'data')]
)
def update_book_info(active_cell, data):
    """
    Update book information panel and search link.
    
    Features:
    1. Display book metadata (title, author, source)
    2. Generate Google search URL
    3. Handle missing author information
    
    Args:
        active_cell: Currently selected table cell
        data: Current table data
        
    Returns:
        Tuple of (info_text, google_url)
    """
    if active_cell:
        row = data[active_cell['row']]
        title = row['title']
        author = row['author']
        source = row['source']
        date = row['date']
        size = f"{row['size']} KB"  # Add KB suffix
        
        # Create info text
        info_text = f"Date: {date} | Title: {title} | Author: {author} | Source: {source} | Size: {size}"
        
        # Create Google search URL - only include author if not null
        search_query = f"{title}"
        if pd.notna(author):
            search_query += f" {author}"
        google_url = f"https://www.google.com/search?q={urllib.parse.quote(search_query)}"
        
        return info_text, google_url
    return "No book selected", "#"

# Add callback for font size control
@app.callback(
    [Output('book-content', 'style'),
     Output('font-size', 'data'),
     Output('margin-size', 'data')],
    [Input('increase-font-button', 'n_clicks'),
     Input('decrease-font-button', 'n_clicks'),
     Input('increase-margin-button', 'n_clicks'),
     Input('decrease-margin-button', 'n_clicks')],
    [State('font-size', 'data'),
     State('margin-size', 'data')]
)
def update_text_style(increase_font_clicks, decrease_font_clicks, 
                     increase_margin_clicks, decrease_margin_clicks,
                     current_font_size, current_margin):
    """
    Update text display style based on user interactions.
    
    Features:
    1. Font size control (8-24px, 2px steps)
    2. Margin control (0-100px, 10px steps)
    3. Style state persistence
    
    Args:
        increase/decrease_font_clicks: Font size button clicks
        increase/decrease_margin_clicks: Margin button clicks
        current_font_size: Current font size in px
        current_margin: Current margin size in px
        
    Returns:
        Tuple of (style_dict, new_font_size, new_margin)
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        # Initialize with default style
        return {
            'width': f'calc(100% - {current_margin * 2}px)',
            'height': 'calc(100vh - 60px)',
            'fontFamily': 'monospace',
            'fontSize': f'{current_font_size}px',
            'padding': '10px',
            'border': '1px solid #ddd',
            'borderRadius': '4px',
            'backgroundColor': '#f8f9fa',
            'resize': 'none',
            'margin': f'0 {current_margin}px'
        }, current_font_size, current_margin
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle font size changes
    if button_id == 'increase-font-button' and current_font_size < 24:
        current_font_size += 2
    elif button_id == 'decrease-font-button' and current_font_size > 8:
        current_font_size -= 2
    
    # Handle margin changes
    if button_id == 'increase-margin-button' and current_margin < 100:
        current_margin += 10
    elif button_id == 'decrease-margin-button' and current_margin > 0:
        current_margin -= 10
        
    return {
        'width': f'calc(100% - {current_margin * 2}px)',
        'height': 'calc(100vh - 60px)',
        'fontFamily': 'monospace',
        'fontSize': f'{current_font_size}px',
        'padding': '10px',
        'border': '1px solid #ddd',
        'borderRadius': '4px',
        'backgroundColor': '#f8f9fa',
        'resize': 'none',
        'margin': f'0 {current_margin}px'
    }, current_font_size, current_margin

# Add clientside callback to reset scroll position
app.clientside_callback(
    """
    function(trigger) {
        const markdown = document.getElementById('book-content');
        if (markdown) {
            markdown.scrollTop = 0;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('book-content', 'children', allow_duplicate=True),
    Input('scroll-reset-trigger', 'data'),
    prevent_initial_call=True
)

# Combined callback for handling ratings and book selection
@app.callback(
    [Output('current-rating', 'children'),
     Output('rate-1-button', 'style'),
     Output('rate-2-button', 'style'),
     Output('rate-3-button', 'style'),
     Output('rating-update-trigger', 'data')],
    [Input('rate-1-button', 'n_clicks'),
     Input('rate-2-button', 'n_clicks'),
     Input('rate-3-button', 'n_clicks'),
     Input('book-table', 'active_cell')],
    [State('book-table', 'data'),
     State('rating-update-trigger', 'data')]
)
def update_rating_and_buttons(rate1_clicks, rate2_clicks, rate3_clicks, active_cell, data, rating_trigger):
    """
    Handle book rating updates and button states.
    
    Features:
    1. Save rating to CSV when button clicked
    2. Update button styles based on current rating
    3. Show current rating as stars
    4. Update on book selection
    5. Trigger table refresh on rating changes
    
    Args:
        rate1_clicks: Number of clicks on 1-star button
        rate2_clicks: Number of clicks on 2-star button
        rate3_clicks: Number of clicks on 3-star button
        active_cell: Currently selected table cell
        data: Current table data
        rating_trigger: Counter to trigger table updates
        
    Returns:
        Tuple of (rating_text, button1_style, button2_style, button3_style, rating_trigger)
    """
    ctx = dash.callback_context
    if not active_cell:
        return "", default_button_style(1), default_button_style(2), default_button_style(3), rating_trigger

    row = data[active_cell['row']]
    file_path = df[df['title'] == row['title']]['full_path'].iloc[0]
    
    # If a rating button was clicked
    rating_changed = False
    if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] in ['rate-1-button', 'rate-2-button', 'rate-3-button']:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # Determine which rating was clicked
        rating = None
        if button_id == 'rate-1-button':
            rating = 1
        elif button_id == 'rate-2-button':
            rating = 2
        elif button_id == 'rate-3-button':
            rating = 3
        
        if rating:
            save_rating(file_path, rating)
            rating_changed = True
    
    # Get current rating for the book
    ratings = load_ratings()
    current_rating = ratings.get(file_path, 0)
    
    # Update button styles based on current rating
    button_styles = [
        get_button_style(1, current_rating, True),
        get_button_style(2, current_rating, False),
        get_button_style(3, current_rating, False)
    ]
    
    rating_text = '★' * current_rating if current_rating else ""
    # Increment trigger if rating changed
    if rating_changed:
        rating_trigger = (rating_trigger or 0) + 1
    return rating_text, *button_styles, rating_trigger

def default_button_style(stars):
    """Generate default style for rating buttons."""
    return {
        'padding': '2px 8px',
        'fontSize': '16px',
        'backgroundColor': '#FFD700',
        'color': 'white',
        'border': 'none',
        'borderRadius': '4px 0 0 4px' if stars == 1 else ('0 4px 4px 0' if stars == 3 else 'none'),
        'borderLeft': '1px solid rgba(255,255,255,0.2)' if stars > 1 else 'none',
        'cursor': 'pointer',
        'width': '40px',
        'opacity': '0.5'
    }

def get_button_style(stars, current_rating, is_first):
    """Generate style for rating button based on current rating."""
    style = default_button_style(stars)
    style['opacity'] = '1' if stars <= current_rating else '0.5'
    return style

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051) 