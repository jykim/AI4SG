"""
Reading with AI Dashboard
A Dash web application for managing and reading text files with encoding support.
Specifically designed for handling Korean text files with various encodings.
"""

import sys
from pathlib import Path
# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import orjson
import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import os
import urllib.parse
import re
import csv
from datetime import datetime, timedelta
import subprocess
import logging
import random

# Import extraction script functionality
from extract_reading import Config, index_txt_files, save_to_csv, save_to_markdown

# Initialize the Dash app with specific orjson settings
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Configuration
app.server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Constants for encodings
DEFAULT_ENCODINGS = ['utf-8', 'cp949', 'euc-kr']  # List of default encodings to try

# Color mapping for sources
SOURCE_COLORS = {
    'reader': '#4CAF50',     # Green
    'ibooks': '#2196F3',     # Blue
    'kindle': '#FF9800',     # Orange
    'medium': '#9C27B0',     # Purple
    'instapaper': '#03A9F4', # Light Blue
    'pocket': '#E91E63',     # Pink
    'Other': '#FFEEAD'       # Light Yellow for any other sources
}

# Get root directory
ROOT_DIR = Path(__file__).parent.parent

def run_extraction():
    """Run the extraction script to update reading entries."""
    try:
        # Initialize config
        config = Config()
        
        # Run extraction
        print("Running extraction script to update reading entries...")
        entries_info = index_txt_files(str(config.reading_dir))
        
        # Save results
        save_to_csv(entries_info)
        save_to_markdown(entries_info)
        
        print(f"Successfully processed {len(entries_info)} reading entries")
        return True
    except Exception as e:
        print(f"Error running extraction script: {str(e)}")
        return False

def create_timeline_figure(df: pd.DataFrame) -> go.Figure:
    """Create the timeline visualization"""
    if df is None or df.empty:
        return {}
        
    # Create a copy of the DataFrame to avoid the warning
    df = df.copy()
    
    # Convert date to datetime for proper grouping and remove rows with NaT dates
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['date'])
    
    # Filter to show only last 60 days
    last_60_days = pd.Timestamp.now() - pd.Timedelta(days=60)
    df = df[df['date'] >= last_60_days]
    
    if df.empty:
        return {}
    
    # Group entries by date and calculate normalized positions within each date
    df.loc[:, 'Entry_Order'] = df.groupby('date').cumcount()
    df.loc[:, 'Entries_Per_Date'] = df.groupby('date')['Entry_Order'].transform('count')
    # Reverse the y-axis ordering so newer entries are at the top
    df.loc[:, 'Y_Normalized'] = 100 - (df['Entry_Order'] / df['Entries_Per_Date']) * 100

    fig = go.Figure()

    # Add vertical shade for today's date
    today = pd.Timestamp.now().normalize()
    fig.add_vrect(
        x0=today,
        x1=today + pd.Timedelta(days=1),
        fillcolor="rgba(200, 200, 200, 0.2)",
        line_width=0,
        layer="below"
    )

    # Add scatter points for each reading entry
    for idx, row in df.iterrows():
        # Create hover text
        hover_text = f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
        hover_text += f"Title: {row['title']}<br>"
        hover_text += f"Author: {row['author'] if pd.notna(row['author']) else 'Unknown'}<br>"
        hover_text += f"Source: {row['source'] if pd.notna(row['source']) else 'Unknown'}"
        
        # Get color from source
        color = SOURCE_COLORS.get(row['source'], '#FFEEAD') if pd.notna(row['source']) else '#FFEEAD'  # Default to 'Other' color
        
        # Check if this is today's entry
        is_today = row['date'].date() == pd.Timestamp.now().date()
        
        fig.add_trace(go.Scatter(
            x=[row['date']],
            y=[row['Y_Normalized']],
            mode='markers',
            marker=dict(
                size=15,
                color=color,
                line=dict(
                    color='#2c5282' if is_today else 'black',
                    width=3 if is_today else 1
                )
            ),
            text=hover_text,
            hoverinfo='text',
            showlegend=False,
            customdata=[row['title']]  # Add title as custom data for click handling
        ))

    # Update layout
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=10, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
            range=[last_60_days, today + pd.Timedelta(days=1)]  # Set x-axis range to last 60 days
        ),
        yaxis=dict(
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
            range=[0, 110]
        )
    )
    return fig

def get_random_article(df: pd.DataFrame) -> tuple:
    """
    Select a random article from the DataFrame.
    
    Args:
        df: DataFrame containing reading entries
        
    Returns:
        Tuple of (row_index, row_data) or (None, None) if no data
    """
    if df is None or df.empty:
        return None, None
    
    # Get a random index
    random_index = random.randint(0, len(df) - 1)
    return random_index, df.iloc[random_index]

# Load and prepare data
def load_library_data():
    """Load and sort the library catalog with ratings."""
    try:
        df = pd.read_csv(ROOT_DIR / 'output' / 'reading_entries.csv')
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
    ratings_file = ROOT_DIR / 'output' / 'reading_ratings.csv'
    try:
        with open(ratings_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ratings[row['file_path']] = int(row['rating'])
    except FileNotFoundError:
        # Create the file with headers if it doesn't exist
        with open(ratings_file, 'w', encoding='utf-8', newline='') as f:
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
    
    ratings_file = ROOT_DIR / 'output' / 'reading_ratings.csv'
    with open(ratings_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file_path', 'rating'])
        writer.writeheader()
        for path, r in ratings.items():
            writer.writerow({'file_path': path, 'rating': r})

# Initialize data
df = load_library_data()

# Get the index of the latest entry (first row since df is sorted by date descending)
latest_entry_index = 0 if not df.empty else None

# Get a random article index
random_entry_index, random_entry = get_random_article(df)

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
    
    return filtered_content.strip() or ""

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
        if line and line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            if level <= 3:  # Only handle h1-h3
                formatted_lines.append(f'{"#" * level} {text}')
            continue
        
        # Handle links
        if line and '[' in line and '](' in line:
            # Keep markdown links as is
            formatted_lines.append(line)
            continue
        
        # Regular text
        if line and line.strip():
            formatted_lines.append(line)
        else:
            formatted_lines.append('')
    
    return '\n'.join(formatted_lines) or ""

# Create the layout
app.layout = html.Div([
    # Add URL component for pathname tracking
    dcc.Location(id='url', refresh=False),
    
    html.Div([
        # Left side panel
        html.Div([
            # Title and Source Filter
            html.Div([
                html.H1('Reading with AI', 
                    style={
                        'margin': '0',
                        'padding': '10px',
                        'color': '#2c3e50',
                        'fontSize': '24px',
                        'fontWeight': 'bold',
                        'borderBottom': '1px solid #eee'
                    }
                ),
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='genre-filter',
                            options=[{'label': source, 'value': source} 
                                   for source in df['source'].dropna().unique().tolist()],
                            value='All Sources',
                            placeholder='Filter Source',
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1', 'marginRight': '10px'}),
                    html.Div([
                        dcc.Dropdown(
                            id='rating-filter',
                            options=[
                                {'label': '★', 'value': 1},
                                {'label': '★★', 'value': 2},
                                {'label': '★★★', 'value': 3}
                            ],
                            value=[],
                            multi=True,
                            placeholder='Filter Rating',
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1'}),
                    html.Div([
                        dcc.Input(
                            id='text-filter',
                            type='text',
                            placeholder='Search title or author...',
                            style={
                                'width': '100%', 
                                'padding': '5px', 
                                'border': '1px solid #ddd', 
                                'borderRadius': '4px',
                                'height': '22px'  # Adjusted height to 22px
                            }
                        ),
                    ], style={'flex': '1'})
                ], style={
                    'display': 'flex',
                    'margin': '10px',
                    'gap': '10px',
                    'height': '40px'  # Fixed height for the filter row
                }),
            ], style={
                'backgroundColor': 'white',
                'borderBottom': '1px solid #ddd'
            }),
            # Timeline visualization
            html.Div([
                dcc.Graph(
                    id='timeline-graph',
                    figure=create_timeline_figure(df),
                    clickData=None,
                    style={'height': '180px'}
                )
            ], style={'margin': '10px'}),
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
                    active_cell={'row': random_entry_index, 'column': 0} if random_entry_index is not None else None,
                    style_table={
                        'height': 'calc(100vh - 380px)',  # Adjusted to match the height of the filter row
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
                    ] + [
                        {
                            'if': {
                                'filter_query': f'{{source}} = "{source}"',
                                'column_id': 'source'
                            },
                            'backgroundColor': SOURCE_COLORS.get(source, '#FFEEAD')
                        }
                        for source in df['source'].unique() if source
                    ],
                    virtualization=True,
                    page_action='none',
                    fixed_rows={'headers': True},
                    sort_action="native",
                    sort_mode="multi",
                    filter_action="none",
                    row_selectable=False,
                    cell_selectable=True
                ),
                # Book Info Panel
                html.Div([
                    html.Div([
                        html.Div([
                            html.Strong('Selected Book Info:', style={'marginBottom': '5px'}),
                            html.A(
                                'Search on Google',
                                id='google-search-link',
                                target='_blank',
                                rel="noopener noreferrer",
                                style={
                                    'color': 'white',
                                    'textDecoration': 'none',
                                    'marginLeft': '10px',
                                    'padding': '2px 8px',
                                    'backgroundColor': '#2196F3',
                                    'borderRadius': '4px',
                                    'fontSize': '12px',
                                    'display': 'inline-block',
                                    'verticalAlign': 'middle'
                                }
                            )
                        ], style={'display': 'flex', 'alignItems': 'center'}),
                        html.Div([
                            html.Div(id='selected-book-info'),
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
            'flex': '0.5',  # Changed from 0.45 to 0.5 for 50%
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
                    dcc.Store(id='font-size', data=16),
                    dcc.Store(id='margin-size', data=40),
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
                    'fontSize': '16px',
                    'padding': '10px',
                    'border': '1px solid #ddd',
                    'borderRadius': '4px',
                    'backgroundColor': '#f8f9fa',
                    'overflowY': 'auto',
                    'whiteSpace': 'pre-wrap',
                    'margin': '0 40px'
                }
            )
        ], style={
            'flex': '0.5',  # Changed from 0.55 to 0.5 for 50%
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

# Update the callbacks
@app.callback(
    [Output('book-table', 'data'),
     Output('timeline-graph', 'figure'),
     Output('book-table', 'active_cell')],
    [Input('genre-filter', 'value'),
     Input('rating-filter', 'value'),
     Input('text-filter', 'value'),
     Input('url', 'pathname')]  # Add URL pathname as input
)
def update_table_and_timeline(selected_source, selected_ratings, search_text, pathname):
    """Update both the table and timeline based on selected filters"""
    if df is None or df.empty:
        return [], {}, None
    
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
    
    # Apply text search filter if search text is provided
    if search_text and search_text.strip():
        search_text = search_text.lower().strip()
        search_mask = (
            filtered_df['title'].str.lower().str.contains(search_text, na=False) |
            filtered_df['author'].str.lower().str.contains(search_text, na=False)
        )
        filtered_df = filtered_df[search_mask]
    
    # Determine which cell to activate
    active_cell = None
    if pathname == '/':  # If it's the root path (initial load or refresh)
        # Get a random article from the filtered data
        random_index, _ = get_random_article(filtered_df)
        if random_index is not None:
            active_cell = {'row': random_index, 'column': 0}
    else:
        # Keep the latest entry selected for other cases
        active_cell = {'row': 0, 'column': 0} if not filtered_df.empty else None
    
    return filtered_df.to_dict('records'), create_timeline_figure(filtered_df), active_cell

# Update the callbacks to handle both table selection and timeline clicks
@app.callback(
    [Output('book-content', 'children'),
     Output('selected-book-title', 'children'),
     Output('book-table', 'active_cell', allow_duplicate=True),
     Output('selected-book-info', 'children', allow_duplicate=True),
     Output('google-search-link', 'href', allow_duplicate=True)],
    [Input('book-table', 'active_cell'),
     Input('timeline-graph', 'clickData')],
    [State('book-table', 'data'),
     State('book-content', 'children')],
    prevent_initial_call=True
)
def update_content_and_info(active_cell, clickData, data, current_content):
    """Handle content updates and info updates from both table selection and timeline clicks"""
    if not data:
        return get_welcome_message(), "Choose a book!", None, "No book selected", "#"
    
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Handle timeline clicks
    if trigger_id == 'timeline-graph' and clickData:
        clicked_title = clickData['points'][0]['customdata'][0]
        # Find the matching row in the table
        for i, row in enumerate(data):
            if row['title'] == clicked_title:
                active_cell = {'row': i, 'column': 0}
                break
    
    # Handle book content loading (from either table selection or timeline click)
    if active_cell is not None:
        row = data[active_cell['row']]
        current_file['path'] = df[df['title'] == row['title']]['full_path'].iloc[0]
        title = row['title']
        author = row['author']
        source = row['source']
        date = row['date']
        size = f"{row['size']} KB"
        
        # Create info text
        info_text = f"Date: {date} | Title: {title} | Author: {author} | Source: {source} | Size: {size}"
        
        # Create Google search URL
        search_query = f"{title}"
        if pd.notna(author):
            search_query += f" {author}"
        google_url = f"https://www.google.com/search?q={urllib.parse.quote(search_query)}"
        
        content, error, used_encoding = read_book_content(current_file['path'], 'cp949')
        if error:
            return error, title, active_cell, info_text, google_url
        
        # Apply markdown formatting
        formatted_content = format_markdown(content) if content else "Failed to read file content"
        return formatted_content, title, active_cell, info_text, google_url
    
    return get_welcome_message(), "Choose a book!", None, "No book selected", "#"

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
    1. Font size control (10-26px, 2px steps)  # Updated range
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
            'margin': f'0 {current_margin}px',
            'whiteSpace': 'pre-wrap',
            'overflowY': 'auto'
        }, current_font_size, current_margin
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle font size changes with new range
    if button_id == 'increase-font-button' and current_font_size < 26:  # Increased max size
        current_font_size += 2
    elif button_id == 'decrease-font-button' and current_font_size > 10:  # Increased min size
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
        'margin': f'0 {current_margin}px',
        'whiteSpace': 'pre-wrap',
        'overflowY': 'auto'
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
    # Run extraction first
    if not run_extraction():
        print("Warning: Extraction script failed, but continuing with dashboard...")
    
    # Configure server to suppress GET/POST messages
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    # Start the server
    app.run_server(debug=False, port=8051) 