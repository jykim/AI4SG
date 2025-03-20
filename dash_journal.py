"""
Journal Dashboard with AI Assistant

This module implements a Dash-based web application that provides an interactive
interface for viewing and analyzing journal entries. It includes:

- A timeline visualization of journal entries
- A searchable and filterable table of entries
- An AI chat assistant for analyzing journal content
- Real-time updates when new entries are added
- Emoji-based tagging and filtering

The dashboard integrates with OpenAI's GPT-4 for intelligent analysis and
maintains chat history with proper timestamps.
"""

# Standard library imports
import argparse
import logging
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# Third-party imports
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH
import pandas as pd
import plotly.graph_objects as go
import yaml

# Local imports
from agent_utils import get_chat_response, get_todays_chat_log

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
                'input_dir': 'input',
                'output_dir': 'output',
                'api_cache_dir': 'api_cache',
                'min_process_interval': 600
            }
        
        # Set configuration values
        self.input_dir = Path(config.get('input_dir', 'input'))
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.api_cache_dir = Path(config.get('api_cache_dir', 'api_cache'))
        self.min_process_interval = config.get('min_process_interval', 600)
        
        # API key should be set via environment variable
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')

    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [self.input_dir, self.output_dir, self.api_cache_dir]:
            directory.mkdir(exist_ok=True)

# Initialize configuration
config = Config()
SCRIPT_DIR = Path(__file__).parent.absolute()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.output_dir / 'dashboard.log'),
        logging.StreamHandler()
    ]
)

# Filter out Dash's GET request logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)

class DashboardState:
    """Manages the global state of the dashboard"""
    def __init__(self, config: Optional[Config] = None):
        self.is_processing = False
        self.df: Optional[pd.DataFrame] = None
        self.last_entries_count = 0
        self.update_event = threading.Event()
        self.last_process_time = 0
        self.processing_lock = threading.Lock()
        self.processing_requested = False
        self.last_content_hash = None  # Track content changes
        self.config = config or Config()  # Use provided config or create new one
        # Chat state
        self.chat_messages = []  # List of dicts with 'role' and 'content' keys
        self.chat_history = []  # List of dicts with 'role', 'content', and 'timestamp' keys

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load and process the journal data"""
        try:
            # Use journal_entries_annotated.csv as the base file
            df = pd.read_csv(self.config.output_dir / 'journal_entries_annotated.csv')
            
            # Clean up and parse dates, dropping any rows with invalid dates
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
            
            # Replace blank/null emotions with empty string
            df['emotion'] = df['emotion'].fillna('')
            df['emotion'] = df['emotion'].apply(lambda x: '' if pd.isna(x) or str(x).strip() == '' else x)
            
            df = df.sort_values('Date', ascending=False)
            
            # Clean up title formatting - remove quotes and asterisks
            df['Title'] = df['Title'].apply(lambda x: str(x).strip('"*') if pd.notna(x) else x)
            
            # Create Tags column with emojis, filtering out NaN values
            df['Tags'] = df.apply(lambda row: ' '.join(filter(None, [
                row['topic_visual'] if pd.notna(row['topic_visual']) else '',
                row['etc_visual'] if pd.notna(row['etc_visual']) else ''
            ])), axis=1)
            
            # Create Tags_Tooltip with filtered values
            df['Tags_Tooltip'] = df.apply(lambda row: ' '.join(filter(None, [
                f"{row['topic_visual']} ({row['topic']})" if pd.notna(row['topic_visual']) and pd.notna(row['topic']) else '',
                f"{row['etc_visual']} ({row['etc']})" if pd.notna(row['etc_visual']) and pd.notna(row['etc']) else ''
            ])), axis=1)
            
            # Format dates for display
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            # Calculate content hash to detect changes
            content_hash = hash(str(df.to_dict()))
            if self.last_content_hash is not None and content_hash != self.last_content_hash:
                logging.info("Content changes detected in journal entries")
                self.update_event.set()
            self.last_content_hash = content_hash
            
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None

    def create_timeline_figure(self, df: pd.DataFrame) -> go.Figure:
        """Create the timeline visualization"""
        # Create a copy of the DataFrame to avoid the warning
        df = df.copy()
        
        # Convert Date back to datetime for proper grouping
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
        
        # Group entries by date and calculate normalized positions within each date
        df.loc[:, 'Entry_Order'] = df.groupby('Date').cumcount()
        df.loc[:, 'Entries_Per_Date'] = df.groupby('Date')['Entry_Order'].transform('count')
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

        # Add scatter points for each journal entry
        for idx, row in df.iterrows():
            # Handle NaN or empty content
            content = row['Content']
            if pd.isna(content):
                content = "(No content)"
            else:
                # Add newline after each sentence (assuming sentences end with .!?)
                content = str(content).replace('. ', '.<br>').replace('! ', '!<br>').replace('? ', '?<br>')
            
            hover_text = f"Date: {row['Date'].strftime('%Y-%m-%d')}<br>"
            if not pd.isna(row['Title']):
                hover_text += f"Title: {row['Title']}<br>"
            hover_text += f"#{row['emotion']} {row['Tags']}<br>"
            hover_text += f"Content: {content[:200]}..." if len(content) > 200 else f"Content: {content}"
            
            # Get color from emotion_visual field using full DataFrame
            color = get_emotion_color(df, row['emotion'], self.df)
            
            # Check if this is today's entry
            is_today = row['Date'].date() == pd.Timestamp.now().date()
            
            fig.add_trace(go.Scatter(
                x=[row['Date']],
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
                showlegend=False
            ))

        # Update layout
        fig.update_layout(
            height=240,  # Reduced by 20% from 300
            margin=dict(l=20, r=20, t=10, b=10),  # Reduced top and bottom margins
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgray',
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                gridcolor='lightgray',
                showgrid=True,
                zeroline=False,  # Hide y-axis labels since they're normalized
                range=[0, 110]  # Extended range to prevent clipping at top
            )
        )
        return fig

    def run_extraction_and_annotation(self, retag_all: bool = False, force: bool = False) -> bool:
        """Run the extraction and annotation scripts in sequence"""
        current_time = time.time()
        
        # Prevent processing if already running or if last run was too recent
        with self.processing_lock:
            if self.is_processing:
                logging.debug("Processing already in progress, skipping...")
                return False
            # Only check time interval if not forced
            if not force and current_time - self.last_process_time < self.config.min_process_interval:
                logging.debug("Skipping processing - too soon since last run")
                return False
            self.is_processing = True
            self.last_process_time = current_time
            self.processing_requested = False  # Reset the request flag

        try:
            env = os.environ.copy()
            env['OUTPUT_DIR'] = str(self.config.output_dir)
            env['INPUT_DIR'] = str(self.config.input_dir)
            env['API_CACHE_DIR'] = str(self.config.api_cache_dir)
            if self.config.openai_api_key:
                env['OPENAI_API_KEY'] = self.config.openai_api_key

            # Run extraction
            logging.info("Starting journal extraction and annotation process...")
            extract_script = SCRIPT_DIR / 'extract_journal.py'
            if not self._run_script(['python', str(extract_script)], env):
                return False

            # Run annotation with the new input file
            annotate_script = SCRIPT_DIR / 'annotate_journal.py'
            cmd = ['python', str(annotate_script), '--input', 'journal_entries.csv']
            if retag_all:
                cmd.append('--retag-all')
                logging.info("Re-tagging all entries...")
            if not self._run_script(cmd, env):
                return False

            # Update data if new entries found
            new_df = self.load_data()
            if new_df is not None and (retag_all or len(new_df) > self.last_entries_count):
                self._handle_new_data(new_df, retag_all)
                return True

            logging.info("Processing completed - no new entries found")
            return False
        except Exception as e:
            logging.error(f"Error in background processing: {str(e)}")
            return False
        finally:
            self.is_processing = False

    def _run_script(self, cmd: list, env: Dict[str, str]) -> bool:
        """Run a Python script and handle its output"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                logging.error(f"Script failed: {result.stderr}")
                return False
            return True
        except Exception as e:
            logging.error(f"Error running script: {str(e)}")
            return False

    def _handle_new_data(self, new_df: pd.DataFrame, retag_all: bool) -> None:
        """Handle newly loaded data"""
        if retag_all:
            logging.info("All entries have been re-tagged!")
            self.df = new_df
            self.last_entries_count = len(self.df)
            self.update_event.set()
            return

        # If we have existing data and new data
        if self.df is not None and not self.df.empty and not new_df.empty:
            # Get the last entry from existing data and first entry from new data
            last_entry = self.df.iloc[0]  # Most recent entry
            new_entry = new_df.iloc[0]   # Most recent entry
            
            # Compare content lengths and check if new entry is a superset
            if (isinstance(new_entry['Content'], str) and 
                isinstance(last_entry['Content'], str) and
                len(new_entry['Content']) > len(last_entry['Content']) and 
                last_entry['Content'] in new_entry['Content']):
                logging.info("New entry is a superset of the last entry. Removing the last entry.")
                # Remove the last entry from existing data
                self.df = self.df.iloc[1:]
                self.last_entries_count -= 1
        
        # Update with new data
        self.df = new_df
        self.last_entries_count = len(self.df)
        self.update_event.set()
        logging.info(f"Updated entries count: {self.last_entries_count}")
        
        # Force refresh the dashboard if new entries are detected
        if not retag_all and len(new_df) > self.last_entries_count:
            logging.info("New entries detected, forcing dashboard refresh")
            self.update_event.set()

def hex_to_rgba(hex_color: str, alpha: float = 0.3) -> str:
    """Convert hex color to rgba format with specified alpha."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

def get_emotion_color(df: pd.DataFrame, emotion: str, full_df: pd.DataFrame = None) -> str:
    """Get the color for an emotion from GPT-4's emotion_visual field."""
    if full_df is None:
        full_df = df
    if full_df is None or full_df.empty:
        return '#FFFFFF'
    if not emotion or str(emotion).lower() == 'blank':  # Handle empty emotion or 'blank'
        return '#FFFFFF'
    emotion_rows = full_df[full_df['emotion'] == emotion]
    if emotion_rows.empty:
        return '#FFFFFF'
    color = emotion_rows['emotion_visual'].iloc[0]
    # Handle NaN or invalid color values
    if pd.isna(color) or not str(color).strip():
        return '#FFFFFF'
    # Convert color to string and ensure it's valid
    color_str = str(color).strip()
    if color_str.lower() == 'blank' or not color_str.startswith('#'):
        return '#FFFFFF'
    return color_str

def format_chat_message(content, is_user=False, is_latest=False, timestamp=None):
    """Format a chat message with proper styling and truncation."""
    # Generate a unique ID for the message using timestamp, content hash, and random component
    content_str = str(content) if isinstance(content, list) else content
    random_component = hash(f"{content_str}{time.time()}{is_user}")
    message_id = f"msg-{int(time.time() * 1000)}-{abs(random_component) % 100000}"
    
    # Use provided timestamp or current time
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str):
        try:
            # Try both formats: with and without seconds
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
                try:
                    timestamp = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
            else:
                timestamp = datetime.now()
        except Exception:
            timestamp = datetime.now()
    
    # Format time for display (HH:MM)
    time_str = timestamp.strftime("%H:%M")
    
    is_long = len(content) > 100 and not is_latest
    display_content = content if not is_long else content[:100]
    
    message_content = [
        html.Div([
            html.Strong("You: " if is_user else "Assistant: "),
            html.Div(
                [
                    html.Span(display_content),
                    html.Button(
                        "...",
                        id={'type': 'expand-button', 'index': message_id},
                        n_clicks=0,
                        style={
                            'background': 'none',
                            'border': 'none',
                            'color': '#0d6efd',
                            'cursor': 'pointer',
                            'padding': '0 4px',
                            'display': 'inline' if is_long else 'none'
                        }
                    )
                ],
                id={'type': 'message-content', 'index': message_id},
                style={
                    'display': 'inline',
                    'word-wrap': 'break-word'
                }
            )
        ]),
        html.Div(
            time_str,
            style={
                'font-size': '0.8em',
                'color': '#6c757d',
                'text-align': 'right' if is_user else 'left',
                'margin-top': '4px',
                'padding-right' if is_user else 'padding-left': '8px'
            }
        )
    ]
    
    # Store the full content in a hidden div
    if is_long:
        message_content.append(
            html.Div(
                content,
                id={'type': 'full-content', 'index': message_id},
                style={'display': 'none'}
            )
        )
    
    return dbc.Alert(
        message_content,
        color="primary" if is_user else "info",
        style={
            'text-align': 'right' if is_user else 'left',
            'margin': '5px',
            'white-space': 'pre-wrap'
        }
    )

def create_layout(state: Optional[DashboardState] = None) -> dbc.Container:
    """Create the dashboard layout"""
    if state is None:
        state = DashboardState()
    
    # Calculate default date range (past 14 days + future 7 days)
    if state.df is not None and not state.df.empty:
        today = pd.Timestamp.now().normalize()
        start_date = (today - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
        end_date = (today + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    else:
        start_date = None
        end_date = None

    # Get today's chat history for initial load
    chat_log = get_todays_chat_log()
    
    # Prepare initial messages
    initial_messages = []
    has_todays_chat = False
    if chat_log:
        # Parse chat log and format messages
        entries = chat_log.split('### Chat Entry - ')
        for entry in entries:
            if entry.strip():
                # Extract timestamp and messages
                lines = entry.strip().split('\n')
                if lines:
                    has_todays_chat = True  # Mark that we have chat history
                    current_message = ""
                    current_role = None
                    entry_timestamp = None
                    
                    # Parse the timestamp from the first line
                    if lines[0].strip():
                        try:
                            # Try both formats: with and without seconds
                            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
                                try:
                                    entry_timestamp = datetime.strptime(lines[0].strip(), fmt)
                                    break
                                except ValueError:
                                    continue
                            if entry_timestamp is None:
                                entry_timestamp = datetime.now()
                                logging.warning(f"Could not parse timestamp with any format: {lines[0].strip()}")
                        except Exception as e:
                            entry_timestamp = datetime.now()
                            logging.warning(f"Error parsing timestamp: {lines[0].strip()} - {str(e)}")
                    
                    for line in lines[1:]:  # Skip the timestamp line
                        if line.startswith('**User**:'):
                            # If we have a previous message, add it
                            if current_message and current_role:
                                initial_messages.append(format_chat_message(
                                    current_message.strip(),
                                    is_user=(current_role == 'user'),
                                    timestamp=entry_timestamp
                                ))
                            current_role = 'user'
                            current_message = line.replace('**User**:', '').strip()
                        elif line.startswith('**Assistant**:'):
                            # If we have a previous message, add it
                            if current_message and current_role:
                                initial_messages.append(format_chat_message(
                                    current_message.strip(),
                                    is_user=(current_role == 'user'),
                                    timestamp=entry_timestamp
                                ))
                            current_role = 'assistant'
                            current_message = line.replace('**Assistant**:', '').strip()
                        else:
                            # Append to current message if it's a continuation
                            if current_message and line.strip():
                                current_message += "\n" + line.strip()
                    
                    # Add the last message if there is one
                    if current_message and current_role:
                        initial_messages.append(format_chat_message(
                            current_message.strip(),
                            is_user=(current_role == 'user'),
                            timestamp=entry_timestamp
                        ))
    
    # Add welcome message only if there's no chat history for today
    if not has_todays_chat:
        initial_messages.append(format_chat_message(
            "Hi! I'm your own AI assistant. What can I do for you?",
            is_user=False,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    return dbc.Container([
        # Store component for journal data
        dcc.Store(
            id='journal-data-store',
            data=state.df.to_dict('records') if state.df is not None else []
        ),
        
        # Header row with title
        dbc.Row([
            dbc.Col([
                html.H1("Journaling with AI", className="mb-0")
            ], width=12, className="d-flex align-items-center")
        ], className="mb-2"),  # Reduced from mb-3
        
        # Main content area with journal and chat
        dbc.Row([
            # Left column - Journal content
            dbc.Col([
                # Search and Date filter row
                dbc.Row([
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("Search:", className="bg-light"),
                            dbc.Input(
                                id='search-input',
                                type='text',
                                placeholder='Search in content...',
                                className="form-control",
                                style={'width': '250px'}
                            )
                        ], className="me-3"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Tags:", className="bg-light"),
                            dcc.Dropdown(
                                id='emoji-filter',
                                options=[],  # Will be populated in callback
                                placeholder='Select tag...',
                                style={'width': '180px'},
                                clearable=True
                            )
                        ], className="me-3")
                    ], width=6, className="d-flex align-items-center"),
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText("Date Range:", className="bg-light"),
                            dcc.DatePickerRange(
                                id='date-picker',
                                start_date=start_date,
                                end_date=end_date,
                                display_format='YYYY-MM-DD',
                                style={'width': 'auto'}
                            ),
                            dbc.Button(
                                "Reset",
                                id="reset-range-button",
                                color="secondary",
                                size="sm",
                                className="ms-2"
                            )
                        ], className="justify-content-end")
                    ], width=6, className="d-flex align-items-center")
                ], className="mb-2"),  # Reduced from mb-3
                
                # Timeline visualization
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='timeline-graph',
                            figure=state.create_timeline_figure(state.df) if state.df is not None else {},
                            clickData=None,
                            style={'height': '180px'}  # Reduced from 200px
                        )
                    ], width=12)
                ], className="mb-2"),  # Reduced from mb-3
                
                # DataTable
                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(
                            id='journal-table',
                            columns=[
                                {"name": "Date", "id": "Date"},
                                {"name": "Time", "id": "Time"},
                                {"name": "Title", "id": "Title"},
                                {"name": "Emotion", "id": "emotion"},
                                {"name": "Topic", "id": "topic"},
                                {"name": "Tags", "id": "Tags"},
                                {"name": "Content", "id": "Content"}
                            ],
                            data=state.df.to_dict('records') if state.df is not None and not state.df.empty else [],
                            style_table={
                                'height': 'calc(100vh - 300px)',  # Adjusted from 500px
                                'overflowY': 'auto'
                            },
                            style_cell={
                                'textAlign': 'left',
                                'padding': '8px',
                                'whiteSpace': 'pre-wrap',
                                'height': 'auto',
                                'minWidth': '50px',
                                'maxWidth': '500px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'fontSize': '14px'
                            },
                            style_cell_conditional=[
                                {'if': {'column_id': 'Date'}, 'width': '100px'},
                                {'if': {'column_id': 'Time'}, 'width': '80px'},
                                {'if': {'column_id': 'Title'}, 'width': '150px'},
                                {'if': {'column_id': 'emotion'}, 'width': '100px'},
                                {'if': {'column_id': 'topic'}, 'width': '100px'},
                                {'if': {'column_id': 'Tags'}, 'width': '150px'},
                                {'if': {'column_id': 'Content'}, 'width': '400px'},
                            ],
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold',
                                'fontSize': '14px'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ] + [
                                {
                                    'if': {
                                        'filter_query': f'{{emotion}} = "{emotion}"',
                                        'column_id': 'emotion'
                                    },
                                    'backgroundColor': hex_to_rgba(get_emotion_color(state.df, emotion, state.df))
                                }
                                for emotion in state.df['emotion'].unique() if state.df is not None and emotion
                            ] + [
                                {
                                    'if': {
                                        'filter_query': f'{{Date}} = "{pd.Timestamp.now().strftime("%Y-%m-%d")}"',
                                        'column_id': ['Date', 'Time', 'Title']
                                    },
                                    'fontWeight': 'bold',
                                    'color': '#2c5282',
                                    'backgroundColor': 'rgba(44, 82, 130, 0.1)'
                                },
                                {
                                    'if': {
                                        'filter_query': f'{{Date}} = "{pd.Timestamp.now().strftime("%Y-%m-%d")}"',
                                        'column_id': ['Content', 'topic', 'Tags', 'Section']
                                    },
                                    'backgroundColor': 'rgba(44, 82, 130, 0.1)'
                                }
                            ]
                        )
                    ], width=12)
                ])
            ], width=8, className="pe-2"),  # Added padding to the right
            
            # Right column - Chat interface
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Chat with AI Assistant", className="bg-primary text-white py-2"),
                    dbc.CardBody([
                        # Chat messages container with initial messages and auto-scroll
                        html.Div(
                            id='chat-messages',
                            style={
                                'height': 'calc(100vh - 300px)',
                                'overflowY': 'auto',
                                'padding': '10px',
                                'marginBottom': '10px',
                                'display': 'flex',
                                'flexDirection': 'column-reverse'  # Reverse order for bottom-up layout
                            },
                            children=html.Div(
                                initial_messages,
                                style={'display': 'flex', 'flexDirection': 'column'}
                            )
                        ),
                        # Chat input area
                        dbc.InputGroup([
                            dbc.Textarea(
                                id='chat-input',
                                placeholder='Type your message here...',
                                style={'resize': 'none', 'height': '60px', 'fontSize': '14px'},
                                className="me-2"
                            ),
                            dbc.Button(
                                "Send",
                                id='send-button',
                                color="primary",
                                className="px-4"
                            )
                        ]),
                        # Suggested questions
                        html.Div([
                            html.Label("Suggested Questions:", className="mt-2 mb-1", style={'fontSize': '14px', 'color': '#6c757d'}),
                            dbc.ButtonGroup([
                                dbc.Button(
                                    "What's the status so far?",
                                    id='suggested-q1',
                                    color="light",
                                    className="me-2",
                                    style={'fontSize': '13px', 'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'flex': '1'}
                                ),
                                dbc.Button(
                                    "What should I do next?",
                                    id='suggested-q2',
                                    color="light",
                                    className="me-2",
                                    style={'fontSize': '13px', 'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'flex': '1'}
                                ),
                                dbc.Button(
                                    "Anything to reflect on?",
                                    id='suggested-q3',
                                    color="light",
                                    className="me-2",
                                    style={'fontSize': '13px', 'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'flex': '1'}
                                )
                            ], className="w-100 d-flex justify-content-between")
                        ], className="mt-2")
                    ], className="p-2")
                ], className="h-100")
            ], width=4, className="ps-2")
        ], className="g-0")
    ], fluid=True, className="px-4 h-100")

# Initialize global state
state = DashboardState()
state.df = state.load_data()
if state.df is not None:
    state.last_entries_count = len(state.df)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = create_layout(state)

# Add callback for expanding messages
@app.callback(
    Output({'type': 'message-content', 'index': MATCH}, 'children'),
    Input({'type': 'expand-button', 'index': MATCH}, 'n_clicks'),
    State({'type': 'full-content', 'index': MATCH}, 'children'),
    prevent_initial_call=True
)
def expand_message(n_clicks, full_content):
    """Handle message expansion when ... is clicked"""
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    
    # Get the current message ID from the callback context
    ctx = dash.callback_context
    message_id = ctx.triggered[0]['prop_id'].split('.')[0].split('"index":')[1].strip('}').strip('"')
    
    # Return the full content with the expand button hidden
    return [
        html.Span(full_content),
        html.Button(
            "...",
            id={'type': 'expand-button', 'index': message_id},
            n_clicks=0,
            style={
                'background': 'none',
                'border': 'none',
                'color': '#0d6efd',
                'cursor': 'pointer',
                'padding': '0 4px',
                'display': 'none'  # Hide the button after expansion
            }
        )
    ]

@app.callback(
    [
        Output('chat-messages', 'children'),
        Output('chat-input', 'value'),
    ],
    [
        Input('send-button', 'n_clicks'),
        Input('chat-input', 'n_submit'),
        Input('suggested-q1', 'n_clicks'),
        Input('suggested-q2', 'n_clicks'),
        Input('suggested-q3', 'n_clicks')
    ],
    [
        State('chat-input', 'value'),
        State('chat-messages', 'children'),
    ],
    prevent_initial_call=True
)
def handle_chat_message(n_clicks, n_submit, q1_clicks, q2_clicks, q3_clicks, message, current_messages):
    """Handle chat messages and get AI responses"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_messages or html.Div([], style={'display': 'flex', 'flexDirection': 'column'}), ""
    
    # Handle suggested questions
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id in ['suggested-q1', 'suggested-q2', 'suggested-q3']:
        question_map = {
            'suggested-q1': "What's the status so far?",
            'suggested-q2': "What should I do next?",
            'suggested-q3': "Anything to reflect on?"
        }
        message = question_map[triggered_id]
    
    if not message:  # Don't process empty messages
        return current_messages or html.Div([], style={'display': 'flex', 'flexDirection': 'column'}), ""
    
    # Initialize messages list if None
    if current_messages is None:
        current_messages = html.Div([], style={'display': 'flex', 'flexDirection': 'column'})
    
    # Get current timestamp
    current_time = datetime.now()
    
    # Get existing messages and mark them as not latest
    existing_messages = []
    if current_messages and isinstance(current_messages, dict) and current_messages.get('props', {}).get('children'):
        messages_list = current_messages['props']['children']
        if isinstance(messages_list, dict):
            messages_list = [messages_list]
            
        for msg in messages_list:
            if not isinstance(msg, dict) or 'props' not in msg:
                continue
                
            msg_children = msg['props'].get('children', [])
            if not msg_children or len(msg_children) < 2:
                continue
                
            # Skip loading message
            if any(isinstance(child, dict) and 
                  isinstance(child.get('props', {}).get('children'), list) and 
                  any(isinstance(c, dict) and c.get('type') == 'Spinner' for c in child['props']['children'])
                  for child in msg_children):
                continue
                
            # Extract timestamp from the message
            timestamp_div = next(
                (child for child in msg_children
                 if isinstance(child, dict) and 
                 child.get('props', {}).get('style', {}).get('color') == '#6c757d'),
                None
            )
            timestamp_str = timestamp_div.get('props', {}).get('children') if timestamp_div else None
            
            # Get message content
            content_div = msg_children[0]
            if not isinstance(content_div, dict):
                continue
                
            content_children = content_div.get('props', {}).get('children', [])
            if not content_children:
                continue
                
            # Get the content after "You: " or "Assistant: "
            content_span = content_children[1] if len(content_children) > 1 else None
            if not isinstance(content_span, dict):
                continue
                
            content = content_span.get('props', {}).get('children', [{}])[0].get('props', {}).get('children', '')
            is_user = 'You: ' in content_children[0].get('props', {}).get('children', '')
            
            # Format message with original timestamp if available
            if timestamp_str:
                try:
                    msg_time = datetime.strptime(f"{current_time.strftime('%Y-%m-%d')} {timestamp_str}", "%Y-%m-%d %H:%M")
                except ValueError:
                    msg_time = current_time
            else:
                msg_time = current_time
                
            existing_messages.append(format_chat_message(content, is_user=is_user, is_latest=False, timestamp=msg_time))
    
    # Create user message component (latest)
    user_message = format_chat_message(message, is_user=True, is_latest=True, timestamp=current_time)
    
    # Create loading message component
    loading_message = dbc.Alert(
        [
            html.Strong("Assistant: "), 
            dbc.Spinner(size="sm", color="info", spinner_class_name="me-2"),
            "Thinking..."
        ],
        color="light",
        style={
            'text-align': 'left',
            'margin': '5px',
            'white-space': 'pre-wrap'
        }
    )
    
    # Add user message and loading indicator
    temp_messages = html.Div(
        existing_messages + [user_message, loading_message],
        style={'display': 'flex', 'flexDirection': 'column'}
    )
    
    # Create chat history for the API
    chat_history = []
    for msg in existing_messages:
        if not isinstance(msg, dict) or 'props' not in msg:
            continue
            
        msg_children = msg['props'].get('children', [])
        if not msg_children or len(msg_children) < 2:
            continue
            
        # Extract timestamp
        timestamp_div = next(
            (child for child in msg_children
             if isinstance(child, dict) and 
             child.get('props', {}).get('style', {}).get('color') == '#6c757d'),
            None
        )
        timestamp_str = timestamp_div.get('props', {}).get('children') if timestamp_div else None
        
        # Get message content
        content_div = msg_children[0]
        if not isinstance(content_div, dict):
            continue
            
        content_children = content_div.get('props', {}).get('children', [])
        if not content_children:
            continue
            
        # Get the content after "You: " or "Assistant: "
        content_span = content_children[1] if len(content_children) > 1 else None
        if not isinstance(content_span, dict):
            continue
            
        content = content_span.get('props', {}).get('children', [{}])[0].get('props', {}).get('children', '')
        is_user = 'You: ' in content_children[0].get('props', {}).get('children', '')
        
        if timestamp_str:
            try:
                msg_time = datetime.strptime(f"{current_time.strftime('%Y-%m-%d')} {timestamp_str}", "%Y-%m-%d %H:%M")
            except ValueError:
                msg_time = current_time
        else:
            msg_time = current_time
            
        chat_history.append({
            'role': 'user' if is_user else 'assistant',
            'content': content,
            'timestamp': msg_time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Get response from agent using the complete dataset from state
    response_data = get_chat_response(message, state.df, chat_history)
    ai_response = response_data['response']
    chat_log = response_data['chat_log']
    
    # Create AI message component (latest)
    ai_message = format_chat_message(ai_response, is_user=False, is_latest=True, timestamp=current_time)
    
    # Update messages list
    new_messages = existing_messages + [user_message, ai_message]
    
    # Return updated messages in a Div with flex column layout and clear input
    return html.Div(new_messages, style={'display': 'flex', 'flexDirection': 'column'}), ""

# Existing callback for table and timeline
@app.callback(
    [Output('journal-table', 'data'),
     Output('timeline-graph', 'figure'),
     Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date'),
     Output('emoji-filter', 'options')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('timeline-graph', 'clickData'),
     Input('reset-range-button', 'n_clicks'),
     Input('search-input', 'value'),
     Input('emoji-filter', 'value')]
)
def update_table_and_timeline(start_date: Optional[str], end_date: Optional[str], 
                            clickData: Optional[dict], reset_clicks: Optional[int], 
                            search_text: Optional[str], selected_emoji: Optional[str]) -> Tuple[list, dict, str, str, list]:
    """Update both the table and timeline based on selected date range, click event, search text, or emoji filter"""
    if state.df is None or state.df.empty:
        return [], {}, None, None, []
    
    # Get unique emojis and their descriptions from Tags and Tags_Tooltip
    emoji_counts = {}
    emoji_descriptions = {}
    
    # Create a mapping to store unique emoji-description pairs
    emoji_desc_mapping = {}
    
    # First pass: collect all unique emoji-description pairs
    for tags, tooltips in zip(state.df['Tags'].dropna(), state.df['Tags_Tooltip'].dropna()):
        tag_list = tags.split()
        tooltip_list = tooltips.split()
        
        for tag, tooltip in zip(tag_list, tooltip_list):
            tag = tag.strip()
            if tag and not pd.isna(tag):
                desc_start = tooltip.find("(")
                desc_end = tooltip.find(")")
                if desc_start != -1 and desc_end != -1:
                    desc = tooltip[desc_start + 1:desc_end].strip()
                    # Store the emoji-description pair
                    if tag not in emoji_desc_mapping:
                        emoji_desc_mapping[tag] = desc
    
    # Second pass: count frequencies using consistent descriptions
    for tags in state.df['Tags'].dropna():
        tag_list = tags.split()
        for tag in tag_list:
            tag = tag.strip()
            if tag and not pd.isna(tag):
                # Use the consistent description from the mapping
                desc = emoji_desc_mapping.get(tag, '')
                # Create a unique key that combines emoji and description
                emoji_key = tag
                emoji_counts[emoji_key] = emoji_counts.get(emoji_key, 0) + 1
                emoji_descriptions[emoji_key] = desc
    
    # Get top 10 most frequent emojis
    top_emojis = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Create options with emoji and description
    emoji_options = [
        {'label': f"{emoji} {emoji_descriptions[emoji]} ({count})", 'value': emoji}
        for emoji, count in top_emojis
    ]
    
    # Get the triggered input
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Handle reset button click
    if triggered_id == 'reset-range-button':
        today = pd.Timestamp.now().normalize()
        start_date = (today - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
        end_date = (today + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    # Handle click event on timeline
    elif clickData:
        clicked_date = pd.to_datetime(clickData['points'][0]['x'])
        start_date = (clicked_date - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        end_date = clicked_date.strftime('%Y-%m-%d')
    # If no dates selected and no click, use past 14 days + future 7 days
    elif not start_date or not end_date:
        today = pd.Timestamp.now().normalize()
        start_date = (today - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
        end_date = (today + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Apply date filter
    mask = pd.Series(True, index=state.df.index)
    mask &= state.df['Date'] >= start_date
    mask &= state.df['Date'] <= end_date
    
    filtered_df = state.df[mask].copy()
    
    # Apply text search filter if search text is provided
    if search_text and search_text.strip():
        search_text = search_text.lower().strip()
        search_mask = (
            filtered_df['Content'].str.lower().str.contains(search_text, na=False) |
            filtered_df['Title'].str.lower().str.contains(search_text, na=False) |
            filtered_df['emotion'].str.lower().str.contains(search_text, na=False) |
            filtered_df['topic'].str.lower().str.contains(search_text, na=False)
        )
        filtered_df = filtered_df[search_mask]
    
    # Apply emoji filter if selected
    if selected_emoji:
        emoji_mask = filtered_df['Tags'].str.contains(selected_emoji, na=False)
        filtered_df = filtered_df[emoji_mask]
    
    # Replace 'blank', '/', 'blank /', 'neutral', and empty values with empty string for display
    for col in ['emotion', 'topic', 'etc']:
        filtered_df[col] = filtered_df[col].replace(['blank', '/', 'blank /', 'neutral', ''], '')
    
    return filtered_df.to_dict('records'), state.create_timeline_figure(filtered_df), start_date, end_date, emoji_options

# Update journal data store
@app.callback(
    Output('journal-data-store', 'data'),
    [Input('journal-table', 'data')]
)
def update_journal_data_store(table_data):
    """Update the journal data store when table data changes"""
    return table_data

def handle_sigtstp(signum: int, frame: Any) -> None:
    """Handle Ctrl+Z (SIGTSTP) signal"""
    logging.info("Received Ctrl+Z signal, triggering extraction and annotation...")
    state.run_extraction_and_annotation(retag_all=False, force=True)  # Force run regardless of time

def background_processor(retag_all: bool = False) -> None:
    """Background thread that runs extraction and annotation periodically"""
    while True:
        current_time = time.time()
        # Only run if enough time has passed since last run
        if current_time - state.last_process_time >= config.min_process_interval:
            state.run_extraction_and_annotation(retag_all=retag_all, force=False)  # Don't force automatic runs
        # Check for content changes every 5 seconds
        new_df = state.load_data()
        if new_df is not None and state.df is not None:
            # Compare number of rows
            if len(new_df) != len(state.df):
                logging.info("Number of entries changed, updating dashboard")
                state.df = new_df
                state.update_event.set()
            else:
                # Compare content of each row
                for col in new_df.columns:
                    if col in state.df.columns:
                        if not (new_df[col].astype(str) == state.df[col].astype(str)).all():
                            logging.info(f"Content changed in column {col}, updating dashboard")
                            state.df = new_df
                            state.update_event.set()
                            break
        time.sleep(5)  # Check every 5 seconds

def main() -> None:
    """Main entry point for the dashboard"""
    parser = argparse.ArgumentParser(description='Journal Reflections Dashboard')
    parser.add_argument('--retag-all', action='store_true',
                       help='Re-tag all entries from scratch on startup')
    args = parser.parse_args()

    signal.signal(signal.SIGTSTP, handle_sigtstp)
    
    background_thread = threading.Thread(
        target=background_processor,
        args=(args.retag_all,),
        daemon=True
    )
    background_thread.start()

    app.run_server(debug=False, port=8050)

if __name__ == '__main__':
    main()
