import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import yaml
import plotly.graph_objects as go
from datetime import datetime
import threading
import time
import subprocess
import logging
import signal
import argparse
from pathlib import Path
import os
from typing import Optional, Tuple, Dict, Any

# Constants
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)
SCRIPT_DIR = Path(__file__).parent.absolute()
MIN_PROCESS_INTERVAL = 600  # Minimum seconds between processing runs (10 minutes)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'dashboard.log'),
        logging.StreamHandler()
    ]
)

class DashboardState:
    """Manages the global state of the dashboard"""
    def __init__(self):
        self.is_processing = False
        self.df: Optional[pd.DataFrame] = None
        self.last_entries_count = 0
        self.update_event = threading.Event()
        self.last_process_time = 0
        self.processing_lock = threading.Lock()
        self.processing_requested = False
        self.last_content_hash = None  # Track content changes

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load and process the journal data"""
        try:
            # Use journal_entries_annotated.csv as the base file
            df = pd.read_csv(OUTPUT_DIR / 'journal_entries_annotated.csv')
            
            # Clean up and parse dates, dropping any rows with invalid dates
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
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
            
            fig.add_trace(go.Scatter(
                x=[row['Date']],
                y=[row['Y_Normalized']],
                mode='markers',
                marker=dict(
                    size=15,
                    color=color,
                    line=dict(color='black', width=1)
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
            if not force and current_time - self.last_process_time < MIN_PROCESS_INTERVAL:
                logging.debug("Skipping processing - too soon since last run")
                return False
            self.is_processing = True
            self.last_process_time = current_time
            self.processing_requested = False  # Reset the request flag

        try:
            env = os.environ.copy()
            env['OUTPUT_DIR'] = str(OUTPUT_DIR)

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
        return '#808080'
    emotion_rows = full_df[full_df['emotion'] == emotion]
    if emotion_rows.empty:
        return '#808080'
    return emotion_rows['emotion_visual'].iloc[0]

# Initialize global state
state = DashboardState()
state.df = state.load_data()
if state.df is not None:
    state.last_entries_count = len(state.df)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_layout() -> dbc.Container:
    """Create the dashboard layout"""
    # Calculate default date range (last 30 days + next 7 days)
    if state.df is not None and not state.df.empty:
        end_date = state.df['Date'].max()
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = (pd.to_datetime(end_date) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    else:
        start_date = None
        end_date = None

    return dbc.Container([
        html.H1("Journal Reflections Dashboard", className="text-center my-4"),
        
        # Status indicator
        dbc.Row([
            dbc.Col([
                html.Div(id='status-indicator', className="text-center mb-3")
            ], width=12)
        ]),
        
        # Timeline visualization
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='timeline-graph',
                    figure=state.create_timeline_figure(state.df) if state.df is not None else {},
                    clickData=None  # Add clickData property
                )
            ], width=12)
        ], className="mb-4"),
        
        # Date range picker
        dbc.Row([
            dbc.Col([
                html.Label("Select Date Range:"),
                dbc.Row([
                    dbc.Col([
                        dcc.DatePickerRange(
                            id='date-picker',
                            start_date=start_date,
                            end_date=end_date,
                            display_format='YYYY-MM-DD'
                        )
                    ], width=10),
                    dbc.Col([
                        dbc.Button(
                            "Reset Range",
                            id="reset-range-button",
                            color="secondary",
                            className="ms-2"
                        )
                    ], width=2)
                ])
            ], width=12)
        ], className="mb-4"),
        
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
                    style_table={'height': '70vh', 'overflowY': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'pre-wrap',
                        'height': 'auto',
                        'minWidth': '50px',
                        'maxWidth': '500px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
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
                        'fontWeight': 'bold'
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
                        for emotion in state.df['emotion'].unique() if state.df is not None
                    ]
                )
            ], width=12)
        ])
    ])

app.layout = create_layout()

@app.callback(
    [Output('journal-table', 'data'),
     Output('timeline-graph', 'figure'),
     Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('timeline-graph', 'clickData'),
     Input('reset-range-button', 'n_clicks')]
)
def update_table_and_timeline(start_date: Optional[str], end_date: Optional[str], clickData: Optional[dict], reset_clicks: Optional[int]) -> Tuple[list, dict, str, str]:
    """Update both the table and timeline based on selected date range or click event"""
    if state.df is None or state.df.empty:
        return [], {}, None, None
    
    # Get the triggered input
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Handle reset button click
    if triggered_id == 'reset-range-button':
        end_date = state.df['Date'].max()
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = (pd.to_datetime(end_date) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    # Handle click event on timeline
    elif clickData:
        clicked_date = pd.to_datetime(clickData['points'][0]['x'])
        start_date = (clicked_date - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        end_date = clicked_date.strftime('%Y-%m-%d')
    # If no dates selected and no click, use last 30 days
    elif not start_date or not end_date:
        end_date = state.df['Date'].max()
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    mask = pd.Series(True, index=state.df.index)
    mask &= state.df['Date'] >= start_date
    mask &= state.df['Date'] <= end_date
    
    filtered_df = state.df[mask]
    return filtered_df.to_dict('records'), state.create_timeline_figure(filtered_df), start_date, end_date

@app.callback(
    Output('status-indicator', 'children'),
    Input('journal-table', 'data')
)
def update_status(data: list) -> str:
    """Update the status indicator"""
    if not data:
        return "No entries found for the selected date range."
    return f"Showing {len(data)} entries"

def handle_sigtstp(signum: int, frame: Any) -> None:
    """Handle Ctrl+Z (SIGTSTP) signal"""
    logging.info("Received Ctrl+Z signal, triggering extraction and annotation...")
    state.run_extraction_and_annotation(retag_all=False, force=True)  # Force run regardless of time

def background_processor(retag_all: bool = False) -> None:
    """Background thread that runs extraction and annotation periodically"""
    while True:
        current_time = time.time()
        # Only run if enough time has passed since last run
        if current_time - state.last_process_time >= MIN_PROCESS_INTERVAL:
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

    app.run_server(debug=True)

if __name__ == '__main__':
    main()
