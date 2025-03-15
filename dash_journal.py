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
MIN_PROCESS_INTERVAL = 600  # Minimum seconds between processing runs

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
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.df: Optional[pd.DataFrame] = None
        self.last_entries_count = 0
        self.update_event = threading.Event()
        self.emotion_colors: Dict[str, str] = {}
        self.tag_emojis: Dict[str, str] = {}
        self.last_process_time = 0
        self.processing_requested = False
        self.load_configs()

    def load_configs(self) -> None:
        """Load emotion colors and tag emojis from YAML files"""
        with open(SCRIPT_DIR / 'emotion_colors.yaml', 'r') as f:
            self.emotion_colors = yaml.safe_load(f)
        with open(SCRIPT_DIR / 'tag_emojis.yaml', 'r') as f:
            self.tag_emojis = yaml.safe_load(f)

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load and process the journal data"""
        try:
            df = pd.read_csv(OUTPUT_DIR / 'reflections_annotated.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date', ascending=False)
            df['Tags'], df['Tags_Tooltip'] = zip(*df['topic'].apply(self.get_tag_emojis))
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None

    def get_tag_emojis(self, tags: str) -> Tuple[str, str]:
        """Convert tags to emoji strings with tooltips"""
        if pd.isna(tags):
            return '', ''
        tag_list = [tag.strip() for tag in tags.split(',')]
        emoji_list = []
        tooltip_list = []
        for tag in tag_list:
            emoji = self.tag_emojis.get(tag, '')
            if emoji:
                emoji_list.append(emoji)
                tooltip_list.append(f"{emoji} ({tag})")
        return ' '.join(emoji_list), ' '.join(tooltip_list)

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

        # Add scatter points for each journal entry
        for idx, row in df.iterrows():
            # Create hover text with sentence boundaries
            content = row['Content']
            # Add newline after each sentence (assuming sentences end with .!?)
            content = content.replace('. ', '.<br>').replace('! ', '!<br>').replace('? ', '?<br>')
            
            hover_text = f"Date: {row['Date'].strftime('%Y-%m-%d')}<br>"
            hover_text += f"#{row['emotion']} {row['Tags']}<br>"
            hover_text += f"Content: {content[:200]}..." if len(content) > 200 else f"Content: {content}"
            
            # Get color based on emotion, default to gray if emotion not in dictionary
            color = self.emotion_colors.get(row['emotion'], '#808080')
            
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

    def run_extraction_and_annotation(self, retag_all: bool = False) -> bool:
        """Run the extraction and annotation scripts in sequence"""
        current_time = time.time()
        
        # Prevent processing if already running or if last run was too recent
        with self.processing_lock:
            if self.is_processing:
                logging.debug("Processing already in progress, skipping...")
                return False
            if current_time - self.last_process_time < MIN_PROCESS_INTERVAL:
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

            # Run annotation
            annotate_script = SCRIPT_DIR / 'annotate_journal.py'
            cmd = ['python', str(annotate_script)]
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
            with self.processing_lock:
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
            if (len(new_entry['Content']) > len(last_entry['Content']) and 
                last_entry['Content'] in new_entry['Content']):
                logging.info("New entry is a superset of the last entry. Removing the last entry.")
                # Remove the last entry from existing data
                self.df = self.df.iloc[1:]
                self.last_entries_count -= 1
        
        # Update with new data
        self.df = new_df
        self.last_entries_count = len(self.df)
        # Reload configs to get any new emotion colors
        self.load_configs()
        self.update_event.set()
        logging.info(f"Updated entries count: {self.last_entries_count}")

# Initialize global state
state = DashboardState()
state.df = state.load_data()
if state.df is not None:
    state.last_entries_count = len(state.df)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_layout() -> dbc.Container:
    """Create the dashboard layout"""
    # Calculate default date range (last 30 days)
    if state.df is not None and not state.df.empty:
        end_date = state.df['Date'].max()
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
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
                    figure=state.create_timeline_figure(state.df) if state.df is not None else {}
                )
            ], width=12)
        ], className="mb-4"),
        
        # Date range picker
        dbc.Row([
            dbc.Col([
                html.Label("Select Date Range:"),
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=start_date,
                    end_date=end_date,
                    display_format='YYYY-MM-DD'
                )
            ], width=12)
        ], className="mb-4"),
        
        # DataTable
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='journal-table',
                    columns=[
                        {"name": "Date", "id": "Date"},
                        {"name": "Title", "id": "Title"},
                        {"name": "Emotion", "id": "emotion"},
                        {"name": "Tags", "id": "Tags"},
                        {"name": "Content", "id": "Content"}
                    ],
                    data=state.df.to_dict('records') if state.df is not None and not state.df.empty else [],
                    style_table={'height': '70vh', 'overflowY': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'whiteSpace': 'pre-wrap',
                        'height': 'auto'
                    },
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
                            'backgroundColor': color
                        }
                        for emotion, color in state.emotion_colors.items()
                    ]
                )
            ], width=12)
        ])
    ])

app.layout = create_layout()

@app.callback(
    [Output('journal-table', 'data'),
     Output('timeline-graph', 'figure')],
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_table_and_timeline(start_date: Optional[str], end_date: Optional[str]) -> Tuple[list, dict]:
    """Update both the table and timeline based on selected date range"""
    if state.df is None or state.df.empty:
        return [], {}
    
    # If no dates selected, use last 30 days
    if not start_date or not end_date:
        end_date = state.df['Date'].max()
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    mask = pd.Series(True, index=state.df.index)
    mask &= state.df['Date'] >= start_date
    mask &= state.df['Date'] <= end_date
    
    filtered_df = state.df[mask]
    return filtered_df.to_dict('records'), state.create_timeline_figure(filtered_df)

@app.callback(
    Output('status-indicator', 'children'),
    Input('journal-table', 'data')
)
def update_status(data: list) -> str:
    """Update the status indicator"""
    if not data:
        return "No entries found for the selected date range."
    return f"Showing {len(data)} entries"

def background_processor(retag_all: bool = False) -> None:
    """Background thread that runs extraction and annotation periodically"""
    while True:
        current_time = time.time()
        with state.processing_lock:
            if not state.is_processing and current_time - state.last_process_time >= MIN_PROCESS_INTERVAL:
                state.processing_requested = True
        
        if state.processing_requested:
            state.run_extraction_and_annotation(retag_all)
        
        time.sleep(5)  # Check every 5 seconds instead of running continuously

def handle_sigtstp(signum: int, frame: Any) -> None:
    """Handle Ctrl+Z (SIGTSTP) signal"""
    logging.info("Received Ctrl+Z signal, triggering extraction and annotation...")
    with state.processing_lock:
        state.processing_requested = True

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
