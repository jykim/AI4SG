import logging
from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from dash_journal import (
    Config as BaseConfig,
    DashboardState,
    create_layout,
    update_table_and_timeline,
    state as global_state
)
from datetime import datetime
import pandas as pd
from dash import Input, Output, html
import os

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class TestConfig(BaseConfig):
    """Test configuration class for the Dash app"""
    def __init__(self):
        # Don't call parent's __init__ since we want to override everything
        self.output_dir = Path('test_data/output')
        self.max_word_count = 50  # Maximum number of words to show in abbreviated messages
        self.chat_log_path = self.output_dir / 'chat_log_test.md'
        self.journal_entries_path = self.output_dir / 'journal_entries_annotated.csv'
        self.agent_cache_dir = Path('test_data/agent_cache')
        self.suggested_questions = [
            "How was your day?",
            "What did you learn today?",
            "What are your goals for tomorrow?"
        ]
        # Call load_config to set up remaining values and create directories
        self.load_config()
        
    def load_config(self) -> None:
        """Override config to use test data directory"""
        config = {
            'input_dir': 'input',
            'output_dir': 'test_data/output',
            'api_cache_dir': 'test_data/api_cache',
            'agent_cache_dir': 'test_data/agent_cache',  # Add agent cache directory
            'min_process_interval': 0  # No delay for testing
        }
        
        # Set configuration values
        self.input_dir = Path(config.get('input_dir', 'input'))
        self.output_dir = Path(config.get('output_dir', 'test_data/output'))
        self.api_cache_dir = Path(config.get('api_cache_dir', 'test_data/api_cache'))
        self.agent_cache_dir = Path(config.get('agent_cache_dir', 'test_data/agent_cache'))  # Set agent cache directory
        self.min_process_interval = config.get('min_process_interval', 0)
        
        # Use actual API key from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        if not self.openai_api_key:
            logging.warning("No OpenAI API key found in environment variables")
        
        # Create all necessary directories
        self.setup_directories()
        
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [self.input_dir, self.output_dir, self.api_cache_dir, self.agent_cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_chat_log_path(self) -> Path:
        """Override to use test chat log"""
        return self.chat_log_path
        
    def get_todays_chat_log(self) -> str:
        """Override to return empty string, preventing chat history from being loaded"""
        return ""

def set_test_date(df: pd.DataFrame) -> None:
    """Set today's date to match the latest date in test data"""
    if df is not None and not df.empty:
        latest_date = pd.to_datetime(df['Date'].max())
        # Mock today's date to be the latest date in test data
        pd.Timestamp.now = lambda: latest_date
        logging.info(f"Set test date to: {latest_date.strftime('%Y-%m-%d')}")

def create_demo_layout(state):
    """Create the demo dashboard layout without chat history"""
    # Get the base layout
    base_layout = create_layout(state)
    
    # Create a new layout with just the main content, excluding chat history (children[0])
    layout = html.Div([
        # Main content from base layout (excluding chat row which is children[0])
        html.Div([
            base_layout.children[1] if len(base_layout.children) > 1 else None,  # Filters row
            base_layout.children[2] if len(base_layout.children) > 2 else None,  # Timeline row
            base_layout.children[3] if len(base_layout.children) > 3 else None,  # Table row
        ])
    ])
    return layout

def main():
    # Initialize test configuration
    config = TestConfig()
    
    # Create a new Dash app instance for testing
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Journal Dashboard (demo)"
    
    # Initialize dashboard state with test config
    global_state.config = config
    global_state.df = global_state.load_data()
    
    if global_state.df is not None:
        global_state.last_entries_count = len(global_state.df)
        # Set today's date to match the latest date in test data
        set_test_date(global_state.df)
        logging.info(f"Loaded {len(global_state.df)} entries")
        
        # Initialize the app with the processed data
        app.layout = create_demo_layout(global_state)
        
        # Register the callback from dash_journal.py
        app.callback(
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
        )(update_table_and_timeline)
    
    # Run the dashboard on port 8049
    app.run_server(debug=True, port=8049)

if __name__ == '__main__':
    main() 