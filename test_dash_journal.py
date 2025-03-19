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

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class TestConfig(BaseConfig):
    """Test configuration class that overrides paths for testing"""
    def load_config(self) -> None:
        """Override config to use test data directory"""
        config = {
            'input_dir': 'input',
            'output_dir': 'test_data/output',
            'api_cache_dir': 'test_data/api_cache',
            'min_process_interval': 0  # No delay for testing
        }
        
        # Set configuration values
        self.input_dir = Path(config.get('input_dir', 'input'))
        self.output_dir = Path(config.get('output_dir', 'test_data/output'))
        self.api_cache_dir = Path(config.get('api_cache_dir', 'test_data/api_cache'))
        self.min_process_interval = config.get('min_process_interval', 0)
        
        # For testing, we'll use a mock API key
        self.openai_api_key = 'test-api-key'
        
        # Create all necessary directories
        self.setup_directories()

def set_test_date(df: pd.DataFrame) -> None:
    """Set today's date to match the latest date in test data"""
    if df is not None and not df.empty:
        latest_date = pd.to_datetime(df['Date'].max())
        # Mock today's date to be the latest date in test data
        pd.Timestamp.now = lambda: latest_date
        logging.info(f"Set test date to: {latest_date.strftime('%Y-%m-%d')}")

def create_demo_layout(state):
    """Create the demo dashboard layout"""
    layout = create_layout(state)
    # Replace the title with the demo version
    layout.children[0].children[0].children[0] = html.H1("Journaling with AI (demo)", className="mb-0")
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