#!/usr/bin/env python3
"""
Tests for the Journal Analysis Dashboard
"""

import logging
from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from .app import (
    Config as BaseConfig,
    AppState,
    create_layout,
    update_table_and_timeline,
    state as global_state
)
from datetime import datetime
import pandas as pd
from dash import Input, Output, html
import os
import pytest
import sys

# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

@pytest.fixture
def test_config():
    return BaseConfig()

class TestConfig:
    """Test configuration loading and validation"""
    
    def test_load_config(self, test_config):
        """Test loading configuration from file"""
        test_config.load_config()
        assert test_config.input_dir.exists()
        assert test_config.output_dir.exists()
        assert test_config.api_cache_dir.exists()

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