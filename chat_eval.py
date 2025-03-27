#!/usr/bin/env python3
"""
Chat Evaluation Interface

This module provides a dashboard for evaluating chat interactions with the journal agent,
including debug information about RAG retrieval and response generation.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Any, Optional

from agent_utils import Config, get_chat_response, format_journal_entries_as_markdown
from chat_utils import (
    format_chat_message,
    process_existing_messages,
    create_chat_history,
    parse_chat_entry
)
from rag_utils import JournalRAG

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize configuration
config = Config()

# Initialize RAG system
rag = JournalRAG(config)

def create_debug_panel(debug_info: Dict[str, Any]) -> dbc.Card:
    """Create a debug panel showing RAG and response generation information."""
    return dbc.Card([
        dbc.CardHeader("Debug Information"),
        dbc.CardBody([
            # RAG Information
            html.H5("RAG Retrieval", className="mb-3"),
            html.Div([
                html.Strong("Query: "),
                html.Span(debug_info.get('query', 'N/A')),
            ], className="mb-2"),
            html.Div([
                html.Strong("Total Documents in Index: "),
                html.Span(str(debug_info.get('total_docs', 0))),
            ], className="mb-2"),
            html.Div([
                html.Strong("Retrieved Documents: "),
                html.Span(str(debug_info.get('retrieved_docs_count', 0))),
            ], className="mb-2"),
            
            # RAG Timing Metrics
            html.H6("RAG Timing Metrics", className="mt-4 mb-2"),
            html.Div([
                html.Div([
                    html.Strong("Retrieval Time: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('retrieval_time', 0):.3f}s"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Processing Time: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('processing_time', 0):.3f}s"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Total RAG Time: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('total_time', 0):.3f}s"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Initial Documents Retrieved: "),
                    html.Span(str(debug_info.get('rag_metrics', {}).get('num_docs_retrieved', 0))),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Unique Documents After Filtering: "),
                    html.Span(str(debug_info.get('rag_metrics', {}).get('num_unique_docs', 0))),
                ], className="mb-2"),
            ], className="bg-light p-3 rounded mb-3"),
            
            # Retrieved Documents
            html.H6("Retrieved Documents", className="mt-4 mb-2"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Strong(f"Document {i+1} (Score: {doc.get('match_score', 0):.3f})"),
                        html.Div([
                            html.Strong("Date: "), html.Span(doc.get('Date', '')),
                            html.Br(),
                            html.Strong("Title: "), html.Span(doc.get('Title', '')),
                            html.Br(),
                            html.Strong("Emotion: "), html.Span(doc.get('emotion', '')),
                            html.Br(),
                            html.Strong("Topic: "), html.Span(doc.get('topic', '')),
                            html.Br(),
                            html.Strong("Tags: "), html.Span(doc.get('Tags', '')),
                            html.Br(),
                            html.Strong("Content: "), html.Span(doc.get('Content', '')),
                        ], className="ml-3 mt-2")
                    ], className="border-bottom pb-3 mb-3")
                ])
                for i, doc in enumerate(debug_info.get('retrieved_documents', []))
            ]),
            
            # Entries in LLM Prompt
            html.H6("Entries in LLM Prompt", className="mt-4 mb-2"),
            html.Div([
                html.Div([
                    html.Strong("Recent Entries (Past 7 Days):"),
                    html.Div([
                        html.Div([
                            html.Strong("Date: "), html.Span(entry.get('Date', '')),
                            html.Br(),
                            html.Strong("Title: "), html.Span(entry.get('Title', '')),
                            html.Br(),
                            html.Strong("Content: "), html.Span(entry.get('Content', '')),
                        ], className="ml-3 mt-2")
                        for entry in debug_info.get('recent_entries', [])
                    ])
                ], className="mb-3"),
                html.Div([
                    html.Strong("Relevant Past Entries:"),
                    html.Div([
                        html.Div([
                            html.Strong("Date: "), html.Span(entry.get('Date', '')),
                            html.Br(),
                            html.Strong("Title: "), html.Span(entry.get('Title', '')),
                            html.Br(),
                            html.Strong("Content: "), html.Span(entry.get('Content', '')),
                        ], className="ml-3 mt-2")
                        for entry in debug_info.get('relevant_entries', [])
                    ])
                ], className="mb-3"),
            ], className="bg-light p-3 rounded mb-3"),
            
            # Response Generation
            html.H5("Response Generation", className="mt-4 mb-3"),
            html.Div([
                html.Strong("Model: "),
                html.Span(debug_info.get('model', 'N/A')),
            ], className="mb-2"),
            html.Div([
                html.Strong("Tokens Used: "),
                html.Span(str(debug_info.get('tokens_used', 0))),
            ], className="mb-2"),
            html.Div([
                html.Strong("Processing Time: "),
                html.Span(f"{debug_info.get('processing_time', 0):.2f}s"),
            ], className="mb-2"),
            
            # Cache Information
            html.H5("Cache Information", className="mt-4 mb-3"),
            html.Div([
                html.Strong("Cache Hit: "),
                html.Span("Yes" if debug_info.get('cache_hit', False) else "No"),
            ], className="mb-2"),
            html.Div([
                html.Strong("Cache Key: "),
                html.Span(debug_info.get('cache_key', 'N/A')),
            ], className="mb-2"),
        ])
    ])

def create_chat_panel() -> dbc.Card:
    """Create the chat interface panel."""
    return dbc.Card([
        dbc.CardHeader("Chat Interface"),
        dbc.CardBody([
            # Chat messages container
            html.Div(id="chat-messages", className="chat-container mb-3"),
            
            # Input area
            dbc.InputGroup([
                dbc.Textarea(
                    id="user-input",
                    placeholder="Type your message...",
                    className="me-2",
                    style={"height": "100px"}
                ),
                dbc.Button("Send", id="send-button", color="primary")
            ]),
            
            # Toggles
            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="debug-toggle",
                        label="Show Debug Info",
                        value=True,
                        className="mt-3"
                    )
                ], width=6),
                dbc.Col([
                    dbc.Switch(
                        id="chat-history-toggle",
                        label="Include Chat History",
                        value=True,
                        className="mt-3"
                    )
                ], width=6)
            ])
        ])
    ], className="h-100")

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Journal Chat Evaluation", className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        # Chat Panel (8 columns)
        dbc.Col([
            create_chat_panel()
        ], width=8),
        
        # Debug Panel (4 columns)
        dbc.Col([
            html.Div(id="debug-panel", style={"display": "none"})
        ], width=4)
    ], className="h-100")
], fluid=True, className="vh-100")

def load_journal_entries() -> pd.DataFrame:
    """Load journal entries from the annotated CSV file in the output directory."""
    try:
        # Look for the annotated CSV file in the output directory
        annotated_file = config.output_dir / 'journal_entries_annotated.csv'
        if not annotated_file.exists():
            logging.warning("No annotated journal entries found in output directory")
            return pd.DataFrame()
        
        logging.info(f"Loading journal entries from {annotated_file}")
        
        df = pd.read_csv(annotated_file)
        if df.empty:
            logging.warning("Loaded CSV file is empty")
            return pd.DataFrame()
            
        # Clean up and parse dates, dropping any rows with invalid dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
        
        # Replace blank/null emotions with empty string
        df['emotion'] = df['emotion'].fillna('')
        df['emotion'] = df['emotion'].apply(lambda x: '' if pd.isna(x) or str(x).strip() == '' else x)
        
        # Clean up title formatting - remove quotes and asterisks
        df['Title'] = df['Title'].apply(lambda x: str(x).strip('"*') if pd.notna(x) else x)
        
        # Create Tags column with emojis, filtering out NaN values
        df['Tags'] = df.apply(lambda row: ' '.join(filter(None, [
            row['topic_visual'] if pd.notna(row['topic_visual']) else '',
            row['etc_visual'] if pd.notna(row['etc_visual']) else ''
        ])), axis=1)
        
        return df
    except Exception as e:
        logging.error(f"Error loading journal entries: {e}")
        return pd.DataFrame()

# Callback to handle message sending and debug panel toggling
@callback(
    [Output("chat-messages", "children"),
     Output("user-input", "value"),
     Output("debug-panel", "children"),
     Output("debug-panel", "style")],
    [Input("send-button", "n_clicks"),
     Input("user-input", "n_submit"),
     Input("debug-toggle", "value"),
     Input("chat-history-toggle", "value")],
    [State("user-input", "value"),
     State("chat-messages", "children"),
     State("debug-toggle", "value"),
     State("chat-history-toggle", "value")]
)
def handle_message_and_debug(n_clicks, n_submit, debug_toggle, chat_history_toggle, user_input, current_messages, show_debug, include_chat_history):
    """Handle sending/receiving messages and debug panel toggling."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_messages, "", None, {"display": "none"}
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Handle toggles
    if trigger_id in ["debug-toggle", "chat-history-toggle"]:
        if show_debug:
            return current_messages, user_input, create_debug_panel({}), {"display": "block"}
        return current_messages, user_input, None, {"display": "none"}
    
    # Handle message sending
    if not user_input or (not n_clicks and not n_submit):
        return current_messages, "", None, {"display": "none"}
    
    # Initialize messages list if None
    if current_messages is None:
        current_messages = []
    
    # Add user message
    user_message = format_chat_message(
        user_input,
        is_user=True,
        is_latest=True,
        timestamp=datetime.now()
    )
    current_messages.append(user_message)
    
    # Create chat history only if enabled
    chat_history = create_chat_history(current_messages, datetime.now()) if include_chat_history else []
    
    # Load journal entries
    journal_entries = load_journal_entries()
    if journal_entries.empty:
        error_message = "No journal entries found. Please ensure there are CSV files in the input directory."
        error_response = format_chat_message(
            error_message,
            is_user=False,
            is_latest=True,
            timestamp=datetime.now()
        )
        current_messages.append(error_response)
        return current_messages, "", None, {"display": "none"}
    
    # Get chat response with timing
    import time
    start_time = time.time()
    
    response = get_chat_response(
        user_input,
        journal_entries,
        chat_history,
        config,
        rag_only=True,  # Always use RAG
        include_chat_history=include_chat_history
    )
    
    processing_time = time.time() - start_time
    
    # Add assistant message
    assistant_message = format_chat_message(
        response['response'],
        is_user=False,
        is_latest=True,
        timestamp=datetime.now()
    )
    current_messages.append(assistant_message)
    
    # Create debug information
    debug_info = {
        'query': user_input,
        'total_docs': len(journal_entries),
        'retrieved_docs_count': len(response.get('relevant_entries', [])),
        'retrieved_documents': response.get('relevant_entries', []),
        'rag_metrics': response.get('rag_metrics', {}),
        'model': 'gpt-4-1106-preview',
        'tokens_used': response.get('response_metadata', {}).get('total_tokens', 0),
        'processing_time': processing_time,
        'cache_hit': response.get('cache_hit', False),
        'cache_key': response.get('cache_key', 'N/A'),
        'include_chat_history': include_chat_history,
        'recent_entries': response.get('recent_entries', []),
        'relevant_entries': response.get('relevant_entries', [])
    }
    
    # Create debug panel if enabled
    debug_panel = create_debug_panel(debug_info) if show_debug else None
    debug_style = {"display": "block"} if show_debug else {"display": "none"}
    
    return current_messages, "", debug_panel, debug_style

if __name__ == "__main__":
    app.run_server(debug=True, port=8051) 