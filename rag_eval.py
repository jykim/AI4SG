#!/usr/bin/env python3
"""
RAG Evaluation Interface

This module provides a dashboard for evaluating the RAG retrieval pipeline,
including debug information about document retrieval and matching.
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

from rag_utils import JournalRAG

# Initialize configuration
def load_config():
    """Load configuration from config_rag.yaml"""
    config_path = Path("config_rag.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config_rag.yaml not found")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize configuration
config = load_config()

# Initialize RAG system
rag = JournalRAG(config)

def create_debug_panel(debug_info: Dict[str, Any]) -> dbc.Card:
    """Create a debug panel showing RAG retrieval information."""
    return dbc.Card([
        dbc.CardHeader("RAG Retrieval Information"),
        dbc.CardBody([
            # Query Information
            html.H5("Query Information", className="mb-3"),
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
                    html.Strong("Unique Documents: "),
                    html.Span(str(debug_info.get('rag_metrics', {}).get('num_unique_docs', 0))),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Total Chunks: "),
                    html.Span(str(debug_info.get('rag_metrics', {}).get('num_chunks', 0))),
                ], className="mb-2"),
            ], className="bg-light p-3 rounded mb-3"),
            
            # Document Diversity Information
            html.H6("Document Diversity", className="mt-4 mb-2"),
            html.Div([
                html.Div([
                    html.Strong("Documents with Multiple Chunks: "),
                    html.Span(str(len([doc for doc in debug_info.get('retrieved_documents', []) 
                                     if doc.get('chunk_index', 0) > 0]))),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Average Chunks per Document: "),
                    html.Span(f"{sum(1 + doc.get('chunk_index', 0) for doc in debug_info.get('retrieved_documents', [])) / max(1, len(set(doc.get('doc_id', '') for doc in debug_info.get('retrieved_documents', [])))):.2f}"),
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
                            html.Strong("Chunk Index: "), html.Span(str(doc.get('chunk_index', 0))),
                            html.Br(),
                            html.Strong("Content: "), html.Span(doc.get('Content', '')),
                        ], className="ml-3 mt-2")
                    ], className="border-bottom pb-3 mb-3")
                ])
                for i, doc in enumerate(debug_info.get('retrieved_documents', []))
            ]),
        ])
    ])

def get_random_documents(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    """Get n random documents from the dataframe."""
    if df.empty:
        return []
    random_docs = df.sample(n=min(n, len(df))).to_dict('records')
    return random_docs

def create_query_panel() -> dbc.Card:
    """Create the query interface panel."""
    return dbc.Card([
        dbc.CardHeader("Query Interface"),
        dbc.CardBody([
            # Keyword Query Section
            html.Div([
                html.H5("Keyword Query", className="mb-3"),
                # Input area
                dbc.InputGroup([
                    dbc.Textarea(
                        id="user-input",
                        placeholder="Enter your query...",
                        className="me-2",
                        style={"height": "100px"}
                    ),
                    dbc.Button("Search", id="search-button", color="primary")
                ]),
            ]),
            # Document Query Section
            html.Div([
                html.H5("Document Query", className="mt-4 mb-3"),
                html.Div(id="random-docs-list", className="list-group")
            ])
        ])
    ], className="h-100")

def create_random_doc_item(doc: Dict[str, Any], index: int) -> html.Div:
    """Create a clickable random document item."""
    content = doc.get('Content', '')
    # Handle NaN values
    if pd.isna(content):
        content = ''
    content = str(content)
    
    # Create metadata string for query
    metadata = {
        'date': doc.get('Date', ''),
        'title': doc.get('Title', ''),
        'emotion': doc.get('emotion', ''),
        'topic': doc.get('topic', ''),
        'tags': doc.get('Tags', '')
    }
    metadata_str = f"Date: {metadata['date']}\nTitle: {metadata['title']}\nEmotion: {metadata['emotion']}\nTopic: {metadata['topic']}\nTags: {metadata['tags']}\n\nContent: {content}"
    
    return html.Div([
        html.Div([
            html.Strong(f"{doc.get('Date', '')} - {doc.get('Title', '')}"),
            html.Br(),
            html.Small(content[:100] + '...' if len(content) > 100 else content),
            # Store the full content and metadata in a hidden div
            html.Div(metadata_str, 
                    id={'type': 'doc-content', 'index': index},
                    style={'display': 'none'})
        ], className="list-group-item list-group-item-action", 
           id={'type': 'random-doc', 'index': index},
           style={'cursor': 'pointer'})
    ])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("RAG Retrieval Evaluation", className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        # Query Panel (4 columns)
        dbc.Col([
            create_query_panel()
        ], width=4),
        
        # Results Panel (8 columns)
        dbc.Col([
            html.Div(id="results-panel")
        ], width=8)
    ], className="h-100")
], fluid=True, className="vh-100")

def load_journal_entries() -> pd.DataFrame:
    """Load journal entries from the annotated CSV file in the output directory."""
    try:
        # Look for the annotated CSV file in the output directory
        output_dir = Path(config['output_dir'])
        annotated_file = output_dir / 'journal_entries_annotated.csv'
        if not annotated_file.exists():
            logging.warning(f"No annotated journal entries found in {output_dir}")
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

# Callback to update random documents
@callback(
    Output("random-docs-list", "children"),
    Input("search-button", "n_clicks")
)
def update_random_docs(n_clicks):
    """Update the list of random documents."""
    journal_entries = load_journal_entries()
    if journal_entries.empty:
        return []
    
    random_docs = get_random_documents(journal_entries)
    return [create_random_doc_item(doc, i) for i, doc in enumerate(random_docs)]

# Combined callback to handle both keyword search and document-based retrieval
@callback(
    Output("results-panel", "children"),
    [Input("user-input", "n_submit"),
     Input({'type': 'random-doc', 'index': dash.ALL}, 'n_clicks')],
    [State("user-input", "value"),
     State({'type': 'random-doc', 'index': dash.ALL}, 'n_clicks'),
     State({'type': 'doc-content', 'index': dash.ALL}, 'children')]
)
def handle_search_and_doc_retrieval(n_submit, random_clicks, user_input, random_clicks_state, doc_contents):
    """Handle both keyword search and document-based retrieval."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return [None]
    
    # Get the trigger ID
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Load journal entries
    journal_entries = load_journal_entries()
    if journal_entries.empty:
        error_message = "No journal entries found. Please ensure there are CSV files in the input directory."
        return [html.Div(error_message, className="alert alert-danger")]
    
    # Initialize vector store if not already initialized
    if not rag.vector_store:
        logging.info("Initializing vector store...")
        rag.initialize_vector_store(journal_entries.to_dict('records'))
        logging.info("Vector store initialized")
    
    # Get RAG results with timing
    import time
    start_time = time.time()
    
    # Determine the query based on the trigger
    if trigger_id == 'user-input':
        if not user_input:
            return [None]
        query = user_input
        logging.info(f"Performing keyword search for query: {query}")
    else:
        # Handle random document click
        clicked_index = next((i for i, v in enumerate(random_clicks_state) if v is not None), None)
        if clicked_index is None or clicked_index >= len(doc_contents):
            return [None]
        query = doc_contents[clicked_index]
        logging.info("Performing document-based retrieval")
    
    # Perform RAG retrieval
    relevant_entries, timing_metrics = rag.get_relevant_entries(query)
    logging.info(f"Retrieved {len(relevant_entries)} relevant entries")
    
    processing_time = time.time() - start_time
    
    # Create debug information
    debug_info = {
        'query': query,
        'total_docs': len(journal_entries),
        'retrieved_docs_count': len(relevant_entries),
        'retrieved_documents': relevant_entries,
        'rag_metrics': timing_metrics,
    }
    
    # Create and return results panel
    return [create_debug_panel(debug_info)]

if __name__ == "__main__":
    app.run_server(debug=True, port=8051) 