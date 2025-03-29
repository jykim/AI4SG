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
from ir_utils import BM25Retriever
from rag_graph import create_graph_panel

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

# Add persistence configuration if not present
if 'bm25' not in config:
    config['bm25'] = {}
config['bm25']['persistence'] = {
    'enabled': True,
    'directory': 'bm25_index'
}

# Initialize RAG system and BM25 retriever
rag = JournalRAG(config)
bm25_retriever = BM25Retriever(config)

def create_debug_panel(debug_info: Dict[str, Any]) -> dbc.Card:
    """Create a debug panel showing RAG retrieval information."""
    retrieval_method = debug_info.get('retrieval_method', 'N/A').upper()
    
    # Create metrics section based on retrieval method
    if retrieval_method == "BM25":
        metrics_section = html.Div([
            html.H6("BM25 Metrics", className="mt-4 mb-2"),
            html.Div([
                html.Div([
                    html.Strong("Total Time: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('total_time', 0):.3f}s"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Documents Retrieved: "),
                    html.Span(str(debug_info.get('rag_metrics', {}).get('num_docs_retrieved', 0))),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Score Range: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('min_score', 0):.3f} to {debug_info.get('rag_metrics', {}).get('max_score', 0):.3f}"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Index Status: "),
                    html.Span(debug_info.get('index_status', 'Unknown')),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Indexed Documents: "),
                    html.Span(str(debug_info.get('indexed_docs', 0))),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Query Tokens: "),
                    html.Span(debug_info.get('query_tokens', 'N/A')),
                ], className="mb-2"),
            ], className="bg-light p-3 rounded mb-3"),
        ])
    elif retrieval_method == "EXACT":
        metrics_section = html.Div([
            html.H6("Exact Match Metrics", className="mt-4 mb-2"),
            html.Div([
                html.Div([
                    html.Strong("Total Time: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('total_time', 0):.3f}s"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Documents Retrieved: "),
                    html.Span(str(debug_info.get('rag_metrics', {}).get('num_docs_retrieved', 0))),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Match Score: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('exact_match_score', 0):.3f}"),
                ], className="mb-2"),
            ], className="bg-light p-3 rounded mb-3"),
        ])
    else:  # Vector Search
        metrics_section = html.Div([
            html.H6("Vector Search Metrics", className="mt-4 mb-2"),
            html.Div([
                html.Div([
                    html.Strong("Total Time: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('total_time', 0):.3f}s"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Documents Retrieved: "),
                    html.Span(str(debug_info.get('rag_metrics', {}).get('num_docs_retrieved', 0))),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Unique Documents: "),
                    html.Span(str(debug_info.get('rag_metrics', {}).get('num_unique_docs', 0))),
                ], className="mb-2"),
            ], className="bg-light p-3 rounded mb-3"),
        ])
    
    # Create diversity section based on retrieval method
    if retrieval_method == "BM25":
        diversity_section = html.Div([
            html.H6("Document Statistics", className="mt-4 mb-2"),
            html.Div([
                html.Div([
                    html.Strong("Average Score: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('avg_score', 0):.3f}"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Score Standard Deviation: "),
                    html.Span(f"{debug_info.get('rag_metrics', {}).get('score_std', 0):.3f}"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Index Size: "),
                    html.Span(f"{debug_info.get('index_size', 0)} tokens"),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Average Document Length: "),
                    html.Span(f"{debug_info.get('avg_doc_length', 0):.1f} tokens"),
                ], className="mb-2"),
            ], className="bg-light p-3 rounded mb-3"),
        ])
    elif retrieval_method == "EXACT":
        diversity_section = html.Div([
            html.H6("Match Information", className="mt-4 mb-2"),
            html.Div([
                html.Div([
                    html.Strong("Match Fields: "),
                    html.Span(", ".join(debug_info.get('exact_match_fields', []))),
                ], className="mb-2"),
                html.Div([
                    html.Strong("Matches per Field: "),
                    html.Div([
                        html.Div([
                            html.Strong(f"{field}: "),
                            html.Span(str(count)),
                        ], className="mb-1")
                        for field, count in debug_info.get('matches_per_field', {}).items()
                    ]),
                ], className="mb-2"),
            ], className="bg-light p-3 rounded mb-3"),
        ])
    else:  # Vector Search
        diversity_section = html.Div([
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
        ])
    
    # Format retrieved documents
    retrieved_docs_section = html.Div([
        html.H6("Retrieved Documents", className="mt-4 mb-2"),
        html.Div([
            html.Div([
                html.Div([
                    html.Strong(f"Document {i+1} (Score: {doc.get('match_score', 0):.3f})"),
                    html.Div([
                        html.Div([
                            html.Strong("Date: "), 
                            html.Span(doc.get('Date', '')),
                        ], className="mb-1"),
                        html.Div([
                            html.Strong("Title: "), 
                            html.Span(doc.get('Title', '')),
                        ], className="mb-1"),
                        html.Div([
                            html.Strong("Emotion: "), 
                            html.Span(doc.get('emotion', '')),
                        ], className="mb-1"),
                        html.Div([
                            html.Strong("Topic: "), 
                            html.Span(doc.get('topic', '')),
                        ], className="mb-1"),
                        html.Div([
                            html.Strong("Tags: "), 
                            html.Span(doc.get('Tags', '')),
                        ], className="mb-1"),
                        html.Div([
                            html.Strong("Content: "), 
                            html.Span(doc.get('Content', '')),
                        ], className="mb-1"),
                    ], className="ml-3 mt-2")
                ], className="border-bottom pb-3 mb-3")
            ])
            for i, doc in enumerate(debug_info.get('retrieved_documents', []))
        ])
    ])
    
    return dbc.Card([
        dbc.CardHeader("RAG Retrieval Information"),
        dbc.CardBody([
            # Graph Visualization
            create_graph_panel(debug_info, config),
            
            # Query Information
            html.H5("Query Information", className="mb-3"),
            html.Div([
                html.Strong("Query: "),
                html.Span(debug_info.get('query', 'N/A')),
            ], className="mb-2"),
            html.Div([
                html.Strong("Retrieval Method: "),
                html.Span(retrieval_method),
            ], className="mb-2"),
            html.Div([
                html.Strong("Total Documents in Index: "),
                html.Span(str(debug_info.get('total_docs', 0))),
            ], className="mb-2"),
            html.Div([
                html.Strong("Retrieved Documents: "),
                html.Span(str(debug_info.get('retrieved_docs_count', 0))),
            ], className="mb-2"),
            
            # Metrics Section
            metrics_section,
            
            # Diversity Section
            diversity_section,
            
            # Retrieved Documents Section
            retrieved_docs_section,
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
            # Retrieval Method Selector
            html.Div([
                html.H5("Retrieval Method", className="mb-3"),
                dbc.RadioItems(
                    id="retrieval-method",
                    options=[
                        {"label": "BM25 (Default)", "value": "bm25"},
                        {"label": "Vector Search", "value": "vector"},
                        {"label": "Exact Match", "value": "exact"}
                    ],
                    value="bm25",
                    inline=True,
                    className="mb-3"
                ),
            ]),
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
        
        logging.info(f"Loaded {len(df)} journal entries")
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
     State({'type': 'doc-content', 'index': dash.ALL}, 'children'),
     State("retrieval-method", "value")]
)
def handle_search_and_doc_retrieval(n_submit, random_clicks, user_input, random_clicks_state, doc_contents, retrieval_method):
    """
    Handle both keyword search and document-based retrieval in the RAG evaluation interface.
    """
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
    
    logging.info(f"Processing query with {len(journal_entries)} journal entries")
    
    # Initialize retrievers if needed
    if retrieval_method in ["vector", "exact"]:
        if not rag.vector_store:
            logging.info("Initializing vector store...")
            rag.initialize_vector_store(journal_entries.to_dict('records'))
            logging.info("Vector store initialized")
        else:
            # Check if we need to update the vector store
            collection = rag.vector_store.get()
            current_docs = len(collection['documents'])
            if current_docs != len(journal_entries):
                logging.info(f"Updating vector store: {current_docs} docs -> {len(journal_entries)} docs")
                rag.initialize_vector_store(journal_entries.to_dict('records'))
                logging.info("Vector store updated")
    
    elif retrieval_method == "bm25":
        # Check if we need to create or update the index
        if not bm25_retriever.documents:
            logging.info("Initializing BM25 index...")
            bm25_retriever.index_documents(journal_entries.to_dict('records'))
            logging.info("BM25 index initialized")
        elif len(journal_entries) != len(bm25_retriever.documents):
            logging.info("Updating BM25 index with new documents...")
            bm25_retriever.index_documents(journal_entries.to_dict('records'))
            logging.info("BM25 index updated")
    
    # Get RAG results with timing
    import time
    start_time = time.time()
    
    # Determine the query based on the trigger
    if trigger_id == 'user-input':
        if not user_input:
            return [None]
        query = user_input
        logging.info(f"Performing keyword search for query: {query}")
        is_doc_query = False
    else:
        # Handle random document click
        clicked_index = next((i for i, v in enumerate(random_clicks_state) if v is not None), None)
        if clicked_index is None or clicked_index >= len(doc_contents):
            return [None]
        query = doc_contents[clicked_index]
        logging.info("Performing document-based retrieval")
        is_doc_query = True
    
    # Perform retrieval based on selected method
    if retrieval_method == "bm25":
        try:
            # Get BM25 index status
            index_status = "Initialized" if bm25_retriever.documents else "Not Initialized"
            indexed_docs = len(bm25_retriever.documents) if bm25_retriever.documents else 0
            
            # Get query tokens for debugging
            from bm25s import tokenize
            query_tokens = tokenize([query], stopwords='en')[0]  # This returns a list of tokens
            query_tokens_str = ', '.join(str(token) for token in query_tokens)
            
            # Get index statistics
            if bm25_retriever.documents:
                # Extract content from documents
                doc_contents = [doc.get('Content', '') for doc in bm25_retriever.documents]
                total_tokens = sum(len(tokenize([content], stopwords='en')[0]) for content in doc_contents)
                avg_doc_length = total_tokens / len(bm25_retriever.documents)
            else:
                total_tokens = 0
                avg_doc_length = 0
            
            # Try to get relevant entries with error handling
            relevant_entries, timing_metrics = bm25_retriever.get_relevant_entries(query)
            
            # Filter out entries with zero scores
            relevant_entries = [entry for entry in relevant_entries if entry.get('match_score', 0) > 0]
            
            # If this is a document query, filter out the query document itself
            if is_doc_query:
                # Parse the query document's title from the metadata
                query_lines = query.split('\n')
                query_title = next((line.split(':', 1)[1].strip() 
                                  for line in query_lines 
                                  if line.startswith('Title:')), None)
                
                if query_title:
                    relevant_entries = [
                        entry for entry in relevant_entries 
                        if entry.get('Title', '').strip() != query_title.strip()
                    ]
                    logging.info(f"Filtered out query document '{query_title}' from results")
            
            # Format relevant entries to ensure all fields are properly handled
            formatted_entries = []
            for entry in relevant_entries:
                formatted_entry = {
                    'match_score': entry.get('match_score', 0),
                    'Date': str(entry.get('Date', '')),
                    'Title': str(entry.get('Title', '')),
                    'emotion': str(entry.get('emotion', '')),
                    'topic': str(entry.get('topic', '')),
                    'Tags': str(entry.get('Tags', '')),
                    'Content': str(entry.get('Content', ''))
                }
                formatted_entries.append(formatted_entry)
            
            # Add BM25-specific metrics
            if relevant_entries:
                scores = [entry.get('match_score', 0) for entry in relevant_entries]
                timing_metrics.update({
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'avg_score': sum(scores) / len(scores),
                    'score_std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
                })
            
            # Add debug information
            debug_info = {
                'query': query,
                'total_docs': len(journal_entries),
                'retrieved_docs_count': len(formatted_entries),
                'retrieved_documents': formatted_entries,
                'rag_metrics': timing_metrics,
                'retrieval_method': retrieval_method,
                'index_status': index_status,
                'indexed_docs': indexed_docs,
                'query_tokens': query_tokens_str,
                'index_size': total_tokens,
                'avg_doc_length': avg_doc_length,
                'error_message': None  # Will be updated if there's an error
            }
            
            return [create_debug_panel(debug_info)]
            
        except Exception as e:
            logging.error(f"Error in BM25 processing: {str(e)}")
            debug_info = {
                'query': query,
                'total_docs': len(journal_entries),
                'retrieved_docs_count': 0,
                'retrieved_documents': [],
                'rag_metrics': {},
                'retrieval_method': retrieval_method,
                'index_status': "Error",
                'indexed_docs': 0,
                'query_tokens': "Error processing query",
                'index_size': 0,
                'avg_doc_length': 0,
                'error_message': str(e)
            }
            return [create_debug_panel(debug_info)]
    elif retrieval_method == "exact":
        try:
            # Convert query to lowercase for case-insensitive matching
            query_lower = query.lower()
            
            # Search for substring matches in entries
            exact_match_entries = []
            matches_per_field = {field: 0 for field in rag.config['exact_match_fields']}
            
            # Search in entries
            for _, entry in journal_entries.iterrows():
                entry_dict = entry.to_dict()
                # Check configured metadata fields
                for field in rag.config['exact_match_fields']:
                    field_value = str(entry_dict.get(field, entry_dict.get(field.capitalize(), ''))).lower()
                    if query_lower in field_value:
                        matches_per_field[field] += 1
                        exact_match_entries.append(entry_dict)
                        break  # Break after finding first match in this document
            
            # If this is a document query, filter out the query document itself
            if is_doc_query:
                # Parse the query document's title from the metadata
                query_lines = query.split('\n')
                query_title = next((line.split(':', 1)[1].strip() 
                                  for line in query_lines 
                                  if line.startswith('Title:')), None)
                
                if query_title:
                    exact_match_entries = [
                        entry for entry in exact_match_entries 
                        if entry.get('Title', '').strip() != query_title.strip()
                    ]
                    logging.info(f"Filtered out query document '{query_title}' from results")
            
            # Format entries
            formatted_entries = []
            for entry in exact_match_entries:
                formatted_entry = {
                    'match_score': rag.config['exact_match_score'],
                    'Date': str(entry.get('Date', '')),
                    'Title': str(entry.get('Title', '')),
                    'emotion': str(entry.get('emotion', '')),
                    'topic': str(entry.get('topic', '')),
                    'Tags': str(entry.get('Tags', '')),
                    'Content': str(entry.get('Content', ''))
                }
                formatted_entries.append(formatted_entry)
            
            # Add timing metrics
            timing_metrics = {
                'total_time': time.time() - start_time,
                'num_docs_retrieved': len(formatted_entries),
                'num_exact_matches': len(formatted_entries)
            }
            
            debug_info = {
                'query': query,
                'total_docs': len(journal_entries),
                'retrieved_docs_count': len(formatted_entries),
                'retrieved_documents': formatted_entries,
                'rag_metrics': timing_metrics,
                'retrieval_method': retrieval_method,
                'matches_per_field': matches_per_field,
                'exact_match_fields': rag.config['exact_match_fields']
            }
            return [create_debug_panel(debug_info)]
            
        except Exception as e:
            logging.error(f"Error in exact match processing: {str(e)}")
            debug_info = {
                'query': query,
                'total_docs': len(journal_entries),
                'retrieved_docs_count': 0,
                'retrieved_documents': [],
                'rag_metrics': {},
                'retrieval_method': retrieval_method,
                'error_message': str(e)
            }
            return [create_debug_panel(debug_info)]
    else:
        # Vector search (without exact matches)
        relevant_entries, timing_metrics = rag.get_relevant_entries(query)
        
        # If this is a document query, filter out the query document itself
        if is_doc_query:
            # Parse the query document's title from the metadata
            query_lines = query.split('\n')
            query_title = next((line.split(':', 1)[1].strip() 
                              for line in query_lines 
                              if line.startswith('Title:')), None)
            
            if query_title:
                relevant_entries = [
                    entry for entry in relevant_entries 
                    if entry.get('Title', '').strip() != query_title.strip()
                ]
                logging.info(f"Filtered out query document '{query_title}' from results")
        
        # Format relevant entries to ensure all fields are properly handled
        formatted_entries = []
        for entry in relevant_entries:
            formatted_entry = {
                'match_score': entry.get('match_score', 0),
                'Date': str(entry.get('Date', '')),
                'Title': str(entry.get('Title', '')),
                'emotion': str(entry.get('emotion', '')),
                'topic': str(entry.get('topic', '')),
                'Tags': str(entry.get('Tags', '')),
                'Content': str(entry.get('Content', ''))
            }
            formatted_entries.append(formatted_entry)
            
        debug_info = {
            'query': query,
            'total_docs': len(journal_entries),
            'retrieved_docs_count': len(formatted_entries),
            'retrieved_documents': formatted_entries,
            'rag_metrics': timing_metrics,
            'retrieval_method': retrieval_method
        }
        return [create_debug_panel(debug_info)]

if __name__ == "__main__":
    app.run_server(debug=True, port=8051) 