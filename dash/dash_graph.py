#!/usr/bin/env python3
"""
Graph Visualization Dashboard

This module provides a dashboard for visualizing and exploring the knowledge graph,
including document indexing and retrieval capabilities.
"""

import sys
from pathlib import Path
# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import yaml
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from ir_utils import BM25Retriever

# Initialize configuration
def load_config():
    """Load configuration from config_rag.yaml"""
    config_path = Path(__file__).parent.parent / "config_rag.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config_rag.yaml not found")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class KnowledgeGraph:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the knowledge graph system."""
        self.config = config
        self.bm25_retriever = BM25Retriever(config)
        self.documents = []
        self.indexed = False
        
    def load_journal_entries(self) -> pd.DataFrame:
        """Load journal entries from the annotated CSV file."""
        try:
            output_dir = Path(self.config['output_dir'])
            annotated_file = output_dir / 'journal_entries_annotated.csv'
            if not annotated_file.exists():
                logging.warning(f"No annotated journal entries found in {output_dir}")
                return pd.DataFrame()
            
            logging.info(f"Loading journal entries from {annotated_file}")
            
            df = pd.read_csv(annotated_file)
            if df.empty:
                logging.warning("Loaded CSV file is empty")
                return pd.DataFrame()
                
            # Clean up and parse dates
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            
            # Clean up fields
            df['emotion'] = df['emotion'].fillna('')
            df['emotion'] = df['emotion'].apply(lambda x: '' if pd.isna(x) or str(x).strip() == '' else x)
            df['Title'] = df['Title'].apply(lambda x: str(x).strip('"*') if pd.notna(x) else x)
            
            # Create Tags column
            df['Tags'] = df.apply(lambda row: ' '.join(filter(None, [
                row['topic_visual'] if pd.notna(row['topic_visual']) else '',
                row['etc_visual'] if pd.notna(row['etc_visual']) else ''
            ])), axis=1)
            
            logging.info(f"Loaded {len(df)} journal entries")
            return df
        except Exception as e:
            logging.error(f"Error loading journal entries: {e}")
            return pd.DataFrame()
            
    def load_reading_entries(self) -> pd.DataFrame:
        """Load reading entries from the annotated CSV file."""
        try:
            output_dir = Path(self.config['output_dir'])
            reading_file = output_dir / 'reading_entries.csv'
            if not reading_file.exists():
                logging.warning(f"No reading entries found in {output_dir}")
                return pd.DataFrame()
            
            logging.info(f"Loading reading entries from {reading_file}")
            
            df = pd.read_csv(reading_file)
            if df.empty:
                logging.warning("Loaded CSV file is empty")
                return pd.DataFrame()
                
            # Clean up and parse dates
            df['Date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            
            # Map column names to match our schema
            df['Title'] = df['title'].apply(lambda x: str(x).strip('"*') if pd.notna(x) else x)
            df['Content'] = df['content'] if 'content' in df.columns else ''  # Map content field
            df['Tags'] = df['tags'] if 'tags' in df.columns else ''
            df['author'] = df['author'] if 'author' in df.columns else ''
            df['source'] = df['source'] if 'source' in df.columns else ''
            
            logging.info(f"Loaded {len(df)} reading entries")
            return df
        except Exception as e:
            logging.error(f"Error loading reading entries: {e}")
            return pd.DataFrame()
            
    def convert_to_json(self, df: pd.DataFrame, doc_type: str) -> List[Dict[str, Any]]:
        """Convert DataFrame rows to JSON blobs."""
        json_docs = []
        for _, row in df.iterrows():
            # Common fields for both types
            doc = {
                'doc_type': doc_type,
                'Date': row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else '',
                'Title': str(row['Title']) if pd.notna(row['Title']) else '',
                'Content': str(row['Content']) if pd.notna(row['Content']) else '',
                'Tags': str(row['Tags']) if pd.notna(row['Tags']) else '',
            }
            
            # Add type-specific fields
            if doc_type == 'journal':
                doc.update({
                    'emotion': str(row['emotion']) if pd.notna(row['emotion']) else '',
                    'topic': str(row['topic']) if pd.notna(row['topic']) else '',
                })
            elif doc_type == 'reading':
                doc.update({
                    'author': str(row['author']) if pd.notna(row['author']) else '',
                    'source': str(row['source']) if pd.notna(row['source']) else '',
                })
                
            json_docs.append(doc)
        return json_docs
        
    def index_documents(self):
        """Index all documents using BM25."""
        if self.indexed:
            logging.info("Documents already indexed")
            return
            
        # Load and process journal entries
        journal_df = self.load_journal_entries()
        if not journal_df.empty:
            journal_docs = self.convert_to_json(journal_df, 'journal')
            self.documents.extend(journal_docs)
            
        # Load and process reading entries
        reading_df = self.load_reading_entries()
        if not reading_df.empty:
            reading_docs = self.convert_to_json(reading_df, 'reading')
            self.documents.extend(reading_docs)
            
        # Index all documents
        if self.documents:
            logging.info(f"Indexing {len(self.documents)} documents")
            self.bm25_retriever.index_documents(self.documents)
            self.indexed = True
            logging.info("Document indexing complete")
        else:
            logging.warning("No documents to index")
            
    def search(self, query: str, top_k: int = 10, doc_type: str = 'all') -> List[Dict[str, Any]]:
        """Search for relevant documents.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            doc_type: Type of documents to search ('journal', 'reading', or 'all')
        """
        # Handle empty query
        if not query.strip():
            return []
            
        # Handle no index
        if not self.indexed:
            logging.warning("Documents not indexed. Indexing now...")
            self.index_documents()
            
        # Handle no documents
        if not self.documents:
            logging.warning("No documents to search")
            return []
            
        try:
            # Get k from config or use default
            k = self.config.get('bm25', {}).get('final_k', top_k)
            
            # If filtering by type, retrieve more documents initially
            # This ensures we have enough results after filtering
            if doc_type != 'all':
                k = k * 5  # Retrieve 5x more documents when filtering by type
            
            # Ensure k is not larger than the number of documents
            k = min(k, len(self.documents))
            
            # Get results from the main index
            results, metrics = self.bm25_retriever.get_relevant_entries(query, k=k)
            
            # Filter results by document type if specified
            if doc_type != 'all':
                results = [doc for doc in results if doc.get('doc_type') == doc_type]
                # Adjust k to match filtered results
                k = min(top_k, len(results))
                results = results[:k]
            
            # Filter out results with zero score
            results = [doc for doc in results if doc.get('match_score', 0) > 0]
            
            # Ensure doc_type is preserved in results
            for result in results:
                if 'doc_type' not in result and doc_type != 'all':
                    result['doc_type'] = doc_type
                    
            return results[:top_k]
        except Exception as e:
            logging.error(f"Error during search: {e}")
            return []
            
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID."""
        if not self.indexed:
            logging.warning("Documents not indexed")
            return None
            
        try:
            # Find document in the documents list
            for doc in self.documents:
                if doc.get('doc_id') == doc_id:
                    return doc
            return None
        except Exception as e:
            logging.error(f"Error retrieving document: {e}")
            return None

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Initialize configuration and knowledge graph
config = load_config()
kg = KnowledgeGraph(config)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Knowledge Graph Explorer", className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        # Search Panel (4 columns)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Search Interface"),
                dbc.CardBody([
                    # Document Type Filter
                    html.Div([
                        html.H5("Document Type", className="mb-3"),
                        dbc.RadioItems(
                            id="doc-type-filter",
                            options=[
                                {"label": "All Documents", "value": "all"},
                                {"label": "Journal Entries", "value": "journal"},
                                {"label": "Reading Entries", "value": "reading"}
                            ],
                            value="all",
                            inline=True,
                            className="mb-3"
                        ),
                    ]),
                    
                    # Search Input
                    dbc.InputGroup([
                        dbc.Textarea(
                            id="search-input",
                            placeholder="Enter your query...",
                            className="me-2",
                            style={"height": "100px"}
                        ),
                        dbc.Button("Search", id="search-button", color="primary")
                    ]),
                    
                    # Search Results
                    html.Div([
                        html.H5("Search Results", className="mt-4 mb-3"),
                        html.Div(id="search-results-list", className="list-group")
                    ])
                ])
            ], className="h-100")
        ], width=4),
        
        # Graph Panel (8 columns)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Knowledge Graph"),
                dbc.CardBody([
                    dcc.Graph(id="knowledge-graph")
                ])
            ], className="h-100")
        ], width=8)
    ], className="h-100")
], fluid=True, className="vh-100")

def create_search_result_item(doc: Dict[str, Any], index: int) -> html.Div:
    """Create a clickable search result item with title, content preview, and metadata."""
    # Format date
    date_str = doc.get('Date', '')
    
    # Get match score
    match_score = doc.get('match_score', 0)
    
    # Get document type
    doc_type = doc.get('doc_type', 'NO_TYPE_FOUND')
    
    return dbc.Card([
        dbc.CardBody([
            # Title (clickable)
            html.H5(
                dcc.Link(
                    doc.get('Title', 'Untitled'),
                    href="#",
                    id={'type': 'result-title', 'index': index},
                    className="text-primary"
                ),
                className="mb-1"
            ),
            
            # Content preview
            html.P(
                doc.get('Content', '')[:200] + "..." if len(doc.get('Content', '')) > 200 else doc.get('Content', ''),
                className="mb-2"
            ),
            
            # Metadata section
            html.Div([
                # Left column: Date and Type
                html.Div([
                    html.Small(f"Date: {date_str}", className="text-muted d-block"),
                    html.Small(f"Type: {doc_type}", className="text-muted d-block"),
                ], className="float-start"),
                
                # Right column: Score and Tags
                html.Div([
                    html.Small(f"Score: {match_score:.3f}", className="text-muted d-block"),
                    html.Small(f"Tags: {doc.get('Tags', '')}", className="text-muted d-block"),
                ], className="float-end"),
                
                # Clear float
                html.Div(style={'clear': 'both'})
            ], className="mt-2"),
            
            # Store the full document data in a hidden div
            html.Div(
                json.dumps(doc),
                id={'type': 'result-data', 'index': index},
                style={'display': 'none'}
            )
        ])
    ], className="mb-3")

# Callback to handle search
@callback(
    [Output("search-results-list", "children"),
     Output("knowledge-graph", "figure")],
    [Input("search-button", "n_clicks"),
     Input("search-input", "n_submit")],
    [State("search-input", "value"),
     State("doc-type-filter", "value")]
)
def handle_search(n_clicks, n_submit, query, doc_type):
    """Handle search queries and update both the results list and graph visualization."""
    if not query:
        return [], {
            'data': [],
            'layout': {
                'title': 'Knowledge Graph',
                'showlegend': True
            }
        }
        
    # Perform search with document type filter
    results = kg.search(query, doc_type=doc_type)
    
    # Create search result items for the list
    result_items = [create_search_result_item(doc, i) for i, doc in enumerate(results)]
    
    # Create graph visualization
    if results:
        # Extract keywords from query (simple space-based splitting for now)
        query_keywords = query.strip().split()
        query_node = f"Query: {' '.join(query_keywords)}"
        
        # Create nodes and edges
        nodes_x = []  # X coordinates
        nodes_y = []  # Y coordinates
        node_labels = []  # Node text labels
        node_colors = []  # Node colors
        node_shapes = []  # Node shapes
        edge_x = []  # Edge X coordinates
        edge_y = []  # Edge Y coordinates
        edge_weights = []  # Edge weights for line width
        
        # Add center (query) node
        nodes_x.append(0)
        nodes_y.append(0)
        node_labels.append(query_node)
        node_colors.append('red')  # Query node in red
        node_shapes.append('circle')  # Query node as circle
        
        # Add document nodes in a circle around the query node
        import math
        radius = 1
        for i, doc in enumerate(results):
            angle = (2 * math.pi * i) / len(results)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Add document node
            nodes_x.append(x)
            nodes_y.append(y)
            
            # Format title with line breaks
            title = doc.get('Title', 'Untitled')
            words = title.split()
            if len(words) > 4:
                # Insert line break after every 4 words
                title = '<br>'.join([' '.join(words[i:i+4]) for i in range(0, len(words), 4)])
            
            # Add date for journal entries
            doc_type = doc.get('doc_type', 'journal')
            date = doc.get('Date', '')
            if doc_type == 'journal' and date:
                node_labels.append(f"{title}<br>{date}")
            else:
                node_labels.append(title)
                
            node_colors.append('blue')  # Document nodes in blue
            node_shapes.append('square' if doc_type == 'reading' else 'circle')
            
            # Add edge between query and document
            edge_x.extend([0, x, None])  # None creates a break in the line
            edge_y.extend([0, y, None])
            edge_weights.append(doc.get('match_score', 0))
        
        # Create the graph figure
        graph_figure = {
            'data': [
                # Nodes
                {
                    'x': nodes_x,
                    'y': nodes_y,
                    'mode': 'markers+text',
                    'marker': {
                        'size': 20,
                        'color': node_colors,
                        'symbol': node_shapes
                    },
                    'text': node_labels,
                    'textposition': 'bottom center',
                    'hoverinfo': 'text',
                    'name': 'Nodes'
                },
                # Edges
                {
                    'x': edge_x,
                    'y': edge_y,
                    'mode': 'lines',
                    'line': {
                        'width': 1,
                        'color': '#888'
                    },
                    'hoverinfo': 'none',
                    'name': 'Edges'
                }
            ],
            'layout': {
                'title': 'Knowledge Graph Visualization',
                'showlegend': True,
                'hovermode': 'closest',
                'xaxis': {
                    'showgrid': False,
                    'zeroline': False,
                    'showticklabels': False,
                    'range': [-1.5, 1.5]  # Extend axis range to accommodate labels
                },
                'yaxis': {
                    'showgrid': False,
                    'zeroline': False,
                    'showticklabels': False,
                    'range': [-1.5, 1.5]  # Extend axis range to accommodate labels
                },
                'margin': {'b': 60, 't': 40, 'l': 60, 'r': 60}  # Increase margins to prevent text cutoff
            }
        }
    else:
        # Empty graph if no results
        graph_figure = {
            'data': [],
            'layout': {
                'title': 'No Results Found',
                'showlegend': True
            }
        }
    
    return result_items, graph_figure

if __name__ == "__main__":
    # Initialize the knowledge graph
    kg.index_documents()
    
    # Run the app
    app.run_server(debug=True, port=8052) 