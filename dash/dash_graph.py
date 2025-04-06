#!/usr/bin/env python3
"""
Graph Visualization Dashboard

This module provides a dashboard for visualizing and exploring the knowledge graph,
including document indexing and retrieval capabilities.
"""

import sys
from pathlib import Path
import argparse
# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import pandas as pd
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

# Import from search directory
from search.ir_utils import BM25Retriever

# Register Cytoscape component
cyto.load_extra_layouts()

# Initialize configuration
def load_config():
    """Load configuration from config_rag.yaml"""
    config_path = Path(__file__).parent.parent / "config_rag.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config_rag.yaml not found")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class DocumentManager:
    def __init__(self, config: Dict[str, Any], force_reindex: bool = False):
        """Initialize the knowledge graph system.
        
        Args:
            config: Configuration dictionary
            force_reindex: Whether to force re-indexing of documents
        """
        self.config = config
        self.bm25_retriever = BM25Retriever(config)
        self.documents = []
        self.indexed = False # Initialize indexed state
        self.doc_types = set()  # Track available document types

        if force_reindex:
            logging.info("Force reindex requested. Running index_documents.py...")
            try:
                # Run index_documents.py script
                index_script = Path(__file__).parent.parent / "index_documents.py"
                if not index_script.exists():
                    raise FileNotFoundError(f"index_documents.py not found at {index_script}")
                
                import subprocess
                result = subprocess.run([sys.executable, str(index_script), '--force', '--build-bm25', '--debug'], 
                                     capture_output=True, text=True)
                
                if result.returncode != 0:
                    logging.error(f"Error running index_documents.py: {result.stderr}")
                    raise RuntimeError("Failed to run index_documents.py")
                    
                # Log timing information from the output
                if result.stdout:
                    logging.info(result.stdout)
                
                logging.info("index_documents.py completed successfully")
                
                # After reindexing, load the new index
                self.load_index()
                logging.info("Re-indexing complete. Continuing with dashboard.")
            except Exception as e:
                logging.error(f"Error during re-indexing: {e}")
                raise

        # If not forcing reindex, check if index exists and load as needed
        elif not self.bm25_retriever.documents:
            logging.info("No existing index found. Please run index_documents.py first.")
            self.load_index() # Try to load index from disk
        else:
            self.documents = self.bm25_retriever.documents
            self.indexed = True
            logging.info("Using existing document index")
        
    def has_index(self) -> bool:
        """Check if the system has an index."""
        return bool(self.bm25_retriever.documents)
        
    def load_indexed_documents(self) -> pd.DataFrame:
        """Load documents from the indexed CSV file."""
        try:
            # Get path relative to dash directory
            output_dir = Path(__file__).parent.parent / 'output'
            indexed_file = output_dir / 'repo_index.csv'
            if not indexed_file.exists():
                logging.warning(f"No indexed documents found in {output_dir}")
                return pd.DataFrame()
            
            logging.info(f"Loading indexed documents from {indexed_file}")
            
            df = pd.read_csv(indexed_file)
            if df.empty:
                logging.warning("Loaded CSV file is empty")
                return pd.DataFrame()
                
            # Clean up and parse dates
            df['Date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            
            # Extract document type from the top folder of the path
            df['doc_type'] = df['path'].apply(lambda x: x.split('/')[0] if '/' in x else 'other')
            
            # Add to doc_types set
            self.doc_types.update(df['doc_type'].unique())
            
            # Map column names to match our schema
            df['Title'] = df['title'].apply(lambda x: str(x).strip('"*') if pd.notna(x) else x)
            df['Content'] = df['content'] if 'content' in df.columns else ''
            df['Tags'] = df['tags'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
            df['Tags'] = df['Tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
            
            logging.info(f"Loaded {len(df)} indexed documents")
            return df
        except Exception as e:
            logging.error(f"Error loading indexed documents: {e}")
            return pd.DataFrame()
            
    def convert_to_json(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame rows to JSON blobs."""
        json_docs = []
        for _, row in df.iterrows():
            # Common fields for all types
            doc = {
                'doc_type': row['doc_type'],
                'Date': row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else '',
                'Title': str(row['Title']) if pd.notna(row['Title']) else '',
                'Content': str(row['Content']) if pd.notna(row['Content']) else '',
                'Tags': str(row['Tags']) if pd.notna(row['Tags']) else '',
                'path': str(row['path']) if pd.notna(row['path']) else '',
                'size': str(row['size']) if pd.notna(row['size']) else '',
            }
            
            # Add properties if available
            if 'properties' in row and pd.notna(row['properties']):
                try:
                    properties = json.loads(row['properties'])
                    doc.update(properties)
                except:
                    pass
                
            json_docs.append(doc)
        return json_docs
        
    def load_index(self):
        """Load the BM25 index from disk."""
        try:
            # Get path relative to dash directory
            output_dir = Path(__file__).parent.parent / 'output'
            bm25_index_dir = output_dir / 'bm25_index'
            
            if not bm25_index_dir.exists():
                logging.warning(f"No BM25 index found at {bm25_index_dir}")
                return
                
            # Load the index
            self.bm25_retriever._load_index(str(bm25_index_dir))
            self.documents = self.bm25_retriever.metadata
            self.indexed = True
            logging.info(f"Loaded BM25 index from {bm25_index_dir}")
            
            # Update doc_types
            for doc in self.documents:
                if 'doc_type' in doc:
                    self.doc_types.add(doc['doc_type'])
                    
        except Exception as e:
            logging.error(f"Error loading BM25 index: {e}")
            
    def search(self, query: str, top_k: int = 10, doc_type: str = 'all', exclude_doc_id: str = None) -> List[Dict[str, Any]]:
        """Search for relevant documents.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            doc_type: Type of documents to search ('journal', 'reading', or 'all')
            exclude_doc_id: Optional document ID to exclude from results
        """
        # Handle empty query
        if not query.strip():
            return []
            
        # Handle no index
        if not self.has_index():
            logging.warning("No index found. Please run index_documents.py first.")
            return []
            
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
            results, metrics = self.bm25_retriever.get_relevant_entries(query, k=k, exclude_doc_id=exclude_doc_id)
            
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

    def get_recent_entries(self, days: int = 7, doc_type: str = 'all', page: int = 0, entries_per_page: int = 10) -> Tuple[List[Dict[str, Any]], bool]:
        """Get recent entries from both journal and reading entries.
        
        Args:
            days: Number of days to look back
            doc_type: Type of documents to return ('journal', 'reading', or 'all')
            page: Page number (0-based)
            entries_per_page: Number of entries per page
            
        Returns:
            Tuple of (entries for current page, has_next_page)
        """
        # Load indexed documents
        indexed_df = self.load_indexed_documents()
        
        # Convert DataFrame to JSON format
        all_docs = self.convert_to_json(indexed_df)
        
        # Filter by document type if specified
        if doc_type != 'all':
            all_docs = [doc for doc in all_docs if doc.get('doc_type') == doc_type]
        
        # Sort by date
        all_docs.sort(key=lambda x: x.get('Date', ''), reverse=True)
        
        # Get entries from the last 7 days
        recent_docs = []
        today = datetime.now().date()
        
        for doc in all_docs:
            try:
                doc_date = datetime.strptime(doc.get('Date', ''), '%Y-%m-%d').date()
                if (today - doc_date).days <= days:
                    recent_docs.append(doc)
                else:
                    break
            except (ValueError, TypeError):
                # Skip documents with invalid dates
                continue
            
        # Calculate pagination
        start_idx = page * entries_per_page
        end_idx = start_idx + entries_per_page
        has_next_page = end_idx < len(recent_docs)
        
        return recent_docs[start_idx:end_idx], has_next_page

class GraphManager:
    def __init__(self):
        self.current_graph = []
        
    def _create_document_node(self, doc: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
        """Create a document node with consistent formatting.
        
        Args:
            doc: Document dictionary
            doc_id: ID for the document node
            
        Returns:
            Dictionary containing document node data
        """
        # Get title and format label
        title = doc.get('Title', 'Untitled')
        doc_type = doc.get('doc_type', 'journal')
        date = doc.get('Date', '')
        
        # Create label with title and date for journal entries
        label = f"{title} ({date})" if doc_type == 'journal' and date else title
        
        return {
            'data': {
                'id': doc_id,
                'label': label,
                'type': 'document',
                'content': doc.get('Content', ''),
                'doc_type': doc_type,
                'Date': date,
                'Title': title,
                'emotion': doc.get('emotion', ''),
                'topic': doc.get('topic', ''),
                'Tags': doc.get('Tags', '')
            }
        }
        
    def _create_query_node(self, title: str, content: str, node_id: str, 
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a query node with consistent formatting.
        
        Args:
            title: Node title/label
            content: Node content
            node_id: ID for the query node
            metadata: Additional metadata to include
            
        Returns:
            Dictionary containing query node data
        """
        node_data = {
            'id': node_id,
            'label': title,
            'type': 'query',
            'content': content,
            'doc_type': 'query'
        }
        
        if metadata:
            node_data.update(metadata)
            
        return {'data': node_data}
        
    def _create_edge(self, source: str, target: str, weight: float) -> Dict[str, Any]:
        """Create an edge with consistent formatting.
        
        Args:
            source: Source node ID
            target: Target node ID
            weight: Edge weight
            
        Returns:
            Dictionary containing edge data
        """
        return {
            'data': {
                'source': source,
                'target': target,
                'weight': f"{weight:.3f}"
            }
        }
        
    def create_graph_elements(self, query: str, results: List[Dict[str, Any]], 
                            query_node_id: str = 'query') -> List[Dict[str, Any]]:
        """Create graph elements from query and results.
        
        Args:
            query: Search query string
            results: List of document dictionaries
            query_node_id: ID for the query node (default: 'query')
            
        Returns:
            List of graph elements (nodes and edges)
        """
        elements = []
        
        # Add query node
        elements.append(self._create_query_node(query, query, query_node_id))
        
        # Add document nodes and edges
        for i, doc in enumerate(results):
            doc_id = f'doc_{i}'
            
            # Add document node
            elements.append(self._create_document_node(doc, doc_id))
            
            # Add edge from query to document
            elements.append(self._create_edge(
                query_node_id, 
                doc_id, 
                doc.get('match_score', 0)
            ))
        
        self.current_graph = elements
        return elements
        
    def expand_graph(self, clicked_node: Dict[str, Any], related_docs: List[Dict[str, Any]], 
                    current_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand existing graph with new related documents.
        
        Args:
            clicked_node: Dictionary containing clicked node data (with 'data' wrapper)
            related_docs: List of related document dictionaries
            current_elements: Current graph elements
            
        Returns:
            List of graph elements (nodes and edges)
        """
        # Get existing nodes and edges
        new_elements = []
        
        # Keep existing nodes and edges except clicked node
        clicked_node_id = clicked_node['data']['id']
        for elem in current_elements:
            if elem['data']['id'] != clicked_node_id:
                new_elements.append(elem)
        
        # Convert clicked node to query node
        new_elements.append(self._create_query_node(
            clicked_node['data'].get('Title', ''),
            clicked_node['data'].get('content', ''),
            clicked_node_id,
            {
                'doc_type': clicked_node['data'].get('doc_type', ''),
                'Date': clicked_node['data'].get('Date', ''),
                'Title': clicked_node['data'].get('Title', ''),
                'emotion': clicked_node['data'].get('emotion', ''),
                'topic': clicked_node['data'].get('topic', ''),
                'Tags': clicked_node['data'].get('Tags', '')
            }
        ))
        
        # Get existing document titles to avoid duplicates
        existing_titles = {
            elem['data'].get('Title', '').strip()
            for elem in new_elements
            if elem['data'].get('type') == 'document'
        }
        
        # Add new document nodes and edges
        for i, doc in enumerate(related_docs):
            # Skip if document already exists
            if doc.get('Title', '').strip() in existing_titles:
                continue
                
            doc_id = f'doc_{len(new_elements)}'
            
            # Add document node
            new_elements.append(self._create_document_node(doc, doc_id))
            
            # Add edge from clicked node to document
            new_elements.append(self._create_edge(
                clicked_node_id,
                doc_id,
                doc.get('match_score', 0)
            ))
        
        self.current_graph = new_elements
        return new_elements

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the knowledge graph visualization dashboard')
parser.add_argument('--reindex', action='store_true', help='Force re-indexing of documents at startup')
args = parser.parse_args()

# Initialize configuration and knowledge graph
config = load_config()
kg = DocumentManager(config, force_reindex=args.reindex)
graph_manager = GraphManager()

# Global state for pagination
current_page = 0
entries_per_page = 10

# App layout
app.layout = dbc.Container([
    dbc.Row([
        # Search Panel (4 columns)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Search Interface"),
                dbc.CardBody([
                    # Document Type Filter
                    html.Div([
                        html.H5("Document Type", className="mb-3"),
                        dcc.Dropdown(
                            id="doc-type-filter",
                            options=[
                                {"label": "All Documents", "value": "all"}
                            ],
                            value="all",
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
                    
                    # Recent Entries Placeholder
                    html.Div([
                        html.H5("Recent Entries", className="mt-4 mb-3"),
                        html.Div(id="recent-entries-list", className="list-group"),
                        html.Div([
                            dbc.Button("Next Page", id="next-page-button", color="primary", className="mt-3", disabled=True),
                        ], className="d-flex justify-content-center")
                    ])
                ])
            ], className="h-100")
        ], width=4),
        
        # Graph and Details Panel (8 columns)
        dbc.Col([
            # Graph Panel
            dbc.Card([
                dbc.CardHeader("Knowledge Graph"),
                dbc.CardBody([
                    cyto.Cytoscape(
                        id='knowledge-graph',
                        layout={
                            'name': 'cose',
                            'idealEdgeLength': 150,
                            'refresh': 20,
                            'fit': True,
                            'padding': 30,
                            'zoom': 1.5  # Increase default zoom level
                        },
                        style={'width': '100%', 'height': '100%'},
                        elements=[],
                        stylesheet=[
                            # Query node style
                            {
                                'selector': 'node[type="query"]',
                                'style': {
                                    'background-color': '#ff0000',
                                    'label': 'data(label)',
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'width': 120,
                                    'height': 60,
                                    'font-size': 12,
                                    'font-weight': 'bold',
                                    'text-wrap': 'wrap',
                                    'text-max-width': 110,
                                    'color': '#ffffff',
                                    'text-outline-color': '#000000',
                                    'text-outline-width': 2
                                }
                            },
                            # Document node style
                            {
                                'selector': 'node[type="document"]',
                                'style': {
                                    'background-color': '#1f77b4',
                                    'label': 'data(label)',
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'width': 120,
                                    'height': 60,
                                    'font-size': 12,
                                    'text-wrap': 'wrap',
                                    'text-max-width': 110,
                                    'color': '#ffffff',
                                    'text-outline-color': '#000000',
                                    'text-outline-width': 2
                                }
                            },
                            # Edge style
                            {
                                'selector': 'edge',
                                'style': {
                                    'width': 1,
                                    'line-color': '#666',
                                    'target-arrow-color': '#666',
                                    'target-arrow-shape': 'triangle',
                                    'curve-style': 'bezier',
                                    'label': 'data(weight)',
                                    'font-size': 10,
                                    'text-outline-color': '#ffffff',
                                    'text-outline-width': 1
                                }
                            },
                            # Hover styles
                            {
                                'selector': 'node:selected',
                                'style': {
                                    'background-color': '#ffd700',
                                    'width': 130,
                                    'height': 70,
                                    'color': '#000000',
                                    'text-outline-color': '#ffffff',
                                    'text-outline-width': 2
                                }
                            },
                            {
                                'selector': 'edge:selected',
                                'style': {
                                    'line-color': '#ffd700',
                                    'target-arrow-color': '#ffd700',
                                    'width': 3
                                }
                            }
                        ],
                        userZoomingEnabled=True,
                        userPanningEnabled=True,
                        boxSelectionEnabled=True
                    )
                ])
            ], className="h-70 mb-3", style={'height': '70%'}),
            
            # Details Panel
            dbc.Card([
                dbc.CardHeader("View Details"),
                dbc.CardBody([
                    html.Div(id="details-content", className="p-3")
                ])
            ], className="h-30", style={'height': '30%'})
        ], width=8, style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'})
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
    # Format document type for display
    doc_type_display = doc_type.replace('_', ' ').capitalize()
    
    return dbc.Card([
        dbc.CardBody([
            # Title (clickable)
            html.Button(
                doc.get('Title', 'Untitled'),
                id={'type': 'result-title', 'index': index},
                className="btn btn-link text-primary p-0 text-start",
                style={'text-decoration': 'none', 'font-size': '1.25rem', 'font-weight': '500'},
                n_clicks=0
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
                    html.Small(f"Type: {doc_type_display}", className="text-muted d-block"),
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

def create_details_content(doc: Dict[str, Any]) -> html.Div:
    """Create the details content for a selected document."""
    # Get document type
    doc_type = doc.get('doc_type', '')
    # Format document type for display
    doc_type_display = doc_type.replace('_', ' ').capitalize()
    
    # For reading documents, only show the content since it already includes title and metadata
    if doc_type == 'reading':
        return html.Div([
            dcc.Markdown(
                doc.get('Content', ''),
                className="mt-3 mb-0",
                style={
                    'fontFamily': 'inherit',
                    'fontSize': '1rem',
                    'lineHeight': '1.5',
                    'margin': '0'
                }
            )
        ])
    
    # For other document types, show title and metadata
    metadata = []
    if doc.get('Date'):
        metadata.append(f"Date: {doc.get('Date')}")
    if doc.get('doc_type'):
        metadata.append(f"Type: {doc_type_display}")
    if doc.get('emotion'):
        metadata.append(f"Emotion: {doc.get('emotion')}")
    if doc.get('topic'):
        metadata.append(f"Topic: {doc.get('topic')}")
    if doc.get('Tags'):
        metadata.append(f"Tags: {doc.get('Tags')}")
    if doc.get('author'):
        metadata.append(f"Author: {doc.get('author')}")
    if doc.get('source'):
        metadata.append(f"Source: {doc.get('source')}")
    if doc.get('path'):
        metadata.append(f"Path: {doc.get('path')}")
    if doc.get('size'):
        metadata.append(f"Size: {doc.get('size')} KB")
    if doc.get('match_score'):
        metadata.append(f"Match Score: {doc.get('match_score'):.3f}")
    
    return html.Div([
        # Title
        html.H2(doc.get('Title', 'Untitled'), className="mb-3"),
        
        # Metadata
        html.Div([
            html.Small(meta, className="text-muted me-3") for meta in metadata
        ], className="mb-4"),
        
        # Content with markdown formatting
        dcc.Markdown(
            doc.get('Content', ''),
            className="mt-3 mb-0",
            style={
                'fontFamily': 'inherit',
                'fontSize': '1rem',
                'lineHeight': '1.5',
                'margin': '0'
            }
        )
    ])

# Callback to update document type dropdown options
@callback(
    Output("doc-type-filter", "options"),
    [Input("search-button", "n_clicks"),
     Input("search-input", "n_submit")],
    prevent_initial_call=False
)
def update_doc_type_options(n_clicks, n_submit):
    """Update document type dropdown options based on available document types."""
    # Get available document types
    doc_types = sorted(list(kg.doc_types))
    
    # Create options list
    options = [{"label": "All Documents", "value": "all"}]
    
    # Add document type options
    for doc_type in doc_types:
        # Capitalize first letter and replace underscores with spaces
        label = doc_type.replace('_', ' ').capitalize()
        options.append({"label": label, "value": doc_type})
    
    return options

# Callback to load recent entries on startup and when doc type changes
@callback(
    [Output("recent-entries-list", "children"),
     Output("next-page-button", "disabled")],
    [Input("search-button", "n_clicks"),
     Input("search-input", "n_submit"),
     Input("doc-type-filter", "value"),
     Input("next-page-button", "n_clicks")],
    prevent_initial_call=False  # Allow initial call to load entries on startup
)
def load_recent_entries(n_clicks, n_submit, doc_type, next_page_clicks):
    """Load recent entries when the app starts or when document type changes."""
    global current_page
    
    # Reset page when document type changes
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'doc-type-filter.value':
        current_page = 0
    
    # Get recent entries with document type filter
    recent_docs, has_next_page = kg.get_recent_entries(doc_type=doc_type, page=current_page)
    
    # Create list of entry components
    entries = []
    for i, doc in enumerate(recent_docs):
        entries.append(create_search_result_item(doc, i))
        
    return entries, not has_next_page

# Callback to handle next page button clicks
@callback(
    Output("recent-entries-list", "children", allow_duplicate=True),
    [Input("next-page-button", "n_clicks")],
    [State("doc-type-filter", "value")],
    prevent_initial_call=True
)
def handle_next_page(n_clicks, doc_type):
    """Handle next page button clicks."""
    global current_page
    
    if n_clicks is None:
        return dash.no_update
        
    current_page += 1
    recent_docs, has_next_page = kg.get_recent_entries(doc_type=doc_type, page=current_page)
    
    # Create list of entry components
    entries = []
    for i, doc in enumerate(recent_docs):
        entries.append(create_search_result_item(doc, i))
        
    return entries

# Callback to handle search
@callback(
    Output("knowledge-graph", "elements"),
    [Input("search-button", "n_clicks"),
     Input("search-input", "n_submit")],
    [State("search-input", "value"),
     State("doc-type-filter", "value")]
)
def handle_search(n_clicks, n_submit, query, doc_type):
    """Handle search queries and update the graph visualization."""
    if not query:
        return []
        
    # Perform search with document type filter
    results = kg.search(query, doc_type=doc_type)
    
    # Create graph elements using GraphManager
    return graph_manager.create_graph_elements(query, results)

# Callback to handle node clicks
@callback(
    [Output("knowledge-graph", "elements", allow_duplicate=True),
     Output("details-content", "children", allow_duplicate=True)],
    [Input("knowledge-graph", "tapNodeData")],
    [State("knowledge-graph", "elements")],
    prevent_initial_call=True
)
def handle_node_click(node_data, current_elements):
    """Handle clicks on graph nodes to expand the graph with related documents."""
    if not node_data:
        return current_elements, dash.no_update
        
    # Get the clicked node's content
    node_id = node_data.get('id')
    if not node_id or node_id == 'query':
        return current_elements, dash.no_update
        
    # Find the node in elements
    node = next((elem for elem in current_elements if elem['data']['id'] == node_id), None)
    if not node:
        return current_elements, dash.no_update
        
    # Get the node's content and metadata
    title = node['data'].get('Title', '')
    content = node['data'].get('content', '')
    date = node['data'].get('Date', '')
    doc_id = f"{date}_{title}"  # Create document ID
    query = f"{title} {content}"
    
    if not query.strip():
        return current_elements, dash.no_update
        
    # Perform search using combined query, excluding the clicked document
    results = kg.search(query, top_k=10, exclude_doc_id=doc_id)  # Pass doc_id to exclude
    
    # Expand graph using GraphManager
    new_elements = graph_manager.expand_graph(node, results, current_elements)
    
    # Create details content for the clicked node
    details_content = create_details_content({
        'Title': node['data'].get('Title', ''),
        'Content': node['data'].get('content', ''),
        'Date': node['data'].get('Date', ''),
        'doc_type': node['data'].get('doc_type', ''),
        'emotion': node['data'].get('emotion', ''),
        'topic': node['data'].get('topic', ''),
        'Tags': node['data'].get('Tags', '')
    })
    
    return new_elements, details_content

# Callback to handle hover events
@callback(
    Output("details-content", "children", allow_duplicate=True),
    [Input("knowledge-graph", "mouseoverNodeData")],
    [State("knowledge-graph", "elements")],
    prevent_initial_call=True
)
def handle_graph_hover(node_data, current_elements):
    """Handle hover events on graph nodes to show document details."""
    if not node_data:
        return html.Div("Hover over a node to view details")
        
    # Find the node in elements
    node_id = node_data.get('id')
    if not node_id:
        return html.Div("Hover over a node to view details")
        
    node = next((elem for elem in current_elements if elem['data']['id'] == node_id), None)
    if not node:
        return html.Div("Hover over a node to view details")
        
    # Create and return the details content
    return create_details_content({
        'Title': node['data'].get('Title', ''),
        'Content': node['data'].get('content', ''),
        'Date': node['data'].get('Date', ''),
        'doc_type': node['data'].get('doc_type', ''),
        'emotion': node['data'].get('emotion', ''),
        'topic': node['data'].get('topic', ''),
        'Tags': node['data'].get('Tags', '')
    })

# Callback to handle recent entry clicks
@callback(
    [Output("knowledge-graph", "elements", allow_duplicate=True),
     Output("details-content", "children", allow_duplicate=True)],
    [Input({'type': 'result-title', 'index': dash.ALL}, 'n_clicks')],
    [State({'type': 'result-data', 'index': dash.ALL}, 'children')],
    prevent_initial_call=True
)
def handle_recent_entry_click(n_clicks_list, result_data_list):
    """Handle clicks on recent entries to update the graph and details."""
    # Check if this is a real click (not initial load)
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
        
    # Get the trigger info
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    trigger_value = ctx.triggered[0]['value']
    
    # If this is the initial load or no clicks, return no update
    if trigger_id == 'dash.no_update' or not trigger_value:
        return dash.no_update, dash.no_update
        
    # Find which button was clicked
    button_id = json.loads(trigger_id)
    index = button_id['index']
    
    # Get the document data for the clicked result
    doc_data = json.loads(result_data_list[index])
    
    # Create query from document content and get document ID
    title = doc_data.get('Title', '')
    content = doc_data.get('Content', '')
    date = doc_data.get('Date', '')
    doc_id = f"{date}_{title}"  # Create document ID
    query = f"{title} {content}"
    
    if not query.strip():
        return dash.no_update, dash.no_update
        
    # Perform search using document content, excluding the clicked document
    results = kg.search(query, top_k=5, exclude_doc_id=doc_id)  # Pass doc_id to exclude
    
    # Create graph elements with the clicked document as query node
    elements = []
    
    # Add the clicked document as query node
    elements.append({
        'data': {
            'id': 'query',
            'label': title,
            'type': 'query',
            'content': content,
            'doc_type': doc_data.get('doc_type', ''),
            'Date': doc_data.get('Date', ''),
            'Title': title,
            'emotion': doc_data.get('emotion', ''),
            'topic': doc_data.get('topic', ''),
            'Tags': doc_data.get('Tags', '')
        }
    })
    
    # Add related document nodes and edges
    for i, doc in enumerate(results):
        doc_id = f'doc_{i}'
        
        # Get title and format label
        doc_title = doc.get('Title', 'Untitled')
        doc_type = doc.get('doc_type', 'journal')
        date = doc.get('Date', '')
        
        # Create label with title and date for journal entries
        label = f"{doc_title} ({date})" if doc_type == 'journal' and date else doc_title
        
        # Add document node
        elements.append({
            'data': {
                'id': doc_id,
                'label': label,
                'type': 'document',
                'content': doc.get('Content', ''),
                'doc_type': doc_type,
                'Date': date,
                'Title': doc_title,
                'emotion': doc.get('emotion', ''),
                'topic': doc.get('topic', ''),
                'Tags': doc.get('Tags', '')
            }
        })
        
        # Add edge from query to document
        elements.append({
            'data': {
                'source': 'query',
                'target': doc_id,
                'weight': f"{doc.get('match_score', 0):.3f}"
            }
        })
    
    # Create details content for the clicked document
    details_content = create_details_content(doc_data)
    
    return elements, details_content

if __name__ == "__main__":
    # Run the app
    app.run_server(debug=True, port=8052) 