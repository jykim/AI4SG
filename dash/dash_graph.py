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
import dash_cytoscape as cyto
import pandas as pd
import yaml
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from ir_utils import BM25Retriever

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
    def __init__(self, config: Dict[str, Any]):
        """Initialize the knowledge graph system."""
        self.config = config
        self.bm25_retriever = BM25Retriever(config)
        self.documents = []
        
        # If BM25Retriever has documents, sync them to our documents list
        if self.bm25_retriever.documents:
            self.documents = self.bm25_retriever.documents
            self.indexed = True
        else:
            self.indexed = False
        
    def has_index(self) -> bool:
        """Check if the system has an index."""
        return bool(self.bm25_retriever.documents)
        
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
            
            # logging.info(f"Loaded {len(df)} journal entries")
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
            
            # logging.info(f"Loaded {len(df)} reading entries")
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
            # logging.info("Documents already indexed")
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
            # logging.info(f"Indexing {len(self.documents)} documents")
            
            # Create document texts that include both content and metadata
            doc_texts = []
            for doc in self.documents:
                text = f"{doc['Title']} {doc['Content']} {doc['Tags']}"
                if 'emotion' in doc:
                    text += f" {doc['emotion']}"
                if 'topic' in doc:
                    text += f" {doc['topic']}"
                doc_texts.append(text)
            
            # Index documents using BM25Retriever
            self.bm25_retriever.index_documents(self.documents)
            self.indexed = True
            # logging.info("Document indexing complete")
            
            # Log some sample tokens for debugging
            tokenizer = self.bm25_retriever.tokenizer
            for i, text in enumerate(doc_texts[:2]):  # Show first 2 docs
                tokens = tokenizer(text)
                # logging.info(f"Doc {i} tokens: {tokens}")
            
            # Remove incorrect debug logging statements
            # logging.info(f"Query: {query}")
            # logging.info(f"Found {len(results)} results")
            # for result in results:
            #     logging.info(f"Score: {result['match_score']:.3f}, Title: {result['Title']}")
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
        if not self.has_index():
            logging.warning("No index found. Indexing now...")
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
                    
            # Log search results for debugging
            # logging.info(f"Query: {query}")
            # logging.info(f"Found {len(results)} results")
            for result in results:
                # logging.info(f"Score: {result['match_score']:.3f}, Title: {result['Title']}")
                pass
                    
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

    def get_recent_entries(self, days: int = 7, doc_type: str = 'all') -> List[Dict[str, Any]]:
        """Get recent entries from both journal and reading entries.
        
        Args:
            days: Number of days to look back
            doc_type: Type of documents to return ('journal', 'reading', or 'all')
        """
        # Load both types of entries
        journal_df = self.load_journal_entries()
        reading_df = self.load_reading_entries()
        
        # Convert both DataFrames to JSON format
        journal_docs = self.convert_to_json(journal_df, 'journal')
        reading_docs = self.convert_to_json(reading_df, 'reading')
        
        # Combine all documents based on type filter
        if doc_type == 'all':
            all_docs = journal_docs + reading_docs
        elif doc_type == 'journal':
            all_docs = journal_docs
        elif doc_type == 'reading':
            all_docs = reading_docs
        else:
            all_docs = []
        
        # Sort by date
        all_docs.sort(key=lambda x: x.get('Date', ''), reverse=True)
        
        # Get entries from the last 7 days
        recent_docs = []
        today = datetime.now().date()
        
        for doc in all_docs:
            doc_date = datetime.strptime(doc.get('Date', ''), '%Y-%m-%d').date()
            if (today - doc_date).days <= days:
                recent_docs.append(doc)
            else:
                break
            
        return recent_docs[:10]  # Limit to 10 most recent entries

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

# Initialize configuration and knowledge graph
config = load_config()
kg = DocumentManager(config)
graph_manager = GraphManager()

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
                    
                    # Recent Entries Placeholder
                    html.Div([
                        html.H5("Recent Entries", className="mt-4 mb-3"),
                        html.Div(id="recent-entries-list", className="list-group")
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

def create_details_content(doc: Dict[str, Any]) -> html.Div:
    """Create the details content for a selected document."""
    # Get document type
    doc_type = doc.get('doc_type', '')
    
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
        metadata.append(f"Type: {doc.get('doc_type')}")
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

# Callback to load recent entries on startup and when doc type changes
@callback(
    Output("recent-entries-list", "children"),
    [Input("search-button", "n_clicks"),
     Input("search-input", "n_submit"),
     Input("doc-type-filter", "value")],
    prevent_initial_call=False  # Allow initial call to load entries on startup
)
def load_recent_entries(n_clicks, n_submit, doc_type):
    """Load recent entries when the app starts or when document type changes."""
    # Get recent entries with document type filter
    recent_docs = kg.get_recent_entries(doc_type=doc_type)
    
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
        
    # Get the node's content
    title = node['data'].get('Title', '')
    content = node['data'].get('content', '')
    query = f"{title} {content}"
    
    if not query.strip():
        return current_elements, dash.no_update
        
    # Perform search using combined query
    results = kg.search(query, top_k=10)  # Increased from 5 to 10 to account for deduplication
    
    # Filter out the clicked document from results
    results = [doc for doc in results if doc.get('Title', '').strip() != title.strip()]
    
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
    Output("knowledge-graph", "elements", allow_duplicate=True),
    [Input({'type': 'result-title', 'index': dash.ALL}, 'n_clicks')],
    [State({'type': 'result-data', 'index': dash.ALL}, 'children')],
    prevent_initial_call=True
)
def handle_recent_entry_click(n_clicks_list, result_data_list):
    """Handle clicks on recent entries to update the graph."""
    # Check if this is a real click (not initial load)
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
        
    # Get the trigger info
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    trigger_value = ctx.triggered[0]['value']
    
    # If this is the initial load or no clicks, return no update
    if trigger_id == 'dash.no_update' or not trigger_value:
        return dash.no_update
        
    # Find which button was clicked
    button_id = json.loads(trigger_id)
    index = button_id['index']
    
    # Get the document data for the clicked result
    doc_data = json.loads(result_data_list[index])
    
    # Create query from document content
    title = doc_data.get('Title', '')
    content = doc_data.get('Content', '')
    query = f"{title} {content}"
    
    if not query.strip():
        return dash.no_update
        
    # Perform search using document content
    results = kg.search(query, top_k=5)  # Limit to 5 related documents
    
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
    
    return elements

if __name__ == "__main__":
    # Run the app
    app.run_server(debug=True, port=8052) 
    app.run_server(debug=True, port=8052) 