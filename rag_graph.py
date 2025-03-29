#!/usr/bin/env python3
"""
RAG Graph Visualization

This module provides a graph visualization component for RAG retrieval results,
showing the query as a central node and retrieved documents as surrounding nodes.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
from typing import Dict, List, Any
import logging
import pandas as pd
from pathlib import Path

# Register Cytoscape component
cyto.load_extra_layouts()

def load_emotion_colors(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Load emotion colors from the annotated CSV file.
    
    Args:
        config: Configuration dictionary containing output_dir
        
    Returns:
        Dictionary mapping emotions to their colors
    """
    try:
        # Look for the annotated CSV file in the output directory
        output_dir = Path(config['output_dir'])
        annotated_file = output_dir / 'journal_entries_annotated.csv'
        if not annotated_file.exists():
            logging.warning(f"No annotated journal entries found in {output_dir}")
            return {}
        
        logging.info(f"Loading emotion colors from {annotated_file}")
        
        # Read the CSV file
        df = pd.read_csv(annotated_file)
        
        # Create emotion to color mapping
        emotion_colors = {}
        for _, row in df.iterrows():
            emotion = row.get('emotion', '')
            color = row.get('emotion_visual', '')
            if emotion and color and pd.notna(emotion) and pd.notna(color):
                emotion_colors[emotion] = color
        
        logging.info(f"Loaded {len(emotion_colors)} emotion colors")
        return emotion_colors
    except Exception as e:
        logging.error(f"Error loading emotion colors: {e}")
        return {}

def create_rag_graph(debug_info: Dict[str, Any], config: Dict[str, Any]) -> cyto.Cytoscape:
    """
    Create a Cytoscape graph visualization of RAG retrieval results.
    
    Args:
        debug_info: Dictionary containing RAG retrieval information
        config: Configuration dictionary containing output_dir
        
    Returns:
        Cytoscape component with the graph visualization
    """
    if not debug_info or 'retrieved_documents' not in debug_info:
        return cyto.Cytoscape(
            id='rag-graph',
            layout={'name': 'grid'},
            style={'width': '100%', 'height': '400px'},
            elements=[],
            stylesheet=[],
            userZoomingEnabled=True,
            userPanningEnabled=True,
            boxSelectionEnabled=True
        )
    
    # Load emotion colors
    emotion_colors = load_emotion_colors(config)
    
    # Create nodes and edges
    nodes = []
    edges = []
    
    # Add query node
    query = debug_info.get('query', 'Query')
    
    # Check if query is in document format (contains metadata)
    if isinstance(query, str) and '\n' in query:
        # Parse document-style query
        query_lines = query.split('\n')
        query_metadata = {}
        query_content = []
        
        for line in query_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                query_metadata[key.strip()] = value.strip()
            else:
                query_content.append(line)
        
        # Get title and tags from metadata
        title = query_metadata.get('Title', 'Query')
        tags = query_metadata.get('Tags', '')
        
        # Create node label with title and tags
        node_label = f"{title[:20]}{'...' if len(title) > 20 else ''}\n{tags}"
        
        # Create query node with document-style formatting
        nodes.append({
            'data': {
                'id': 'query',
                'label': node_label,
                'content': '\n'.join(query_content),
                'type': 'query',
                'emotion_color': '#808080'  # Neutral gray for query node
            }
        })
    else:
        # Regular query text
        nodes.append({
            'data': {
                'id': 'query',
                'label': f"Query\n{query[:30]}{'...' if len(query) > 30 else ''}",
                'content': query,
                'type': 'query',
                'emotion_color': '#808080'  # Neutral gray for query node
            }
        })
    
    # Add document nodes and edges
    for i, doc in enumerate(debug_info['retrieved_documents']):
        doc_id = f'doc_{i}'
        score = doc.get('match_score', 0)
        
        # Get title and tags
        title = doc.get('Title', '')
        tags = doc.get('Tags', '')
        
        # Get emotion and its color
        emotion = doc.get('emotion', '')
        emotion_color = emotion_colors.get(emotion, '#1f77b4')  # Default blue if no emotion color
        
        # Create node label with title and tags
        node_label = f"{title[:20]}{'...' if len(title) > 20 else ''}\n{tags}"
        
        # Create node
        nodes.append({
            'data': {
                'id': doc_id,
                'label': node_label,
                'content': doc.get('Content', ''),
                'score': score,
                'type': 'document',
                'emotion_color': emotion_color,
                'emotion': emotion  # Store emotion for tooltip
            }
        })
        
        # Create edge from query to document
        edges.append({
            'data': {
                'source': 'query',
                'target': doc_id,
                'weight': score,
                'label': f'{score:.3f}'
            }
        })
    
    # Define stylesheet
    stylesheet = [
        # Query node style
        {
            'selector': 'node[type="query"]',
            'style': {
                'background-color': 'data(emotion_color)',
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': 120,
                'height': 60,
                'font-size': 12,
                'font-weight': 'bold',
                'text-wrap': 'wrap',
                'text-max-width': 110,
                'color': '#ffffff',  # White text for better contrast
                'text-outline-color': '#000000',  # Black outline
                'text-outline-width': 2  # Outline width
            }
        },
        # Document node style
        {
            'selector': 'node[type="document"]',
            'style': {
                'background-color': 'data(emotion_color)',
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': 120,
                'height': 60,
                'font-size': 12,
                'text-wrap': 'wrap',
                'text-max-width': 110,
                'color': '#ffffff',  # White text for better contrast
                'text-outline-color': '#000000',  # Black outline
                'text-outline-width': 2  # Outline width
            }
        },
        # Edge style
        {
            'selector': 'edge',
            'style': {
                'width': 'data(weight)',
                'line-color': '#666',
                'target-arrow-color': '#666',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'label': 'data(label)',
                'font-size': 10,
                'text-outline-color': '#ffffff',  # White outline for edge labels
                'text-outline-width': 1  # Thinner outline for edge labels
            }
        },
        # Hover styles
        {
            'selector': 'node:selected',
            'style': {
                'background-color': '#ffd700',
                'width': 130,
                'height': 70,
                'color': '#000000',  # Black text for selected nodes
                'text-outline-color': '#ffffff',  # White outline for selected nodes
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
    ]
    
    return cyto.Cytoscape(
        id='rag-graph',
        layout={
            'name': 'cose',  # Use force-directed layout
            'idealEdgeLength': 150,  # Increased for better spacing
            'refresh': 20,
            'fit': True,
            'padding': 30
        },
        style={'width': '100%', 'height': '400px'},
        elements=nodes + edges,
        stylesheet=stylesheet,
        userZoomingEnabled=True,
        userPanningEnabled=True,
        boxSelectionEnabled=True
    )

def create_graph_panel(debug_info: Dict[str, Any], config: Dict[str, Any]) -> dbc.Card:
    """
    Create a panel containing the RAG graph visualization.
    
    Args:
        debug_info: Dictionary containing RAG retrieval information
        config: Configuration dictionary containing output_dir
        
    Returns:
        Card component containing the graph visualization
    """
    return dbc.Card([
        dbc.CardHeader("RAG Retrieval Graph"),
        dbc.CardBody([
            create_rag_graph(debug_info, config),
            html.Div([
                html.H6("Graph Legend", className="mt-3"),
                html.Div([
                    html.Div([
                        html.Div(style={
                            'width': '20px',
                            'height': '20px',
                            'background-color': '#808080',
                            'display': 'inline-block',
                            'margin-right': '10px'
                        }),
                        html.Span("Query Node")
                    ], className="mb-2"),
                    html.Div([
                        html.Div(style={
                            'width': '20px',
                            'height': '20px',
                            'background-color': '#1f77b4',
                            'display': 'inline-block',
                            'margin-right': '10px'
                        }),
                        html.Span("Document Node (Color reflects emotion)")
                    ], className="mb-2"),
                    html.Div([
                        html.Div(style={
                            'width': '2px',
                            'height': '20px',
                            'background-color': '#666',
                            'display': 'inline-block',
                            'margin-right': '10px'
                        }),
                        html.Span("Edge Weight (Match Score)")
                    ])
                ])
            ])
        ])
    ]) 