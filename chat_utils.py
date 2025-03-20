"""
Chat utility functions for the journal dashboard.

This module contains utility functions for handling chat messages, including
parsing, formatting, and processing chat history.
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dash import html
import dash_bootstrap_components as dbc

def parse_message_timestamp(timestamp_str: str) -> datetime:
    """Parse message timestamp from string."""
    try:
        # Try both formats: with and without seconds
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        logging.warning(f"Could not parse timestamp with any format: {timestamp_str}")
        return datetime.now()
    except Exception as e:
        logging.warning(f"Error parsing timestamp: {timestamp_str} - {str(e)}")
        return datetime.now()

def extract_message_content(msg_children: List[Dict]) -> Tuple[str, bool]:
    """Extract message content and determine if it's from user."""
    if not msg_children or len(msg_children) < 2:
        return "", False
        
    # Skip loading message
    if any(isinstance(child, dict) and 
          isinstance(child.get('props', {}).get('children'), list) and 
          any(isinstance(c, dict) and c.get('type') == 'Spinner' for c in child['props']['children'])
          for child in msg_children):
        return "", False
        
    # Get message content
    content_div = msg_children[0]
    if not isinstance(content_div, dict):
        return "", False
        
    content_children = content_div.get('props', {}).get('children', [])
    if not content_children:
        return "", False
        
    # Get the content after "You: " or "Assistant: "
    content_span = content_children[1] if len(content_children) > 1 else None
    if not isinstance(content_span, dict):
        return "", False
        
    content = content_span.get('props', {}).get('children', [{}])[0].get('props', {}).get('children', '')
    is_user = 'You: ' in content_children[0].get('props', {}).get('children', '')
    
    return content, is_user

def extract_message_timestamp(msg_children: List[Dict], current_time: datetime) -> datetime:
    """Extract timestamp from message children."""
    timestamp_div = next(
        (child for child in msg_children
         if isinstance(child, dict) and 
         child.get('props', {}).get('style', {}).get('color') == '#6c757d'),
        None
    )
    timestamp_str = timestamp_div.get('props', {}).get('children') if timestamp_div else None
    
    if timestamp_str:
        try:
            return datetime.strptime(f"{current_time.strftime('%Y-%m-%d')} {timestamp_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            return current_time
    return current_time

def format_chat_message(content: str, is_user: bool = False, is_latest: bool = False, 
                       timestamp: Optional[datetime] = None, max_word_count: int = 30) -> dbc.Alert:
    """Format a chat message with proper styling and truncation."""
    import time
    
    # Generate a unique ID for the message using timestamp, content hash, and random component
    content_str = str(content) if isinstance(content, list) else content
    random_component = hash(f"{content_str}{time.time()}{is_user}")
    message_id = f"msg-{int(time.time() * 1000)}-{abs(random_component) % 100000}"
    
    # Use provided timestamp or current time
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str):
        try:
            # Try both formats: with and without seconds
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
                try:
                    timestamp = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
            else:
                timestamp = datetime.now()
        except Exception:
            timestamp = datetime.now()
    
    # Format time for display (HH:MM)
    time_str = timestamp.strftime("%H:%M")
    
    # Split content into words and check if it needs truncation
    words = content_str.split()
    is_long = len(words) > max_word_count and not is_latest
    display_content = ' '.join(words[:max_word_count]) if is_long else content_str
    
    # Create message content with conditional expand button
    message_content = [
        html.Div([
            html.Strong("You: " if is_user else "Assistant: "),
            html.Div(
                [
                    html.Span(display_content),
                    html.Button(
                        "...",
                        id={'type': 'expand-button', 'index': message_id},
                        n_clicks=0,
                        style={
                            'background': 'none',
                            'border': 'none',
                            'color': '#0d6efd',
                            'cursor': 'pointer',
                            'padding': '0 4px',
                            'display': 'inline' if is_long else 'none'
                        }
                    )
                ],
                id={'type': 'message-content', 'index': message_id},
                style={
                    'display': 'inline',
                    'word-wrap': 'break-word'
                }
            )
        ]),
        html.Div(
            time_str,
            style={
                'font-size': '0.8em',
                'color': '#6c757d',
                'text-align': 'right' if is_user else 'left',
                'margin-top': '4px',
                'padding-right' if is_user else 'padding-left': '8px'
            }
        )
    ]
    
    # Store the full content in a hidden div
    message_content.append(
        html.Div(
            content_str,
            id={'type': 'full-content', 'index': message_id},
            style={'display': 'none'}
        )
    )
    
    return dbc.Alert(
        message_content,
        color="primary" if is_user else "info",
        style={
            'text-align': 'right' if is_user else 'left',
            'margin': '5px',
            'white-space': 'pre-wrap'
        }
    )

def parse_chat_entry(entry: str, current_time: datetime, max_word_count: int = 30) -> List[Dict]:
    """Parse a single chat entry into formatted messages."""
    messages = []
    lines = entry.strip().split('\n')
    if not lines:
        return messages
        
    # Parse timestamp from first line
    entry_timestamp = parse_message_timestamp(lines[0].strip())
    
    current_message = ""
    current_role = None
    
    for line in lines[1:]:  # Skip the timestamp line
        if line.startswith('**User**:'):
            # If we have a previous message, add it
            if current_message and current_role:
                messages.append(format_chat_message(
                    current_message.strip(),
                    is_user=(current_role == 'user'),
                    is_latest=False,
                    timestamp=entry_timestamp,
                    max_word_count=max_word_count
                ))
            current_role = 'user'
            current_message = line.replace('**User**:', '').strip()
        elif line.startswith('**Assistant**:'):
            # If we have a previous message, add it
            if current_message and current_role:
                messages.append(format_chat_message(
                    current_message.strip(),
                    is_user=(current_role == 'user'),
                    is_latest=False,
                    timestamp=entry_timestamp,
                    max_word_count=max_word_count
                ))
            current_role = 'assistant'
            current_message = line.replace('**Assistant**:', '').strip()
        else:
            # Append to current message if it's a continuation
            if current_message and line.strip():
                current_message += "\n" + line.strip()
    
    # Add the last message if there is one
    if current_message and current_role:
        messages.append(format_chat_message(
            current_message.strip(),
            is_user=(current_role == 'user'),
            is_latest=True,  # Mark the last message as latest
            timestamp=entry_timestamp,
            max_word_count=max_word_count
        ))
    
    return messages

def process_existing_messages(current_messages: Dict, current_time: datetime, max_word_count: int = 30) -> List[Dict]:
    """Process existing messages from the chat interface."""
    existing_messages = []
    if not current_messages or not isinstance(current_messages, dict) or not current_messages.get('props', {}).get('children'):
        return existing_messages
        
    messages_list = current_messages['props']['children']
    if isinstance(messages_list, dict):
        messages_list = [messages_list]
        
    for msg in messages_list:
        if not isinstance(msg, dict) or 'props' not in msg:
            continue
            
        msg_children = msg['props'].get('children', [])
        content, is_user = extract_message_content(msg_children)
        if not content:
            continue
            
        msg_time = extract_message_timestamp(msg_children, current_time)
        existing_messages.append(format_chat_message(
            content, 
            is_user=is_user, 
            is_latest=False, 
            timestamp=msg_time,
            max_word_count=max_word_count
        ))
    
    return existing_messages

def create_chat_history(existing_messages: List[Dict], current_time: datetime) -> List[Dict]:
    """Create chat history for the API from existing messages."""
    chat_history = []
    for msg in existing_messages:
        if not isinstance(msg, dict) or 'props' not in msg:
            continue
            
        msg_children = msg['props'].get('children', [])
        content, is_user = extract_message_content(msg_children)
        if not content:
            continue
            
        msg_time = extract_message_timestamp(msg_children, current_time)
        chat_history.append({
            'role': 'user' if is_user else 'assistant',
            'content': content,
            'timestamp': msg_time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return chat_history 