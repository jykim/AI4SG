#!/usr/bin/env python3
"""
Journal Chat Agent Utilities

This module handles the chat functionality for the journal dashboard,
including processing journal entries and interacting with the OpenAI API.
"""

import os
import openai
import pandas as pd
import logging
import traceback
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import json
import hashlib
from typing import Optional, Dict, List, Any

class Config:
    """Configuration class to manage application settings"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.load_config()
        self.setup_directories()

    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found. Using default values.")
            config = {
                'input_dir': 'input',
                'output_dir': 'output',
                'api_cache_dir': 'api_cache',
                'agent_cache_dir': 'agent_cache',
                'min_process_interval': 600,
                'max_entries_for_prompt': 10
            }
        
        # Set configuration values
        self.input_dir = Path(config.get('input_dir', 'input'))
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.api_cache_dir = Path(config.get('api_cache_dir', 'api_cache'))
        self.agent_cache_dir = Path(config.get('agent_cache_dir', 'agent_cache'))
        self.min_process_interval = config.get('min_process_interval', 600)
        self.max_entries_for_prompt = config.get('max_entries_for_prompt', 10)
        
        # API key should be set via environment variable
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')

    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [self.input_dir, self.output_dir, self.api_cache_dir, self.agent_cache_dir]:
            directory.mkdir(exist_ok=True)

def get_cache_key(content: str) -> str:
    """Generate a cache key for the given content using SHA-256 hash."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_cached_response(cache_key: str, config: Optional[Config] = None) -> Optional[Dict]:
    """Try to get a cached response for the given key."""
    if config is None:
        config = Config()
    cache_file = config.agent_cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading cache file: {e}")
    return None

def save_to_cache(cache_key: str, response: Dict, config: Optional[Config] = None) -> None:
    """Save an API response to the cache with detailed debug information."""
    if config is None:
        config = Config()
    cache_file = config.agent_cache_dir / f"{cache_key}.json"
    
    # Extract journal entries from the prompt if available
    entries_summary = "No journal entries found in request"
    if 'request' in response and 'messages' in response['request']:
        for msg in response['request']['messages']:
            if msg.get('role') == 'user' and 'content' in msg:
                content = msg['content']
                # Look for entries between the prompt and chat history
                if 'Below are the recent journal entries' in content and 'Today\'s Chat History:' in content:
                    entries_section = content.split('Below are the recent journal entries')[1].split('Today\'s Chat History:')[0]
                    titles = []
                    for line in entries_section.split('\n'):
                        if line.startswith('## '):
                            # Extract date and title from the markdown header
                            title = line.replace('## ', '').strip()
                            titles.append(title)
                    if titles:
                        entries_summary = "Journal entries included in request:\n" + "\n".join(titles)
                break
    
    # Add debug information to the cached response
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'cache_key': cache_key,
        'response_length': len(response.get('content', '')),
        'response': response,
        'entries_summary': entries_summary,
        'debug': {
            'python_version': os.sys.version,
            'openai_key_exists': bool(config.openai_api_key),
            'cache_directory': str(config.agent_cache_dir)
        }
    }
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error saving to cache: {e}")

def load_prompt_template() -> str:
    """Load the prompt template for the chat agent."""
    try:
        with open('agent_prompt.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error("agent_prompt.md not found")
        return ""

def format_journal_entries_as_markdown(entries: List[Dict[str, Any]]) -> str:
    """Convert journal entries to markdown format."""
    markdown = ""
    for entry in entries:
        date = entry.get('Date', '')
        time = entry.get('Time', '')
        title = entry.get('Title', 'Untitled')
        content = entry.get('Content', '')
        emotion = entry.get('emotion', '')
        topic = entry.get('topic', '')
        tags = entry.get('Tags', '')

        markdown += f"## {date} {time} - {title}\n\n"
        if emotion or topic:
            markdown += f"*Emotion: {emotion}* | *Topic: {topic}*\n\n"
        if tags:
            markdown += f"Tags: {tags}\n\n"
        markdown += f"{content}\n\n---\n\n"
    
    return markdown

def get_recent_entries(df: pd.DataFrame, days: int = 7, config: Optional[Config] = None) -> pd.DataFrame:
    """Get entries from both past and future, limited to configured number of entries."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    if config is None:
        config = Config()
    
    # Convert Date column to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate date range (past and future)
    today = datetime.now().date()
    past_threshold = today - timedelta(days=days)
    future_threshold = today + timedelta(days=days)
    
    # Filter entries within the date range
    recent_entries = df[
        (df['Date'].dt.date >= past_threshold) & 
        (df['Date'].dt.date <= future_threshold)
    ].copy()
    
    # Sort by date
    recent_entries = recent_entries.sort_values('Date', ascending=True)
    
    # Limit to configured number of entries
    max_entries = config.max_entries_for_prompt
    if len(recent_entries) > max_entries:
        # Try to balance past and future entries
        past_entries = recent_entries[recent_entries['Date'].dt.date <= today]
        future_entries = recent_entries[recent_entries['Date'].dt.date > today]
        
        # Calculate how many entries to take from each
        past_count = min(len(past_entries), max_entries // 2)
        future_count = max_entries - past_count
        
        # Take entries from both past and future
        past_entries = past_entries.tail(past_count)
        future_entries = future_entries.head(future_count)
        
        # Combine past and future entries
        recent_entries = pd.concat([past_entries, future_entries])
    
    return recent_entries

def save_chat_log(
    user_query: str,
    ai_response: str,
    current_time: datetime,
    config: Optional[Config] = None
) -> None:
    """Save chat interaction to a markdown log file.
    
    Args:
        user_query: The user's message
        ai_response: The AI's response
        current_time: Current datetime
        config: Optional Config object
    """
    if config is None:
        config = Config()
    
    chat_log_file = config.output_dir / 'chat_log.md'
    
    # Format the chat entry with second-level timestamp
    chat_entry = f"""
### Chat Entry - {current_time.strftime("%Y-%m-%d %H:%M:%S")}

**User**: {user_query}

**Assistant**: {ai_response}

"""
    
    # Append to the log file
    try:
        with open(chat_log_file, 'a', encoding='utf-8') as f:
            f.write(chat_entry)
    except Exception as e:
        logging.error(f"Error saving chat log: {e}")

def get_todays_chat_log(config: Optional[Config] = None) -> str:
    """Read today's chat log entries from the markdown file.
    
    Args:
        config: Optional Config object
        
    Returns:
        str: Today's chat log entries in markdown format
    """
    if config is None:
        config = Config()
    
    # Use config's get_chat_log_path method if available, otherwise use default path
    chat_log_file = getattr(config, 'get_chat_log_path', lambda: config.output_dir / 'chat_log.md')()
    today = datetime.now().strftime("%Y-%m-%d")
    
    if not chat_log_file.exists():
        return ""
    
    try:
        with open(chat_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split content into entries
        entries = content.split('### Chat Entry - ')
        
        # Filter for today's entries
        todays_entries = []
        for entry in entries:
            if entry.strip():  # Skip empty entries
                entry_date = entry.split('\n')[0].strip()  # Get the date from first line
                if today in entry_date:
                    todays_entries.append(entry)
        
        if todays_entries:
            return "### Today's Chat History:\n\n### Chat Entry - " + "### Chat Entry - ".join(todays_entries)
        return ""
        
    except Exception as e:
        logging.error(f"Error reading chat log: {e}")
        return ""

def get_chat_response(
    user_query: str,
    journal_entries: pd.DataFrame,
    chat_history: List[Dict[str, str]],
    config: Optional[Config] = None
) -> str:
    """
    Get a response from the chat agent using the OpenAI API.
    
    Args:
        user_query: The user's question or message
        journal_entries: DataFrame containing recent journal entries
        chat_history: List of previous chat messages
        config: Optional Config object
        
    Returns:
        str: The agent's response
    """
    if config is None:
        config = Config()
    
    # Set up OpenAI API
    openai.api_key = config.openai_api_key
    
    # Get current date and time information
    current_time = datetime.now()
    date_str = current_time.strftime("%Y-%m-%d")
    time_str = current_time.strftime("%H:%M")
    time_of_day = (
        "morning" if 5 <= current_time.hour < 12
        else "afternoon" if 12 <= current_time.hour < 17
        else "evening" if 17 <= current_time.hour < 22
        else "night"
    )
    
    # Get recent entries and format as markdown
    recent_entries = get_recent_entries(journal_entries, config=config)
    entries_markdown = format_journal_entries_as_markdown(recent_entries.to_dict('records'))
    
    # Get today's chat log and filter for last 60 minutes
    todays_chat_log = get_todays_chat_log(config)
    if todays_chat_log:
        # Split into entries and filter by time
        entries = todays_chat_log.split('### Chat Entry - ')
        filtered_entries = []
        cutoff_time = current_time - timedelta(minutes=60)
        
        for entry in entries:
            if entry.strip():
                # Extract timestamp from the first line
                lines = entry.strip().split('\n')
                if lines:
                    try:
                        timestamp_str = lines[0].strip()
                        entry_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        if entry_time >= cutoff_time:
                            filtered_entries.append(entry)
                    except ValueError:
                        # If timestamp parsing fails, include the entry
                        filtered_entries.append(entry)
        
        todays_chat_log = '### Chat Entry - '.join(filtered_entries)
    
    # Load and format prompt template
    prompt_template = load_prompt_template()
    formatted_prompt = prompt_template.format(
        journal_entries=entries_markdown,
        user_query=user_query,
        current_date=date_str,
        current_time=time_str,
        time_of_day=time_of_day,
        chat_log=todays_chat_log
    )
    
    # Prepare messages including chat history
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant analyzing journal entries."}
    ]
    
    # Filter chat history to last 60 minutes
    cutoff_time = current_time - timedelta(minutes=60)
    recent_chat_history = []
    for msg in chat_history:
        if 'timestamp' in msg:
            try:
                msg_time = datetime.fromisoformat(msg['timestamp'])
                if msg_time >= cutoff_time:
                    recent_chat_history.append(msg)
            except ValueError:
                # If timestamp parsing fails, include the message
                recent_chat_history.append(msg)
    
    # Add filtered chat history (last 5 messages)
    for msg in recent_chat_history[-5:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add current prompt
    messages.append({"role": "user", "content": formatted_prompt})
    
    try:
        # Check cache first
        cache_key = get_cache_key(str(messages))
        cached_response = get_cached_response(cache_key, config)
        
        if cached_response:
            logging.info("Using cached response")
            ai_response = cached_response['response']['content']
            # Save to chat log even if using cached response
            save_chat_log(user_query, ai_response, current_time, config)
            
            # Return both the chat log and the response
            return {
                'response': ai_response,
                'chat_log': todays_chat_log
            }
        
        # Make API call
        logging.info("Sending request to OpenAI API")
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",  # Using GPT-4 Turbo with 128k context window
            messages=messages,
            temperature=0.7,
            max_tokens=1000  # Increased since we have a larger context window
        )
        
        # Extract response
        ai_response = response.choices[0].message.content
        
        # Save to chat log
        save_chat_log(user_query, ai_response, current_time, config)
        
        # Cache response with additional debug info
        save_to_cache(cache_key, {
            'content': ai_response,
            'request': {
                'messages': messages,
                'model': 'gpt-4-1106-preview',
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'response_metadata': {
                'model': response.model,
                'created': response.created,
                'response_ms': response.usage.total_tokens if response.usage else None
            }
        }, config)
        
        # Return both the chat log and the response
        return {
            'response': ai_response,
            'chat_log': todays_chat_log
        }
        
    except Exception as e:
        logging.error(f"Error getting chat response: {e}")
        error_response = f"I apologize, but I encountered an error: {str(e)}"
        
        # Save error to chat log
        save_chat_log(user_query, error_response, current_time, config)
        
        # Cache the error for debugging
        save_to_cache(f"error_{cache_key}", {
            'content': error_response,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'request': {
                'messages': messages,
                'model': 'gpt-4-1106-preview',
                'temperature': 0.7,
                'max_tokens': 1000
            }
        }, config)
        
        # Return both the error response and chat log
        return {
            'response': error_response,
            'chat_log': todays_chat_log
        }

# Initialize configuration and logging
config = Config()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.output_dir / 'agent.log'),
        logging.StreamHandler()
    ]
) 