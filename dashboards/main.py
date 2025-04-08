#!/usr/bin/env python3
"""
Main entry point for all dashboards.
"""

import argparse
import sys
from pathlib import Path

def main():
    """Parse arguments and run the appropriate dashboard."""
    parser = argparse.ArgumentParser(description='Run a dashboard')
    parser.add_argument('dashboard', choices=['journal', 'reading', 'graph'],
                        help='Which dashboard to run')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    args = parser.parse_args()

    # Remove sys.path manipulation
    # dash_dir = Path(__file__).parent
    # sys.path.insert(0, str(dash_dir.parent)) # Add project root to path
    
    try:
        # Import the appropriate dashboard app dynamically
        if args.dashboard == 'journal':
            from dashboards.journal.app import app
        elif args.dashboard == 'reading':
            # Assuming dash_reading.py was moved to dashboards/reading/app.py
            # You'll need to refactor dashboards/reading/app.py similarly to dashboards/journal/app.py
            from dashboards.reading.app import app
        elif args.dashboard == 'graph':
            # Assuming dash_graph.py was moved to dashboards/graph/app.py
            # You'll need to refactor dashboards/graph/app.py similarly to dashboards/journal/app.py
            from dashboards.graph.app import app
        else:
            print(f"Error: Unknown dashboard '{args.dashboard}'")
            sys.exit(1)

        # Set app title (optional, can also be set in individual app.py files)
        app.title = f"{args.dashboard.capitalize()} Dashboard"

        # Run the dashboard
        print(f"Running {args.dashboard} dashboard on http://127.0.0.1:{args.port}/")
        app.run(debug=args.debug, port=args.port)

    except ImportError as e:
        print(f"Error importing dashboard '{args.dashboard}': {e}")
        print("Please ensure the dashboard module exists and is refactored correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while trying to run the dashboard: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 