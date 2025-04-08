#!/bin/bash

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Welcome to the Journal Dashboard Setup!${NC}"
echo "This script will help you configure the application."
echo

# Function to validate directory path
validate_directory() {
    local dir=$1
    if [ -d "$dir" ]; then
        return 0
    else
        return 1
    fi
}

# Function to expand tilde in path
expand_path() {
    local path=$1
    echo "${path/#\~/$HOME}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a Python package is installed
check_python_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# 0. Check Python environment
echo -e "${YELLOW}Step 0: Checking Python Environment${NC}"

# Check if Python is installed
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command_exists pip3; then
    echo -e "${RED}Error: pip3 is not installed.${NC}"
    echo "Please install pip3 and try again."
    exit 1
fi

# Install requirements
echo "Installing Python requirements..."
if [ -f "requirements.txt" ]; then
    # First, try to install openai specifically
    echo "Installing openai package..."
    pip3 install openai>=1.12.0
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install openai package.${NC}"
        echo "Please check your internet connection and try again."
        exit 1
    fi

    # Then install the rest of the requirements
    echo "Installing other requirements..."
    pip3 install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Requirements installed successfully!${NC}"
        
        # Verify critical packages
        echo "Verifying critical packages..."
        if ! check_python_package "openai"; then
            echo -e "${RED}Error: openai package not properly installed.${NC}"
            echo "Please try running: pip3 install --upgrade openai"
            exit 1
        fi
    else
        echo -e "${RED}Error: Failed to install requirements.${NC}"
        echo "Please check your internet connection and try again."
        exit 1
    fi
else
    echo -e "${RED}Error: requirements.txt not found.${NC}"
    echo "Please make sure you're in the correct directory."
    exit 1
fi

# 1. Obsidian Vault Location
echo -e "\n${YELLOW}Step 1: Obsidian Vault Configuration${NC}"
echo "Please enter the path to your Obsidian vault root directory."
echo "Default: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/OV2024"
read -p "Vault path: " VAULT_PATH
VAULT_PATH=${VAULT_PATH:-"~/Library/Mobile Documents/iCloud~md~obsidian/Documents/OV2024"}
EXPANDED_VAULT_PATH=$(expand_path "$VAULT_PATH")

if ! validate_directory "$EXPANDED_VAULT_PATH"; then
    echo -e "${RED}Warning: The specified vault directory does not exist.${NC}"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup aborted."
        exit 1
    fi
fi

# 2. Create necessary folders
echo -e "\n${YELLOW}Step 2: Creating Application Directories${NC}"
echo "Creating required directories..."

# Create application directories
mkdir -p input output api_cache bm25_index

# Create symlinks to Obsidian directories if they don't exist
JOURNAL_DIR="$EXPANDED_VAULT_PATH/Journal"
READING_DIR="$EXPANDED_VAULT_PATH/Reading"

if [ ! -L "input/journal" ] && [ -d "$JOURNAL_DIR" ]; then
    ln -s "$JOURNAL_DIR" "input/journal"
    echo "Created symlink for journal directory"
fi

if [ ! -L "input/reading" ] && [ -d "$READING_DIR" ]; then
    ln -s "$READING_DIR" "input/reading"
    echo "Created symlink for reading directory"
fi

# 3. OpenAI API Key
echo -e "\n${YELLOW}Step 3: OpenAI API Key Configuration${NC}"
echo "Please enter your OpenAI API key."
echo "You can find your API key at: https://platform.openai.com/api-keys"
read -p "OpenAI API Key: " OPENAI_KEY

# Update config.yaml with the new values
echo -e "\n${YELLOW}Updating configuration...${NC}"

# Create a temporary file
TMP_CONFIG=$(mktemp)

# Update the config file
awk -v vault="$VAULT_PATH" -v key="$OPENAI_KEY" '
    /journal_dir:/ { print "journal_dir: \"" vault "/Journal\""; next }
    /reading_dir:/ { print "reading_dir: \"" vault "/Reading\""; next }
    /index_dir:/ { print "index_dir: \"" vault "\""; next }
    /openai_api_key:/ { print "openai_api_key: \"" key "\""; next }
    { print }
' config.yaml > "$TMP_CONFIG"

# Replace the original config file
mv "$TMP_CONFIG" config.yaml

echo -e "\n${GREEN}Setup completed successfully!${NC}"
echo "Your configuration has been updated with:"
echo "- Python requirements installed"
echo "- Obsidian vault path: $VAULT_PATH"
echo "- Required directories created"
echo "- OpenAI API key configured"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Review the updated config.yaml file"
echo "2. Run the application using: python main.py"
echo "3. Check the logs for any issues"

# Make the script executable
chmod +x "$0" 