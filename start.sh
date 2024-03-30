#!/bin/bash

# Check if the virtual environment already exists
if [ ! -d "dsbootcamp" ]; then
    # Create a virtual environment named dsbootcamp if it doesn't exist
    python3 -m venv dsbootcamp
fi

# Activate the virtual environment and install the required packages
source dsbootcamp/bin/activate
pip install -r requirements.txt >> requirements.log 2>&1

# Pull updates from Git repository
echo "Pulling updates from Git repository..."
git pull