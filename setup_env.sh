#!/bin/bash

echo "Installing virtualenv..."
pip install virtualenv

echo "Creating virtual environment..."
python3 -m virtualenv env

echo "Activating virtual environment..."
source env/bin/activate

echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "Installing Playwright + Chromium (for the McKesson ordering tool)..."
pip install playwright
python -m playwright install chromium

echo "✅ Environment setup complete!"
