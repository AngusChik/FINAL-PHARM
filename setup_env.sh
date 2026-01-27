#!/bin/bash

echo "Installing virtualenv..."
pip install virtualenv

echo "Creating virtual environment..."
python3 -m virtualenv env

echo "Activating virtual environment..."
source env/bin/activate

echo "Installing packages..."
pip install django
pip install psycopg2-binary
pip install python-dateutil
pip install reportlab

echo "✅ Environment setup complete!"
