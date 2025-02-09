#!/bin/bash

python3 -m venv PreGawk

source PreGawk/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

echo "Setup complete. Starting virtual environment. To deactivate, run 'deactivate'."
