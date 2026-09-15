#!/bin/bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py makemigrations predictor
python manage.py migrate
python manage.py seed_demo
python manage.py check
echo "Setup complete. Run ./run_mac.sh and open http://127.0.0.1:8000/"
