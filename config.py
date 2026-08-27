# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from .env into the environment

# All real values now live in .env (gitignored) - see .env.example for the
# variables you need to set. Nothing sensitive is hardcoded here anymore.
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-only-insecure-key-change-me')

DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_NAME = os.environ.get('DB_NAME', 'datasense_ai')

# Leave unset (empty) if you don't have a key - insights.py and
# query_engine.py both already fall back to rule-based/template output
# automatically when this is empty, so the app still works fine without it.
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
