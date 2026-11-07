"""Configuração do app, lida das variáveis de ambiente ou do arquivo .env."""

import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("MODEL", "openai:gpt-4.1-mini")
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", 20))
API_URL = os.environ.get("API_URL", "http://localhost:8000")
