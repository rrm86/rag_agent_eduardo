import uuid
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants for our modularized application
API_KEY = os.getenv("API_KEY")
MODEL = "gemini-2.0-flash"
TEMPERATURE = 1.0

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE_NAME = os.getenv("SUPABASE_TABLE_NAME")
SUPABASE_COLLECTION_NAME = os.getenv("SUPABASE_COLLECTION_NAME")



def get_config():
    """Gera um objeto de configuração com um identificador de thread."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}
