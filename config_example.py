import uuid

# Configuration constants for our modularized application
API_KEY = "sua chave"
MODEL = "gemini-2.0-flash"
TEMPERATURE = 1.0


def get_config():
    """Gera um objeto de configuração com um identificador de thread."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}
