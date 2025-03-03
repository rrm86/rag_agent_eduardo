from langchain_google_genai import ChatGoogleGenerativeAI
from config import API_KEY, MODEL, TEMPERATURE

# Inicializa o modelo LLM
llm = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=TEMPERATURE,
    api_key=API_KEY
)

