import os
import numpy as np
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from llm import llm
from config import API_KEY

# Caminho para o índice FAISS
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "index", "faiss_index")

# Define the RAG prompt template para produtos
RAG_PROMPT_TEMPLATE = """
Você é uma assistente especializado em um inventário de produtos.
Use o seguinte contexto da base de conhecimento para responder à pergunta da usuária.
Se você não souber a resposta com base no contexto, diga que não sabe
e não tente inventar uma resposta.

Contexto: {context}

Pergunta da usuária: {question}

Regras Importantes:
- Seja direto e objetivo
- Sempre inclua o preço nos resultados
- Sempre inclua os tamanhos nos resultados
- Sempre inclua as cores nos resultados
- Quando perguntado sobre combinações, sugira combinações entre os produtos encontrados e inclua o preço da combinação

Sua resposta:
"""

# Função de ajuda para converter distância L2 para similaridade cosseno
def l2_to_cosine(l2_distance):
    """Converte a distância L2 para similaridade cosseno."""
    return 1 - (l2_distance ** 2 / 2)

def get_vectorstore():
    """Obtenha ou crie uma instância de vectorstore."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )
    
    if os.path.exists(FAISS_INDEX_PATH) and len(os.listdir(FAISS_INDEX_PATH)) > 0:
        return FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        raise ValueError("Vectorstore não encontrado. Execute o script de indexação primeiro.")

@tool
def search_documents(query: str) -> str:
    """
    Aciona o vectorstore para buscar documentos similares a uma consulta. Chamar quando o usuário digitar: 'docs' <query>
    
    Args:
        query: A consulta para buscar documentos similares
        
    Returns:
        A lista de documentos e suas respectivas similaridades.
    """
    vectorstore = get_vectorstore()
    docs_with_scores = vectorstore.similarity_search_with_score(query=query, k=5)
    
    results = []
    for i, (doc, l2_distance) in enumerate(docs_with_scores):
        cosine_sim = l2_to_cosine(l2_distance)
        similarity_percent = f"{cosine_sim * 100:.2f}%"
        source = doc.metadata.get("source", "Desconhecido")
        results.append(f"Documento {i+1} (Similaridade: {similarity_percent}, Fonte: {source}):\n{doc.page_content}\n")
    
    return "\n".join(results) if results else "Nenhum documento encontrado para esta consulta."

@tool
def rag_query(query: str) -> str:
    """
    Busca informações sobre produtos no catálogo usando RAG (Retrieval-Augmented Generation).
    
    Args:
        query: A pergunta sobre produtos do catálogo
        
    Returns:
        Uma resposta detalhada baseada nos documentos recuperados
    """
    vectorstore = get_vectorstore()
    docs_with_scores = vectorstore.similarity_search_with_score(query=query, k=5)
    
    context_docs = []
    results = []
    
    for i, (doc, l2_distance) in enumerate(docs_with_scores):
        cosine_sim = l2_to_cosine(l2_distance)
        similarity_percent = f"{cosine_sim * 100:.2f}%"
        source = doc.metadata.get("source", "Desconhecido")
        
        context_docs.append(doc.page_content)
        results.append(f"Documento {i+1} (Similaridade: {similarity_percent}, Fonte: {source}):\n{doc.page_content}\n")
    
    # Combina os documentos para o contexto
    context = "\n\n".join(context_docs)
    
    # Cria e executa o chain
    prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": query})
    
    return f"{result.content}\n\nDocumentos usados para responder à pergunta:\n{''.join(results)}"

# Exporta as ferramentas para uso em outros modulos
rag_tools = [rag_query, search_documents]
