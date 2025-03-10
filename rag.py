import os
import numpy as np
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from llm import llm
from config import API_KEY, SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE_NAME, SUPABASE_COLLECTION_NAME
from supabase import create_client, Client

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

def get_supabase_client():
    """Obtém um cliente Supabase."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_embeddings_model():
    """Obtém o modelo de embeddings do Google."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )

def similarity_search_with_score(query, k=5):
    """
    Realiza uma busca de similaridade no Supabase usando pgvector.
    
    Args:
        query: A string de consulta
        k: Número de resultados a retornar
    
    Returns:
        Uma lista de tuplas (Document, score)
    """
    # Obter o embedding para a consulta
    embeddings = get_embeddings_model()
    query_embedding = embeddings.embed_query(query)
    
    # Realizar a busca no Supabase
    supabase = get_supabase_client()
    
    # Usar o operador de similaridade cosine do pgvector diretamente via RPC
    rpc_payload = {
        "query_embedding": query_embedding,
        "filter_collection_name": SUPABASE_COLLECTION_NAME.split(",")[0],
        "match_count": k
    }
    
    results = supabase.rpc(
        "match_documents", 
        rpc_payload
    ).execute()
    
    # Converter resultados em documentos com pontuações
    docs_with_scores = []
    
    for item in results.data:
        # Criar documento a partir do conteúdo e metadados retornados
        doc = Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        # A similaridade é retornada como "similarity"
        score = item["similarity"]
        docs_with_scores.append((doc, score))
    
    return docs_with_scores

@tool
def search_documents(query: str) -> str:
    """
    Aciona o vectorstore para buscar documentos similares a uma consulta. Chamar quando o usuário digitar: 'docs' <query>
    
    Args:
        query: A consulta para buscar documentos similares
        
    Returns:
        A lista de documentos e suas respectivas similaridades.
    """
    docs_with_scores = similarity_search_with_score(query=query, k=5)
    
    results = []
    for i, (doc, score) in enumerate(docs_with_scores):
        # No Supabase/pgvector, os scores já são similaridades (não distâncias L2)
        similarity_percent = f"{score * 100:.2f}%"
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
    docs_with_scores = similarity_search_with_score(query=query, k=5)
    
    context_docs = []
    results = []
    
    for i, (doc, score) in enumerate(docs_with_scores):
        # No pgvector, os scores já são similaridades (não distâncias L2)
        similarity_percent = f"{score * 100:.2f}%"
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
