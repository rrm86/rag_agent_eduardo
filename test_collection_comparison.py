#!/usr/bin/env python3
import os
import json
import csv
import pandas as pd
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from supabase import create_client
from config import API_KEY, SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE_NAME, SUPABASE_COLLECTION_NAME

# Lista de coleções a serem comparadas
collections = SUPABASE_COLLECTION_NAME.split(",")

def get_supabase_client():
    """Obtém um cliente Supabase."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_embeddings_model():
    """Obtém o modelo de embeddings do Google."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )

def similarity_search_with_score(query, collection_name, k=3):
    """
    Realiza uma busca de similaridade no Supabase usando pgvector para uma coleção específica.
    
    Args:
        query: A string de consulta
        collection_name: Nome da coleção a ser pesquisada
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
        "filter_collection_name": collection_name,
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
            metadata=item.get("metadata", {}) or {}  # Ensure metadata is always a dict, even if None
        )
        # A similaridade é retornada como "similarity"
        score = item["similarity"]
        docs_with_scores.append((doc, score))
    
    return docs_with_scores

def extract_product_info(content):
    """
    Extrai informações de produto de um conteúdo JSON.
    
    Args:
        content: Conteúdo do documento
        
    Returns:
        Tupla com (id, nome, preço) ou None se não for possível extrair
    """
    try:
        if content.strip().startswith("{") and content.strip().endswith("}"):
            json_obj = json.loads(content)
            if isinstance(json_obj, dict):
                product_id = json_obj.get("id", "N/A")
                product_name = json_obj.get("nome", "N/A")
                price = json_obj.get("preco", "N/A")
                return (product_id, product_name, price)
    except:
        pass
    
    return None

def format_result_for_dataframe(doc, score):
    """
    Formata um resultado para inclusão no DataFrame.
    
    Args:
        doc: Documento
        score: Pontuação de similaridade
    
    Returns:
        Dicionário com informações formatadas
    """
    # Formata a pontuação como porcentagem
    score_percent = f"{score * 100:.2f}%"
    score_value = score * 100  # Valor numérico para ordenação
    
    # Extrai informações do produto se disponíveis
    product_info = extract_product_info(doc.page_content)
    
    result = {
        "score": score_percent,
        "score_value": score_value,
        "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
        "product_id": "",
        "product_name": "",
        "price": ""
    }
    
    if product_info:
        product_id, product_name, price = product_info
        result["product_id"] = product_id
        result["product_name"] = product_name
        result["price"] = price
    
    return result

def compare_collections_pandas(queries, save_csv=False, k=3):
    """
    Compara os resultados de consultas em todas as coleções e retorna um DataFrame pandas.
    
    Args:
        queries: Lista de strings de consulta
        save_csv: Se True, salva os resultados em um arquivo CSV
        k: Número de resultados a retornar por coleção
    
    Returns:
        DataFrame pandas com os resultados
    """
    # Prepara os dados para o DataFrame
    df_data = []
    
    # Para cada consulta
    for query in queries:
        print(f"Processando consulta: '{query}'")
        
        # Obtém resultados para cada coleção
        for collection in collections:
            results = similarity_search_with_score(query, collection, k)
            
            # Para cada resultado
            for rank, (doc, score) in enumerate(results, 1):
                # Formata o resultado
                result_data = format_result_for_dataframe(doc, score)
                
                # Adiciona informações adicionais
                result_data["query"] = query
                result_data["collection"] = collection
                result_data["rank"] = rank
                
                # Adiciona aos dados do DataFrame
                df_data.append(result_data)
    
    # Cria o DataFrame
    df = pd.DataFrame(df_data)
    
    # Reorganiza as colunas para melhor visualização
    columns = [
        "query", "collection", "rank", "score", "score_value",
        "product_id", "product_name", "price", 
        "content_preview"
    ]
    df = df[columns]
    
    # Salva em CSV se solicitado
    if save_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"collection_comparison_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"Resultados salvos no arquivo: {output_file}")
    
    return df

def display_comparison_by_query(df):
    """
    Exibe uma comparação dos resultados agrupados por consulta.
    
    Args:
        df: DataFrame pandas com os resultados
    """
    # Remove a coluna de conteúdo para exibição mais limpa
    display_df = df.drop(columns=["content_preview", "score_value"])
    
    # Para cada consulta
    for query in df["query"].unique():
        print("\n" + "=" * 100)
        print(f"CONSULTA: '{query}'")
        print("=" * 100)
        
        # Filtra resultados para esta consulta
        query_df = display_df[display_df["query"] == query]
        
        # Agrupa por coleção
        for collection in collections:
            collection_df = query_df[query_df["collection"] == collection]
            
            if not collection_df.empty:
                print(f"\nCOLEÇÃO: {collection}")
                print("-" * 80)
                # Exibe sem o índice e sem as colunas query e collection que já sabemos
                print(collection_df.drop(columns=["query", "collection"]).to_string(index=False))
    
    print("\n" + "=" * 100)

def display_top_results_comparison(df):
    """
    Exibe uma comparação dos melhores resultados para cada consulta entre as coleções.
    
    Args:
        df: DataFrame pandas com os resultados
    """
    print("\n" + "=" * 100)
    print("COMPARAÇÃO DOS MELHORES RESULTADOS POR COLEÇÃO")
    print("=" * 100)
    
    # Para cada consulta
    for query in df["query"].unique():
        print(f"\nCONSULTA: '{query}'")
        print("-" * 80)
        
        # Filtra resultados para esta consulta e apenas rank 1
        top_results = df[(df["query"] == query) & (df["rank"] == 1)]
        
        # Ordena por score (maior primeiro)
        top_results = top_results.sort_values(by="score_value", ascending=False)
        
        # Exibe sem o índice e sem as colunas query e rank que já sabemos
        display_cols = ["collection", "score", "product_id", "product_name", "price"]
        print(top_results[display_cols].to_string(index=False))
        print()

def main():
    """Função principal para testar a comparação de coleções."""
    # Lista de consultas para testar
    queries = [
        "Quais são as características de um blazer feminino em couro?",
        "Quanto é o preço de uma calça jeans? O que combinar com ela?",
        "Existe algum vestido em promoção?",
        "Camiseta básica branca",
        "Vestido floral",
        "Produto VF003"
    ]
    
    # Obtém os resultados como DataFrame pandas
    print("Comparando resultados nas coleções:", ", ".join(collections))
    df = compare_collections_pandas(queries)
    
    # Exibe comparações
    display_comparison_by_query(df)
    display_top_results_comparison(df)
    
    print("\nAnálise concluída!")

if __name__ == "__main__":
    main()
