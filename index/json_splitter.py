from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.documents import Document
import os
import json
import sys
from typing import Dict, List, Any
from supabase import create_client

# Adiciona o diretório pai ao sys.path para permitir importação absoluta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE_NAME, SUPABASE_COLLECTION_NAME

# Caminho para a pasta de dados (dentro do diretório index)
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def process_json_file(filename: str) -> List[Dict]:
    """
    Realiza o parseamento de um arquivo JSON.
    
    Args:
        filename: Nome do arquivo JSON a ser processado
        
    Returns:
        Lista de documentos divididos
    """
    # Caminho completo para o arquivo
    file_path = os.path.join(data_folder, filename)
    
    print(f"Processando arquivo: {file_path}")
    
    # Lê o arquivo JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Divisão dos documentos (split documents):
    # Após a obtenção e formatação dos dados dos produtos, os documentos são divididos em partes menores.
    # Essa divisão facilita o processamento pelo modelo de linguagem, permitindo que cada pedaço seja analisado
    # de maneira mais eficaz.
    # Documentação: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/
    splitter = RecursiveJsonSplitter(max_chunk_size=300)
    doc_splits = splitter.split_json(json_data=json_data)
    
    return doc_splits


def upload_to_supabase(doc_splits: List[Dict], collection_name: str = None, clear_collection: bool = False):
    """
    Faz upload dos documentos divididos para o Supabase
    
    Args:
        doc_splits: Lista de documentos divididos
        collection_name: Nome da coleção (opcional)
        clear_collection: Se True, limpa a coleção antes de inserir novos documentos
    """
    # Cria embeddings utilizando o modelo do Google
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )
    
    # Se não for fornecido um nome de coleção, usa o quarto da lista em SUPABASE_COLLECTION_NAME
    if collection_name is None:
        collection_name = SUPABASE_COLLECTION_NAME.split(",")[3]
    
    print(f"Conectando ao Supabase: {SUPABASE_URL}")
    print(f"Usando tabela: {SUPABASE_TABLE_NAME} com a collection: {collection_name}")

    # Cria cliente Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    try:
        # Verificar se a tabela existe usando supabase.table().select()
        print(f"Verificando acesso à tabela {SUPABASE_TABLE_NAME}...")
        
        # Tentar acessar a tabela para ver se existe
        table_check = supabase.table(SUPABASE_TABLE_NAME).select("id").limit(1).execute()
        print(f"Tabela {SUPABASE_TABLE_NAME} encontrada.")
        
        # Limpar dados existentes para evitar duplicatas (apenas se clear_collection for True)
        if clear_collection:
            print(f"Limpando dados existentes da coleção {collection_name}...")
            supabase.table(SUPABASE_TABLE_NAME).delete().eq('collection_name', collection_name).execute()
        
        # Agora inserimos os documentos como embeddings
        print("Inserindo documentos como embeddings...")
        
        for i, doc in enumerate(doc_splits):
            # Converter o documento (dicionário) para string JSON
            doc_str = json.dumps(doc, ensure_ascii=False)
            
            # Obter embedding para o documento
            doc_embedding = embeddings.embed_query(doc_str)
            
            # Inserir no Supabase
            data = {
                "content": doc_str,
                "embedding": doc_embedding,
                "collection_name": collection_name
            }
            
            result = supabase.table(SUPABASE_TABLE_NAME).insert(data).execute()
            
            if (i + 1) % 10 == 0 or i + 1 == len(doc_splits):
                print(f"Progresso: {i + 1}/{len(doc_splits)} documentos inseridos")
        
        print(f"Total de {len(doc_splits)} documentos inseridos com sucesso na coleção {collection_name}!")
        
    except Exception as e:
        print(f"Erro ao interagir com o Supabase: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Função principal que processa todos os arquivos JSON na pasta de dados
    """
    # Verifica se a pasta de dados existe
    if not os.path.exists(data_folder):
        print(f"Pasta de dados '{data_folder}' não encontrada.")
        return
    
    # Lista todos os arquivos JSON na pasta de dados
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"Nenhum arquivo JSON encontrado na pasta '{data_folder}'.")
        return
    
    print(f"Encontrados {len(json_files)} arquivos JSON para processar.")
    
    # Definir o nome da coleção
    collection_name = SUPABASE_COLLECTION_NAME.split(",")[3]  # json_split
    
    # Limpar a coleção apenas uma vez no início do processo
    try:
        # Cria cliente Supabase
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"Limpando dados existentes da coleção {collection_name} antes de iniciar o processamento...")
        supabase.table(SUPABASE_TABLE_NAME).delete().eq('collection_name', collection_name).execute()
        print(f"Coleção {collection_name} limpa com sucesso.")
    except Exception as e:
        print(f"Erro ao limpar a coleção: {e}")
        import traceback
        traceback.print_exc()
    
    # Processa cada arquivo JSON
    for filename in json_files:
        doc_splits = process_json_file(filename)
        # Não limpar a coleção para cada arquivo (clear_collection=False)
        upload_to_supabase(doc_splits, collection_name=collection_name, clear_collection=False)


if __name__ == "__main__":
    main()