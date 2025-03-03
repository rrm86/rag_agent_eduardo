from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import json
import sys
from typing import Dict, List, Any

# Adiciona o diretório pai ao sys.path para permitir importação absoluta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY

# Caminho para a pasta de dados
data_folder = "data"

def format_product_data(data: Dict[str, Any]) -> str:
    """
    Formata os dados do produto a partir de um arquivo JSON, convertendo-os em um texto legível.
    Essa função realiza o parseamento dos dados extraindo informações relevantes, como ID, nome, categoria, descrição,
    preços, cores, tamanhos, instruções de cuidado e outras características, para que possam ser facilmente convertidos
    em documentos para indexação e buscas posteriores.

    Parâmetros:
        data: Dados em formato JSON de um produto de vestuário.

    Retorna:
        Texto formatado contendo as informações do produto.
    """
    # Cria a informação do produto
    product_info = []
    
    # Informações básicas do produto
    if 'id' in data:
        product_info.append(f"ID: {data['id']}")
    
    if 'nome' in data:
        product_info.append(f"Nome: {data['nome']}")
    
    if 'categoria' in data:
        product_info.append(f"Categoria: {data['categoria']}")
    
    if 'descricao' in data:
        product_info.append(f"Descrição: {data['descricao']}")
    
    if 'detalhes_tecnicos' in data:
        product_info.append(f"Detalhes Técnicos: {data['detalhes_tecnicos']}")
    
    # Preço e promocao
    if 'preco' in data:
        price_info = f"Preço: R$ {data['preco']:.2f}"
        if data.get('em_promocao', False) and 'preco_promocional' in data:
            price_info += f" (Preço Promocional: R$ {data['preco_promocional']:.2f})"
        product_info.append(price_info)
    
    # Cores e Tamanhos  
    if 'cores_disponiveis' in data:
        product_info.append(f"Cores Disponíveis: {', '.join(data['cores_disponiveis'])}")
    
    if 'tamanhos_disponiveis' in data:
        product_info.append(f"Tamanhos Disponíveis: {', '.join(data['tamanhos_disponiveis'])}")
    
    # Instruções de Cuidado e Sustentabilidade
    if 'instrucoes_cuidado' in data:
        product_info.append(f"Instruções de Cuidado: {data['instrucoes_cuidado']}")
    
    if 'sustentabilidade' in data:
        product_info.append(f"Sustentabilidade: {data['sustentabilidade']}")
    
    # Informações Adicionais
    if 'caracteristicas_adicionais' in data:
        char_info = ["Características Adicionais:"]
        for key, value in data['caracteristicas_adicionais'].items():
            char_info.append(f"  - {key.capitalize()}: {value}")
        product_info.append("\n".join(char_info))
    
    # Tags
    if 'tags' in data:
        product_info.append(f"Tags: {', '.join(data['tags'])}")
    
    # Junta todos os elementos em uma string
    return "\n".join(product_info)

# Realiza o parseamento dos arquivos JSON:
# Essa etapa envolve a leitura dos arquivos JSON na pasta de dados, extraindo e formatando as informações de cada produto.
# O processo de parseamento converte os dados estruturados em um formato textual, possibilitando a criação de documentos
# que serão posteriormente utilizados para indexação e busca.
documents = []
for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)
    if os.path.isfile(file_path) and filename.endswith('.json') and not filename.startswith('.'):
        try:
            # Lê arquivos
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Formata conteúdo
            content = format_product_data(data)
            
            # Cria metadados
            metadata = {
                "source": filename,
                "id": data.get("id", "unknown"),
                "nome": data.get("nome", "unknown"),
                "categoria": data.get("categoria", "unknown")
            }
            
            # Cria documento
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            
            print(f"Loaded {filename}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

# Divisão dos documentos (split documents):
# Após a obtenção e formatação dos dados dos produtos, os documentos são divididos em partes menores.
# Essa divisão facilita o processamento pelo modelo de linguagem, permitindo que cada pedaço seja analisado
# de maneira mais eficaz. Utilizamos o RecursiveCharacterTextSplitter com um tamanho de chunk de 500 caracteres
# e sobreposição de 100 caracteres para manter o contexto entre os pedaços.
# Documentação:https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,  # Tamanho maior de chunk para descrições de produtos
    chunk_overlap=100
)
doc_splits = text_splitter.split_documents(documents)

# Cria embeddings utilizando o modelo do Google
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=API_KEY
)

# Adiciona os documentos divididos ao banco de dados de vetores FAISS
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=embeddings,
)


# Salva o índice FAISS para uso futuro
vectorstore.save_local("faiss_index")