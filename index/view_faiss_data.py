import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY


def view_faiss_data(index_path="faiss_index"):
    """
    Carrega e exibe os documentos do FAISS.
    """
    print(f"Loading FAISS index from {index_path}...")
    
    # Verifica se o índice existe
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}")
        return
    
    # Cria o modelo de embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )
    
    # Carrega o índice FAISS
    vectorstore = FAISS.load_local(
        index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Obtem o store de documentos
    docstore = vectorstore.docstore
    
    # Cria uma lista para armazenar os dados
    rows = []
    
    
    for doc_id, doc in docstore._dict.items():
        rows.append({
            "ID": doc_id[:8] + "...",  
            "Content": doc.page_content,
            "Source": os.path.basename(doc.metadata.get("source", "Unknown"))
        })
    
   
    df = pd.DataFrame(rows)
    
    
    print(f"\nFAISS Index Summary:")
    print(f"Total documents: {len(rows)}")
    print(f"Index location: {os.path.abspath(index_path)}")
    
    
    pd.set_option('display.max_colwidth', 200)  
    pd.set_option('display.max_rows', None)  
    pd.set_option('display.width', 1000)  
    
    print("\nDocument Contents:")
    print(df)
    
    # Imprime os chunks de cada documento
    print("\nDetailed Document Chunks:")
    for i, row in df.iterrows():
        print(f"\n[Chunk {i+1}] ID: {row['ID']}")
        print(f"Source: {row['Source']}")
        print("Content:")
        print("-" * 80)
        print(row['Content'])
        print("-" * 80)
    
    return df

if __name__ == "__main__":
    view_faiss_data()
