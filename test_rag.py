#!/usr/bin/env python3
from rag import rag_query

# Exemplos de consultas para testar
queries = [
    "Quais são as características de um blazer feminino em couro?",
    "Quanto é o preço de uma calça jeans? O que combinar com ela?",
    "Existe algum vestido em promoção?",
    "docs VF003"
]

# Testar a consulta específica sobre calça jeans
print("\n\n======== TESTANDO RAG QUERY ========\n")
result = rag_query(queries[3])
print(result)
print("\n\n======== FIM DO TESTE ========\n")
