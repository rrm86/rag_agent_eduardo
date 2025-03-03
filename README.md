# RAG Agent: Sistema de Recuperação Aumentada por Geração

Este projeto implementa um sistema de Recuperação Aumentada por Geração (RAG), uma técnica poderosa que combina a capacidade de recuperação de informações com a geração de texto por modelos de linguagem.

## O que é RAG?

RAG (Retrieval-Augmented Generation) é uma técnica que melhora as respostas dos modelos de linguagem (LLMs) com informações externas relevantes. O processo envolve três etapas principais:

1. **Armazenamento de Documentos**: Conversão de textos em representações vetoriais
2. **Recuperação de Informações**: Busca de documentos relevantes para uma consulta específica
3. **Geração Aumentada**: Uso dos documentos recuperados para contextualizar a resposta do LLM

## Estrutura do Projeto

- `main.py`: Ponto de entrada do aplicativo
- `rag.py`: Implementação do sistema RAG com FAISS
- `tools.py`: Ferramentas disponíveis para o agente
- `llm.py`: Factory para instanciação de modelos de linguagem
- `config.py`: Configurações e chaves de API

## Componentes Essenciais do RAG

### 1. Dados e Documentos

Os dados são o alicerce do sistema RAG. Neste projeto, você aprenderá:

- Como preparar documentos para inserção na base de conhecimento
- Técnicas de chunking (divisão de textos em partes menores)
- Processamento e normalização de texto
- Considerações sobre qualidade e relevância dos dados

### 2. Vectorstore (FAISS)

A vectorstore é o coração do RAG. Você aprenderá:

- Como embeddings transformam texto em vetores numéricos
- Métodos de inserção de documentos na FAISS
- Estratégias de indexação e armazenamento eficiente
- Técnicas de busca vetorial (similaridade de cosseno)

### 3. Recuperação

A recuperação eficaz é crucial para o sucesso do RAG:

- Otimização de consultas para recuperação precisa
- Métricas de similaridade e seus impactos (conversão de distância L2 para similaridade de cosseno)
- Como determinar o valor ideal de k (número de documentos a recuperar)
- **Oportunidades de melhoria**: implementação de filtragem e reordenação de resultados

### 4. Prompts

Os prompts determinam a qualidade das respostas:

- Estruturação de prompts para integrar contexto recuperado
- Técnicas de engenharia de prompts para RAG
- Balanceamento entre informação externa e conhecimento do modelo
- Estratégias para lidar com informações contraditórias

## Como Usar

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Altere o arquivo `config_example.py` para `config.py` e adicione sua chave de API

3. Rode o loader: Na pasta index, execute: `python load.py`. Para verificar se o load foi bem-sucedido, execute `python view_faiss_data.py`
4. Execute o sistema:
Volte para a pasta raiz e execute:

```bash
python main.py
```

5. Se quiser, execute teste da RAG com: `python test_rag.py`

## Exercícios Práticos

Este projeto é ideal para praticar:

1. **Explorar ferramentas**: Verifique as ferramentas disponíveis e descubra como elas podem ser úteis"
2. **Recuperação**: Teste a recuperação com diferentes tipos de consultas: "O que você sabe sobre FAISS?"
3. **Engenharia de Prompts**: Modifique o template RAG_PROMPT_TEMPLATE em `rag.py` para otimizar as respostas
4. **Medição de Desempenho**: Avalie a qualidade das respostas baseada nas suas necessidades. Anote quando o sistema não responder de acordo e identifique oportunidade de melhoria nos dados.

## Conclusão

O RAG representa uma evolução significativa na aplicação de LLMs, combinando a precisão da recuperação de informações com a fluência da geração de texto. Este projeto serve como um laboratório para experimentar e compreender cada componente deste poderoso paradigma.
