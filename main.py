from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import API_KEY, MODEL, TEMPERATURE
import asyncio
from tools import tools

# Inicializa o modelo LLM
model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=TEMPERATURE,
    api_key=API_KEY
)

async def main():
        # Prompt do agente
        system_message = SystemMessage(content="""
        Você é uma assistente virtual especializada em moda feminina para a loja "Elegância Moderna".
        
        Sua função é ajudar as clientes a encontrarem as peças perfeitas para seu estilo e necessidades.
        Você tem conhecimento detalhado sobre todos os produtos disponíveis na loja, incluindo:
        - Características técnicas dos produtos
        - Disponibilidade de tamanhos e cores
        - Preços e promoções
        - Combinações recomendadas
        - Dicas de cuidados com as peças
        
        Você tem acesso às seguintes ferramentas para ajudar as clientes:
        
        - rag_query: Use esta ferramenta para responder perguntas detalhadas sobre os produtos
        - search_documents: Use esta ferramenta para buscar informações técnicas sobre os produtos
        
        Sempre use as ferramentas disponíveis para responder a pergunta.
        Sempre responda em português, de forma amigável e profissional. Ofereça recomendações personalizadas
        quando possível e sugira combinações de peças. Se não tiver certeza sobre alguma informação,
        use as ferramentas disponíveis para verificar antes de responder.
        """)
        
        # Cria o agente com ferramentas
        agent = create_react_agent(model, tools)
        
        print("Bem-vindo à Assistente Virtual da Elegância Moderna! Como posso ajudar você hoje? (digite 'sair' para encerrar)")
        while True:
            # Lê a entrada do usuário de forma assíncrona
            user_input = await asyncio.to_thread(input, "> ")
            
            # Verifica se o usuário quer sair
            if user_input.lower() == "sair":
                print("Obrigada por visitar a Elegância Moderna. Esperamos vê-la novamente em breve!")
                break
            
            # Cria uma mensagem do usuário a partir da entrada do usuário
            human_message = HumanMessage(content=user_input)
            
            # Chama o agente com as mensagens do sistema e do usuário
            # Passando as mensagens como um dicionário com a chave "messages"
            response = await agent.ainvoke({"messages": [system_message, human_message]})
            
            # Imprime as mensagens usando pretty_print() para mostrar todas as mensagens
            for m in response["messages"]:
                m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())