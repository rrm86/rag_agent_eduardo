from langchain_core.tools import tool
from rag import rag_query, search_documents

@tool
def agent_tool_debug() -> str:
    """Essa ferramenta é para testar o comportamento do agente. Chame essa ferramenta quando o cliente digitar 'debug_tool'."""
    return "Yes, I can call a tool"


# List of available tools
tools = [agent_tool_debug, rag_query, search_documents]

# Dictionary mapping tool names to tool instances for easy lookup
tools_by_name = {t.name: t for t in tools}
