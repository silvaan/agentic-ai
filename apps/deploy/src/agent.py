"""Agente da livraria: junta modelo, ferramentas, prompt e memória por conversa."""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from src.config import MODEL
from src.prompts import SYSTEM
from src.tools import get_stock, search_books

agent = create_agent(
    init_chat_model(MODEL, temperature=0.0),
    tools=[search_books, get_stock],
    system_prompt=SYSTEM,
    checkpointer=InMemorySaver(),
)


def answer(message: str, thread_id: str) -> str:
    """Envia uma mensagem na conversa indicada e devolve a resposta do agente."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
    return result["messages"][-1].text
