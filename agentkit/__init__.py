"""agentkit: biblioteca de referência da disciplina de IA Agêntica, Unidade I.

Ela não é um framework. Foi escrita para ser lida, inclusive com
inspect.getsource dentro do notebook.
"""

from .agent import run_agent
from .memory import (
    add_message,
    forget,
    recall,
    recent_messages,
    remember,
    summarize_messages,
)
from .model import LLM, Embeddings
from .tools import (
    TOOL_SYSTEM_PROMPT,
    parse_tool_call,
    render_tools,
    run_tool,
    tool,
)

__all__ = [
    "LLM",
    "Embeddings",
    "tool",
    "render_tools",
    "parse_tool_call",
    "run_tool",
    "TOOL_SYSTEM_PROMPT",
    "add_message",
    "recent_messages",
    "summarize_messages",
    "remember",
    "recall",
    "forget",
    "run_agent",
]
