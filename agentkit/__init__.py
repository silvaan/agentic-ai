"""agentkit: biblioteca de referência da disciplina de IA Agêntica, Unidade I.

Ela não é um framework. Foi escrita para ser lida, inclusive com
inspect.getsource dentro do notebook.
"""

import os

# O transformers importa o TensorFlow quando ele está instalado, e ele registra
# avisos de compilação no primeiro import. Silenciar aqui evita que a saída da
# primeira célula de qualquer notebook comece com eles.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from .agent import Agent, run_agent
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
    "Agent",
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
