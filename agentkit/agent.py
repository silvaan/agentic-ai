"""O laço do agente.

Uma função só. A sequência precisa ser legível de cima para baixo: pergunta,
contexto, chamada ao modelo, parsing, ferramenta, observação, nova chamada,
resposta final.
"""

from __future__ import annotations

import time

from .model import LLM
from .tools import TOOL_SYSTEM_PROMPT, parse_tool_call, render_tools, run_tool

DEFAULT_SYSTEM_PROMPT = "You are a concise and helpful assistant."


class Agent:
    """Modelo com ferramentas ligadas e o laço que alterna chamada e execução.

    Usa o protocolo de ferramentas do template de conversa do modelo. Para
    modelos sem esse protocolo, use run_agent, que escreve o protocolo textual
    no prompt de sistema.
    """

    def __init__(self, llm: LLM, tools: list, max_steps: int = 5) -> None:
        self.llm = llm.bind_tools(tools)
        self.tools = {fn.tool_schema["name"]: fn for fn in tools}
        self.max_steps = max_steps

    def run(self, input: str | list[dict]) -> list[dict]:
        """Responde à pergunta, ou continua a conversa, e devolve o histórico."""
        messages = [{"role": "user", "content": input}] if isinstance(input, str) else list(input)
        for _ in range(self.max_steps):
            message = self.llm.invoke(messages)
            messages.append(message)
            if "tool_calls" not in message:
                return messages
            for call in message["tool_calls"]:
                observation = run_tool(call, self.tools)
                messages.append({"role": "tool", "name": call["name"], "content": observation})
        return messages + [{"role": "assistant", "content": "limite de passos atingido"}]


def run_agent(
    llm: LLM,
    question: str,
    tools: tuple | list = (),
    system_prompt: str | None = None,
    context: str | None = None,
    max_steps: int = 5,
    max_tokens: int | None = None,
    max_seconds: float | None = None,
) -> dict:
    """Executa o laço de raciocínio e ação até a resposta final ou o fim do orçamento.

    O orçamento é argumento, não classe: max_tokens soma o uso acumulado desta
    execução e max_seconds mede o tempo decorrido, ambos verificados no topo de
    cada passo. Devolve answer, messages, trace, stop_reason e usage.
    """
    tools = list(tools)
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    if tools:
        system = f"{system}\n\n{TOOL_SYSTEM_PROMPT.format(tools=render_tools(tools))}"

    messages = [{"role": "system", "content": system}]
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}"})
    messages.append({"role": "user", "content": question})

    trace: list[dict] = []
    answer: str | None = None
    stop_reason = "max_steps"
    tokens_used = 0
    started = time.perf_counter()

    for step in range(max_steps):
        if max_seconds is not None and time.perf_counter() - started >= max_seconds:
            stop_reason = "timeout"
            break
        if max_tokens is not None and tokens_used >= max_tokens:
            stop_reason = "token_budget"
            break

        text = llm.invoke(messages)
        messages.append({"role": "assistant", "content": text})
        trace.append({"step": step, "type": "model", "content": text, **llm.last_usage})
        tokens_used += llm.last_usage.get("tokens_in", 0) + llm.last_usage.get("tokens_out", 0)

        call = parse_tool_call(text)
        if call is None:
            answer = text.strip()
            stop_reason = "final_answer"
            break

        observation = run_tool(call, {fn.tool_schema["name"]: fn for fn in tools})
        trace.append(
            {
                "step": step,
                "type": "tool",
                "name": call["name"],
                "arguments": call["arguments"],
                "observation": observation,
            }
        )
        messages.append({"role": "tool", "name": call["name"], "content": observation})

    usage = {
        "tokens_in": sum(item.get("tokens_in", 0) for item in trace if item["type"] == "model"),
        "tokens_out": sum(item.get("tokens_out", 0) for item in trace if item["type"] == "model"),
        "seconds": round(time.perf_counter() - started, 3),
        "steps": sum(1 for item in trace if item["type"] == "model"),
    }
    return {
        "answer": answer,
        "messages": messages,
        "trace": trace,
        "stop_reason": stop_reason,
        "usage": usage,
    }
