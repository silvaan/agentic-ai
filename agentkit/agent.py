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

        result = run_tool(call, tools)
        trace.append(
            {
                "step": step,
                "type": "tool",
                "name": result["name"],
                "arguments": call["arguments"],
                "output": result["output"],
                "error": result["error"],
            }
        )
        observation = result["error"] if result["error"] else result["output"]
        messages.append(
            {
                "role": "tool",
                "name": result["name"],
                "content": str(observation),
            }
        )

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
