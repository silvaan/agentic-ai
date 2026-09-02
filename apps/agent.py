"""Agente no terminal, com ferramentas e várias sessões.

    python apps/agent.py
"""

from datetime import datetime

from agentkit import LLM, run_agent, tool

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Você é um assistente prestativo. Responda em poucas frases."


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, such as 12 * (3 + 4)."""
    return str(eval(expression, {"__builtins__": {}}, {}))


@tool
def current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%d/%m/%Y %H:%M")


@tool
def count_words(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


TOOLS = [calculate, current_time, count_words]


def chat(llm, history, message):
    """Executa um turno com ferramentas e acrescenta as mensagens ao histórico."""
    # O run_agent recebe uma pergunta, e não uma conversa: o histórico vai como
    # texto no contexto.
    context = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    result = run_agent(llm, message, tools=TOOLS, system_prompt=SYSTEM, context=context or None)
    answer = result["answer"] or "não cheguei a uma resposta dentro do limite de passos"
    history += [{"role": "user", "content": message},
                {"role": "assistant", "content": answer}]
    return answer, result["trace"]


def show_trace(trace):
    """Imprime os passos do modelo e as chamadas de ferramenta do último turno."""
    for item in trace:
        if item["type"] == "tool":
            print(f"  {item['name']}({item['arguments']}) -> {item['observation']}")
        else:
            print(f"  modelo: {' '.join(item['content'].split())[:70]}")


def main():
    print(f"carregando {MODEL_NAME}...")
    llm = LLM(MODEL_NAME, temperature=0.7, max_tokens=200)
    # Cada sessão tem o seu histórico, e o nome é a chave que separa uma da outra.
    sessions = {"geral": []}
    current = "geral"
    trace = []
    print(f"pronto em {llm.device}")
    print("comandos: /sessao <nome>, /sessoes, /traco, /limpar, /sair\n")

    while True:
        try:
            message = input(f"{current}> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-C ou fim da entrada encerram o programa.
            print()
            break

        if not message:
            continue
        if message == "/sair":
            break
        if message.startswith("/sessao "):
            # A sessão é criada na primeira vez que o nome é usado.
            current = message.removeprefix("/sessao ").strip()
            sessions.setdefault(current, [])
            print(f"{len(sessions[current])} mensagens nesta sessão")
            continue
        if message == "/sessoes":
            for name, history in sessions.items():
                print(f"  {name}: {len(history)} mensagens")
            continue
        if message == "/traco":
            show_trace(trace)
            continue
        if message == "/limpar":
            sessions[current] = []
            print("histórico apagado")
            continue

        answer, trace = chat(llm, sessions[current], message)
        print("assistente>", answer)

    print(f"até logo | {len(sessions)} sessões descartadas")


if __name__ == "__main__":
    main()
