"""Agente no terminal, com ferramentas e várias sessões.

    python apps/interfaces/agent.py
"""

from datetime import datetime

from agentkit import LLM, Agent, tool

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


def chat(agent, history, message):
    """Executa um turno com ferramentas e acrescenta as mensagens ao histórico."""
    # O agente recebe a conversa inteira, e não só a pergunta. O prompt de
    # sistema entra a cada turno e fica fora do histórico, que assim guarda
    # apenas o que a sessão trocou.
    conversa = [{"role": "system", "content": SYSTEM}, *history,
                {"role": "user", "content": message}]
    messages = agent.run(conversa)
    # O que o turno produziu começa na pergunta, depois do sistema e do que já
    # estava no histórico.
    turno = messages[1 + len(history):]
    history += turno
    return turno[-1]["content"], turno


def show_trace(turno):
    """Imprime as chamadas de ferramenta e as respostas do último turno."""
    for m in turno:
        if m["role"] == "tool":
            print(f"  {m['name']} -> {m['content']}")
        elif m["role"] == "assistant":
            for call in m.get("tool_calls", []):
                print(f"  modelo: {call['name']}({call['arguments']})")
            if m["content"]:
                print(f"  modelo: {' '.join(m['content'].split())[:70]}")


def main():
    print(f"carregando {MODEL_NAME}...")
    llm = LLM(MODEL_NAME, temperature=0.7, max_tokens=200)
    agent = Agent(llm, TOOLS)
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

        answer, trace = chat(agent, sessions[current], message)
        print("assistente>", answer)

    print(f"até logo | {len(sessions)} sessões descartadas")


if __name__ == "__main__":
    main()
