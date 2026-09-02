"""Chat no terminal. O histórico vive em memória e morre com o processo.

    python apps/chat.py
"""

from agentkit import LLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Você é um assistente prestativo. Responda em poucas frases."


def chat(llm, history, message):
    """Executa um turno e acrescenta as duas mensagens ao histórico."""
    history.append({"role": "user", "content": message})
    # A mensagem de sistema é montada a cada chamada e não entra no histórico.
    messages = [{"role": "system", "content": SYSTEM}, *history]
    answer = llm.invoke(messages)
    history.append({"role": "assistant", "content": answer})
    return answer


def show_history(history):
    """Imprime o histórico atual."""
    print(f"{len(history)} mensagens no histórico")
    for message in history:
        content = " ".join(message["content"].split())
        print(f"  {message['role']}: {content[:70]}")


def main():
    print(f"carregando {MODEL_NAME}...")
    llm = LLM(MODEL_NAME, temperature=0.7, max_tokens=200)
    history = []
    print(f"pronto em {llm.device}")
    print("comandos: /historico, /limpar, /sair\n")

    while True:
        try:
            message = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-C ou fim da entrada encerram o programa.
            print()
            break

        if not message:
            continue
        if message == "/sair":
            break
        if message == "/historico":
            show_history(history)
            continue
        if message == "/limpar":
            history.clear()
            print("histórico apagado")
            continue

        print("assistente>", chat(llm, history, message))

    print(f"até logo | {len(history)} mensagens descartadas")


if __name__ == "__main__":
    main()
