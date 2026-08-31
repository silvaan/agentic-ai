"""Assistente de conversa que roda no terminal.

Este arquivo existe para mostrar o que um notebook não mostra: o programa fica
esperando a próxima entrada e remonta o prompt a cada volta. O histórico vive
em memória e morre com o processo — nada é gravado em disco.

    python apps/chat.py
"""

from __future__ import annotations

from agentkit import LLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Você é um assistente prestativo. Responda em poucas frases."


def build_prompt(history: list[dict]) -> list[dict]:
    """Monta as mensagens da chamada: a de sistema mais o histórico inteiro.

    A mensagem de sistema é escrita a cada chamada e nunca entra no histórico,
    para que ela não dependa do que a conversa fez com a lista.
    """
    return [{"role": "system", "content": SYSTEM}, *history]


def turn(llm: LLM, history: list[dict], message: str) -> str:
    """Executa um turno e acrescenta as duas mensagens ao histórico."""
    history.append({"role": "user", "content": message})
    answer = llm.invoke(build_prompt(history))
    history.append({"role": "assistant", "content": answer})
    return answer


def show_history(history: list[dict]) -> None:
    """Mostra o que o modelo recebe de volta a cada chamada."""
    print(f"{len(history)} mensagens no histórico")
    for message in history:
        content = " ".join(message["content"].split())
        print(f"  {message['role']}: {content[:70]}")


def main() -> None:
    print(f"carregando {MODEL_NAME}...")
    llm = LLM(MODEL_NAME, temperature=0.7, max_tokens=200)
    history: list[dict] = []
    print(f"pronto em {llm.device}")
    print("comandos: /historico, /limpar, /sair\n")

    while True:
        try:
            message = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            # EOF encerra uma entrada canalizada, e Ctrl-C encerra o terminal.
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

        print("assistente>", turn(llm, history, message))

    print(f"até logo | {len(history)} mensagens perdidas com o processo")


if __name__ == "__main__":
    main()
