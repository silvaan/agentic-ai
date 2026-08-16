"""Memória do agente, escrita sobre listas e dicionários.

As duas estruturas são construídas pelo próprio usuário no notebook:

    conversation = []
    vector_memory = {"texts": [], "vectors": [], "metadata": []}

Não existe classe de memória. Toda função recebe a estrutura como argumento e a
modifica no lugar ou devolve uma nova lista.
"""

from __future__ import annotations

import time

import numpy as np


def add_message(history: list, role: str, content: str) -> None:
    """Acrescenta uma mensagem ao histórico, no lugar."""
    history.append({"role": role, "content": content})


def recent_messages(history: list, max_messages: int = 10) -> list[dict]:
    """Devolve as últimas mensagens do histórico.

    O corte é por número de mensagens, nunca por turnos. A partir da aula 6 o
    histórico tem observações de ferramenta, e qualquer cálculo que assuma
    alternância entre usuário e assistente quebra.
    """
    return history[-max_messages:] if max_messages > 0 else []


def summarize_messages(llm, messages: list[dict]) -> str:
    """Resume um trecho de conversa em poucas linhas.

    As mensagens são formatadas como texto corrido no estilo papel: conteúdo.
    Passar str(messages) injetaria a representação Python, com chaves e aspas,
    dentro do prompt.
    """
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt = [
        {
            "role": "system",
            "content": "Summarize conversations while preserving facts, names, and numbers.",
        },
        {
            "role": "user",
            "content": f"Summarize the conversation below in at most five lines.\n\n{conversation}",
        },
    ]
    return llm.invoke(prompt)


def remember(memory: dict, embeddings, text: str, metadata: dict | None = None) -> None:
    """Guarda um texto e seu vetor na memória vetorial.

    O carimbo de tempo e o contador de acessos entram nos metadados porque são o
    que forget usa depois para decidir o que podar.
    """
    memory["texts"].append(text)
    memory["vectors"].append(embeddings.embed([text])[0])
    memory["metadata"].append({"created_at": time.time(), "hits": 0, **(metadata or {})})


def recall(memory: dict, embeddings, query: str, k: int = 3) -> list[dict]:
    """Busca os k textos mais parecidos com a consulta, por similaridade de cosseno."""
    if not memory["texts"]:
        return []
    # A matriz é remontada e renormalizada a cada chamada. É ineficiente de
    # propósito: a aula precisa que a operação inteira fique visível aqui.
    matrix = np.vstack(memory["vectors"])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    vector = embeddings.embed([query])[0]
    vector = vector / np.linalg.norm(vector)
    scores = matrix @ vector
    order = np.argsort(scores)[::-1][:k]
    results = []
    for i in order:
        memory["metadata"][i]["hits"] += 1
        memory["metadata"][i]["accessed_at"] = time.time()
        results.append(
            {
                "text": memory["texts"][i],
                "score": float(scores[i]),
                "metadata": memory["metadata"][i],
            }
        )
    return results


def forget(memory: dict, max_items: int = 100, half_life: float = 3600.0) -> int:
    """Poda a memória combinando recência e relevância, e devolve quantos itens saíram.

    A pontuação decai pela meia-vida desde o último acesso e cresce com o número
    de acessos, de modo que um item antigo mas muito consultado sobrevive.
    """
    total = len(memory["texts"])
    if total <= max_items:
        return 0
    now = time.time()
    scores = []
    for i, meta in enumerate(memory["metadata"]):
        age = now - meta.get("accessed_at", meta.get("created_at", now))
        scores.append(((1 + meta.get("hits", 0)) * 0.5 ** (age / half_life), i))
    keep = sorted(index for _, index in sorted(scores, reverse=True)[:max_items])
    for key in ("texts", "vectors", "metadata"):
        memory[key] = [memory[key][i] for i in keep]
    return total - max_items
