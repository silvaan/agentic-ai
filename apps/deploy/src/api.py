"""API do agente. O front e qualquer outro cliente conversam com o agente por aqui.

    uvicorn src.api:app --port 8000
"""

from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import answer
from src.config import MAX_MESSAGES

app = FastAPI(title="Livraria Sebo Azul")

# Mensagens recebidas por conversa. O limite fica na API, e não no front,
# porque qualquer pessoa pode chamar a API diretamente.
usage = Counter()


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(min_length=1, max_length=100)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: Message) -> dict:
    """Responde a uma mensagem; cada thread_id é uma conversa com memória própria."""
    if usage[request.thread_id] >= MAX_MESSAGES:
        raise HTTPException(429, "Limite de mensagens desta conversa atingido.")
    usage[request.thread_id] += 1
    return {"answer": answer(request.message, request.thread_id)}
