"""API de conversa com FastAPI.

    uvicorn apps.api:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from agentkit import LLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Você é um assistente prestativo. Responda em poucas frases."

# Carregado uma vez, quando o servidor sobe, e não a cada requisição.
llm = LLM(MODEL_NAME, temperature=0.7, max_tokens=200)

# HTTP não guarda estado: o histórico de cada sessão fica aqui, e some quando
# o servidor para.
sessions: dict[str, list[dict]] = {}

app = FastAPI()


class Request(BaseModel):
    session_id: str
    message: str


def build_prompt(history: list[dict]) -> list[dict]:
    """Monta as mensagens da chamada: a de sistema mais o histórico inteiro."""
    return [{"role": "system", "content": SYSTEM}, *history]


def turn(history: list[dict], message: str) -> str:
    """Executa um turno e acrescenta as duas mensagens ao histórico."""
    history.append({"role": "user", "content": message})
    answer = llm.invoke(build_prompt(history))
    history.append({"role": "assistant", "content": answer})
    return answer


@app.post("/chat")
def chat(request: Request) -> dict:
    history = sessions.setdefault(request.session_id, [])
    answer = turn(history, request.message)
    return {"answer": answer, "messages": len(history)}


@app.get("/history/{session_id}")
def history(session_id: str) -> dict:
    return {"messages": sessions.get(session_id, [])}


@app.delete("/history/{session_id}")
def clear(session_id: str) -> dict:
    sessions.pop(session_id, None)
    return {"cleared": session_id}
