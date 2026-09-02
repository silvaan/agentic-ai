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

# HTTP não guarda estado: o histórico de cada sessão fica aqui e some quando o
# servidor para.
sessions = {}

app = FastAPI()


# O FastAPI valida o JSON do corpo da requisição por estes tipos.
class Request(BaseModel):
    session_id: str
    message: str


def chat(history, message):
    """Executa um turno e acrescenta as duas mensagens ao histórico."""
    history.append({"role": "user", "content": message})
    messages = [{"role": "system", "content": SYSTEM}, *history]
    answer = llm.invoke(messages)
    history.append({"role": "assistant", "content": answer})
    return answer


@app.post("/chat")
def send(request: Request):
    history = sessions.setdefault(request.session_id, [])
    answer = chat(history, request.message)
    return {"answer": answer, "messages": len(history)}


@app.get("/history/{session_id}")
def history(session_id: str):
    return {"messages": sessions.get(session_id, [])}


@app.delete("/history/{session_id}")
def clear(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": session_id}
