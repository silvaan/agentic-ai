"""Chat com Streamlit.

    streamlit run apps/web.py
"""

import streamlit as st

from agentkit import LLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Você é um assistente prestativo. Responda em poucas frases."


# O script inteiro reexecuta a cada interação. Sem o cache, o modelo seria
# carregado de novo a cada mensagem enviada.
@st.cache_resource
def load_model() -> LLM:
    return LLM(MODEL_NAME, temperature=0.7, max_tokens=200)


def build_prompt(history: list[dict]) -> list[dict]:
    """Monta as mensagens da chamada: a de sistema mais o histórico inteiro."""
    return [{"role": "system", "content": SYSTEM}, *history]


def turn(llm: LLM, history: list[dict], message: str) -> str:
    """Executa um turno e acrescenta as duas mensagens ao histórico."""
    history.append({"role": "user", "content": message})
    answer = llm.invoke(build_prompt(history))
    history.append({"role": "assistant", "content": answer})
    return answer


llm = load_model()

st.title("Assistente")

# Pelo mesmo motivo do cache: uma lista comum voltaria vazia a cada reexecução.
if "history" not in st.session_state:
    st.session_state.history = []

if st.sidebar.button("Limpar conversa"):
    st.session_state.history = []

for message in st.session_state.history:
    st.chat_message(message["role"]).write(message["content"])

message = st.chat_input("Sua mensagem")
if message:
    st.chat_message("user").write(message)
    with st.spinner("pensando..."):
        answer = turn(llm, st.session_state.history, message)
    st.chat_message("assistant").write(answer)
