"""Chat com Streamlit.

    streamlit run apps/web.py
"""

import streamlit as st

from agentkit import LLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Você é um assistente prestativo. Responda em poucas frases."


# st.cache_resource guarda o objeto entre as reexecuções do script. O script
# inteiro roda de novo a cada interação, e sem o cache o modelo seria carregado
# outra vez a cada mensagem.
@st.cache_resource
def load_model():
    return LLM(MODEL_NAME, temperature=0.7, max_tokens=200)


def chat(llm, history, message):
    """Executa um turno e acrescenta as duas mensagens ao histórico."""
    history.append({"role": "user", "content": message})
    messages = [{"role": "system", "content": SYSTEM}, *history]
    answer = llm.invoke(messages)
    history.append({"role": "assistant", "content": answer})
    return answer


llm = load_model()

# st.title escreve o título da página.
st.title("Assistente")

# st.session_state é o dicionário que sobrevive às reexecuções. Uma lista comum
# voltaria vazia a cada mensagem enviada.
if "history" not in st.session_state:
    st.session_state.history = []

# st.sidebar coloca o componente na barra lateral, e o botão devolve True na
# reexecução causada pelo clique.
if st.sidebar.button("Limpar conversa"):
    st.session_state.history = []

# st.chat_message desenha um balão com o avatar do papel. A tela é reconstruída
# do zero a cada reexecução, então o histórico inteiro é redesenhado aqui.
for message in st.session_state.history:
    st.chat_message(message["role"]).write(message["content"])

# st.chat_input fixa a caixa de entrada no rodapé e devolve None enquanto nada
# for enviado.
message = st.chat_input("Sua mensagem")
if message:
    st.chat_message("user").write(message)
    # st.spinner mostra o indicador de espera enquanto o bloco de dentro roda.
    with st.spinner("pensando..."):
        answer = chat(llm, st.session_state.history, message)
    st.chat_message("assistant").write(answer)
