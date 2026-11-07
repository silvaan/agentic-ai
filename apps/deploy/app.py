"""Interface de chat da livraria, que conversa com o agente pela API.

    streamlit run app.py
"""

import uuid

import requests
import streamlit as st

from src.config import API_URL

st.title("Livraria Sebo Azul")

if "thread_id" not in st.session_state or st.sidebar.button("Nova conversa"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input("Pergunte sobre um livro"):
    st.chat_message("user").write(prompt)
    response = requests.post(f"{API_URL}/chat", json={"message": prompt, "thread_id": st.session_state.thread_id}, timeout=120)
    if response.ok:
        reply = response.json()["answer"]
        st.chat_message("assistant").write(reply)
        st.session_state.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
    else:
        st.error(response.json()["detail"])
