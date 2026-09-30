"""Chat com o agente de notas fiscais.

    streamlit run apps/invoices/app.py

O banco invoices.db é gerado pelo ingest.ipynb, na mesma pasta.
"""

import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

DATABASE = Path(__file__).parent / "invoices.db"


@tool
def run_select(query: str) -> str:
    """Roda uma consulta SELECT no banco de notas fiscais e devolve até 50 linhas."""
    if not query.strip().lower().startswith("select"):
        return "erro: só consultas SELECT são permitidas"
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
        try:
            cursor = connection.execute(query)
        except (sqlite3.Error, sqlite3.Warning) as error:
            return f"erro: {error}"
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(50)
    return "\n".join([" | ".join(columns)] + [" | ".join(map(str, row)) for row in rows])


# st.cache_resource guarda o objeto entre as reexecuções do script. O script
# inteiro roda de novo a cada interação, e sem o cache o agente e o
# checkpointer, que guarda as conversas, seriam recriados a cada mensagem.
@st.cache_resource
def load_agent():
    with sqlite3.connect(DATABASE) as connection:
        schema = "\n\n".join(sql for (sql,) in connection.execute("SELECT sql FROM sqlite_master WHERE type = 'table'"))
    system = f"""Você responde perguntas sobre as notas fiscais do usuário, guardadas em um banco SQLite com este esquema:

{schema}

Consulte os dados com run_select e nunca invente valores. Os valores estão em euros.
Os nomes dos estabelecimentos vêm de OCR e variam na grafia, então liste os valores distintos antes de filtrar por nome.
Responda em uma ou duas frases."""
    model = init_chat_model("openai:gpt-4.1-mini", temperature=0.0)
    return create_agent(model, tools=[run_select], system_prompt=system, checkpointer=InMemorySaver())


agent = load_agent()

st.title("Notas fiscais")

# st.session_state é o dicionário que sobrevive às reexecuções. Aqui ele guarda
# só o id da thread, porque as mensagens ficam no checkpointer do agente.
# O botão da barra lateral devolve True na reexecução causada pelo clique.
new_conversation = st.sidebar.button("Nova conversa")
if new_conversation or "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# A tela é reconstruída do zero a cada reexecução, então a conversa inteira é
# redesenhada a partir do estado da thread. As consultas pedidas pelo modelo
# ficam em um st.expander acima da resposta que elas produziram.
queries = []
for message in agent.get_state(config).values.get("messages", []):
    if message.type == "human":
        st.chat_message("user").write(message.content)
    elif message.type == "ai" and message.tool_calls:
        queries += [call["args"]["query"] for call in message.tool_calls]
    elif message.type == "ai":
        with st.chat_message("assistant"):
            if queries:
                with st.expander(f"{len(queries)} consulta(s)"):
                    st.code("\n\n".join(queries), language="sql")
            st.write(message.content)
        queries = []

# st.chat_input fixa a caixa de entrada no rodapé e devolve None enquanto nada
# for enviado. Depois do agente responder, st.rerun roda o script de novo, e o
# laço acima desenha a pergunta e a resposta junto com o resto da conversa.
if request := st.chat_input("Pergunte sobre suas notas"):
    st.chat_message("user").write(request)
    with st.spinner("consultando..."):
        agent.invoke({"messages": [HumanMessage(request)]}, config)
    st.rerun()
