# Livraria Sebo Azul

Agente mínimo pronto para produção: o atendente da Livraria Sebo Azul responde perguntas sobre o catálogo, com memória por conversa. O agente fica atrás de uma API em FastAPI, e a interface em Streamlit conversa com ele pela API.

```text
Streamlit (app.py) ──HTTP──> FastAPI (src/api.py) ──> agente (src/agent.py) ──> ferramentas (src/tools.py)
```

## Estrutura

```text
├── app.py              front em Streamlit
├── src/
│   ├── config.py       variáveis de ambiente
│   ├── prompts.py      mensagem de sistema
│   ├── tools.py        ferramentas sobre o catálogo
│   ├── agent.py        monta o agente e expõe answer()
│   └── api.py          API com /chat e /health
├── data/books.json     catálogo
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── .env.example
```

## Como rodar

Requer Python 3.11 ou superior.

```bash
make install
cp .env.example .env    # preencha OPENAI_API_KEY
make run                # API em localhost:8000 e front em localhost:8501
```

`make api` e `make front` sobem cada parte separadamente. A documentação da API fica em http://localhost:8000/docs.

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "Quanto custa Duna?", "thread_id": "teste"}'
```

## Docker

```bash
make docker
```

Sobe a API e o front em dois contêineres a partir da mesma imagem, com as variáveis do `.env`.

## Deploy no Render

1. Crie um **Web Service** a partir do repositório, com Root Directory `apps/deploy` e runtime Docker. Cadastre em Environment as variáveis do `.env`. Esse serviço é a API.
2. Crie um segundo **Web Service** com o mesmo repositório e a mesma pasta, e preencha Docker Command com `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`. Cadastre `API_URL` com o endereço da API criada no passo 1.

A chave da API fica só no serviço da API. O front não precisa dela.

## Observabilidade

Com `LANGSMITH_TRACING=true` e uma `LANGSMITH_API_KEY` no `.env`, cada execução do agente vira um traço no LangSmith, com as chamadas ao modelo, as ferramentas, os tokens e a latência. Nenhuma linha de código muda.

## O que muda em produção

- **Memória**: o `InMemorySaver` guarda as conversas na memória do processo. Elas somem quando o servidor reinicia e não são compartilhadas entre instâncias. Em produção, troque por um checkpointer com banco, como o `PostgresSaver`.
- **Limite de uso**: a API aceita até `MAX_MESSAGES` mensagens por conversa. Como o `thread_id` vem do cliente, o limite segura o uso normal, mas não um abuso deliberado; para isso, a API precisa de autenticação e de limite por usuário.
- **Segredos**: as chaves ficam em variáveis de ambiente e o `.env` está no `.gitignore` e no `.dockerignore`.
