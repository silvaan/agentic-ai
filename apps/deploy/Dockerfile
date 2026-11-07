FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Por padrão a imagem sobe a API; o front usa a mesma imagem com outro comando.
# O Render define a porta em $PORT.
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}
