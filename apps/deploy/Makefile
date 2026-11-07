.PHONY: install api front run docker

install:
	pip install -r requirements.txt

api:
	uvicorn src.api:app --reload --port 8000

front:
	streamlit run app.py

# Sobe a API e o front juntos
run:
	$(MAKE) -j2 api front

docker:
	docker compose up --build
