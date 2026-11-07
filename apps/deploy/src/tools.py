"""Ferramentas do agente, que consultam o catálogo em data/books.json."""

import json
from pathlib import Path

from langchain.tools import tool

BOOKS = json.loads((Path(__file__).parents[1] / "data" / "books.json").read_text())


@tool
def search_books(query: str) -> list[dict]:
    """Busca livros no catálogo por título ou autor e devolve id, título, autor e preço."""
    words = set(query.lower().split())
    found = []
    for id, book in BOOKS.items():
        if words & set(f"{book['title']} {book['author']}".lower().split()):
            found.append({"id": id, "title": book["title"], "author": book["author"], "price": book["price"]})
    return found


@tool
def get_stock(book_id: str) -> int | str:
    """Devolve a quantidade em estoque de um livro pelo id."""
    if book_id not in BOOKS:
        return "Livro não encontrado."
    return BOOKS[book_id]["stock"]
