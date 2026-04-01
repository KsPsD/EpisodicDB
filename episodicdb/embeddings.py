"""Thin helpers for generating embeddings from popular providers.

Each function returns a list[float] ready for record_episode(embedding=...).
Providers are imported lazily so the main package has no hard dependency on them.
"""

from __future__ import annotations

from episodicdb.schema import EMBEDDING_DIM


def openai(
    text: str,
    model: str = "text-embedding-3-small",
    dimensions: int = EMBEDDING_DIM,
) -> list[float]:
    """Generate an embedding using OpenAI's API.

    Requires: pip install openai
    """
    try:
        import openai as _openai
    except ImportError:
        raise ImportError("pip install openai  — required for OpenAI embeddings")

    response = _openai.embeddings.create(model=model, input=text, dimensions=dimensions)
    return response.data[0].embedding


def voyage(
    text: str,
    model: str = "voyage-3",
) -> list[float]:
    """Generate an embedding using Voyage AI's API.

    Requires: pip install voyageai
    """
    try:
        import voyageai
    except ImportError:
        raise ImportError("pip install voyageai  — required for Voyage embeddings")

    client = voyageai.Client()
    result = client.embed([text], model=model)
    return result.embeddings[0]


def ollama(
    text: str,
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> list[float]:
    """Generate an embedding using a local Ollama instance.

    Requires: Ollama running locally (no pip install needed).
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("pip install httpx  — required for Ollama embeddings")

    response = httpx.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": text},
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]
