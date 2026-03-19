from __future__ import annotations

from typing import Iterable

from openai import OpenAI

from config import (
    get_openai_api_key,
    get_openai_max_retries,
    get_openai_model,
    get_openai_timeout_seconds,
)


class Embedder:
    def __init__(self, model: str | None = None) -> None:
        api_key = get_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env file.")

        self.model = model or get_openai_model()
        self.client = OpenAI(
            api_key=api_key,
            timeout=get_openai_timeout_seconds(),
            max_retries=get_openai_max_retries(),
        )

    def embed_text(self, text: str) -> list[float]:
        clean_text = str(text).strip()
        if not clean_text:
            raise ValueError("Cannot embed empty text.")

        response = self.client.embeddings.create(
            model=self.model,
            input=clean_text,
        )
        return response.data[0].embedding

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        items = [str(text).strip() for text in texts if str(text).strip()]
        if not items:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=items,
        )
        return [row.embedding for row in response.data]
