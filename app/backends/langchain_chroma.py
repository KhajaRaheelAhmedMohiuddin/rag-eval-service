"""OPTIONAL production RAG backend: LangChain + Chroma + real embeddings.

This module is NOT imported by the default app (which stays dependency-light and
runs anywhere). It demonstrates the same pipeline built on the industry-standard
stack, and is drop-in compatible with `TfidfRetriever.retrieve()`.

Enable it by installing the extras and wiring it into RagService:

    pip install -r requirements-prod.txt

    from app.backends.langchain_chroma import ChromaRetriever
    service = RagService(...)
    service.retriever = ChromaRetriever(load_documents(DATA_DIR))
"""
from __future__ import annotations

import os

from app.chunking import Chunk
from app.retrieval import RetrievedChunk


class ChromaRetriever:
    """RAG retrieval with LangChain embeddings + a Chroma vector store.

    Uses OpenAI embeddings when OPENAI_API_KEY is set, otherwise a local
    HuggingFace sentence-transformers model (no API key, downloads once).
    """

    def __init__(self, chunks: list[Chunk], persist_dir: str = ".chroma"):
        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        embeddings = self._make_embeddings()
        docs = [
            Document(page_content=c.text, metadata={"id": c.id, "source": c.source})
            for c in chunks
        ]
        self._store = Chroma.from_documents(
            docs, embedding=embeddings, persist_directory=persist_dir
        )

    @staticmethod
    def _make_embeddings():
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model="text-embedding-3-small")
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        hits = self._store.similarity_search_with_relevance_scores(query, k=k)
        out: list[RetrievedChunk] = []
        for doc, score in hits:
            out.append(
                RetrievedChunk(
                    chunk=Chunk(
                        id=doc.metadata.get("id", ""),
                        source=doc.metadata.get("source", ""),
                        text=doc.page_content,
                    ),
                    score=float(score),
                )
            )
        return out
