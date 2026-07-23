"""Document chunking.

A production RAG system splits documents into overlapping chunks so retrieval
returns focused, relevant passages. This is a dependency-free recursive
character splitter (the same idea as LangChain's RecursiveCharacterTextSplitter):
it splits on the coarsest separator first and recurses into any piece still
larger than the target size. The optional production backend swaps in the
LangChain splitter directly.
"""
from __future__ import annotations

from dataclasses import dataclass

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """One retrievable passage, traceable back to its source document."""
    id: str
    source: str
    text: str


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Break text into pieces no larger than chunk_size, preferring coarse cuts."""
    if len(text) <= chunk_size or not separators:
        return [text] if text else []
    sep, *rest = separators
    if sep == "":
        # Finest fallback: hard-split oversized runs with no whitespace.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    # Re-attach the separator we split on so the text round-trips.
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]] if len(parts) > 1 else parts

    out: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            if piece:
                out.append(piece)
        else:
            out.extend(_recursive_split(piece, rest, chunk_size))
    return out


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """Split text into ~chunk_size character windows with overlap.

    Overlap preserves context that would otherwise be cut at a boundary, which
    measurably improves retrieval recall.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    pieces = _recursive_split(text.strip(), _SEPARATORS, chunk_size)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            # Start the next chunk with a tail of the previous one (overlap).
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = tail + piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_document(source: str, text: str, **kwargs) -> list[Chunk]:
    """Chunk a single document into traceable Chunk records."""
    return [
        Chunk(id=f"{source}#{i}", source=source, text=t)
        for i, t in enumerate(chunk_text(text, **kwargs))
    ]
