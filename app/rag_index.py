"""
app/rag_index.py

Indexes the 7 knowledge base docs into ChromaDB using a local
sentence-transformer embedding model (no API key, no cost).

Run once at startup, and re-run whenever docs/knowledge_base/*.md changes.
"""

import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from . import config

_chroma_client = chromadb.Client()
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

knowledge_collection = _chroma_client.get_or_create_collection(
    "reader_knowledge", embedding_function=_ef, metadata={"hnsw:space": "cosine"},
)


def naive_chunk(text: str) -> list[str]:
    """Baseline: split on blank lines, ignoring document structure."""
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def structure_aware_chunk(text: str) -> list[str]:
    """Splits on markdown headings (##) so each chunk stays within one section."""
    sections = re.split(r"(?=^##\s)", text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]


def index_knowledge_base(chunk_fn=structure_aware_chunk):
    docs_dir = config.DOCS_DIR / "knowledge_base"
    all_chunks, all_ids, all_metadata = [], [], []

    for doc_path in sorted(docs_dir.glob("*.md")):
        text = doc_path.read_text()
        chunks = chunk_fn(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_path.stem}_chunk_{i+1}")
            all_metadata.append({"source": doc_path.stem})

    knowledge_collection.upsert(documents=all_chunks, ids=all_ids, metadatas=all_metadata)
    print(f"  indexed {len(all_chunks)} chunks from {len(list(docs_dir.glob('*.md')))} documents.")


if __name__ == "__main__":
    index_knowledge_base()