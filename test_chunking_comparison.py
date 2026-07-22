"""
Part 1 chunking comparison test — uses YOUR actual chunking functions from app/rag_index.py.

Run from project root:
    python test_chunking_comparison.py

This does NOT touch your real 'reader_knowledge' collection — it builds two separate,
throwaway collections (kb_naive / kb_structured) purely for comparison.
"""

import chromadb
from chromadb.utils import embedding_functions

from app import config
from app.rag_index import naive_chunk, structure_aware_chunk

_client = chromadb.Client()  # separate in-memory client, isolated from rag_index.py's client
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def build_test_collection(name: str, chunk_fn):
    try:
        _client.delete_collection(name)
    except Exception:
        pass

    collection = _client.create_collection(name=name, embedding_function=_ef, metadata={"hnsw:space": "cosine"})

    docs_dir = config.DOCS_DIR / "knowledge_base"
    all_chunks, all_ids, all_metadata = [], [], []

    for doc_path in sorted(docs_dir.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_fn(text)):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_path.stem}_chunk_{i+1}")
            all_metadata.append({"source": doc_path.stem})

    collection.upsert(documents=all_chunks, ids=all_ids, metadatas=all_metadata)
    print(f"Built '{name}': {len(all_chunks)} chunks")
    return collection


TEST_QUESTIONS = [
    "Hi, I'm new to this. I don't read a ton but I want something good to start with.",
    "I just finished a book I was obsessed with and now nothing else feels as good. What should I read next?",
    "I want something that keeps me on edge but isn't going to give me nightmares.",
    "I keep starting fantasy books and dropping them by book two, what's going on with me?",
    "I loved the found family part of the last book I read, got anything else like that?",
    "Everything I pick up lately feels boring, I think I'm in a slump.",
    "I want a slow build, not something where everything happens in the first chapter.",
    "What's the best font size for reading on an e-reader?",  # out-of-scope control, keep this one
]


def run_comparison(naive_coll, structured_coll):
    for question in TEST_QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        naive_result = naive_coll.query(query_texts=[question], n_results=3)
        structured_result = structured_coll.query(query_texts=[question], n_results=3)

        print("\n--- NAIVE top 3 ---")
        for rank in range(len(naive_result["documents"][0])):
            source = naive_result["metadatas"][0][rank]["source"]
            distance = naive_result["distances"][0][rank]
            text = naive_result["documents"][0][rank]
            print(f"  #{rank+1} [{source}] distance={distance:.4f}")
            print(f"      {text[:200]}")

        print("\n--- STRUCTURE-AWARE top 3 ---")
        for rank in range(len(structured_result["documents"][0])):
            source = structured_result["metadatas"][0][rank]["source"]
            distance = structured_result["distances"][0][rank]
            text = structured_result["documents"][0][rank]
            print(f"  #{rank+1} [{source}] distance={distance:.4f}")
            print(f"      {text[:200]}")


if __name__ == "__main__":
    naive_coll = build_test_collection("kb_naive", naive_chunk)
    structured_coll = build_test_collection("kb_structured", structure_aware_chunk)
    run_comparison(naive_coll, structured_coll)