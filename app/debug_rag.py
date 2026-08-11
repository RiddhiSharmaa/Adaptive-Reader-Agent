# FIX 3: Add this diagnostic to app/main.py startup or run standalone:

# Save as: debug_rag.py in project root

import sys
sys.path.insert(0, '.')

from app.rag_index import knowledge_collection, index_knowledge_base
from pathlib import Path

# Check if index has documents
try:
    # Peek at collection
    all_results = knowledge_collection.get(include=["embeddings", "documents", "metadatas"])
    doc_count = len(all_results.get("documents", []))
    print(f"✓ RAG index has {doc_count} documents")
    
    if doc_count == 0:
        print("\n⚠ Index is empty! Re-indexing...")
        index_knowledge_base()
        print("✓ Re-indexed. Run evals again.")
    else:
        print(f"Sample doc: {all_results['documents'][0][:100]}...")
        
except Exception as e:
    print(f"✗ Error: {e}")
    print("Run: python -c 'from app.rag_index import index_knowledge_base; index_knowledge_base()'")