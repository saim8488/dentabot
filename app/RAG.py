from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_PATH = "app/documents/knowledge_base"
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "dental_knowledge"
CHUNK_SIZE_WORDS = 300          # words per chunk
CHUNK_OVERLAP_WORDS = 50        # overlap between consecutive chunks


# ── Internal RAGEngine class ───────────────────────────────────────────────────

class _RAGEngine:
    """Lazy-initialised singleton that manages embedding + ChromaDB."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._collection = None
        self._ready = False

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init(self) -> None:
        if self._ready:
            return

        print("[RAG] Initialising embedding model...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self._collection = client.get_or_create_collection(name=COLLECTION_NAME)

        count = self._collection.count()
        if count == 0:
            print("[RAG] Collection is empty — indexing knowledge base...")
            self._index()
        else:
            print(f"[RAG] Collection already contains {count} chunks — skipping indexing.")

        self._ready = True

    # ── Document loading & chunking ─────────────────────────────────────────

    def _chunk_text(self, text: str, source: str) -> List[dict]:
        words = text.split()
        chunks = []
        step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + CHUNK_SIZE_WORDS])
            if chunk.strip():
                chunks.append(
                    {
                        "text": chunk,
                        "metadata": {"source": source, "chunk_id": len(chunks)},
                    }
                )
        return chunks

    def _load_documents(self, folder_path: str) -> List[dict]:
        documents: List[dict] = []
        folder = Path(folder_path)
        if not folder.exists():
            print(f"[RAG] Knowledge base folder not found: {folder_path}")
            return documents

        # ── Markdown files ──────────────────────────────────────────────────
        for fp in sorted(folder.glob("*.md")):
            content = fp.read_text(encoding="utf-8")
            documents.extend(self._chunk_text(content, fp.name))

        # ── CSV files (FAQ / Q-A style) ─────────────────────────────────────
        for fp in sorted(folder.glob("*.csv")):
            try:
                with open(fp, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        parts = [
                            f"{k}: {str(v).strip()}"
                            for k, v in row.items()
                            if v and str(v).strip()
                        ]
                        text = " | ".join(parts)
                        if len(text) > 30:
                            documents.append(
                                {
                                    "text": text[:700],
                                    "metadata": {
                                        "source": fp.name,
                                        "chunk_id": len(documents),
                                    },
                                }
                            )
            except Exception as exc:
                print(f"[RAG] Could not load {fp.name}: {exc}")

        # ── Plain text files ────────────────────────────────────────────────
        for fp in sorted(folder.glob("*.txt")):
            content = fp.read_text(encoding="utf-8")
            documents.extend(self._chunk_text(content, fp.name))

        print(f"[RAG] Loaded {len(documents)} chunks from {len(list(folder.iterdir()))} files.")
        return documents

    # ── Indexing ────────────────────────────────────────────────────────────

    def _index(self) -> None:
        docs = self._load_documents(KNOWLEDGE_BASE_PATH)
        if not docs:
            print("[RAG] No documents to index.")
            return

        texts = [d["text"] for d in docs]
        embeddings = self._model.encode(texts, show_progress_bar=True, batch_size=32)  # type: ignore[union-attr]

        BATCH = 100
        for start in range(0, len(docs), BATCH):
            batch_docs = docs[start : start + BATCH]
            batch_emb = embeddings[start : start + BATCH]
            self._collection.add(
                ids=[f"doc_{start + j}" for j in range(len(batch_docs))],
                embeddings=[e.tolist() for e in batch_emb],
                metadatas=[d["metadata"] for d in batch_docs],
                documents=[d["text"] for d in batch_docs],
            )

        print(f"[RAG] Indexed {len(docs)} chunks successfully.")

    # ── Public retrieval ────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        self._init()
        count = self._collection.count()  # type: ignore[union-attr]
        if count == 0:
            return []

        query_embedding = self._model.encode([query])[0].tolist()  # type: ignore[union-attr]
        results = self._collection.query(  # type: ignore[union-attr]
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents"],
        )
        return results["documents"][0] if results["documents"] else []

    def reindex(self) -> None:
        """Force a full re-index (e.g., after adding new documents)."""
        if self._collection is not None:
            # delete all existing docs
            existing_ids = self._collection.get(include=[])["ids"]
            if existing_ids:
                self._collection.delete(ids=existing_ids)
        self._ready = False
        self._init()


# ── Module-level singleton ─────────────────────────────────────────────────────

_engine = _RAGEngine()


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[str]:
    """Public API used by engine.py — same signature as before."""
    return _engine.retrieve(query, top_k)
