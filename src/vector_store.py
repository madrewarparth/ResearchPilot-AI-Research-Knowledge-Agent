"""
Vector Store & Retrieval Engine.
Integrates ChromaDB with Ollama embeddings (nomic-embed-text) and metadata preservation.
Includes robust fallback embeddings for local offline execution and testing.
"""

import os
import math
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config import settings
from src.chunking import DocumentChunk

@dataclass
class VectorStoreResult:
    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    similarity_score: float
    distance: float
    metadata: Dict[str, Any]

    def format_citation(self) -> str:
        """Standard citation string: e.g. [paper.pdf, Page 3, Chunk 1]"""
        chunk_idx = self.metadata.get("chunk_index", 0)
        return f"[{self.doc_name}, Page {self.page_number}, Chunk {chunk_idx}]"


class FallbackDeterministicEmbedder:
    """
    Lightweight, deterministic high-dimensional embedding generator.
    Used when Ollama is offline or in testing environments to guarantee
    reliable cosine-similarity ranking without crashing or requiring remote services.
    """
    DIM = 256

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.DIM
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Deterministic hash to dimension bucket
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.DIM
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)


class ResearchVectorStore:
    """
    ChromaDB-backed vector database for ResearchPilot with Ollama nomic-embed-text.
    """
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        force_fallback_embedder: bool = False
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.force_fallback = force_fallback_embedder
        self.fallback_embedder = FallbackDeterministicEmbedder()
        self.client = None
        self.collection = None
        self._is_ollama_available = False
        
        self._init_client_and_collection()

    def _check_ollama_embed_service(self) -> bool:
        if self.force_fallback:
            return False
        import urllib.request
        import json
        try:
            req = urllib.request.Request(
                f"{settings.OLLAMA_HOST.rstrip('/')}/api/tags",
                headers={"User-Agent": "ResearchPilot"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return any(settings.EMBEDDING_MODEL in m for m in models)
        except Exception:
            return False
        return False

    def _init_client_and_collection(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        self._is_ollama_available = self._check_ollama_embed_service()

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            # If chromadb is not yet installed in active python, initialize mock in-memory store
            self.client = None
            self.collection = None
            self._in_memory_docs: List[Dict[str, Any]] = []

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates embedding using Ollama nomic-embed-text if active, else deterministic fallback.
        """
        if self._is_ollama_available:
            import urllib.request
            import json
            try:
                payload = json.dumps({
                    "model": settings.EMBEDDING_MODEL,
                    "prompt": text
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{settings.OLLAMA_HOST.rstrip('/')}/api/embeddings",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        emb = data.get("embedding", [])
                        if emb:
                            return emb
            except Exception:
                pass
        
        return self.fallback_embedder.embed_text(text)

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Indexes chunks into Chroma with metadata preservation.
        Returns count of indexed chunks.
        """
        if not chunks:
            return 0

        ids = [c.metadata.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata.to_dict() for c in chunks]
        embeddings = [self.get_embedding(t) for t in texts]

        if self.collection is not None:
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return len(chunks)
        else:
            # Fallback memory store
            if not hasattr(self, "_in_memory_docs"):
                self._in_memory_docs = []
            for i, cid in enumerate(ids):
                self._in_memory_docs.append({
                    "id": cid,
                    "text": texts[i],
                    "embedding": embeddings[i],
                    "metadata": metadatas[i]
                })
            return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        filter_doc: Optional[str] = None
    ) -> List[VectorStoreResult]:
        """
        Performs vector similarity search against indexed document passages.
        Returns sorted VectorStoreResults.
        """
        if not query or len(query.strip()) == 0:
            return []

        query_emb = self.get_embedding(query)
        results: List[VectorStoreResult] = []

        if self.collection is not None:
            where_clause = {"file_name": filter_doc} if filter_doc else None
            # Check if collection is empty
            if self.collection.count() == 0:
                return []

            k = min(top_k, self.collection.count())
            search_args = {
                "query_embeddings": [query_emb],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"]
            }
            if where_clause:
                search_args["where"] = where_clause

            res = self.collection.query(**search_args)

            if res and res.get("ids") and len(res["ids"][0]) > 0:
                ids = res["ids"][0]
                docs = res["documents"][0]
                metas = res["metadatas"][0]
                distances = res["distances"][0]

                for i, cid in enumerate(ids):
                    dist = float(distances[i]) if distances else 0.0
                    # Cosine distance to similarity: 1 - dist (clipped between 0 and 1)
                    sim = max(0.0, min(1.0, 1.0 - dist))
                    meta = metas[i] if metas else {}
                    results.append(VectorStoreResult(
                        chunk_id=cid,
                        doc_name=meta.get("file_name", "Unknown"),
                        page_number=meta.get("page_number", 1),
                        text=docs[i],
                        similarity_score=round(sim, 4),
                        distance=round(dist, 4),
                        metadata=meta
                    ))
        else:
            # Memory search fallback
            if not hasattr(self, "_in_memory_docs") or not self._in_memory_docs:
                return []
            
            scored = []
            for item in self._in_memory_docs:
                if filter_doc and item["metadata"].get("file_name") != filter_doc:
                    continue
                # Cosine similarity between query_emb and item["embedding"]
                dot = sum(a * b for a, b in zip(query_emb, item["embedding"]))
                sim = max(0.0, min(1.0, (dot + 1.0) / 2.0))
                dist = 1.0 - sim
                scored.append((sim, dist, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            for sim, dist, item in scored[:top_k]:
                results.append(VectorStoreResult(
                    chunk_id=item["id"],
                    doc_name=item["metadata"].get("file_name", "Unknown"),
                    page_number=item["metadata"].get("page_number", 1),
                    text=item["text"],
                    similarity_score=round(sim, 4),
                    distance=round(dist, 4),
                    metadata=item["metadata"]
                ))

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Returns stats about total chunks and distinct indexed documents.
        """
        if self.collection is not None:
            count = self.collection.count()
            distinct_docs = set()
            if count > 0:
                peek = self.collection.peek(limit=count)
                for meta in peek.get("metadatas", []):
                    if meta and "file_name" in meta:
                        distinct_docs.add(meta["file_name"])
            return {
                "total_chunks": count,
                "distinct_documents": sorted(list(distinct_docs)),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_dir,
                "embedding_engine": "Ollama nomic-embed-text" if self._is_ollama_available else "Deterministic Fallback"
            }
        else:
            docs = getattr(self, "_in_memory_docs", [])
            distinct = {d["metadata"].get("file_name") for d in docs if "metadata" in d}
            return {
                "total_chunks": len(docs),
                "distinct_documents": sorted(list(distinct)),
                "collection_name": self.collection_name,
                "persist_directory": "In-Memory",
                "embedding_engine": "Deterministic Fallback"
            }

    def delete_document(self, doc_name: str) -> int:
        """
        Removes all chunks associated with a specific file_name.
        """
        if self.collection is not None:
            peek = self.collection.peek(limit=self.collection.count() or 1)
            ids_to_del = []
            for i, meta in enumerate(peek.get("metadatas", [])):
                if meta and meta.get("file_name") == doc_name:
                    ids_to_del.append(peek["ids"][i])
            if ids_to_del:
                self.collection.delete(ids=ids_to_del)
                return len(ids_to_del)
            return 0
        else:
            if hasattr(self, "_in_memory_docs"):
                before = len(self._in_memory_docs)
                self._in_memory_docs = [d for d in self._in_memory_docs if d["metadata"].get("file_name") != doc_name]
                return before - len(self._in_memory_docs)
            return 0

    def clear_collection(self) -> None:
        """
        Resets and clears all indexed documents in the collection.
        """
        if self.client is not None:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        if hasattr(self, "_in_memory_docs"):
            self._in_memory_docs = []
