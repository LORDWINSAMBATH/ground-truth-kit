"""Moteur hybride : BM25 (lexical) + dense (bge-m3), fusionnés par Reciprocal Rank Fusion.

- Les embeddings du corpus sont calculés UNE fois à l'indexation (emb.npy), jamais par question.
- Le cosinus + top-k viennent de src/dense_scratch.py (bloc 1, écrit par toi).
"""
from __future__ import annotations

import numpy as np

try:
    from .dense_scratch import cosine_topk
except ImportError:  # exécution en script
    from dense_scratch import cosine_topk

EMBED_MODEL = "BAAI/bge-m3"   # multilingue, bon en français (listé dans l'énoncé)
MAX_SEQ = 512
_model = None


def get_embedder():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Hybride] Chargement de {EMBED_MODEL} sur {dev}...")
        m = SentenceTransformer(EMBED_MODEL, device=dev)
        m.max_seq_length = MAX_SEQ
        if dev == "cuda":
            m.half()
        _model = m
    return _model


def embed_texts(texts, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """Embeddings normalisés (float32), forme (n, d)."""
    m = get_embedder()
    v = m.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=show_progress)
    return np.asarray(v, dtype=np.float32)


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None):
    """Reciprocal Rank Fusion (Cormack et al. 2009) : score(d) = somme_r w_r / (k + rang_r(d))."""
    scores: dict[int, float] = {}
    for r, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[r]
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(query: str, bm25, emb_matrix: np.ndarray, mode: str = "hybrid",
                  n_candidates: int = 40, top_k: int = 30, rrf_k: int = 60,
                  w_bm25: float = 1.0, w_dense: float = 1.0) -> list[dict]:
    """mode : 'hybrid' | 'bm25' | 'dense' (utile pour l'ablation).
    Renvoie [{'idx', 'score' (RRF), 'cos' (cosinus ou None)}, ...] trié par score décroissant."""
    bm25_rank: list[int] = []
    dense_rank: list[int] = []
    cos: dict[int, float] = {}

    if mode in ("hybrid", "bm25"):
        bm25_rank = [r["doc_id"] for r in bm25.search(query, top_k=n_candidates)]
    if mode in ("hybrid", "dense"):
        q = embed_texts([query])[0]
        for i, s in cosine_topk(q, emb_matrix, n_candidates):
            dense_rank.append(i)
            cos[i] = s

    fused = rrf([bm25_rank, dense_rank], k=rrf_k, weights=[w_bm25, w_dense])
    return [{"idx": int(i), "score": float(s), "cos": cos.get(i)} for i, s in fused[:top_k]]
