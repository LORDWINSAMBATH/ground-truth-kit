import json
import os
import unicodedata
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Chargement d'un modèle d'embedding léger, performant et multilingue
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        print(f"[Hybride] Chargement du modèle d'embedding : {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def normalize_text(text: str) -> str:
    """Normalise le texte pour uniformiser la recherche."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()

def hybrid_search_robust(query_text, retriever, corpus_chunks, top_k=10, alpha=0.5):
    """
    Moteur de recherche hybride combinant BM25 (mots-clés exacts/tableaux) 
    et Recherche Dense (sémantique) avec fusion des scores (Alpha blending).
    """
    # 1. Recherche Lexicale (BM25)
    bm25_raw_results = retriever.search(query_text, top_k=top_k * 2)
    
    # Normalisation des scores BM25 entre 0 et 1
    bm25_scores = {}
    if bm25_raw_results:
        max_bm25 = max([res.get("score", 1.0) for res in bm25_raw_results]) or 1.0
        for res in bm25_raw_results:
            doc_idx = res["doc_id"]
            bm25_scores[doc_idx] = res.get("score", 0.0) / max_bm25

    # 2. Recherche Sémantique Dense (Embeddings)
    model = get_embedding_model()
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    
    corpus_embeddings = model.encode(corpus_chunks, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    
    # Normalisation des scores cosinus entre 0 et 1 (ils le sont déjà globalement via cos_sim, mais on s'assure du format)
    dense_scores = {idx: float(score) for idx, score in enumerate(cos_scores)}

    # 3. Fusion Hybride des scores (Score final = alpha * BM25 + (1 - alpha) * Sémantique)
    # alpha = 0.5 donne autant de poids aux mots-clés (tableaux, chiffres) qu'au sens global.
    all_indices = set(bm25_scores.keys()).union(set(dense_scores.keys()))
    combined_results = []

    for idx in all_indices:
        b_score = bm25_scores.get(idx, 0.0)
        d_score = dense_scores.get(idx, 0.0)
        final_score = (alpha * b_score) + ((1 - alpha) * d_score)
        combined_results.append({"doc_id": int(idx), "score": float(final_score)})

    # Tri par score décroissant et sélection des top_k meilleurs
    combined_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)[:top_k]
    
    return combined_results