"""BLOC 1 : recherche dense from scratch (à écrire TOI-MÊME, ~8 lignes de NumPy)."""
import numpy as np


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Renvoie [(indice, cosinus), ...] trié par cosinus décroissant, longueur min(k, n).

    - query_vec : forme (d,) ; matrix : forme (n, d). Les vecteurs ne sont PAS supposés normalisés.
    - cosinus = produit scalaire / (norme(q) * norme(doc)), sans util.cos_sim ni FAISS.
    - Une norme nulle ne doit pas provoquer de division par zéro (score = 0.0).
    - Top-k : np.argpartition pour isoler les k meilleurs (O(n)), puis tri de ces k seulement.
    - Types de sortie : indice = int Python, score = float Python.
    """
    raise NotImplementedError("À écrire toi-même (bloc 1)")
