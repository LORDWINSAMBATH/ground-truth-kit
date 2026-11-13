"""BLOC 1 : métriques de recherche, à écrire TOI-MÊME.

`retrieved` : liste ordonnée d'identifiants (ex. (doc_id, page)), le meilleur d'abord.
`gold`      : ensemble des identifiants pertinents (pertinence binaire).
Si `gold` est vide, chaque métrique renvoie 0.0.
"""
import math  # noqa: F401  (utile pour nDCG)


def recall_at_k(retrieved: list, gold: set, k: int) -> float:
    """Part des éléments de `gold` présents dans les k premiers de `retrieved`."""
    raise NotImplementedError("À écrire toi-même (bloc 1)")


def mrr(retrieved: list, gold: set) -> float:
    """1 / rang (à partir de 1) du premier élément pertinent, 0.0 s'il n'y en a aucun."""
    raise NotImplementedError("À écrire toi-même (bloc 1)")


def ndcg_at_k(retrieved: list, gold: set, k: int = 10) -> float:
    """nDCG@k à pertinence binaire : DCG = somme de 1/log2(rang+1) sur les pertinents du top-k,
    normalisé par le DCG idéal (tous les pertinents placés en tête, dans la limite de k)."""
    raise NotImplementedError("À écrire toi-même (bloc 1)")
