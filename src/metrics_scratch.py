"""BLOC 1 : métriques de recherche, à écrire TOI-MÊME.

`retrieved` : liste ordonnée d'identifiants (ex. (doc_id, page)), le meilleur d'abord.
`gold`      : ensemble des identifiants pertinents (pertinence binaire).
Si `gold` est vide, chaque métrique renvoie 0.0.
"""
import math  # noqa: F401  (utile pour nDCG)


def recall_at_k(retrieved: list, gold: set, k: int) -> float:
    """Part des éléments de `gold` présents dans les k premiers de `retrieved`."""
    
    if not gold:
        return 0.0
    top = retrieved[:k]
    found = len(set(top) & gold)         
    return found / len(gold)   


def mrr(retrieved: list, gold: set) -> float:
    """1 / rang (à partir de 1) du premier élément pertinent, 0.0 s'il n'y en a aucun."""
    for rang, x in enumerate(retrieved, 1):   # rang commence à 1
       if x in gold:       
            return 1 / rang 
    return 0.0

def ndcg_at_k(retrieved: list, gold: set, k: int = 10) -> float:
    """nDCG@k à pertinence binaire : DCG = somme de 1/log2(rang+1) sur les pertinents du top-k,
    normalisé par le DCG idéal (tous les pertinents placés en tête, dans la limite de k)."""

def ndcg_at_k(retrieved, gold, k=10):
    if not gold:
        return 0.0
    dcg = 0.0
    for rang, x in enumerate(retrieved[:k], 1):
        if x in gold:
            dcg += 1 / math.log2(rang + 1)       
    nb = min(len(gold), k)                        
    idcg = sum(1 / math.log2(rang + 1) for rang in range(1, nb + 1))
    return  dcg / idcg
