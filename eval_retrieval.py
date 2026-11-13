"""Métriques de recherche sur la validation (utilise TES fonctions de src/metrics_scratch.py).

Usage :  python eval_retrieval.py --pred runs/hybrid.json --gold data_val.json
Les questions sans page gold (questions pièges) sont ignorées ici.
À lancer sur une sortie de `answer --retrieval-only --k 20` pour comparer BM25 seul / dense seul / hybride.
"""
import argparse

from diagnose import _load, _pairs
from src.metrics_scratch import mrr, ndcg_at_k, recall_at_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    a = ap.parse_args()

    pred = {r["qid"]: r for r in _load(a.pred)}
    gold = {r["qid"]: r for r in _load(a.gold)}
    rows = []
    for qid, p in pred.items():
        g = _pairs(gold.get(qid, {}))
        if not g:
            continue
        ret = [(str(r["doc_id"]), int(r["page"])) for r in p.get("retrieved", [])]
        rows.append((recall_at_k(ret, g, 1), recall_at_k(ret, g, 5), recall_at_k(ret, g, 10),
                     recall_at_k(ret, g, 20), mrr(ret, g), ndcg_at_k(ret, g, 10)))
    if not rows:
        print("Aucune question avec page gold trouvée.")
        return
    n = len(rows)
    names = ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "MRR", "nDCG@10"]
    print(f"{n} questions avec page gold")
    for i, name in enumerate(names):
        print(f"{name:10s} {sum(r[i] for r in rows) / n:.3f}")


if __name__ == "__main__":
    main()
