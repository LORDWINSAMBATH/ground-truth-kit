"""Vérifie ton BM25 contre rank_bm25 (pip install rank_bm25).
Les IDF diffèrent un peu (ton log(1+x) vs log(x) de rank_bm25) : on compare donc les CLASSEMENTS,
avec une tolérance. Un grand écart = bug à chercher (tokenisation, longueur moyenne, saturation)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_bm25_ranking_close_to_reference():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 absent : pip install rank_bm25")
        return
    from src.bm25_scratch import BM25FromScratch

    rng = np.random.default_rng(1)
    vocab = [f"mot{i}" for i in range(300)]
    p = 1 / np.arange(1, 301) ** 0.8
    p /= p.sum()
    docs = [list(rng.choice(vocab, size=int(rng.integers(40, 160)), p=p)) for _ in range(60)]
    df = {w: sum(w in d for d in docs) for w in vocab}
    rare = [w for w in vocab if 0 < df[w] <= 0.3 * len(docs)]  # évite les mots à IDF négatif chez rank_bm25

    mine = BM25FromScratch(k1=1.5, b=0.75)
    mine.fit([" ".join(d) for d in docs])
    ref = BM25Okapi(docs, k1=1.5, b=0.75)

    top1, overlap = 0, []
    for _ in range(60):
        q = list(rng.choice(rare, size=3, replace=False))
        a = [r["doc_id"] for r in mine.search(" ".join(q), top_k=5)]
        s = ref.get_scores(q)
        b = list(np.argsort(-s)[:5])
        top1 += a[0] == b[0]
        overlap.append(len(set(a) & set(b)) / 5)
    print(f"accord top-1 : {top1 / 60:.2f}   recouvrement top-5 : {np.mean(overlap):.2f}")
    assert top1 / 60 >= 0.8 and np.mean(overlap) >= 0.8


if __name__ == "__main__":
    test_bm25_ranking_close_to_reference()
