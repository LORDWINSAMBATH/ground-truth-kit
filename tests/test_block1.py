"""Tests du bloc 1.  Lancer :  python tests/test_block1.py   (ou :  python -m pytest tests -q)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.chunking_scratch import chunk_text
from src.dense_scratch import cosine_topk
from src.metrics_scratch import mrr, ndcg_at_k, recall_at_k


def _cos(q, d):
    nq = math.sqrt(sum(x * x for x in q))
    nd = math.sqrt(sum(x * x for x in d))
    if nq == 0 or nd == 0:
        return 0.0
    return sum(a * b for a, b in zip(q, d)) / (nq * nd)


def test_cosine_topk():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(50, 8))
    q = rng.normal(size=8)
    exp = sorted(((i, _cos(q, M[i])) for i in range(50)), key=lambda x: -x[1])[:5]
    got = cosine_topk(q, M, 5)
    assert [i for i, _ in got] == [i for i, _ in exp], "mauvais indices ou mauvais ordre"
    assert all(abs(a[1] - b[1]) < 1e-6 for a, b in zip(got, exp)), "scores faux"
    assert all(isinstance(i, int) and isinstance(s, float) for i, s in got), "types : int et float Python"
    assert len(cosine_topk(q, M, 500)) == 50, "k > n doit renvoyer n résultats"
    M2 = M.copy()
    M2[3] = 0.0
    out = dict(cosine_topk(q, M2, 50))
    assert out[3] == 0.0, "vecteur nul : score 0.0, pas de NaN"
    s = [x for _, x in cosine_topk(q, M, 50)]
    assert s == sorted(s, reverse=True)


def test_chunk_text():
    t = "".join(chr(97 + i % 26) for i in range(2500))
    c = chunk_text(t, size=1000, overlap=200)
    assert len(c) == 3, f"attendu 3 chunks, obtenu {len(c)}"
    assert all(len(x) <= 1000 for x in c)
    assert c[0][-200:] == c[1][:200] and c[1][-200:] == c[2][:200], "recouvrement"
    assert c[0] + c[1][200:] + c[2][200:] == t, "recomposition"
    assert chunk_text("court", 1000, 200) == ["court"]
    assert chunk_text("", 1000, 200) == [] and chunk_text("   \n ", 1000, 200) == []
    try:
        chunk_text("abc", 100, 100)
        raise AssertionError("size <= overlap doit lever ValueError")
    except ValueError:
        pass


def test_metrics():
    r, g = ["a", "b", "c", "d"], {"c", "z"}
    assert recall_at_k(r, g, 3) == 0.5 and recall_at_k(r, g, 2) == 0.0
    assert abs(mrr(r, g) - 1 / 3) < 1e-9 and mrr(["x"], g) == 0.0
    dcg = 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert abs(ndcg_at_k(r, g, 10) - dcg / idcg) < 1e-9
    assert abs(ndcg_at_k(["c", "z"], g, 10) - 1.0) < 1e-9
    assert recall_at_k(r, set(), 3) == 0.0 and mrr(r, set()) == 0.0 and ndcg_at_k(r, set()) == 0.0


if __name__ == "__main__":
    for f in (test_cosine_topk, test_chunk_text, test_metrics):
        try:
            f()
            print("OK   ", f.__name__)
        except NotImplementedError:
            print("TODO ", f.__name__, "(fonction pas encore écrite)")
        except AssertionError as e:
            print("ÉCHEC", f.__name__, "->", e)
