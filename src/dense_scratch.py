import numpy as np

def cosine_topk(query_vec, matrix, k):
    q = np.asarray(query_vec, dtype=np.float64)
    M = np.asarray(matrix, dtype=np.float64)

    dots =M @ q     
    norms = np.linalg.norm(M, axis=1) * np.linalg.norm(q)     

    scores = np.zeros(len(M))
    ok = norms > 0                      # évite la division par zéro
    scores[ok] = dots[ok] / norms[ok]

    k = min(k, len(scores))
    best = np.argsort(-scores)[:min(k, len(scores))]     
    return [(int(i), float(scores[i])) for i in best]
