"""Reranker cross-encoder (bge-reranker-v2-m3) : score de pertinence dans [0, 1] pour (question, passage)."""
from __future__ import annotations

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_LEN = 512
_state = None


def _get():
    global _state
    if _state is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Reranker] Chargement de {RERANK_MODEL} sur {dev}...")
        tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL).to(dev).eval()
        if dev == "cuda":
            model.half()
        _state = (tok, model, dev)
    return _state


def rerank(query: str, texts: list[str], batch_size: int = 8) -> list[float]:
    import torch

    tok, model, dev = _get()
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok([query] * len(batch), batch, padding=True, truncation="only_second",
                      max_length=MAX_LEN, return_tensors="pt").to(dev)
            logits = model(**enc).logits.view(-1).float()
            out += torch.sigmoid(logits).cpu().tolist()
    return out
