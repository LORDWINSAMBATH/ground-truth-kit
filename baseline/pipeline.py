"""Baseline officielle : le RAG le plus simple qui respecte le contrat du run final.

Ce n'est PAS un modèle à recopier pour le bloc 1 (qui doit être codé from scratch, sans
librairie de recherche). C'est le point de comparaison du leaderboard et l'exemple du contrat :

    python -m baseline.pipeline index  --pdf-dir corpus/pdf --index-dir index/
    python -m baseline.pipeline update --index-dir index/ --pdf-dir corpus/pdf_v2      # palier 3
    python -m baseline.pipeline answer --index-dir index/ --questions data/validation.json --out runs/val.json

Ton dépôt doit exposer exactement ces deux commandes (même arguments), via un fichier
`run.sh` à la racine qui les enchaîne : voir README, section « Contrat du run final ».

Recette : texte brut PyMuPDF, une page = un chunk, BM25 (bm25s), top-4 pages tronquées pour
tenir dans 4 096 tokens, un seul appel au générateur, abstention si le modèle le décide.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROMPT_SYSTEM = (
    "Tu réponds à des questions sur des rapports publics français, uniquement à partir des extraits fournis. "
    "Chaque extrait est précédé de sa référence [doc_id p.N]. "
    "Si la réponse n'est pas dans les extraits, réponds exactement : INTROUVABLE. "
    "Sinon, réponds en une ou deux phrases, puis indique tes sources sur une dernière ligne au format "
    "SOURCES: [doc_id p.N], [doc_id p.N]"
)


def _tokenize(text: str) -> list[str]:
    import unicodedata

    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", text)


def _read_pages(pdf_dir: str) -> list[dict]:
    import pymupdf

    pages = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            for i, page in enumerate(doc, 1):
                pages.append({"doc_id": pdf.stem, "page": i, "text": page.get_text("text")})
    return pages


def build_index(pdf_dir: str, index_dir: str, pages: list[dict] | None = None) -> None:
    import bm25s

    pages = _read_pages(pdf_dir) if pages is None else pages
    retriever = bm25s.BM25()
    retriever.index([_tokenize(p["text"]) for p in pages])
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    retriever.save(out / "bm25")
    (out / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    print(f"{len(pages)} pages indexées")


def update_index(index_dir: str, pdf_dir: str) -> None:
    """Ajoute de nouveaux PDF à un index existant (palier 3).

    La baseline relit seulement les nouveaux PDF puis reconstruit l'index BM25, ce qui ne coûte que
    quelques secondes. Un pipeline avec OCR et embeddings doit, lui, n'encoder que les nouvelles
    pages : c'est tout l'intérêt de la contrainte de 20 minutes.
    """
    idx = Path(index_dir)
    pages = json.loads((idx / "pages.json").read_text(encoding="utf-8"))
    known = {p["doc_id"] for p in pages}
    new = [p for p in _read_pages(pdf_dir) if p["doc_id"] not in known]
    print(f"{len({p['doc_id'] for p in new})} nouveaux documents, {len(new)} pages")
    build_index(pdf_dir, index_dir, pages=pages + new)


def _parse_citations(text: str) -> tuple[str, list[dict]]:
    cits = [{"doc_id": d, "page": int(p)} for d, p in re.findall(r"\[([\w\-]+) p\.?\s*(\d+)\]", text)]
    answer = re.split(r"\n?SOURCES\s*:", text)[0].strip()
    seen, uniq = set(), []
    for c in cits:
        k = (c["doc_id"], c["page"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return answer, uniq[:5]


def answer_questions(index_dir: str, questions_path: str, out_path: str, backend: str, k: int = 4) -> None:
    import bm25s

    from groundtruth.generator import OfficialGenerator
    from groundtruth.io import load_questions, write_predictions

    idx = Path(index_dir)
    retriever = bm25s.BM25.load(idx / "bm25")
    pages = json.loads((idx / "pages.json").read_text(encoding="utf-8"))
    gen = OfficialGenerator(backend=backend)
    preds = []
    for q in load_questions(questions_path):
        gen.start_question(q["qid"])
        res, scores = retriever.retrieve([_tokenize(q["question"])], k=10)
        hits = [pages[int(i)] for i in res[0]]
        # remplit le contexte page par page tant que le budget le permet
        budget_chars = 2600
        chunks = []
        for h in hits[:k]:
            chunks.append(f"[{h['doc_id']} p.{h['page']}]\n{h['text'][:budget_chars]}")
        while True:
            msgs = [{"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": "Extraits :\n\n" + "\n\n".join(chunks) + f"\n\nQuestion : {q['question']}"}]
            if gen.count_tokens(msgs) <= 4096 - 16 or budget_chars < 300:
                break
            budget_chars = int(budget_chars * 0.8)
            chunks = [c[: budget_chars + 40] for c in chunks]
        raw = gen.chat(msgs, max_new_tokens=200)
        answer, cits = _parse_citations(raw)
        if "INTROUVABLE" in raw.upper():
            answer, cits = "INTROUVABLE", []
        preds.append({"qid": q["qid"], "answer": answer, "citations": cits,
                      "retrieved": [{"doc_id": h["doc_id"], "page": h["page"]} for h in hits]})
        print(f"{q['qid']}: {answer[:80]}")
    write_predictions(out_path, preds, run_name="baseline-bm25")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--pdf-dir", required=True)
    a.add_argument("--index-dir", required=True)
    u = sub.add_parser("update")
    u.add_argument("--index-dir", required=True)
    u.add_argument("--pdf-dir", required=True)
    b = sub.add_parser("answer")
    b.add_argument("--index-dir", required=True)
    b.add_argument("--questions", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--backend", default="hf", choices=["hf", "openai", "echo"])
    args = ap.parse_args()
    if args.cmd == "index":
        build_index(args.pdf_dir, args.index_dir)
    elif args.cmd == "update":
        update_index(args.index_dir, args.pdf_dir)
    else:
        answer_questions(args.index_dir, args.questions, args.out, args.backend)


if __name__ == "__main__":
    main()
