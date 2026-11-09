
from __future__ import annotations

import argparse
import json
import re 
from pathlib import Path
import sys
import os
import pickle
import unicodedata

# Importations des modules du projet
from src.hybrid_retrieved import hybrid_search_robust
from src.bm25_scratch import BM25FromScratch


PROMPT_SYSTEM = (
    "Tu es un expert en analyse de rapports publics français. Réponds précisément et complètement à la question posée "
    "en te basant **uniquement** sur les extraits fournis.\n"
    "Chaque extrait est précédé de sa référence [doc_id p.N]. Tu dois citer tes sources à la fin de ta réponse.\n"
    "Si la réponse ne figure absolument pas dans les extraits, réponds exactement et uniquement : INTROUVABLE."
)

def _tokenize(text: str) -> list[str]:
    """Normalisation et tokenisation robuste pour la recherche lexicale."""
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", text)

def _read_pages(pdf_dir: str) -> list[dict]:
    """Lecture rapide et ultra-robuste des PDF (texte brut)."""
    import pymupdf

    pages = []
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"[Attention] Le dossier {pdf_dir} est introuvable ou vide.")
        return pages

    for pdf in sorted(pdf_path.glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            for i, page in enumerate(doc, 1):
                try:
                    page_text = page.get_text("text")
                except Exception:
                    page_text = page.get_text()
                
                pages.append({
                    "doc_id": pdf.stem, 
                    "page": i, 
                    "text": page_text
                })
    return pages

def build_index(pdf_dir: str, index_dir: str, pages: list[dict] | None = None) -> None:
    pages = _read_pages(pdf_dir) if pages is None else pages
    
    retriever = BM25FromScratch()
    corpus_texts = [p["text"] for p in pages]
    retriever.fit(corpus_texts)
    
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump(retriever, f)
        
    (out / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    print(f"{len(pages)} pages indexées avec succès !")

def update_index(index_dir: str, pdf_dir: str) -> None:
    idx = Path(index_dir)
    pages = json.loads((idx / "pages.json").read_text(encoding="utf-8"))
    known = {p["doc_id"] for p in pages}
    new_pages = _read_pages(pdf_dir)
    new = [p for p in new_pages if p["doc_id"] not in known]
    print(f"{len(new)} nouvelles pages ajoutées.")
    build_index(pdf_dir, index_dir, pages=pages + new)

def _parse_citations(text: str) -> tuple[str, list[dict]]:
    cits = [{"doc_id": d, "page": int(p)} for d, p in re.findall(r"\[([\w-]+)\s+p\.(\d+)\]", text)]
    answer = re.split(r"\n?SOURCES\s*:", text)[0].strip()
    seen, uniq = set(), []
    for c in cits:
        k = (c["doc_id"], c["page"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return answer, uniq[:5]

def answer_questions(index_dir: str, questions_path: str, out_path: str, backend: str, k: int = 12) -> None:
    from groundtruth.generator import OfficialGenerator
    from groundtruth.io import load_questions, write_predictions
    idx = Path(index_dir)
    
    with open(idx / "bm25.pkl", "rb") as f:
        retriever = pickle.load(f)
        
    pages = json.loads((idx / "pages.json").read_text(encoding="utf-8"))
    gen = OfficialGenerator(backend=backend)
    preds = []
    
    for q in load_questions(questions_path):
        gen.start_question(q["qid"])
        
        # 1. Recherche hybride avec alpha=0.5 et top_k ajusté
        search_results = hybrid_search_robust(
            query_text=q["question"],
            retriever=retriever,
            corpus_chunks=[p["text"] for p in pages],
            top_k=k,
            alpha=0.5  # Équilibre parfait 50/50 entre BM25 et Dense
        )
        
        if not search_results:
            answer, cits = "INTROUVABLE", []
            hits = []
        else:
            valid_results = search_results[:k]
            hits = [pages[res["doc_id"]] for res in valid_results]
            
            budget_chars = 4000
            chunks = []
            for h in hits[:k]:
                chunks.append(f"[{h['doc_id']}] p.{h['page']}\n{h['text'][:budget_chars]}")
                
            while True:
                msgs = [
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": "Extraits :\n\n" + "\n\n".join(chunks) + f"\n\nQuestion : {q['question']}"}
                ]
                if gen.count_tokens(msgs) <= 4096 - 16 or budget_chars < 500:
                    break
                budget_chars = int(budget_chars * 0.8)
                chunks = [c[:budget_chars + 40] for c in chunks]
                
            raw = gen.chat(msgs, max_new_tokens=400)
            answer, cits = _parse_citations(raw)
            if "INTROUVABLE" in raw.upper():
                answer, cits = "INTROUVABLE", []

        preds.append({
            "qid": q["qid"], 
            "answer": answer, 
            "citations": cits,
            "retrieved": [{"doc_id": h["doc_id"], "page": h["page"]} for h in hits[:k]]
        })
        print(f"[{q['qid']}] -> {answer[:80]}")
        
    write_predictions(out_path, preds, run_name="moteur-hybride-k12-alpha05")

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
    b.add_argument("--k", type=int, default=12)
    
    args = ap.parse_args()
    if args.cmd == "index":
        build_index(args.pdf_dir, args.index_dir)
    elif args.cmd == "update":
        update_index(args.index_dir, args.pdf_dir)
    else:
        answer_questions(args.index_dir, args.questions, args.out, args.backend, k=args.k)

if __name__ == "__main__":
    main()