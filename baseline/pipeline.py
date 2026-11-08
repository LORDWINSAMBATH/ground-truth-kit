
from __future__ import annotations

import argparse
import json
import re 
from pathlib import Path
import sys
import os
import pickle
import unicodedata

# Importations propres des modules de ton projet (depuis le dossier src)
from src.hybrid_retrieved import hybrid_search_robust
from src.bm25_scratch import BM25FromScratch


PROMPT_SYSTEM = (
    "Tu réponds à des questions sur des rapports publics français, uniquement à partir des extraits fournis.\n"
    "Chaque extrait est précédé de sa référence [doc_id p.N].\n"
    "Si la réponse n'est pas dans les extraits, réponds exactement : INTROUVABLE."
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
                # Extraction propre et directe du texte de la page (sans surcharge IA)
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
    
    # Utilisation de ton BM25 fait maison
    retriever = BM25FromScratch()
    corpus_texts = [p["text"] for p in pages]
    retriever.fit(corpus_texts)
    
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde de l'objet retriever avec pickle
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

def answer_questions(index_dir: str, questions_path: str, out_path: str, backend: str, k: int) -> None:
    from groundtruth.generator import OfficialGenerator
    from groundtruth.io import load_questions, write_predictions
    idx = Path(index_dir)
    
    # Recharge le BM25 sauvegardé
    with open(idx / "bm25.pkl", "rb") as f:
        retriever = pickle.load(f)
        
    pages = json.loads((idx / "pages.json").read_text(encoding="utf-8"))
    gen = OfficialGenerator(backend=backend)
    preds = []
    
    for q in load_questions(questions_path):
        gen.start_question(q["qid"])
        
        # 1. Appel direct du moteur hybride robuste (Fusion Score BM25 + Sémantique Dense)
        search_results = hybrid_search_robust(
            query_text=q["question"],
            retriever=retriever,
            corpus_chunks=[p["text"] for p in pages],
            top_k=10,
            alpha=0.2  # Équilibre optimal pour privilégier la précision lexicale et sémantique
        )
        
        # 2. Calibration de l'abstention et filtrage des faux positifs (Évite les hallucinations / pénalités -1 pt)
        CONFIDENCE_THRESHOLD = 0.07
        valid_results = [res for res in search_results if res.get("score", 0.0) >= CONFIDENCE_THRESHOLD]
        
        if not valid_results:
            answer, cits = "INTROUVABLE", []
            hits = []
        else:
            hits = [pages[res["doc_id"]] for res in valid_results]
            
            budget_chars = 2600
            chunks = []
            for h in hits[:k]:
                chunks.append(f"[{h['doc_id']}] p.{h['page']}\n{h['text'][:budget_chars]}")
                
            while True:
                msgs = [
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": "Extraits :\n\n" + "\n\n".join(chunks) + f"\n\nQuestion : {q['question']}"}
                ]
                if gen.count_tokens(msgs) <= 4096 - 16 or budget_chars < 300:
                    break
                budget_chars = int(budget_chars * 0.8)
                chunks = [c[:budget_chars + 40] for c in chunks]
                
            raw = gen.chat(msgs, max_new_tokens=200)
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
        
    write_predictions(out_path, preds, run_name="moteur-hybride-robuste-calibre")

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
        answer_questions(args.index_dir, args.questions, args.out, args.backend, k=10)

if __name__ == "__main__":
    main()