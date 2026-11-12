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
    "Tu es un expert en analyse de rapports publics français. Réponds à la question en te basant "
    "uniquement sur les extraits fournis.\n"
    "Chaque extrait est précédé de sa référence [doc_id p.N]. Les lignes [TABLEAU] contiennent des "
    "tableaux au format « colonne: valeur » : lis-les attentivement, la réponse y figure souvent.\n"
    "Règles strictes :\n"
    "- Une mauvaise réponse est pénalisée, alors qu'un INTROUVABLE ne l'est pas : "
    "ne réponds que si un extrait contient explicitement l'information demandée.\n"
    "- Si un extrait contient l'information, réponds : n'hésite pas à lire les chiffres dans les tableaux "
    "et à combiner plusieurs extraits.\n"
    "- Réponds directement, sans introduction (pas de « D'après les extraits... »), "
    "en 1 à 4 phrases précises : chiffres, dates, noms exacts tels qu'ils figurent dans l'extrait. "
    "N'invente rien et n'utilise aucune connaissance extérieure.\n"
    "- Termine par les sources au format [doc_id p.N], uniquement des pages qui contiennent la réponse.\n"
    "- Si l'information n'est pas explicitement dans les extraits, ou seulement en partie, ou si tu dois "
    "deviner ou extrapoler, réponds exactement et uniquement : INTROUVABLE.\n"
    "- N'écris jamais de phrase du type « les extraits ne permettent pas... » ou « aucun document ne mentionne... » : "
    "dans ce cas, réponds simplement INTROUVABLE."
)
 
MAX_NEW_TOKENS = 512
CONTEXT_TOKENS = 4096
REQUIRE_CITATION = True  # True : une réponse sans citation valide devient INTROUVABLE
 
 
def _tokenize(text: str) -> list[str]:
    """Normalisation et tokenisation robuste pour la recherche lexicale."""
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", text)
 
 
# Phrases "méta" typiques d'une non-réponse déguisée (testées uniquement au début de la sortie)
_META = re.compile(
    r"aucun (document|extrait|passage)"
    r"|extraits?( fournis?)? (ne |n')"
    r"|ne (permet|permettent|fournit|fournissent|mentionne|mentionnent|contient|contiennent) pas"
    r"|impossible de (déterminer|répondre|conclure)"
    r"|informations? (n'est|ne sont) pas (disponible|présente|mentionn)",
    re.IGNORECASE,
)
 
 
def _abstention_reason(raw: str, valid_cits: list[dict]) -> str:
    """Renvoie la raison de l'abstention, ou '' si la réponse est conservée."""
    if not raw.strip():
        return "reponse_vide"
    if re.sub(r"[^A-Za-z]", "", raw).upper() == "INTROUVABLE" or "INTROUVABLE" in raw:
        return "modele_a_dit_introuvable"
    if _META.search(raw[:300]):
        return "non_reponse_deguisee"
    if REQUIRE_CITATION and not valid_cits:
        return "aucune_citation_valide"
    return ""
 
 
def _best_window(text: str, question: str, size: int) -> str:
    """Retourne la fenêtre de `size` caractères la plus proche de la question
    (plus grand nombre de mots de la question présents), au lieu des N premiers caractères."""
    if len(text) <= size:
        return text
    q = set(_tokenize(question))
    step = max(size // 2, 1)
    best_start, best_score = 0, -1
    for start in range(0, len(text) - size + step, step):
        score = len(q & set(_tokenize(text[start:start + size])))
        if score > best_score:
            best_start, best_score = start, score
    return text[best_start:best_start + size]
 
 
def _tables_to_text(page) -> str:
    """Extrait les tableaux d'une page PyMuPDF et les met sous forme 'colonne: valeur'
    (une ligne par ligne du tableau), beaucoup plus lisible pour le modèle que le texte brut."""
    try:
        tables = page.find_tables().tables
    except Exception:
        return ""
 
    blocks = []
    for t in tables:
        try:
            rows = t.extract()
        except Exception:
            continue
        clean = []
        for r in rows:
            cells = [re.sub(r"\s+", " ", (c or "")).strip() for c in r]
            if any(cells):
                clean.append(cells)
        if len(clean) < 2:
            continue
        header = [h or f"col{i + 1}" for i, h in enumerate(clean[0])]
        lines = ["[TABLEAU] " + " | ".join(header)]
        for r in clean[1:]:
            pairs = [f"{h}: {v}" for h, v in zip(header, r) if v]
            if pairs:
                lines.append("; ".join(pairs))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
 
 
def _read_pages(pdf_dir: str) -> list[dict]:
    """Lecture des PDF : texte brut + tableaux structurés."""
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
 
                tables = _tables_to_text(page)
                if tables:
                    page_text = page_text.rstrip() + "\n\n" + tables
 
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
    cits = [{"doc_id": d.strip(), "page": int(p)} for d, p in re.findall(r"\[([^\[\]]+?)\s+p\.(\d+)\]", text)]
    answer = text.strip()
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
    preds, debug = [], []
 
    for q in load_questions(questions_path):
        gen.start_question(q["qid"])
 
        # 1. Recherche hybride
        search_results = hybrid_search_robust(
            query_text=q["question"],
            retriever=retriever,
            corpus_chunks=[p["text"] for p in pages],
            top_k=k,
            alpha=0.5
        )
 
        raw, reason = "", ""
        if not search_results:
            answer, cits, hits = "INTROUVABLE", [], []
            reason = "aucun_resultat_recherche"
        else:
            valid_results = search_results[:k]
            hits = [pages[res["doc_id"]] for res in valid_results]
 
            # 2. Contexte : fenêtre pertinente de chaque page, réduite pour tenir dans le budget
            window = 1500
            while True:
                chunks = [
                    f"[{h['doc_id']} p.{h['page']}]\n{_best_window(h['text'], q['question'], window)}"
                    for h in hits
                ]
                msgs = [
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": "Extraits :\n\n" + "\n\n".join(chunks)
                                                 + f"\n\nQuestion : {q['question']}"},
                ]
                if gen.count_tokens(msgs) <= CONTEXT_TOKENS - MAX_NEW_TOKENS - 16 or window < 400:
                    break
                window = int(window * 0.8)
 
            raw = gen.chat(msgs, max_new_tokens=MAX_NEW_TOKENS)
            answer, cits = _parse_citations(raw)
 
            # 3. Garde-fous : on ne garde que les citations de pages réellement récupérées
            retrieved_pages = {(h["doc_id"], h["page"]) for h in hits}
            cits = [c for c in cits if (c["doc_id"], c["page"]) in retrieved_pages]
            reason = _abstention_reason(raw, cits)
            if reason:
                answer, cits = "INTROUVABLE", []
 
        retrieved = [{"doc_id": h["doc_id"], "page": h["page"]} for h in hits[:k]]
        preds.append({
            "qid": q["qid"],
            "answer": answer,
            "citations": cits,
            "retrieved": retrieved
        })
        debug.append({
            "qid": q["qid"],
            "question": q["question"],
            "reason": reason,
            "raw": raw,
            "retrieved": retrieved
        })
        print(f"[{q['qid']}] -> {answer[:80]}" + (f"   ({reason})" if reason else ""))
 
    write_predictions(out_path, preds, run_name="moteur-hybride-k12-alpha05")
 
    # Fichier de diagnostic : sortie brute du modèle + raison de chaque abstention
    debug_path = Path(str(out_path) + ".debug.jsonl")
    debug_path.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in debug), encoding="utf-8"
    )
    print(f"Diagnostic écrit dans {debug_path}")
 
 
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
 
    