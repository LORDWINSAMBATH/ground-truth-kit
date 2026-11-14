
from __future__ import annotations
 
import argparse
import json
import os
import re
from pathlib import Path
 
import numpy as np
 
try:  # exécution en package : python -m src.pipeline  (ou python -m baseline.pipeline)
    from .bm25_scratch import BM25FromScratch
    from .chunking_scratch import chunk_text
    from . import hybrid_retrieved as hy
    from .reranker import rerank
except ImportError:  # exécution en script
    from bm25_scratch import BM25FromScratch
    from chunking_scratch import chunk_text
    import hybrid_retrieved as hy
    from reranker import rerank
 
 
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
 
# --- Budgets imposés par le règlement : 4096 tokens de contexte, 20 s par question sur T4 ---
MAX_NEW_TOKENS = 256          # 512 risque de dépasser les 20 s ; les réponses font 1 à 4 phrases
CONTEXT_TOKENS = 4096
 
# --- Réglages de la recherche ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 30
N_CANDIDATES = 40             # candidats pris dans BM25 et dans le dense avant fusion
N_RERANK = 30                 # nombre de chunks fusionnés envoyés au reranker
N_CTX = 6                     # nombre de chunks maximum donnés au générateur
 
# --- Garde-fous ---
REQUIRE_CITATION = True
FALLBACK_CITATION = True      # réponse sans citation valide -> on cite la page du meilleur chunk (+1 au lieu de +0,5 si correct)
 
_META = re.compile(
    r"aucun (document|extrait|passage)"
    r"|extraits?( fournis?)? (ne |n')"
    r"|ne (permet|permettent|fournit|fournissent|mentionne|mentionnent|contient|contiennent) pas"
    r"|impossible de (déterminer|répondre|conclure)"
    r"|informations? (n'est|ne sont) pas (disponible|présente|mentionn)",
    re.IGNORECASE,
)
 
 
def _abstention_reason(raw: str) -> str:
    """Raison de l'abstention d'après la sortie brute du modèle, ou '' si la réponse est conservée."""
    if not raw.strip():
        return "reponse_vide"
    if "INTROUVABLE" in raw.upper():
        return "modele_a_dit_introuvable"
    if _META.search(raw[:300]):
        return "non_reponse_deguisee"
    return ""
 
 
# ----------------------------------------------------------------------------- lecture des PDF
# Pages avec énormément de dessins vectoriels (graphiques, cartes) : find_tables y devient très lent
# (c'est de la géométrie sur CPU, pas un modèle). On les saute : le texte brut de la page reste indexé.
MAX_DRAWINGS = int(os.environ.get("MAX_DRAWINGS", "1500"))
 
 
def _tables_to_text(page) -> str:
    """Tableaux d'une page PyMuPDF sous la forme 'colonne: valeur' (une ligne par ligne de tableau)."""
    try:
        if len(page.get_cdrawings()) > MAX_DRAWINGS:
            return ""
    except Exception:
        pass
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
 
 
def _read_pages(pdf_dir: str, skip: set[str] | None = None, max_docs: int | None = None,
                tables: bool = True) -> list[dict]:
    """Lecture des PDF : texte brut + tableaux structurés. `skip` : doc_id déjà indexés.
    `max_docs` : ne lire que les N premiers PDF (test rapide). `tables=False` : sans extraction de tableaux (plus rapide)."""
    import time
 
    import pymupdf
 
    skip = skip or set()
    pages = []
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"[Attention] Le dossier {pdf_dir} est introuvable ou vide.")
        return pages
 
    files = [f for f in sorted(pdf_path.glob("*.pdf")) if f.stem not in skip]
    if max_docs:
        files = files[:max_docs]
    for n, pdf in enumerate(files, 1):
        t0 = time.time()
        with pymupdf.open(pdf) as doc:
            for i, page in enumerate(doc, 1):
                try:
                    page_text = page.get_text("text")
                except Exception:
                    page_text = page.get_text()
                tab = _tables_to_text(page) if tables else ""
                if tab:
                    page_text = page_text.rstrip() + "\n\n" + tab
                pages.append({"doc_id": pdf.stem, "page": i, "text": page_text})
        print(f"[PDF {n}/{len(files)}] {pdf.name} : {time.time() - t0:.1f} s", flush=True)
    return pages
 
 
# ----------------------------------------------------------------------------- indexation
def _index_text(c: dict) -> str:
    """Texte indexé (BM25 + embeddings) : le nom du document aide les questions qui le citent."""
    return f"{c['doc_id']}\n{c['text']}"
 
 
def _make_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    for p in pages:
        for t in chunk_text(p["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            if len(t.strip()) >= MIN_CHUNK_CHARS:
                chunks.append({"doc_id": p["doc_id"], "page": p["page"], "text": t})
    return chunks
 
 
def _write_index(index_dir: str, pages: list[dict], chunks: list[dict], emb: np.ndarray) -> None:
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(out / "emb.npy", emb)
    (out / "meta.json").write_text(json.dumps({"embed_model": hy.EMBED_MODEL}), encoding="utf-8")
 
 
def build_index(pdf_dir: str, index_dir: str, pages: list[dict] | None = None,
                max_docs: int | None = None, tables: bool = True, reuse_pages: bool = False) -> None:
    cache = Path(index_dir) / "pages.json"
    if pages is None and reuse_pages and cache.exists():
        pages = json.loads(cache.read_text(encoding="utf-8"))
        print(f"{len(pages)} pages relues depuis {cache} (PDF non relus).")
    if pages is None:
        pages = _read_pages(pdf_dir, max_docs=max_docs, tables=tables)
        cache.parent.mkdir(parents=True, exist_ok=True)       # sauvegarde AVANT les embeddings : la lecture des PDF est la partie lente
        cache.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    empty = sum(1 for p in pages if len(p["text"].strip()) < 30)
    chunks = _make_chunks(pages)
    emb = hy.embed_texts([_index_text(c) for c in chunks], show_progress=True)
    _write_index(index_dir, pages, chunks, emb)
    print(f"{len(pages)} pages, {len(chunks)} chunks indexés ({empty} pages vides/scannées : OCR à envisager).")
 
 
def update_index(index_dir: str, pdf_dir: str) -> None:
    idx = Path(index_dir)
    pages = json.loads((idx / "pages.json").read_text(encoding="utf-8"))
    chunks = json.loads((idx / "chunks.json").read_text(encoding="utf-8"))
    emb = np.load(idx / "emb.npy")
    meta = idx / "meta.json"
    if meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("embed_model") != hy.EMBED_MODEL:
        raise SystemExit("Index construit avec un autre modèle d'embedding : reconstruis l'index.")
 
    known = {p["doc_id"] for p in pages}
    new_pages = _read_pages(pdf_dir, skip=known)
    new_chunks = _make_chunks(new_pages)
    if new_chunks:
        new_emb = hy.embed_texts([_index_text(c) for c in new_chunks], show_progress=True)
        emb = np.vstack([emb, new_emb])
        chunks = chunks + new_chunks
    pages = pages + new_pages
    _write_index(index_dir, pages, chunks, emb)
    print(f"{len(new_pages)} nouvelles pages, {len(new_chunks)} nouveaux chunks ajoutés.")
 
 
def _load_index(index_dir: str):
    idx = Path(index_dir)
    chunks = json.loads((idx / "chunks.json").read_text(encoding="utf-8"))
    emb = np.load(idx / "emb.npy")
    meta = idx / "meta.json"
    if meta.exists():
        built_with = json.loads(meta.read_text(encoding="utf-8")).get("embed_model")
        if built_with != hy.EMBED_MODEL:
            raise SystemExit(f"Index construit avec {built_with} mais EMBED_MODEL={hy.EMBED_MODEL} : "
                             "même modèle pour indexer et répondre (ou reconstruis l'index).")
    bm25 = BM25FromScratch()          # reconstruit au chargement (quelques secondes) : pas de pickle fragile
    bm25.fit([_index_text(c) for c in chunks])
    return chunks, emb, bm25
 
 
# ----------------------------------------------------------------------------- réponses
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
 
 
def _build_messages(question: str, ctx_chunks: list[dict]) -> list[dict]:
    parts = [f"[{c['doc_id']} p.{c['page']}]\n{c['text']}" for c in ctx_chunks]
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": "Extraits :\n\n" + "\n\n".join(parts) + f"\n\nQuestion : {question}"},
    ]
 
 
def answer_questions(index_dir: str, questions_path: str, out_path: str, backend: str, k: int = 12,
                     mode: str = "hybrid", use_rerank: bool = True, tau: float = 0.0,
                     n_ctx: int = N_CTX, retrieval_only: bool = False) -> None:
    from groundtruth.io import load_questions, write_predictions
 
    chunks, emb, bm25 = _load_index(index_dir)
    gen = None
    if not retrieval_only:
        from groundtruth.generator import OfficialGenerator
        gen = OfficialGenerator(backend=backend)
 
    preds, debug = [], []
    for q in load_questions(questions_path):
        if gen is not None:
            gen.start_question(q["qid"])
 
        # 1. Recherche hybride (BM25 + dense, fusion RRF) puis reranking
        cands = hy.hybrid_search(q["question"], bm25, emb, mode=mode,
                                 n_candidates=N_CANDIDATES, top_k=N_RERANK)
        top_score = 0.0
        if cands and use_rerank:
            scores = rerank(q["question"], [chunks[c["idx"]]["text"] for c in cands])
            for c, s in zip(cands, scores):
                c["rr"] = s
            cands.sort(key=lambda c: -c["rr"])
            top_score = cands[0]["rr"]
        elif cands:
            top_score = max([c["cos"] for c in cands if c["cos"] is not None], default=0.0)
 
        # pages distinctes, dans l'ordre du classement (pour diagnose.py et les métriques)
        seen_pages: list[tuple[str, int]] = []
        for c in cands:
            key = (chunks[c["idx"]]["doc_id"], chunks[c["idx"]]["page"])
            if key not in seen_pages:
                seen_pages.append(key)
        retrieved = [{"doc_id": d, "page": p} for d, p in seen_pages[:k]]
 
        ctx = cands[:n_ctx]
        raw, reason, answer, cits = "", "", "INTROUVABLE", []
 
        if not cands:
            reason = "aucun_resultat_recherche"
        elif top_score < tau:
            reason = "score_sous_seuil"
        elif retrieval_only:
            reason = "retrieval_only"
        else:
            # 2. Contexte : on retire les chunks les moins bien classés jusqu'à tenir dans le budget
            while True:
                msgs = _build_messages(q["question"], [chunks[c["idx"]] for c in ctx])
                if gen.count_tokens(msgs) <= CONTEXT_TOKENS - MAX_NEW_TOKENS - 16 or len(ctx) <= 1:
                    break
                ctx = ctx[:-1]
 
            raw = gen.chat(msgs, max_new_tokens=MAX_NEW_TOKENS)
            answer, cits = _parse_citations(raw)
 
            # 3. Garde-fous : citations limitées aux pages réellement envoyées au modèle
            ctx_pages = {(chunks[c["idx"]]["doc_id"], chunks[c["idx"]]["page"]) for c in ctx}
            cits = [c for c in cits if (c["doc_id"], c["page"]) in ctx_pages]
            reason = _abstention_reason(raw)
            if not reason and not cits:
                if FALLBACK_CITATION:
                    top = chunks[ctx[0]["idx"]]
                    cits = [{"doc_id": top["doc_id"], "page": top["page"]}]
                elif REQUIRE_CITATION:
                    reason = "aucune_citation_valide"
            if reason:
                answer, cits = "INTROUVABLE", []
 
        ctx_list = sorted({(chunks[c["idx"]]["doc_id"], chunks[c["idx"]]["page"]) for c in ctx})
        preds.append({"qid": q["qid"], "answer": answer, "citations": cits, "retrieved": retrieved})
        debug.append({"qid": q["qid"], "question": q["question"], "reason": reason, "raw": raw,
                      "top_score": top_score, "retrieved": retrieved,
                      "ctx_pages": [list(p) for p in ctx_list]})
        print(f"[{q['qid']}] score={top_score:.3f} -> {answer[:80]}" + (f"   ({reason})" if reason else ""))
 
    run_name = f"{mode}-rr{int(use_rerank)}-tau{tau}-k{k}-ctx{n_ctx}"
    write_predictions(out_path, preds, run_name=run_name)
 
    debug_path = Path(str(out_path) + ".debug.jsonl")
    debug_path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in debug), encoding="utf-8")
    print(f"Diagnostic écrit dans {debug_path}")
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
 
    a = sub.add_parser("index")
    a.add_argument("--pdf-dir", required=True)
    a.add_argument("--index-dir", required=True)
    a.add_argument("--max-docs", type=int, default=None, help="indexer seulement N PDF (test rapide)")
    a.add_argument("--no-tables", action="store_true", help="sans extraction de tableaux (plus rapide)")
    a.add_argument("--reuse-pages", action="store_true",
                   help="réutiliser pages.json s'il existe (ne relit pas les PDF : utile pour changer chunks/embeddings)")
 
    u = sub.add_parser("update")
    u.add_argument("--index-dir", required=True)
    u.add_argument("--pdf-dir", required=True)
 
    b = sub.add_parser("answer")
    b.add_argument("--index-dir", required=True)
    b.add_argument("--questions", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--backend", default="hf", choices=["hf", "openai", "echo"])
    b.add_argument("--k", type=int, default=12)
    b.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "dense"])
    b.add_argument("--no-rerank", action="store_true")
    b.add_argument("--tau", type=float, default=0.0, help="seuil d'abstention sur le score de pertinence")
    b.add_argument("--n-ctx", type=int, default=N_CTX)
    b.add_argument("--retrieval-only", action="store_true", help="n'appelle pas le générateur (évaluer la recherche)")
 
    args = ap.parse_args()
    if args.cmd == "index":
        build_index(args.pdf_dir, args.index_dir, max_docs=args.max_docs, tables=not args.no_tables,
                    reuse_pages=args.reuse_pages)
    elif args.cmd == "update":
        update_index(args.index_dir, args.pdf_dir)
    else:
        answer_questions(args.index_dir, args.questions, args.out, args.backend, k=args.k,
                         mode=args.mode, use_rerank=not args.no_rerank, tau=args.tau,
                         n_ctx=args.n_ctx, retrieval_only=args.retrieval_only)
 
 
if __name__ == "__main__":
    main()