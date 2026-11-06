"""Palier 4 « Red team » : vérifie ton fichier de questions pièges avant de le pousser.

    python -m groundtruth.redteam --questions redteam/questions.json --pdf-dir corpus/pdf --pdf-dir corpus/pdf_v2

Règles (voir README, palier 4) :
  - exactement 15 questions, sur le corpus v2 complet (130 documents) ;
  - au moins 4 questions sans réponse dans le corpus (`answerable: false`) ;
  - au moins 4 questions dont la preuve est un tableau, un graphique ou une page numérisée ;
  - pour chaque question avec réponse : réponse de référence, `key_values` (valeurs qui doivent
    apparaître), `evidence` (groupes de pages {doc_id, page}) et `source_quote` (extrait exact) ;
  - pour chaque question sans réponse : `notes` qui explique pourquoi c'est introuvable (recherches faites).

Ce script contrôle le format, l'existence des pages citées et, quand la page a une couche texte,
que l'extrait et les valeurs clés y figurent. Il ne juge pas si la question est bonne : c'est le
mentor qui tranche en cas de contestation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from .normalize import value_present

N_QUESTIONS = 15
MIN_UNANSWERABLE = 4
MIN_VISUAL = 4
TYPES = {"texte", "tableau", "graphique", "scan", "multi-doc", "temporel", "synthese", "piege"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace(" ", " ").replace(" ", " ")
    s = s.replace("’", "'").replace("«", '"').replace("»", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def load_pages(pdf_dirs: list[str]) -> dict[tuple[str, int], str]:
    import pymupdf

    pages = {}
    for d in pdf_dirs:
        for pdf in Path(d).glob("*.pdf"):
            with pymupdf.open(pdf) as doc:
                for i, page in enumerate(doc, 1):
                    pages[(pdf.stem, i)] = page.get_text("text")
    return pages


def check(questions: list[dict], pages: dict[tuple[str, int], str]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if len(questions) != N_QUESTIONS:
        errors.append(f"il faut exactement {N_QUESTIONS} questions (trouvé {len(questions)})")
    unans = sum(not q.get("answerable", True) for q in questions)
    if unans < MIN_UNANSWERABLE:
        errors.append(f"au moins {MIN_UNANSWERABLE} questions sans réponse (trouvé {unans})")
    visual = sum(q.get("type") in ("tableau", "graphique", "scan") for q in questions)
    if visual < MIN_VISUAL:
        errors.append(f"au moins {MIN_VISUAL} questions tableau/graphique/scan (trouvé {visual})")
    seen = set()
    for i, q in enumerate(questions, 1):
        tag = q.get("id", f"#{i}")
        if tag in seen:
            errors.append(f"{tag}: id en double")
        seen.add(tag)
        for k in ("id", "question", "type", "answerable"):
            if k not in q:
                errors.append(f"{tag}: champ '{k}' manquant")
        if q.get("type") not in TYPES:
            errors.append(f"{tag}: type inconnu {q.get('type')!r} (attendu : {sorted(TYPES)})")
        if not q.get("answerable", True):
            if not q.get("notes"):
                errors.append(f"{tag}: question sans réponse : explique tes recherches dans 'notes'")
            continue
        for k in ("answer", "evidence", "source_quote"):
            if not q.get(k):
                errors.append(f"{tag}: champ '{k}' manquant ou vide")
        all_text = ""
        for g in q.get("evidence", []):
            if not g:
                errors.append(f"{tag}: groupe d'evidence vide")
            for e in g:
                key = (e.get("doc_id"), e.get("page"))
                if key not in pages:
                    errors.append(f"{tag}: page inexistante {key}")
                else:
                    all_text += " " + pages[key]
        if all_text.strip() and q.get("type") in ("texte", "multi-doc", "temporel", "synthese"):
            # avertissements seulement : une valeur calculée (écart) ou un extrait coupé par la mise en
            # page peuvent être légitimes ; vérifie-les à la main
            if q.get("source_quote") and _norm(q["source_quote"])[:60] not in _norm(all_text):
                warnings.append(f"{tag}: source_quote non retrouvée telle quelle sur les pages citées")
            for kv in q.get("key_values", []):
                if not value_present(kv, all_text):
                    warnings.append(f"{tag}: valeur clé {kv!r} absente du texte des pages citées (calculée ?)")
        for kv in q.get("key_values", []):
            if not value_present(kv, q.get("answer", "")):
                errors.append(f"{tag}: valeur clé {kv!r} absente de ta propre réponse de référence")
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="redteam/questions.json")
    ap.add_argument("--pdf-dir", action="append", default=None,
                    help="dossier de PDF (répéter l'option : corpus/pdf et corpus/pdf_v2)")
    args = ap.parse_args()
    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    pages = load_pages(args.pdf_dir or ["corpus/pdf", "corpus/pdf_v2"])
    errors, warnings = check(questions, pages)
    for w in warnings:
        print("  (à vérifier)", w)
    if errors:
        print(f"{len(errors)} problème(s) :")
        for e in errors:
            print("  -", e)
        return 1
    print(f"OK : {len(questions)} questions au bon format. Pousse redteam/questions.json avant la date limite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
