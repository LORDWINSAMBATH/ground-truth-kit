"""Lecture des questions et écriture / validation des fichiers de réponses."""
from __future__ import annotations

import json
from pathlib import Path

from . import ABSTAIN, MAX_CITATIONS


def load_questions(path: str | Path) -> list[dict]:
    """Charge un fichier de questions (validation ou classement)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def is_abstention(answer: str) -> bool:
    return answer.strip().strip(".").upper() == ABSTAIN


def write_predictions(path: str | Path, predictions: list[dict], run_name: str = "", commit: str = "") -> None:
    """Écrit un fichier de réponses au format officiel.

    Chaque prédiction : {"qid": str, "answer": str, "citations": [{"doc_id": str, "page": int}],
                         "retrieved": [{"doc_id": str, "page": int}]  (optionnel, classé)}
    """
    payload = {"run_name": run_name, "commit": commit, "predictions": predictions}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_predictions(path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    preds = data["predictions"] if isinstance(data, dict) else data
    return {p["qid"]: p for p in preds}


def validate_predictions(preds: dict[str, dict], questions: list[dict], doc_ids: set[str] | None = None) -> list[str]:
    """Renvoie la liste des problèmes de format (vide si tout va bien)."""
    problems = []
    qids = {q["qid"] for q in questions}
    missing = qids - set(preds)
    extra = set(preds) - qids
    if missing:
        problems.append(f"{len(missing)} question(s) sans réponse, ex. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} qid inconnu(s), ex. {sorted(extra)[:3]}")
    for qid, p in preds.items():
        if not isinstance(p.get("answer"), str):
            problems.append(f"{qid}: 'answer' doit être une chaîne")
            continue
        cits = p.get("citations", [])
        if not isinstance(cits, list):
            problems.append(f"{qid}: 'citations' doit être une liste")
            continue
        if len(cits) > MAX_CITATIONS:
            problems.append(f"{qid}: {len(cits)} citations (max {MAX_CITATIONS}, les citations seront jugées invalides)")
        for c in cits:
            if not (isinstance(c, dict) and isinstance(c.get("doc_id"), str) and isinstance(c.get("page"), int)):
                problems.append(f"{qid}: citation mal formée {c!r} (attendu {{'doc_id': str, 'page': int}})")
            elif doc_ids is not None and c["doc_id"] not in doc_ids:
                problems.append(f"{qid}: doc_id inconnu {c['doc_id']!r}")
    return problems
