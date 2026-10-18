"""Script d'évaluation officiel.

    python -m groundtruth.evaluate --pred runs/val.json --gold data/validation.json --judge hf
    python -m groundtruth.evaluate --pred runs/val.json --gold data/validation.json --judge none   # rapide, CPU

Score de confiance (par question) :
    bonne réponse + citation valide ............ +1
    bonne réponse, citation invalide/absente ... +0,5
    INTROUVABLE sur une question qui a une réponse  0
    INTROUVABLE sur une question piège ......... +1
    question piège à fausse prémisse : corriger la prémisse (« en réalité... ») compte +1 aussi
    réponse fausse, ou réponse à une question piège  -1

Question de synthèse (champ `key_points`) : la réponse est juste si elle contient au moins
`min_points` des éléments attendus (chacun vérifié par le juge).

Une citation est valide si, pour CHAQUE groupe de preuves de la question (une question
multi-documents a un groupe par document), au moins une page citée appartient au groupe.
Plus de 5 citations : citation invalide.

Si les prédictions contiennent un champ optionnel `retrieved` (liste classée de pages
{doc_id, page}), le script calcule aussi Recall@5, Recall@10, MRR et nDCG@10 sur les
questions qui ont une réponse.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from . import MAX_CITATIONS
from .io import is_abstention, load_predictions, load_questions, validate_predictions
from .normalize import all_values_present


def citation_valid(citations: list[dict], evidence_groups: list[list[dict]]) -> bool:
    if not citations or len(citations) > MAX_CITATIONS:
        return False
    cited = {(c["doc_id"], int(c["page"])) for c in citations}
    for group in evidence_groups:
        if not cited & {(e["doc_id"], int(e["page"])) for e in group}:
            return False
    return True


def retrieval_metrics(retrieved: list[dict], evidence_groups: list[list[dict]]) -> dict:
    relevant = {(e["doc_id"], int(e["page"])) for g in evidence_groups for e in g}
    ranked = [(r["doc_id"], int(r["page"])) for r in retrieved]
    seen, dedup = set(), []
    for r in ranked:
        if r not in seen:
            seen.add(r)
            dedup.append(r)
    ranked = dedup

    def recall_at(k):  # proportion de groupes couverts dans le top-k
        top = set(ranked[:k])
        return sum(bool(top & {(e["doc_id"], int(e["page"])) for e in g}) for g in evidence_groups) / len(evidence_groups)

    rr = 0.0
    for i, r in enumerate(ranked, 1):
        if r in relevant:
            rr = 1 / i
            break
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(ranked[:10], 1) if r in relevant)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), 10) + 1))
    return {"recall@5": recall_at(5), "recall@10": recall_at(10), "mrr": rr, "ndcg@10": dcg / idcg if idcg else 0.0}


def score_question(q: dict, p: dict | None, judge) -> dict:
    answer = (p or {}).get("answer", "")
    citations = (p or {}).get("citations", []) or []
    abstained = p is None or is_abstention(answer)
    res = {"qid": q["qid"], "type": q["type"], "answerable": q["answerable"], "abstained": abstained,
           "correct": None, "citation_valid": None}
    if not q["answerable"]:
        if abstained:
            res["points"], res["outcome"] = 1.0, "abstention juste"
        elif q.get("false_premise") and judge.rejects_premise(q["question"], q["premise_correction"], answer):
            res["points"], res["outcome"] = 1.0, "prémisse corrigée"
        else:
            res["points"], res["outcome"] = -1.0, "hallucination (piège)"
        return res
    if abstained:
        res["points"] = 0.0
        res["outcome"] = "abstention"
        return res
    ok = all_values_present(q.get("key_values", []), answer)
    if ok and q.get("key_points"):  # question de synthèse : assez d'éléments attendus
        res["points_couverts"] = judge.count_points(q["question"], q["key_points"], answer)
        ok = res["points_couverts"] >= q.get("min_points", len(q["key_points"]))
    elif ok:
        ok = judge.is_correct(q["question"], q["answer"], answer)
    res["correct"] = ok
    if not ok:
        res["points"] = -1.0
        res["outcome"] = "réponse fausse"
        return res
    cv = citation_valid(citations, q["evidence"])
    res["citation_valid"] = cv
    res["points"] = 1.0 if cv else 0.5
    res["outcome"] = "juste + citation" if cv else "juste, citation invalide"
    return res


def evaluate(pred_path: str, gold_path: str, judge, verbose: bool = False) -> dict:
    questions = load_questions(gold_path)
    preds = load_predictions(pred_path)
    problems = validate_predictions(preds, questions)
    rows = [score_question(q, preds.get(q["qid"]), judge) for q in questions]

    n = len(rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["points"])
    outcomes = defaultdict(int)
    for r in rows:
        outcomes[r["outcome"]] += 1

    report = {
        "score_confiance": round(sum(r["points"] for r in rows) / n, 4),
        "n_questions": n,
        "issues": dict(outcomes),
        "par_type": {t: {"n": len(v), "score": round(sum(v) / len(v), 4)} for t, v in sorted(by_type.items())},
        "taux_abstention": round(sum(r["abstained"] for r in rows) / n, 4),
        "taux_hallucination": round(sum(r["points"] == -1.0 for r in rows) / n, 4),
        "problemes_format": problems,
    }

    ret = [retrieval_metrics(preds[q["qid"]]["retrieved"], q["evidence"])
           for q in questions if q["answerable"] and preds.get(q["qid"], {}).get("retrieved")]
    if ret:
        report["retrieval"] = {k: round(sum(m[k] for m in ret) / len(ret), 4) for k in ret[0]} | {"n": len(ret)}
    if verbose:
        report["details"] = rows
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--judge", default="hf", choices=["hf", "openai", "none"])
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-url", default=None, help="base_url du serveur OpenAI-compatible")
    ap.add_argument("--out", default=None, help="écrit le rapport JSON complet ici")
    ap.add_argument("--details", action="store_true", help="inclut le détail par question")
    args = ap.parse_args()

    from .judge import JUDGE_MODEL_ID, Judge

    judge = Judge(args.judge, model_id=args.judge_model or JUDGE_MODEL_ID, base_url=args.judge_url)
    report = evaluate(args.pred, args.gold, judge, verbose=args.details or bool(args.out))
    if args.judge == "none":
        report["avertissement"] = "juge heuristique (--judge none) : score indicatif, pas officiel"
    summary = {k: v for k, v in report.items() if k != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
