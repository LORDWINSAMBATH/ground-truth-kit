"""Prépare une soumission au leaderboard (set de classement).

    python -m groundtruth.submit --pred runs/classement_v1.json --phase v1
    python -m groundtruth.submit --pred runs/classement_v2.json --phase v2     # à partir du 2 novembre

Le script :
  1. vérifie le format (toutes les questions du set de classement, citations bien formées) ;
  2. vérifie que tu n'as pas déjà 2 soumissions aujourd'hui (heure de Paris) ;
  3. copie le fichier dans submissions/<phase>/AAAA-MM-JJ_N.json avec le hash du commit courant.
Les 2 soumissions par jour sont comptées par phase.
Il te reste à faire `git add submissions/ && git commit && git push`.
Le mentor relève les soumissions poussées sur ton dépôt et met à jour LEADERBOARD.md.
Une soumission compte dès qu'elle est poussée : ne pousse pas de fichier de test dans submissions/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .io import load_predictions, load_questions, validate_predictions

MAX_PER_DAY = 2
_DATA = Path(__file__).resolve().parents[1] / "data"
PHASES = {"v1": _DATA / "classement_v1_questions.json", "v2": _DATA / "classement_v2_questions.json"}
V2_OPENS = "2026-11-02"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--dir", default="submissions")
    args = ap.parse_args()

    today = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d")
    if args.phase == "v2" and today < V2_OPENS:
        print(f"Le leaderboard v2 ouvre le {V2_OPENS}. D'ici là, soumets sur la phase v1.")
        return 1
    qs = load_questions(PHASES[args.phase])
    preds = load_predictions(args.pred)
    problems = validate_predictions(preds, qs)
    if problems:
        print("Soumission refusée :")
        for p in problems[:20]:
            print("  -", p)
        return 1

    d = Path(args.dir) / args.phase
    d.mkdir(parents=True, exist_ok=True)
    existing = sorted(d.glob(f"{today}_*.json"))
    if len(existing) >= MAX_PER_DAY:
        print(f"Déjà {MAX_PER_DAY} soumissions aujourd'hui ({today}). Réessaie demain.")
        return 1
    dest = d / f"{today}_{len(existing) + 1}.json"
    data = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    payload = {"phase": args.phase, "submitted_at": datetime.now(ZoneInfo("Europe/Paris")).isoformat(timespec="seconds"),
               "commit": _commit(),
               "predictions": [{k: v for k, v in p.items() if k in ("qid", "answer", "citations")}
                               for p in (data["predictions"] if isinstance(data, dict) else data)]}
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Soumission prête : {dest}")
    print(f'Maintenant : git add {dest} && git commit -m "soumission {dest.stem}" && git push')
    return 0


if __name__ == "__main__":
    sys.exit(main())
