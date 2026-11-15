"""Calibre le seuil d'abstention (--tau) SANS lancer le générateur.

1) python -m src.pipeline answer ... --retrieval-only --out runs/val_ret.json   (toutes les questions de validation)
2) python calibrate.py --debug runs/val_ret.json.debug.jsonl --gold <fichier gold> --gen-acc 0.7

Score de confiance estimé par question (barème de l'énoncé) :
  - on s'abstient (score < tau) : +1 si question piège, 0 sinon ;
  - on répond à une question piège : -1 ;
  - on répond à une vraie question : si la page gold est dans le contexte, le générateur a raison avec la
    probabilité --gen-acc (+1 / -1), sinon la réponse est supposée fausse (-1).
C'est une APPROXIMATION : confirme toujours avec le juge officiel (groundtruth.evaluate) sur 2-3 valeurs de tau.
Avec peu de questions la courbe est bruitée : utilise les 100 questions de validation.
"""
import argparse
import json
from pathlib import Path

from diagnose import _load, _pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--gen-acc", type=float, default=0.7)
    ap.add_argument("--target-rate", type=float, default=None,
                    help="taux d'abstention visé par le seuil (ex. 0.30) : affiche le tau correspondant")
    a = ap.parse_args()

    dbg = [json.loads(l) for l in Path(a.debug).read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = {r["qid"]: _pairs(r) for r in _load(a.gold)}

    items = []
    for d in dbg:
        g = gold.get(d["qid"], set())
        ctx = {(str(x[0]), int(x[1])) for x in d.get("ctx_pages", [])}
        items.append((d["top_score"], not g, bool(g & ctx)))   # (score, est_piège, gold_dans_ctx)
    n = len(items)
    n_trap = sum(1 for _, t, _ in items if t)
    print(f"{n} questions dont {n_trap} pièges (sans page gold)")

    def value(tau):
        tot = 0.0
        for s, trap, hit in items:
            if s < tau:
                tot += 1.0 if trap else 0.0
            elif trap:
                tot -= 1.0
            else:
                tot += (a.gen_acc * 1 + (1 - a.gen_acc) * -1) if hit else -1.0
        return tot / n

    if a.target_rate is not None:
        sc = sorted(s for s, _, _ in items)
        tau = sc[min(n - 1, int(a.target_rate * n))]
        ab = [(s, t, h) for s, t, h in items if s < tau]
        print(f"\n--- Cible : {a.target_rate:.0%} d'abstention par le seuil ---")
        print(f"tau = {tau:.6f}  -> {len(ab)}/{n} questions écartées avant le générateur")
        print(f"  dont pièges écartés (bien) : {sum(1 for _, t, _ in ab if t)}/{n_trap}")
        print(f"  dont vraies questions avec la bonne page écartées (perte) : {sum(1 for _, t, h in ab if (not t) and h)}")
        print(f"  score estimé à ce tau : {value(tau):+.3f}")
        print("Rappel : l'abstention totale = ce seuil + les INTROUVABLE dits par le modèle ; vise un peu moins que 30 % ici.\n")

    taus = sorted({0.0} | {round(s, 4) for s, _, _ in items})
    best = max(taus, key=value)
    print("tau       score estimé")
    step = max(1, len(taus) // 25)
    for t in taus[::step]:
        print(f"{t:<9.4f} {value(t):+.3f}{'   <- meilleur' if t == best else ''}")
    print(f"\nMeilleur tau = {best:.4f}  (score estimé {value(best):+.3f} ; sans seuil : {value(0.0):+.3f})")
    print("Prends plutôt une valeur un peu en dessous du meilleur tau si la courbe est plate (moins de sur-ajustement).")


if __name__ == "__main__":
    main()
