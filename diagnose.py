
"""Diagnostic : pour chaque question, la page gold est-elle récupérée ? Le modèle s'est-il abstenu, et pourquoi ?
 
Usage :
    python diagnose.py --pred runs/val20.json --gold runs/val20_gold.json
"""
import argparse
import json
from pathlib import Path
 
 
def _find_records(obj):
    """Trouve la liste de dictionnaires contenant un 'qid', quelle que soit la structure du JSON."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj) and any("qid" in x for x in obj):
            return obj
        for x in obj:
            r = _find_records(x)
            if r:
                return r
    elif isinstance(obj, dict):
        for v in obj.values():
            r = _find_records(v)
            if r:
                return r
    return []
 
 
def _load(path):
    txt = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in txt.splitlines() if line.strip()]
    return _find_records(data)
 
 
def _pairs(obj):
    """Toutes les paires (doc_id, page) présentes dans un enregistrement."""
    out = set()
    if isinstance(obj, dict):
        if "doc_id" in obj and "page" in obj:
            try:
                out.add((str(obj["doc_id"]), int(obj["page"])))
            except (TypeError, ValueError):
                pass
        for v in obj.values():
            out |= _pairs(v)
    elif isinstance(obj, list):
        for x in obj:
            out |= _pairs(x)
    return out
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--debug", default=None, help="défaut : <pred>.debug.jsonl")
    a = ap.parse_args()
 
    pred = {r["qid"]: r for r in _load(a.pred)}
    gold_records = _load(a.gold)
    gold = {r["qid"]: r for r in gold_records}
 
    debug = {}
    dbg_path = Path(a.debug or (a.pred + ".debug.jsonl"))
    if dbg_path.exists():
        for line in dbg_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                debug[d["qid"]] = d
 
    if gold_records and not any(_pairs(r) for r in gold_records):
        print("[Info] Aucune paire (doc_id, page) trouvée dans le gold. Premier enregistrement :")
        print(json.dumps(gold_records[0], ensure_ascii=False, indent=2)[:800])
        print()
 
    counts = {}
    for qid in sorted(pred):
        p, g = pred[qid], gold.get(qid, {})
        g_pairs = _pairs(g)
        retrieved = [(str(r["doc_id"]), int(r["page"])) for r in p.get("retrieved", [])]
        rank = next((i + 1 for i, x in enumerate(retrieved) if x in g_pairs), None)
        abstained = str(p.get("answer", "")).strip().upper() == "INTROUVABLE"
        reason = debug.get(qid, {}).get("reason", "")
 
        if not g_pairs:
            cat = "SANS_PAGE_GOLD_OK" if abstained else "SANS_PAGE_GOLD_REPONDU"
            msg = "pas de page gold (vraiment introuvable ?) → " + ("abstention OK" if abstained else "hallucination probable")
        elif rank is None:
            cat = "RECHERCHE_ABSTENU" if abstained else "RECHERCHE_REPONDU"
            msg = "page gold ABSENTE du top-k → problème de RECHERCHE" + ("" if abstained else " (mais le modèle a répondu : risque)")
        elif abstained:
            cat = "GENERATEUR"
            msg = f"page gold au rang {rank} mais abstention → problème de GÉNÉRATEUR ({reason})"
        else:
            cat = "OK_REPONDU"
            msg = f"répondu, page gold au rang {rank}"
 
        counts[cat] = counts.get(cat, 0) + 1
        print(f"{qid}: {msg}")
 
    print("\n=== Résumé ===")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{cat:25s} {n}")
    print("\nGENERATEUR dominant → prompt / garde-fous (voir 'reason' dans le .debug.jsonl)."
          "\nRECHERCHE_* dominant → améliorer la récupération (chunking, alpha, k).")
 
 
if __name__ == "__main__":
    main()