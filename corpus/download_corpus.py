"""Télécharge le corpus Ground Truth depuis les sites officiels (Insee, Cour des comptes).

Usage :
    python corpus/download_corpus.py      # corpus v1 dans corpus/pdf/, mise à jour v2 dans corpus/pdf_v2/

Le corpus v1 (110 documents) sert aux paliers 1 et 2. Les 20 documents de corpus/pdf_v2/ sont la
« mise à jour » du palier 3 : ne les mets pas dans ton index avant d'y être (voir README).

Les téléchargements se font 3 par 3. Aucune dépendance (bibliothèque standard uniquement).
Le script vérifie l'empreinte SHA-256 de chaque fichier : tout le monde travaille donc exactement
sur le même corpus.
À la fin, 10 documents sont transformés en « numérisations » (PDF image sans texte) par
corpus/make_scans.py, appelé automatiquement (nécessite pymupdf, pillow, numpy).
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = "Mozilla/5.0 (compatible; ground-truth-kit/1.0; +https://piscine.horama.ai)"
WORKERS = 3  # téléchargements en parallèle : assez pour aller vite, sans surcharger les serveurs publics


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, retries: int = 4) -> None:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("la réponse n'est pas un PDF")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"    {dest.stem} : tentative {attempt} échouée ({e}), nouvel essai dans {wait}s", flush=True)
            time.sleep(wait)


def process(i: int, total: int, doc: dict, out: Path, out_v2: Path, scanned: set) -> dict:
    """Télécharge un document si besoin, puis calcule son empreinte (exécuté dans un thread)."""
    doc_id = doc["doc_id"]
    dest = (out_v2 if doc.get("corpus_version") == 2 else out) / f"{doc_id}.pdf"
    expected = doc.get("sha256")
    if doc_id in scanned and dest.exists():
        print(f"[{i}/{total}] déjà présent (numérisé)  {doc_id}", flush=True)
        return {"doc_id": doc_id}
    if dest.exists() and (not expected or sha256(dest) == expected):
        print(f"[{i}/{total}] déjà présent  {doc_id}", flush=True)
    else:
        print(f"[{i}/{total}] téléchargement {doc_id}", flush=True)
        try:
            fetch(doc["url"], dest)
            time.sleep(0.5)  # politesse envers les serveurs publics
        except Exception as e:  # noqa: BLE001
            print(f"    ERREUR {doc_id} : {e}", flush=True)
            return {"doc_id": doc_id, "error": True}
    return {"doc_id": doc_id, "digest": sha256(dest), "expected": expected}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "manifest.json"))
    ap.add_argument("--out", default=str(HERE / "pdf"), help="dossier du corpus v1")
    ap.add_argument("--out-v2", default=str(HERE / "pdf_v2"), help="dossier des documents ajoutés en v2")
    ap.add_argument("--write-checksums", action="store_true",
                    help="(orga) enregistre les SHA-256 dans le manifeste")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_v2 = Path(args.out_v2)
    out_v2.mkdir(parents=True, exist_ok=True)

    errors, mismatches = [], []
    docs = manifest["documents"]
    marker = out / ".scans_done.json"
    scanned = set(json.loads(marker.read_text())) if marker.exists() else set()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda a: process(a[0], len(docs), a[1], out, out_v2, scanned),
                                enumerate(docs, 1)))
    for doc, res in zip(docs, results):
        if res.get("error"):
            errors.append(res["doc_id"])
        elif "digest" in res:
            if args.write_checksums:
                doc["sha256"] = res["digest"]
            elif res["expected"] and res["digest"] != res["expected"]:
                mismatches.append(res["doc_id"])

    if args.write_checksums:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Empreintes SHA-256 écrites dans le manifeste.", flush=True)

    n2 = sum(d.get("corpus_version") == 2 for d in docs)
    print(f"\n{len(docs) - len(errors)}/{len(docs)} documents : {len(docs) - n2} dans {out}, {n2} dans {out_v2}",
          flush=True)
    if not args.write_checksums and not errors:
        try:
            sys.path.insert(0, str(HERE))
            import make_scans
            print("Génération des pages numérisées...", flush=True)
            make_scans.main_dir(out)
        except ImportError as e:
            print(f"Installe d'abord les dépendances (pip install -r requirements.txt), puis lance "
                  f"python corpus/make_scans.py  ({e})", flush=True)
    if errors:
        print("Échecs :", ", ".join(errors), flush=True)
    if mismatches:
        print("ATTENTION, empreinte différente (le document a changé en ligne) :", ", ".join(mismatches), flush=True)
        print("Signale-le au mentor : le classement se fait sur la version du manifeste.", flush=True)
    return 1 if errors or mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
