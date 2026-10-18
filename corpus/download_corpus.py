"""Télécharge le corpus Ground Truth (documents publics de l'Insee et de la Cour des comptes).

Usage :
    python corpus/download_corpus.py      # corpus v1 dans corpus/pdf/, mise à jour v2 dans corpus/pdf_v2/

Le corpus v1 (110 documents) sert aux paliers 1 et 2. Les 20 documents de corpus/pdf_v2/ sont la
« mise à jour » du palier 3 : ne les mets pas dans ton index avant d'y être (voir README).

Source principale : la release GitHub « corpus-v1 » de ce dépôt, qui héberge une copie figée des
130 documents (un site officiel peut modifier un fichier après coup). Si la release est
indisponible, le script se rabat sur l'URL officielle du manifeste.

Les téléchargements se font 3 par 3. Aucune dépendance (bibliothèque standard uniquement).
Le script vérifie l'empreinte SHA-256 de chaque fichier : un fichier qui ne correspond pas au
manifeste est retéléchargé, puis refusé avec un message clair. Tout le monde travaille donc
exactement sur le même corpus.
À la fin, 10 documents sont transformés en « numérisations » (PDF image sans texte) par
corpus/make_scans.py, appelé automatiquement (nécessite pymupdf, pillow, numpy).
"""
from __future__ import annotations

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
RELEASE_URL = "https://github.com/Mathos34/ground-truth-kit/releases/download/corpus-v1/{doc_id}.pdf"
WORKERS = 3  # téléchargements en parallèle : assez pour aller vite, sans surcharger les serveurs publics


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: str | None = None, retries: int = 4) -> None:
    """Télécharge url vers dest. Si expected est donné, un fichier d'empreinte différente est
    traité comme un échec (réessai, puis exception) et n'est jamais écrit sur dest."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("la réponse n'est pas un PDF")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("empreinte SHA-256 différente de celle du manifeste")
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


def obtain(doc: dict, dest: Path, expected: str | None, origin_only: bool) -> None:
    """Récupère un document : release GitHub d'abord, site d'origine en secours."""
    sources = [("le site d'origine", doc["url"])]
    if not origin_only:
        sources.insert(0, ("la release GitHub corpus-v1", RELEASE_URL.format(doc_id=doc["doc_id"])))
    problems = []
    for k, (name, url) in enumerate(sources):
        try:
            fetch(url, dest, expected)
            if url == doc["url"]:
                time.sleep(0.5)  # politesse envers les serveurs publics
            return
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name} : {e}")
            if k + 1 < len(sources):
                print(f"    {doc['doc_id']} : échec depuis {name}, essai sur {sources[k + 1][0]}", flush=True)
    raise RuntimeError("aucune source n'a fourni une version conforme au manifeste (" + " ; ".join(problems) + ")")


def process(i: int, total: int, doc: dict, out: Path, out_v2: Path, scanned: set, write_checksums: bool) -> dict:
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
            # --write-checksums (orga) : on enregistre l'état actuel des sites d'origine, sans vérifier
            obtain(doc, dest, None if write_checksums else expected, origin_only=write_checksums)
        except Exception as e:  # noqa: BLE001
            print(f"    ERREUR {doc_id} : {e}", flush=True)
            return {"doc_id": doc_id, "error": True}
    return {"doc_id": doc_id, "digest": sha256(dest)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "manifest.json"))
    ap.add_argument("--out", default=str(HERE / "pdf"), help="dossier du corpus v1")
    ap.add_argument("--out-v2", default=str(HERE / "pdf_v2"), help="dossier des documents ajoutés en v2")
    ap.add_argument("--write-checksums", action="store_true",
                    help="(orga) enregistre les SHA-256 dans le manifeste, en téléchargeant depuis les sites d'origine")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_v2 = Path(args.out_v2)
    out_v2.mkdir(parents=True, exist_ok=True)

    errors = []
    docs = manifest["documents"]
    marker = out / ".scans_done.json"
    scanned = set(json.loads(marker.read_text())) if marker.exists() else set()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda a: process(a[0], len(docs), a[1], out, out_v2, scanned, args.write_checksums),
                                enumerate(docs, 1)))
    for doc, res in zip(docs, results):
        if res.get("error"):
            errors.append(res["doc_id"])
        elif args.write_checksums and "digest" in res:
            doc["sha256"] = res["digest"]

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
        print("Relance le script. Si l'erreur persiste, signale-la au mentor.", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
