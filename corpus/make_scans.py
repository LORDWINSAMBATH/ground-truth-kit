"""Transforme 10 documents du corpus en « numérisations » (PDF image, sans couche texte).

    python corpus/make_scans.py            # après download_corpus.py

Ces documents simulent des pages scannées : rendu en niveaux de gris, légère rotation,
bruit, flou et compression JPEG. Le PDF natif est remplacé : il n'y a plus de texte à
extraire, il faut passer par l'OCR. Le résultat est déterministe (graine fixe).
Dépendances : pymupdf, pillow, numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCAN_DOCS = [
    "insee-ip2036", "insee-ip2041", "insee-ip2055", "insee-ip2075", "insee-ip2088",
    "insee-ip2103", "insee-ip2119", "insee-ip2127", "cc-pac-synthese", "cc-vaccination-synthese",
]


def degrade(img, seed: int):
    import numpy as np
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(seed)
    img = img.convert("L")
    angle = float(rng.uniform(-1.6, 1.6))
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    arr = np.asarray(img).astype(np.float32)
    arr = arr * rng.uniform(0.86, 0.95) + rng.uniform(8, 22)        # papier un peu gris, encre moins noire
    arr += rng.normal(0, 9, arr.shape)                              # grain
    h, w = arr.shape
    yy = np.linspace(-1, 1, h)[:, None]
    arr -= 18 * (yy ** 2)                                           # ombre de reliure en haut et en bas
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 0.8))))
    return img


def scan_pdf(src: Path, dpi: int = 130) -> bytes:
    import pymupdf
    from PIL import Image

    out = pymupdf.open()
    with pymupdf.open(src) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            seed = int(hashlib.sha256(f"{src.stem}-{i}".encode()).hexdigest()[:8], 16)
            img = degrade(img, seed)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=55)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=buf.getvalue())
    return out.tobytes(deflate=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(HERE / "pdf"))
    args = ap.parse_args()
    main_dir(Path(args.pdf_dir))


def main_dir(pdf_dir: Path) -> None:
    import pymupdf  # noqa: F401  (vérifie la dépendance avant de toucher aux fichiers)
    import numpy  # noqa: F401
    import PIL  # noqa: F401

    marker = pdf_dir / ".scans_done.json"
    done = json.loads(marker.read_text()) if marker.exists() else []
    for doc_id in SCAN_DOCS:
        if doc_id in done:
            print(f"déjà numérisé : {doc_id}")
            continue
        src = pdf_dir / f"{doc_id}.pdf"
        data = scan_pdf(src)
        src.write_bytes(data)
        done.append(doc_id)
        marker.write_text(json.dumps(done))
        print(f"numérisé : {doc_id}")


if __name__ == "__main__":
    main()
