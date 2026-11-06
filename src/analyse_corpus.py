import sys
import io
import os
from pathlib import Path
import pymupdf
import unicodedata
from collections import Counter

# Force l'encodage UTF-8 pour la console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def normalize_text(text: str) -> str:
    """Normalise le texte pour ignorer les variations d'accents lors de la recherche de mots-clés."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()

def analyze_corpus_types(pdf_dir="ground-truth-kit/corpus/pdf"):
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"Le dossier {pdf_dir} est introuvable. Verifie le chemin.")
        return

    types_count = Counter()
    total_docs = 0

    print(f"Analyse detaillee et prise en compte des accents dans '{pdf_dir}'...\n")
    print("--- DETAIL PAR DOCUMENT ---")

    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        total_docs += 1
        doc_id = pdf_file.stem
        doc_type = "Autre / Non classe"
        
        # 1. Analyse basee sur les prefixes du nom de fichier
        if "insee" in doc_id.lower():
            doc_type = "Publication INSEE (Conjoncture / Flash)"
        
        # 2. Analyse du contenu de la premiere page avec normalisation des accents
        try:
            with pymupdf.open(pdf_file) as doc:
                if len(doc) > 0:
                    raw_text = doc[0].get_text("text")
                    first_page_text = normalize_text(raw_text)
                    
                    if "rapport" in first_page_text or "commission" in first_page_text:
                        doc_type = "Rapport Institutionnel / Commission"
                    elif "bilan" in first_page_text or "financier" in first_page_text or "budget" in first_page_text:
                        doc_type = "Document Budgetaire / Financier"
                    elif "synthese" in first_page_text:
                        doc_type = "Note de Synthese"
                    elif "statistique" in first_page_text or "resultats" in first_page_text:
                        doc_type = "Etude / Recueil Statistique"
        except Exception as e:
            print(f"Erreur de lecture sur {doc_id}: {e}")

        types_count[doc_type] += 1
        print(f"- {doc_id} -> {doc_type}")

    # Affichage du resume global avec encodage propre
    print("\n" + "=" * 60)
    print(f"RESUME GLOBAL DU CORPUS (Total : {total_docs} documents)")
    print("=" * 60)
    for dtype, count in types_count.most_common():
        print(f"- {count} document(s) de type : {dtype}")
    print("=" * 60)

if __name__ == "__main__":
    analyze_corpus_types()