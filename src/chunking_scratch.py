"""BLOC 1 : chunker naïf (taille fixe avec recouvrement), à écrire TOI-MÊME."""


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Découpe `text` en morceaux de `size` caractères, avec `overlap` caractères communs.

    Spécification (vérifiée par tests/test_block1.py) :
    - pas entre deux débuts de chunk = size - overlap ;
    - on s'arrête dès qu'un chunk atteint la fin du texte (pas de petit chunk final redondant) ;
    - texte vide ou fait uniquement d'espaces -> [] ;
    - texte plus court que `size` -> [text] ;
    - size <= overlap -> ValueError ;
    - recomposition : chunks[0] + chunks[1][overlap:] + chunks[2][overlap:] ... == text.
    """
    raise NotImplementedError("À écrire toi-même (bloc 1)")
