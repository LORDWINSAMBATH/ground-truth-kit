"""BLOC 1 : chunker naïf (taille fixe avec recouvrement), à écrire TOI-MÊME."""



def chunk_text(text, size=1200, overlap=200):
    if size <= overlap:
        raise ValueError("size doit etre > overlap")
    if not text.strip():
        return []
    if len(text) <= size:
        return [text]

    step = size-overlap         
    chunks = []
    start = 0
    while True:
        chunks.append(text[start:start + size])
        if start+size >= len(text):         
            break
        start += step
    return chunks
