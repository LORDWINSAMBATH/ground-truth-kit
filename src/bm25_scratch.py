import math
import re
from collections import defaultdict, Counter

class BM25FromScratch:
    def __init__(self, k1=1.5, b=0.75):
        """
        Initialise les hyperparamètres de BM25 :
        - k1 : contrôle la saturation du terme.
        - b : contrôle la pénalité liée à la longueur du document.
        """
        self.k1 = k1
        self.b = b
        self.doc_len = []             
        self.avg_doc_len = 0.0        
        self.corpus_size = 0          
        self.doc_freqs = defaultdict(int)  
        self.inverted_index = defaultdict(list) 
        self.documents = []           

    def _tokenize(self, text):
        """Nettoie et découpe le texte en tokens (minuscules, ponctuation)."""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def fit(self, documents):
        """Construit l'index inversé et calcule les statistiques globales."""
        self.documents = documents
        self.corpus_size = len(documents)
        total_len = 0

        for doc_id, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            length = len(tokens)
            self.doc_len.append(length)
            total_len += length

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.doc_freqs[term] += 1
                self.inverted_index[term].append((doc_id, count))

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

    def _calc_idf(self, term):
        """Calcule l'IDF selon la formule Okapi BM25."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query, top_k=3):
        """Calcule le score BM25 et retourne le Top-K."""
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)

        for token in query_tokens:
            idf = self._calc_idf(token)
            if idf == 0.0:
                continue

            for doc_id, freq in self.inverted_index.get(token, []):
                doc_length = self.doc_len[doc_id]
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_len))
                scores[doc_id] += idf * (num / den)

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in ranked_docs[:top_k]:
            results.append({
                "doc_id": doc_id,
                "score": round(score, 4),
                "content": self.documents[doc_id]
            })
        return results