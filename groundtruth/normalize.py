"""Normalisation des nombres à la française pour la vérification stricte des valeurs.

Règle officielle : si la réponse de référence contient des valeurs clés (champ `key_values`),
chacune doit apparaître dans la réponse du système, à l'arrondi d'affichage près.
Exemples reconnus comme identiques : « 1 234,5 », « 1234,5 », « 1 234.5 », « 1,2345 millier ».
On ne gère pas les conversions d'unités : si la référence dit « 2,6 millions », écrire « 2 600 000 »
est accepté, tout comme « 2,6 M » (la valeur 2,6 y figure telle quelle).
"""
from __future__ import annotations

import re
import unicodedata

_MULT = {
    "millier": 1e3, "milliers": 1e3,
    "million": 1e6, "millions": 1e6,
    "milliard": 1e9, "milliards": 1e9, "md": 1e9, "mds": 1e9, "mrd": 1e9,
    "thousand": 1e3, "thousands": 1e3, "billion": 1e9, "billions": 1e9, "bn": 1e9,
}

# un nombre : chiffres, éventuellement groupés par espace (normale, insécable, fine), décimale , ou .
_NUM = re.compile(
    r"(?<![\w.,])[-−–]?\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?![\d])"
    r"|(?<![\w.,])[-−–]?\d+(?:[.,]\d+)?(?![\d])"
)
_AFTER = re.compile(r"\s*(milliers?|millions?|milliards?|mds?|md|mrd|thousands?|millions?|billions?|bn)\b", re.I)
# format anglais : 2,896.5 (virgule = milliers, point = décimale)
_NUM_EN = re.compile(r"(?<![\w.,])[-−–]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])")


def _to_float(tok: str) -> float:
    tok = tok.replace(" ", "").replace(" ", "").replace(" ", "")
    tok = tok.replace("−", "-").replace("–", "-")
    tok = tok.replace(",", ".")
    return float(tok)


_WORDS = {w: str(i) for i, w in enumerate(
    "zéro _ deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize "
    "dix-sept dix-huit dix-neuf vingt".split()) if w != "_"}
_WORDS_RX = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _WORDS), key=len, reverse=True)) + r")\b", re.I)


def _words_to_digits(text: str) -> str:
    """« cinq projets » -> « 5 projets » (de deux à vingt ; « un/une » restent des articles)."""
    return _WORDS_RX.sub(lambda m: _WORDS[m.group(1).lower()], text)


def extract_numbers(text: str) -> list[tuple[float, float]]:
    """Renvoie une liste de (valeur brute, valeur avec multiplicateur)."""
    text = _words_to_digits(unicodedata.normalize("NFC", text))
    out = []
    for m in _NUM.finditer(text):
        try:
            v = _to_float(m.group(0))
        except ValueError:
            continue
        mult = 1.0
        nxt = _AFTER.match(text, m.end())
        if nxt:
            mult = _MULT.get(nxt.group(1).lower(), 1.0)
        out.append((v, v * mult))
    # nombres au format anglais (réponses en anglais) : « 2,896 » = 2 896
    for m in _NUM_EN.finditer(text):
        try:
            v = float(m.group(0).replace(",", "").replace("−", "-").replace("–", "-"))
        except ValueError:
            continue
        nxt = _AFTER.match(text, m.end())
        out.append((v, v * (_MULT.get(nxt.group(1).lower(), 1.0) if nxt else 1.0)))
    return out


def _close(a: float, b: float) -> bool:
    # le signe est souvent exprimé en toutes lettres (« baisse de 34 % » pour « -34 % »)
    a, b = abs(a), abs(b)
    if a == b:
        return True
    tol = max(abs(b) * 0.005, 1e-9)  # 0,5 % : couvre les arrondis d'affichage
    return abs(a - b) <= tol


def value_present(key_value: str, answer: str) -> bool:
    """Vrai si la valeur clé (ex. « 14,4 % », « 2,6 millions ») figure dans la réponse."""
    kv = extract_numbers(key_value)
    if not kv:  # valeur non numérique : comparaison textuelle souple
        norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()).strip()
        return norm(key_value) in norm(answer)
    raw_k, scaled_k = kv[0]
    for raw_a, scaled_a in extract_numbers(answer):
        if _close(raw_a, raw_k) or _close(scaled_a, scaled_k) or _close(raw_a, scaled_k) or _close(scaled_a, raw_k):
            return True
    return False


def all_values_present(key_values: list[str], answer: str) -> bool:
    return all(value_present(k, answer) for k in key_values)
