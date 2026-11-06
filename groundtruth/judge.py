"""Juge LLM officiel : décide si une réponse est correcte par rapport à la référence.

Le juge ne voit jamais les documents : il compare la réponse du système à la réponse de
référence rédigée par les mentors. Les valeurs clés (`key_values`) sont vérifiées avant
le juge par une règle stricte (voir normalize.py) : une valeur fausse est éliminatoire.

Backends :
  - "hf"     : transformers en local (GPU conseillé, ~8 Go en float16)
  - "openai" : n'importe quel serveur compatible OpenAI (vLLM, llama.cpp server, Ollama...)
               qui sert le MÊME modèle juge ; utile pour aller plus vite
  - "none"   : pas de LLM ; heuristique de recouvrement lexical. Approximatif, pour itérer vite
               sur CPU. Le classement officiel utilise toujours le juge LLM.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM = (
    "Tu es un correcteur d'examen rigoureux. On te donne une question, la réponse de référence "
    "et la réponse d'un candidat. Tu décides si la réponse du candidat est CORRECTE.\n"
    "Règles :\n"
    "1. CORRECTE si elle contient l'information essentielle de la référence (même sens, même valeur, "
    "même entité), même formulée autrement ou plus longue.\n"
    "2. INCORRECTE si elle contredit la référence, donne une autre valeur, une autre entité, une autre "
    "année, ou si l'information essentielle est absente.\n"
    "3. INCORRECTE si elle ajoute une affirmation factuelle importante qui contredit la référence, "
    "même si le reste est juste.\n"
    "4. Une réponse qui hésite entre plusieurs valeurs est INCORRECTE.\n"
    "5. Ignore les citations entre crochets et le style. La réponse peut être en français ou en anglais.\n"
    'Réponds uniquement en JSON : {"verdict": "CORRECTE"} ou {"verdict": "INCORRECTE"}.'
)

TEMPLATE = "Question : {question}\n\nRéponse de référence : {reference}\n\nRéponse du candidat : {candidate}"

POINT_SYSTEM = (
    "Tu es un correcteur d'examen rigoureux. On te donne une question, UN élément attendu dans la "
    "réponse, et la réponse d'un candidat (en français ou en anglais). L'élément est PRESENT si la réponse "
    "l'exprime clairement, même avec d'autres mots. Il est ABSENT s'il manque, s'il est seulement "
    "évoqué de façon vague, ou si la réponse dit le contraire.\n"
    'Réponds uniquement en JSON : {"verdict": "PRESENT"} ou {"verdict": "ABSENT"}.'
)
POINT_TEMPLATE = "Question : {question}\n\nÉlément attendu : {point}\n\nRéponse du candidat : {candidate}"

PREMISE_SYSTEM = (
    "Tu es un correcteur d'examen rigoureux. La question posée repose sur une prémisse FAUSSE. "
    "On te donne la question, la correction de référence (ce qui est vrai) et la réponse d'un candidat. "
    "La réponse est CORRECTE seulement si le candidat signale explicitement que la prémisse est fausse "
    "ET que ce qu'il affirme à la place est compatible avec la correction de référence. "
    "Elle est INCORRECTE s'il répond à la question comme si la prémisse était vraie, s'il invente une "
    "explication, ou s'il affirme un fait qui contredit la correction.\n"
    'Réponds uniquement en JSON : {"verdict": "CORRECTE"} ou {"verdict": "INCORRECTE"}.'
)
PREMISE_TEMPLATE = "Question : {question}\n\nCorrection de référence : {reference}\n\nRéponse du candidat : {candidate}"


def _parse(text: str, positive: str = "CORRECTE", negative: str = "INCORRECTE") -> bool:
    m = re.search(r'"verdict"\s*:\s*"([A-ZÉa-zé]+)"', text)
    word = (m.group(1) if m else text).upper()
    return negative not in word and positive in word


class Judge:
    def __init__(self, backend: str = "hf", model_id: str = JUDGE_MODEL_ID,
                 base_url: str | None = None, api_key: str | None = None):
        self.backend = backend
        self.model_id = model_id
        if backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                dtype = torch.float32
            elif torch.cuda.get_device_capability()[0] >= 8:
                dtype = torch.bfloat16
            else:  # T4 : float16 (bf16 seulement émulé)
                dtype = torch.float16
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map={"": 0} if torch.cuda.is_available() else None).eval()
        elif backend == "openai":
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url or os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
                                 api_key=api_key or os.environ.get("JUDGE_API_KEY", "EMPTY"))
        elif backend != "none":
            raise ValueError(f"backend inconnu : {backend}")

    def is_correct(self, question: str, reference: str, candidate: str) -> bool:
        if self.backend == "none":
            return _overlap(reference, candidate) >= 0.5
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TEMPLATE.format(question=question, reference=reference, candidate=candidate)}]
        return self._ask(msgs)

    def rejects_premise(self, question: str, correction: str, candidate: str) -> bool:
        """Question piège à fausse prémisse : le candidat corrige-t-il la prémisse ?"""
        if self.backend == "none":
            low = candidate.lower()
            flagged = any(w in low for w in ("prémisse", "premisse", "en réalité", "contrairement", "n'a pas", "ne recommande pas", "au contraire"))
            return flagged and _overlap(correction, candidate) >= 0.3
        msgs = [{"role": "system", "content": PREMISE_SYSTEM},
                {"role": "user", "content": PREMISE_TEMPLATE.format(question=question, reference=correction, candidate=candidate)}]
        return self._ask(msgs)

    def count_points(self, question: str, key_points: list[str], candidate: str) -> int:
        """Question de synthèse : nombre d'éléments attendus présents dans la réponse."""
        n = 0
        for point in key_points:
            if self.backend == "none":
                n += _overlap(point, candidate) >= 0.6
                continue
            msgs = [{"role": "system", "content": POINT_SYSTEM},
                    {"role": "user", "content": POINT_TEMPLATE.format(question=question, point=point, candidate=candidate)}]
            n += self._ask(msgs, positive="PRESENT", negative="ABSENT")
        return n

    def _ask(self, msgs: list[dict], positive: str = "CORRECTE", negative: str = "INCORRECTE") -> bool:
        if self.backend == "openai":
            r = self.client.chat.completions.create(model=self.model_id, messages=msgs, temperature=0, max_tokens=20)
            return _parse(r.choices[0].message.content or "", positive, negative)
        import torch

        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=20, do_sample=False)
        return _parse(self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True), positive, negative)


_STOP = set("le la les un une des de du d l à au aux et en est sont a ont pour par sur dans que qui ce cet cette ces se sa son ses leur leurs plus moins entre avec ne pas".split())


def _words(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()
    return [w for w in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", s) if w not in _STOP]


def _overlap(reference: str, candidate: str) -> float:
    """Rappel des mots de la référence présents dans la réponse (heuristique hors-ligne)."""
    ref, cand = _words(reference), set(_words(candidate))
    if not ref:
        return 0.0
    return sum(w in cand for w in ref) / len(ref)


if __name__ == "__main__":
    print(json.dumps({"model": JUDGE_MODEL_ID}))
