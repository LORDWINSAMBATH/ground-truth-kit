"""Le LLM générateur officiel, avec les limites de la compétition intégrées.

    from groundtruth.generator import OfficialGenerator
    gen = OfficialGenerator()                      # backend "hf" : transformers sur GPU
    gen.start_question("val-001")
    text = gen.chat([{"role": "system", "content": "..."},
                     {"role": "user", "content": "..."}])

Règles appliquées automatiquement (identiques pour tout le monde) :
  - modèle : mistralai/Ministral-3-3B-Instruct-2512, poids non modifiés ;
  - décodage glouton (do_sample=False), max_new_tokens <= 512 ;
  - au plus MAX_INPUT_TOKENS = 4 096 tokens en entrée par appel (prompt complet : système +
    question + contexte, comptés avec le tokenizer officiel) ;
  - au plus MAX_CALLS = 4 appels par question (réécriture, multi-hop, auto-vérification inclus).
Un dépassement lève BudgetExceeded : à toi de découper ton contexte avec count_tokens().

Chaque appel est journalisé (qid, tokens, durée) dans GT_LOG (par défaut runs/generator_log.jsonl).
L'orga relit ce journal lors du run officiel.

Backends :
  - "hf"     : officiel. transformers, float16 sur T4 (bfloat16 si le GPU le permet).
  - "openai" : pour développer plus vite avec un serveur qui sert le MÊME modèle
               (vLLM >= 0.12, llama.cpp server avec le GGUF officiel mistralai/...-GGUF, Ollama).
               Les tokens sont comptés avec le tokenizer officiel. Le run final utilise "hf".
  - "echo"   : aucun modèle, renvoie un texte factice. Pour tester ton pipeline sans GPU.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 512
MAX_CALLS = 4


class BudgetExceeded(RuntimeError):
    pass


def _load_tokenizer(model_id: str):
    from transformers import MistralCommonBackend

    return MistralCommonBackend.from_pretrained(model_id)


class OfficialGenerator:
    def __init__(self, backend: str = "hf", model_id: str = MODEL_ID, base_url: str | None = None,
                 served_model_name: str | None = None, log_path: str | None = None):
        self.backend = backend
        self.model_id = model_id
        self.calls = 0
        self.qid = None
        self.log_path = Path(log_path or os.environ.get("GT_LOG", "runs/generator_log.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if backend == "echo":
            self.tok = None
        else:
            self.tok = _load_tokenizer(model_id)

        if backend == "hf":
            import torch
            from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration

            if not torch.cuda.is_available():
                print("[generator] pas de GPU : chargement en float32 sur CPU (lent, pour tester seulement)")
                dtype = torch.float32
            elif torch.cuda.get_device_capability()[0] >= 8:  # Ampere et plus récents
                dtype = torch.bfloat16
            else:  # T4 (sm75) : bf16 n'y est qu'émulé, donc lent ; on reste en float16
                dtype = torch.float16
            # Les poids officiels sont en FP8 : on les déquantifie au chargement (T4 ne gère pas le FP8).
            # La déquantification produit du bfloat16 : on convertit ensuite tout le modèle au dtype voulu.
            # Tout le modèle va sur le GPU 0 (pas de délestage CPU : ce serait 10 fois plus lent).
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=FineGrainedFP8Config(dequantize=True),
                dtype=dtype,
                device_map={"": 0} if torch.cuda.is_available() else None,
            )
            self.model = self.model.to(dtype).eval()
            self._torch = torch
        elif backend == "openai":
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url or os.environ.get("GT_BASE_URL", "http://localhost:8000/v1"),
                                 api_key=os.environ.get("GT_API_KEY", "EMPTY"))
            self.served = served_model_name or os.environ.get("GT_SERVED_MODEL", model_id)
        elif backend != "echo":
            raise ValueError(f"backend inconnu : {backend}")

    # ------------------------------------------------------------------ budget
    def count_tokens(self, messages: list[dict]) -> int:
        """Nombre de tokens d'entrée de ce prompt, tel que compté par l'orga."""
        if self.tok is None:  # echo : approximation
            return sum(len(m["content"]) for m in messages) // 3 + 8 * len(messages)
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys") and "input_ids" in ids:  # BatchEncoding (transformers >= 5)
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):  # format par lot
            ids = ids[0]
        return len(ids)

    def start_question(self, qid: str) -> None:
        self.qid = qid
        self.calls = 0

    def remaining_calls(self) -> int:
        return MAX_CALLS - self.calls

    # ------------------------------------------------------------------ génération
    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        if self.qid is None:
            raise RuntimeError("appelle start_question(qid) avant chat()")
        if self.calls >= MAX_CALLS:
            raise BudgetExceeded(f"{self.qid}: plus de {MAX_CALLS} appels au générateur")
        if max_new_tokens > MAX_NEW_TOKENS:
            raise BudgetExceeded(f"max_new_tokens={max_new_tokens} > {MAX_NEW_TOKENS}")
        for m in messages:
            if not isinstance(m.get("content"), str):
                raise ValueError("texte uniquement : 'content' doit être une chaîne (pas d'image envoyée au générateur)")
        n_in = self.count_tokens(messages)
        if n_in > MAX_INPUT_TOKENS:
            raise BudgetExceeded(f"{self.qid}: prompt de {n_in} tokens > {MAX_INPUT_TOKENS}")
        self.calls += 1
        t0 = time.time()

        if self.backend == "echo":
            text = "INTROUVABLE"
            n_out = 1
        elif self.backend == "openai":
            r = self.client.chat.completions.create(model=self.served, messages=messages,
                                                    temperature=0, max_tokens=max_new_tokens)
            text = r.choices[0].message.content or ""
            n_out = getattr(r.usage, "completion_tokens", None) or 0
        else:
            torch = self._torch
            enc = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            new = out[0, enc["input_ids"].shape[1]:]
            n_out = int(new.shape[0])
            text = self.tok.decode(new, skip_special_tokens=True)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"qid": self.qid, "call": self.calls, "n_in": n_in, "n_out": n_out,
                                "seconds": round(time.time() - t0, 2), "backend": self.backend}) + "\n")
        return text.strip()
