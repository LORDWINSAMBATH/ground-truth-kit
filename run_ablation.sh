#!/usr/bin/env bash
# Ablations de la RECHERCHE (rapide, sans générateur) puis runs complets à noter avec le juge officiel.
# Usage (depuis la racine du dépôt, après `index`) :  bash run_ablation.sh index/ data/validation_v1.json
set -e
IDX=${1:-index/}
VAL=${2:-data/validation_v1.json}
mkdir -p runs

# --- 1) Recherche seule : BM25 / dense / hybride, avec et sans reranker (Recall@k, MRR, nDCG@10) ---
for MODE in bm25 dense hybrid; do
  python -m src.pipeline answer --index-dir "$IDX" --questions "$VAL" --out runs/ret_${MODE}_norr.json \
      --mode $MODE --no-rerank --k 20 --retrieval-only
  echo "=== $MODE sans reranker ==="; python eval_retrieval.py --pred runs/ret_${MODE}_norr.json --gold "$VAL"
done
python -m src.pipeline answer --index-dir "$IDX" --questions "$VAL" --out runs/ret_hybrid_rr.json \
    --mode hybrid --k 20 --retrieval-only
echo "=== hybride + reranker ==="; python eval_retrieval.py --pred runs/ret_hybrid_rr.json --gold "$VAL"

# --- 2) Seuil d'abstention estimé sur les 100 questions ---
python calibrate.py --debug runs/ret_hybrid_rr.json.debug.jsonl --gold "$VAL"

# --- 3) Runs complets (générateur + juge) : à lancer ensuite, un par configuration, avec TAU = valeur trouvée ---
# python -m src.pipeline answer --index-dir $IDX --questions $VAL --out runs/full_hybrid_rr.json --tau $TAU
# python -m groundtruth.evaluate --pred runs/full_hybrid_rr.json --gold $VAL --judge hf
# puis les mêmes avec --no-rerank, --mode bm25, --mode dense, --tau 0 : c'est ton tableau d'ablation.
