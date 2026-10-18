#!/usr/bin/env bash
# Contrat du run final. L'orga exécute EXACTEMENT ces commandes sur un GPU T4 16 Go, sans réseau
# (sauf setup.sh) :
#
#   bash setup.sh                                           # réseau autorisé : dépendances + poids
#   bash run.sh index  <dossier_pdf_v1> <dossier_index>     # corpus v1 (110 docs) : 2 h max
#   bash run.sh update <dossier_index> <dossier_pdf_v2>     # ajout des 20 docs v2 : 20 min max
#   bash run.sh answer <dossier_index> <questions.json> <sortie.json>   # 20 s par question en moyenne
#
# Remplace le corps des trois commandes par ton pipeline. Ici : la baseline officielle.
set -euo pipefail
cmd="$1"; shift
case "$cmd" in
  index)  python -m baseline.pipeline index  --pdf-dir "$1" --index-dir "$2" ;;
  update) python -m baseline.pipeline update --index-dir "$1" --pdf-dir "$2" ;;
  answer) python -m baseline.pipeline answer --index-dir "$1" --questions "$2" --out "$3" --backend hf ;;
  *) echo "usage: run.sh index|update|answer ..." >&2; exit 1 ;;
esac
