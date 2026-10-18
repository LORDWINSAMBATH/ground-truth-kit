# Ground Truth : kit de départ

Kit officiel de la piscine I Lab **« Ground Truth : un RAG qui ne ment pas »** (Intelligence Lab, ECE Paris).
Contact : mathis.lacombe@edu.ece.fr

Tu construis un assistant qui répond à des questions sur des rapports publics français (Insee, Cour
des comptes), cite la page exacte de sa source et sait dire « je ne sais pas ». Le LLM générateur est
le même pour tout le monde : seul ton pipeline (lecture des PDF, OCR, découpage, recherche,
reranking, raisonnement, abstention) fait la différence. Le sujet se joue en **4 paliers**, chacun plus
dur que le précédent, puis une finale sur un test caché.

## Démarrage rapide

Ton dépôt de travail est celui que tu as lié à la plateforme : copie-y le contenu de ce kit (ou crée-le
avec le bouton *Use this template* de GitHub). Tes soumissions, ton `run.sh` et ta red team y vivent.

```bash
git clone https://github.com/Mathos34/ground-truth-kit.git && cd ground-truth-kit
pip install -r requirements.txt
python corpus/download_corpus.py          # corpus v1 -> corpus/pdf/ (110 PDF), mise à jour v2 -> corpus/pdf_v2/ (20 PDF)
python -m baseline.pipeline index --pdf-dir corpus/pdf --index-dir index/
python -m baseline.pipeline answer --index-dir index/ --questions data/validation_v1.json \
       --out runs/val1.json --backend echo   # sans GPU : teste la chaîne (répond toujours INTROUVABLE)
python -m groundtruth.evaluate --pred runs/val1.json --gold data/validation_v1.json --judge none
```

Sur Colab ou Kaggle (GPU T4) : ouvre `notebooks/quickstart_colab.ipynb`, ou remplace `--backend echo`
par `--backend hf` et `--judge none` par `--judge hf`.

## Le calendrier

| Palier | Dates | Ce que tu fais | Ce qui est évalué |
|---|---|---|---|
| 1. From scratch | dès le début | BM25, recherche dense, chunker et métriques codés à la main, RAG de base | code + résultats sur `validation_v1` |
| 2. Le pipeline | jusqu'au 1er nov. | Parsing, OCR, chunking, recherche hybride, reranking, abstention | leaderboard **v1** (200 questions) |
| 3. Le corpus bouge | 2 → 8 nov. | Mise à jour incrémentale (+20 documents récents), questions temporelles, en anglais, de synthèse | leaderboard **v2** (130 questions) |
| 4. Red team | 9 → 12 nov. | Tu écris 15 questions pour piéger les systèmes des autres | les autres systèmes sur tes questions |
| Finale | 14 → 16 nov. | Code figé, test caché + questions red team, soutenance | classement final |

Les données des paliers 3 et 4 sont déjà dans le dépôt : tu peux les regarder, mais garde la
discipline des paliers (en particulier, ne mets pas `corpus/pdf_v2/` dans ton index avant le palier 3).

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `corpus/manifest.json` | Les 130 documents : identifiant, source, URL officielle, date, licence, version du corpus, empreinte SHA-256 |
| `corpus/download_corpus.py` | Télécharge le corpus depuis les sites officiels et vérifie les empreintes |
| `corpus/make_scans.py` | Transforme 10 documents en PDF image sans texte (appelé automatiquement) |
| `data/validation_v1.json` | 100 questions **avec** réponses et pages de preuve, corpus v1 |
| `data/classement_v1_questions.json` | 200 questions sans réponses, leaderboard v1 |
| `data/validation_v2.json` | 40 questions avec réponses, corpus v2 (nouveaux types) |
| `data/classement_v2_questions.json` | 130 questions sans réponses, leaderboard v2 |
| `groundtruth/generator.py` | Le LLM générateur officiel, limites de la compétition intégrées |
| `groundtruth/evaluate.py`, `judge.py` | Le script d'évaluation et le juge officiels |
| `groundtruth/submit.py` | Prépare une soumission au leaderboard |
| `groundtruth/redteam.py` | Vérifie ton fichier de questions red team (palier 4) |
| `baseline/pipeline.py` | Baseline BM25 + générateur, exemple du contrat `index` / `update` / `answer` |
| `run.sh`, `setup.sh` | Le contrat de la finale (à adapter à ton pipeline) |
| `notebooks/quickstart_colab.ipynb` | Tout le démarrage sur un GPU T4 gratuit |
| `LEADERBOARD.md` | Les classements v1 et v2, mis à jour par le mentor |

## Le corpus

130 documents publics, **3 292 pages**, en deux versions :

- **v1 (paliers 1 et 2) : 110 documents, `corpus/pdf/`** : 84 Insee Première (4 pages chacun : texte,
  tableaux, graphiques) et 26 rapports et synthèses de la Cour des comptes (20 à 250 pages).
- **v2 (à partir du palier 3) : +20 documents, `corpus/pdf_v2/`** : les éditions les plus récentes de
  16 Insee Première (souvent les mêmes sujets que des documents v1, avec des chiffres révisés) et 4
  rapports de la Cour publiés en octobre 2026.
- **10 documents « numérisés »** dans la v1 : `insee-ip2036`, `ip2041`, `ip2055`, `ip2075`, `ip2088`,
  `ip2103`, `ip2119`, `ip2127`, `cc-pac-synthese`, `cc-vaccination-synthese`. Après téléchargement, ils
  sont remplacés par un rendu image dégradé (gris, rotation, bruit, JPEG), sans couche texte. Le
  contenu est inchangé ; il faut de l'OCR pour le lire. N'utilise pas les originaux : à la finale, ton
  pipeline reçoit le corpus tel que produit par le script.

Licences : Insee, Licence Ouverte 2.0 (« Source : Insee ») ; Cour des comptes, reproduction autorisée
par les mentions légales de www.ccomptes.fr (citer la source, ne pas altérer les contenus).

Chaque document a un `doc_id` (nom du fichier sans `.pdf`). Les **pages sont l'index physique dans le
PDF, à partir de 1** (pas le numéro imprimé). C'est ce format que tu cites.

## Les questions

```json
{
  "qid": "val1-001",
  "question": "…",
  "lang": "fr",
  "type": "texte | tableau | graphique | scan | multi-doc | synthese | temporel | piege",
  "difficulty": "facile | moyen | difficile",
  "answerable": true,
  "answer": "réponse de référence",
  "key_values": ["18,8 %", "1 800"],
  "evidence": [[{"doc_id": "cc-anr", "page": 77}]],
  "source_quote": "extrait qui prouve la réponse",
  "corpus_version": 1
}
```

- Les questions sont formulées **comme par de vrais usagers** : elles reprennent rarement les mots exacts
  des documents. Une recherche lexicale seule retrouve la bonne page dans le top 5 pour environ la
  moitié des questions seulement.
- `evidence` est une liste de **groupes** (un par document quand il faut en croiser plusieurs) ; dans un
  groupe, toutes les pages listées sont acceptées.
- `key_values` : valeurs qui doivent figurer dans ta réponse (à l'arrondi d'affichage près ; « 2 600 000 »
  vaut « 2,6 millions » ; « baisse de 34 % » vaut « -34 % » ; le format anglais « 2,896.5 » est reconnu).
- `synthese` : réponse à plusieurs éléments. `key_points` liste les éléments attendus, `min_points` le
  nombre qu'une bonne réponse doit contenir.
- `temporel` : la question porte sur la donnée **la plus récente** ou sur une révision entre deux
  publications. Les éditions récentes révisent souvent les anciennes : il faut savoir laquelle croire.
- `lang: "en"` : question posée en anglais sur un corpus en français. Tu peux répondre en français ou en
  anglais.
- Questions sans réponse (`answerable: false`, 20 à 30 % selon les jeux) : l'information n'est pas dans le
  corpus. Certaines reposent sur une **fausse prémisse** (`false_premise: true`, avec
  `premise_correction`) ; d'autres portent sur une information **qui n'existe pas encore dans le corpus
  v1** mais apparaîtra en v2.

| Type | val. v1 | classement v1 | val. v2 | classement v2 | test caché |
|---|---|---|---|---|---|
| texte | 26 | 58 | 8 | 36 | 71 |
| tableau | 17 | 34 | 6 | 21 | 44 |
| graphique | 12 | 26 | 4 | 16 | 34 |
| scan | 9 | 17 | | 5 | 19 |
| multi-doc | 5 | 10 | 7 | 13 | 24 |
| synthese | 8 | 13 | 3 | 9 | 22 |
| temporel | | | 12 | 16 | 22 |
| sans réponse | 23 | 57 | | 14 | 47 |
| dont en anglais | | | 6 | 15 | 30 |
| **Total** | **100** | **200** | **40** | **130** | **283** |

## Le score de confiance

| Cas | Points |
|---|---|
| Bonne réponse, citation valide | +1 |
| Bonne réponse, citation invalide ou absente | +0,5 |
| `INTROUVABLE` sur une question qui a une réponse | 0 |
| `INTROUVABLE` sur une question sans réponse | +1 |
| Fausse prémisse : réponse qui corrige explicitement la prémisse | +1 |
| Réponse fausse, ou réponse inventée à une question sans réponse | **−1** |

- **Bonne réponse** : toutes les `key_values` sont présentes, puis le juge LLM la déclare correcte (pour
  une synthèse : au moins `min_points` éléments présents).
- **Citation valide** : au moins une page citée dans chaque groupe d'`evidence`, 5 citations au maximum.
- Score final = moyenne, entre −1 et +1.

```bash
python -m groundtruth.evaluate --pred runs/val1.json --gold data/validation_v1.json --judge hf
python -m groundtruth.evaluate ... --judge none      # heuristique, CPU, indicatif seulement
python -m groundtruth.evaluate ... --details --out runs/rapport.json   # détail par question
```

Juge officiel : `Qwen/Qwen3-4B-Instruct-2507` (open weights, famille différente du générateur, ~0,7 s
par jugement sur T4). Si tes prédictions contiennent un champ optionnel `retrieved` (liste classée de
pages `{doc_id, page}`), le script calcule aussi Recall@5, Recall@10, MRR et nDCG@10.

## Le format de réponse

```json
{"predictions": [
  {"qid": "val1-001",
   "answer": "Le taux de succès serait de 18,8 %, soit environ 1 800 projets par an.",
   "citations": [{"doc_id": "cc-anr", "page": 77}],
   "retrieved": [{"doc_id": "cc-anr", "page": 77}, {"doc_id": "cc-anr", "page": 10}]},
  {"qid": "val1-002", "answer": "INTROUVABLE", "citations": []}
]}
```

`groundtruth.io.write_predictions()` écrit ce format.

## Le générateur officiel

`mistralai/Ministral-3-3B-Instruct-2512` (Mistral AI, Apache 2.0), texte uniquement, décodage glouton.
Utilise **toujours** la classe fournie, qui applique les limites :

```python
from groundtruth.generator import OfficialGenerator
gen = OfficialGenerator(backend="hf")            # "hf" (officiel), "openai" (serveur local), "echo" (test)
gen.start_question(q["qid"])                     # remet le compteur d'appels à zéro
msgs = [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
assert gen.count_tokens(msgs) <= 4096            # pour découper ton contexte
answer = gen.chat(msgs, max_new_tokens=256)
```

| Limite | Valeur |
|---|---|
| Tokens d'entrée par appel (prompt complet) | 4 096 |
| Tokens générés par appel | 512 |
| Appels par question | 4 (réécriture, multi-hop, auto-vérification compris) |
| Temps moyen par question (tout compris : recherche, reranking, génération) | 30 s sur T4 |
| Images envoyées au générateur | Aucune (texte uniquement) |

Mesures sur un T4 de Colab : chargement 7 s (après le premier téléchargement de 4,7 Go), ~7,7 Go de
VRAM au repos, **~13 Go en pic** avec un prompt de 4 000 tokens, **~9 s** pour un appel à 4 000 tokens.
Deux appels pleins par question consomment déjà l'essentiel du budget de temps, et il reste peu de
VRAM pendant la génération : c'est un vrai compromis d'ingénierie (contextes plus courts, modèles de
recherche légers, OCR fait à l'indexation, quand le générateur n'est pas chargé).

Un dépassement lève `BudgetExceeded`. Chaque appel est journalisé dans `runs/generator_log.jsonl`.

- Les poids publiés sont en FP8 : la classe les déquantifie au chargement et passe tout le modèle en
  float16 sur T4 (bfloat16 sur les GPU récents).
- Le backend `openai` permet de développer avec un serveur local qui sert le même modèle (vLLM, llama.cpp
  avec le GGUF officiel `mistralai/Ministral-3-3B-Instruct-2512-GGUF`, Ollama). Les résultats officiels
  utilisent `hf`.
- Tout le reste (OCR, embeddings, reranking, description de graphiques à l'indexation) est libre, avec
  des modèles open weights qui tiennent sur le T4.
- **Test rapide** : `python -m groundtruth.generator` charge le générateur dans son propre processus,
  pose une question et affiche le nombre de tokens d'entrée, la réponse et la durée (`--backend echo`
  pour tester sans GPU).
- **Piège mémoire JAX / bm25s** : `bm25s` importe JAX (préinstallé sur Colab), et JAX réserve 75 % de la
  VRAM dès l'import, ce qui provoque ensuite « CUDA out of memory » au chargement du générateur. Les
  paquets `groundtruth` et `baseline` règlent `JAX_PLATFORMS=cpu` à leur import. Si ton code importe
  `bm25s` ou JAX **avant** `groundtruth`, exporte toi-même `JAX_PLATFORMS=cpu` (dans le shell, ou
  `os.environ["JAX_PLATFORMS"] = "cpu"` tout en haut de ton script).
- **Ne charge pas le générateur deux fois sur un T4 de 15 Go** (~7,7 Go chacun) : par exemple dans un
  notebook et dans un script lancé en parallèle. Lance tes tests dans un processus séparé qui se termine
  (c'est ce que fait le notebook), car `del` + `torch.cuda.empty_cache()` ne libèrent pas tout.

## Palier 1 : RAG from scratch

Codé par toi, sans LangChain, LlamaIndex, FAISS, `bm25s` ni base vectorielle :
- un BM25 (tokenisation, index inversé, scoring Okapi), vérifié contre `bm25s` ;
- une recherche dense (embeddings d'un modèle open weights, cosinus et top-k en NumPy) ;
- un chunker naïf (taille fixe avec recouvrement) ;
- Recall@k, MRR et nDCG@10 ;
- un RAG de base avec le générateur officiel et des citations `[doc_id, page]` ;
- les résultats sur `validation_v1` : BM25 seul, dense seul, RAG de base, et 5 échecs commentés.

La baseline du dépôt (`bm25s` + PyMuPDF, une page = un chunk) sert de point de comparaison.

## Palier 2 : le pipeline et le leaderboard v1

Parsing (au moins 2 approches comparées sur 30 pages difficiles que tu annotes), au moins 2 stratégies
de chunking au-delà du naïf, recherche hybride + reranker, mécanisme d'abstention calibré, rapport
d'ablation. Soumets sur le leaderboard v1 :

```bash
python -m groundtruth.submit --pred runs/classement_v1.json --phase v1
git add submissions/ && git commit -m "soumission" && git push
```

2 soumissions par jour et par phase au maximum. Le mentor relève les soumissions poussées sur ton dépôt
et met à jour `LEADERBOARD.md`.

## Palier 3 : le corpus bouge

Le 2 novembre, le corpus reçoit 20 documents plus récents (`corpus/pdf_v2/`). En production, on ne
réindexe pas tout à chaque nouveau document :

```bash
bash run.sh update <dossier_index> corpus/pdf_v2      # 20 minutes maximum sur T4, sans réindexer la v1
```

Nouveaux défis du leaderboard v2 (`--phase v2`, ouvert le 2 novembre) :
- **questions temporelles** : quand deux éditions se contredisent, la plus récente fait foi ;
- **questions en anglais** sur un corpus en français ;
- **synthèses** : plusieurs éléments à rassembler et condenser dans 4 096 tokens ;
- **informations nouvelles** : certaines questions sans réponse en v1 en ont une en v2. Ton système doit
  changer d'avis.

## Palier 4 : red team

Écris **15 questions** sur le corpus v2 complet, conçues pour piéger les autres systèmes, dans
`redteam/questions.json` (même format que `data/validation_v2.json`, avec `id`, `question`, `type`,
`answerable`, `answer`, `key_values`, `evidence`, `source_quote`, et `notes` pour les questions sans
réponse). Au moins 4 sans réponse, au moins 4 sur un tableau, un graphique ou un scan.

```bash
python -m groundtruth.redteam --questions redteam/questions.json     # vérifie le format et les preuves
```

Pousse le fichier au plus tard le **12 novembre à 23 h 59**. À la finale, chaque système répond aux
questions de tous les autres participants. Ton score red team est la part moyenne de systèmes adverses
qui échouent sur tes questions (score ≤ 0). **Une question invalide** (réponse fausse, ambiguë, ou
réponse présente dans le corpus pour une question « sans réponse ») **compte −1** : chacun peut
contester une question adverse, le mentor tranche.

## La finale

1. **14 novembre, 12 h** : tu fixes ton code avec un tag git (`git tag final && git push --tags`).
2. **14 novembre, 18 h** : le mentor publie `data/test_questions.json` (283 questions) et
   `data/redteam_questions.json` (toutes les questions red team, sans réponses).
3. Avant le **15 novembre, 23 h 59**, depuis le commit `final`, sur un T4 :
   ```bash
   bash setup.sh
   bash run.sh index  corpus/pdf /tmp/index
   bash run.sh update /tmp/index corpus/pdf_v2
   bash run.sh answer /tmp/index data/test_questions.json final/predictions_test.json
   bash run.sh answer /tmp/index data/redteam_questions.json final/predictions_redteam.json
   ```
   puis pousse `final/` avec `runs/generator_log.jsonl` copié dans `final/`.
4. **Vérification** : le mentor ré-exécute le pipeline des 3 premiers, depuis leur tag `final`, sur un
   échantillon du test. Un écart de score inexpliqué de plus de 0,05, un dépassement des limites ou un
   journal du générateur incohérent disqualifie.
5. **16 novembre** : soutenance et démo live (le jury pose 5 questions inédites à ton système).

`run.sh` et `setup.sh` à la racine de ton dépôt suivent le modèle fourni ici (commandes `index`,
`update`, `answer`). `setup.sh` est la seule étape avec réseau.

## Barème

| Critère | Poids |
|---|---|
| Score de confiance sur le test caché | 35 % |
| Red team | 10 % |
| Rapport d'ablation, fiche système et soutenance | 25 % |
| Code : RAG from scratch, pipeline, mise à jour incrémentale, reproductibilité | 15 % |
| Démo live devant le jury | 15 % |

## Règles

- Palier 1 sans librairie de recherche (voir plus haut). Ensuite, librairies libres.
- Pipeline final : modèles open weights uniquement, aucune API propriétaire, aucun réseau après `setup.sh`.
- Interdit : fine-tuner le générateur, annoter à la main les questions de classement ou de test,
  chercher à reconstituer les réponses cachées. Fine-tuner ton embedder ou ton reranker est autorisé et
  encouragé, avec des données que tu génères toi-même à partir du corpus.
- Assistants IA autorisés pour coder, mais tu dois pouvoir expliquer chaque ligne du palier 1.

## Crédits

Corpus : Source : Insee (Insee Première, Licence Ouverte 2.0) ; Cour des comptes, www.ccomptes.fr
(rapports et synthèses 2026). Les versions « numérisées » de 10 documents sont des rendus image
dégradés générés pour l'exercice d'OCR ; leur contenu n'est pas modifié. Questions : Intelligence Lab
by ECE, 2026.
