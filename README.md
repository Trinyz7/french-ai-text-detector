# French AI Text Detector 📝

**Projet Python / NLP / Machine Learning / Streamlit / FastAPI — gratuit, en français.**

Outil expérimental pour **estimer** si un texte français présente des motifs
statistiques associés aux textes produits par un modèle de langage ou rédigés
par un humain. Ce n'est **pas une preuve d'origine**, ni un détecteur infaillible.

> **Statut du dépôt :** le code et les tests sont fournis. Le modèle n'est
> **pas préentraîné dans cette archive** : il est créé sur ton ordinateur
> après téléchargement du jeu de données. Aucun score n'est inventé.

## Démonstration de ce qui est inclus

- Entrée : texte en français collé dans une interface Streamlit.
- Sortie : **indices humains / indices IA / indéterminé**, score du modèle,
  nombre de mots et avertissements.
- Mode CSV : analyser jusqu'à 250 textes d'un coup et exporter les résultats.
- API locale : `POST /predict` et `POST /predict/batch`, documentation `/docs`.
- Entraînement reproductible + métriques sur données de test indépendantes.
- Tests automatisés GitHub Actions.

## ⚡ Installation Windows (PowerShell)

Il faut **Python 3.10+**, Git facultatif et une connexion Internet uniquement
pour installer les librairies et récupérer les données.

```powershell
# Se placer dans le dossier extrait
cd french-ai-text-detector

# Créer et activer un environnement virtuel
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Si l'activation est bloquée :
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements-llm.txt   # ou requirements.txt, sans la perplexité

# Téléchargement du corpus HC3 français (~175 Mo, une seule fois)
python -m detector.download_data

# Entraîner et évaluer le modèle (création de models/detecteur_fr.joblib)
python -m detector.train                      # complet : TF-IDF + style + perplexité
# python -m detector.train --config sans_llm  # plus léger, sans torch

# Lancer le site local
streamlit run app.py
```

Streamlit indique une adresse locale (généralement http://localhost:8501).
Une fois le modèle entraîné, **tous les textes collés sont traités localement**
sans appel réseau vers un LLM. Le site ne demande aucune clé API.

> Sur Linux/macOS : `python3 -m venv .venv` puis
> `source .venv/bin/activate`, et exécute les commandes `python ...`.
>
> Sur Windows, si `py -3` ne fonctionne pas, utilise `python -m venv .venv`.

### Première installation plus rapide

Sans torch et avec moins de données (moins représentatif) :

```powershell
pip install -r requirements.txt
python -m detector.download_data --per-source-train 100
python -m detector.train --config sans_llm --sans-hors-domaine
```

Par défaut : **300 questions par domaine** pour l'entraînement (soit
~3 000 textes), 60 pour la validation, 50 pour le test, plus ~500 textes
natifs. Le mode complet télécharge aussi deux petits modèles de langue
(Qwen2.5-0.5B et sa version Instruct, ~2 Go) au premier lancement. Sur CPU,
la perplexité prend environ **0,5 à 1 s par texte**. Pendant l'entraînement,
les résultats sont mis en cache dans `data/cache/`. Les textes analysés dans
l'application ne sont jamais mis en cache.

## 🐳 Alternative Docker (facultative)

Tu peux aussi tout lancer avec **Docker Compose**, sans installer Python sur
ton système (seul Docker est nécessaire) :

```powershell
docker compose build
docker compose run --rm ui python -m detector.download_data
docker compose run --rm ui python -m detector.train
docker compose up -d
```

Interface : http://localhost:8501 · API : http://localhost:8000/docs.
Arrêt : `docker compose down`. Les données/modèles sont sauvegardés dans les
dossiers `data/` et `models/` locaux. Les ports sont exposés uniquement sur
`127.0.0.1` (machine locale).

## 🌐 API REST locale facultative

Cette API est **celle de ton propre projet** : aucun abonnement / aucune clé.

```powershell
uvicorn api:app --reload
```

Ouvrir **http://127.0.0.1:8000/docs** puis essayer `POST /predict` :

```json
{
  "text": "La transformation numérique fait évoluer les métiers et les méthodes de travail."
}
```

Exemple de réponse (valeurs **indicatives, non issues d'un modèle entraîné**) :

```json
{
  "categorie": "indetermine",
  "message": "Le modèle ne distingue pas clairement l'origine du texte.",
  "score_ia": 0.54,
  "score_humain": 0.46,
  "statistiques": {"mots": 13, "caracteres": 92, "phrases_approx": 1},
  "avertissements": ["Ce résultat statistique ne prouve jamais l'identité de l'auteur.", "Texte court (moins de 80 mots) : interprétation particulièrement fragile."]
}
```

Tu peux appeler l'API depuis **JavaScript, Python, PHP**, etc. Ne pas exposer
`uvicorn` publiquement sans mettre en place les protections nécessaires.

### En terminal

```powershell
python -m detector.predict "Je suis allé au marché ce matin et j'ai retrouvé une amie."
```

## 🧪 Méthode ML

```text
HC3 français (archive Hugging Face)
     |
     v
Échantillonnage APPARIÉ : pour chaque question, 1 réponse humaine + 1 réponse IA,
même quota par domaine (reddit, finance, medicine, open, wiki)
     |
     v
Train / validation / test (splits du dataset) + test_natif (textes écrits en français)
     |
     v
Caractéristiques :
  - TF-IDF mots (1-2) + caractères (2-4)
  - stylométrie (27 mesures : rythme des phrases, ponctuation, vocabulaire,
    connecteurs typiques des LLM, registre familier, 1re personne...)
  - perplexité / Binoculars avec Qwen2.5-0.5B (base + Instruct), facultatif
     |
     v
Régression logistique -> calibration de Platt sur la validation
     |
     v
Seuils humain / indéterminé / IA choisis sur la validation
     |
     v
Évaluation : test HC3, test natif, leave-one-domain-out, détail par source
     |
     v
models/detecteur_fr.joblib + metrics.json -> Streamlit / FastAPI / CLI
```

**Pourquoi l'échantillonnage apparié ?** Dans HC3, la part de réponses IA
varie beaucoup selon le domaine. Avec un échantillon aléatoire, le modèle
apprend « sujet financier → IA », et l'ancienne version classait **80 % des
articles Wikipédia humains** comme IA. Avec des paires de réponses à la même
question, seule la manière d'écrire permet de distinguer les classes.

**Binoculars** (Hans et al., 2024) : on divise la perplexité du texte (modèle
de base) par l'entropie croisée entre le modèle de base et sa version Instruct.
Un texte IA est « prévisible » ; la normalisation évite de confondre un sujet
simple avec un texte généré.

`0` = humain ; `1` = IA. Le score est **calibré** sur la validation, mais
reste une sortie de modèle, **pas une probabilité garantie**. Seuil IA :
au-delà du 95e percentile des scores humains de validation (au moins 0,65) ;
seuil humain : sous le 5e percentile des scores IA (au plus 0,35). Entre les
deux, le texte est « indéterminé ».

Chaque analyse renvoie aussi une **explication** : les indices (mots,
mesures de style, perplexité) qui ont le plus poussé vers « IA » ou « humain ».

Toutes les métriques sont enregistrées dans `models/metrics.json` (`test`,
`test_natif`, `hors_domaine`, avec le détail par source). La matrice de
confusion suit l'ordre `[humain, IA]`.

<!-- RESULTATS -->

## 🧰 Configurations

| `--config` | Caractéristiques | Dépendances |
|---|---|---|
| `complet` (défaut) | TF-IDF + stylométrie + perplexité | `requirements-llm.txt` |
| `sans_llm` | TF-IDF + stylométrie | `requirements.txt` |
| `tfidf` | TF-IDF seul (ancienne méthode) | `requirements.txt` |
| `style_llm` | stylométrie + perplexité, sans vocabulaire | `requirements-llm.txt` |
| `style` | stylométrie seule | `requirements.txt` |

Si torch n'est pas installé, `complet` bascule automatiquement sur `sans_llm`.

## 📁 Structure

```text
french-ai-text-detector/
├── Dockerfile / compose.yaml  # Déploiement local facultatif avec Docker
├── app.py                     # Interface Streamlit
├── api.py                     # API FastAPI (Swagger /docs)
├── detector/
│   ├── download_data.py       # Corpus HC3 FR, échantillonnage apparié, test natif
│   ├── train.py               # Entraînement, calibration, évaluations
│   ├── model.py               # Pipeline, calibration, explications
│   ├── features.py            # Stylométrie + perplexité (Binoculars)
│   ├── inference.py           # Analyse partagée entre interfaces
│   ├── predict.py             # CLI
│   └── config.py              # Chemins / sources / paramètres
├── data/                      # CSV locaux, ignorés dans Git
├── models/                    # Modèle entraîné localement, ignoré dans Git
├── tests/                     # Tests sans accès Internet
├── .github/workflows/tests.yml
├── requirements.txt
├── requirements-llm.txt       # torch + transformers (facultatif)
├── requirements-dev.txt
├── .gitignore
└── LICENSE
```

## 📦 Données : source, provenance et licence

**[almanach/hc3_french_ood](https://huggingface.co/datasets/almanach/hc3_french_ood)**
(Inria / ALMAnaCH). Fichiers utilisés :

- `hc3_fr/*_qa_examples.json` : questions/réponses HC3 **traduites de
  l'anglais** (humain vs ChatGPT) → train / validation / test ;
- `hc3_fr_chatgpt`, `qa_binggpt` : réponses de ChatGPT et BingGPT **écrites
  directement en français** → `test_natif` (IA) ;
- `faq_fr_gouv`, `faq_fr_random`, `qa_binglikehuman` : textes humains natifs
  (FAQ de sites français, réponses web) → `test_natif` (humain).

Le dépôt du dataset indique **CC BY-SA 4.0**, avec possibilité de conditions
plus restrictives pour les sources initiales. Pour cette raison, ce dépôt
**ne redistribue pas de textes du corpus**. Respecte les attributions,
licences spécifiques et modalités d'utilisation avant de redistribuer un
modèle ou des données. La licence MIT de ce dépôt concerne **uniquement notre
code**, pas les données de tiers.

Le script télécharge l'archive `data.tar.gz` du dépôt Hugging Face, car
l'API *Dataset Viewer* ne prend plus en charge ce dataset. Il n'en extrait que
les fichiers français. Aucune clé n'est requise.

### Utiliser tes propres données

Tu peux remplacer les CSV dans `data/` par des fichiers personnels :

- `data/train.csv`, `data/validation.csv`, `data/test.csv`
- Colonnes minimales `text,label` (labels entiers `0` humain, `1` IA).
  La colonne `source` est facultative : elle active le détail par domaine et
  le leave-one-domain-out. `data/test_natif.csv` est facultatif.
- Chaque partition doit contenir les deux classes (au moins 5 exemples chacune)
- **Ne jamais** mettre le même texte dans deux partitions ni entraîner sur test.
- Diversifier thèmes, auteurs, générateurs IA, longueurs et styles. Vérifier
  l'autorisation d'utiliser chaque texte et supprimer les données privées.

`python -m detector.train` fonctionne aussi sur ces CSV personnalisés.

## ✅ Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

Ils créent **uniquement des mini-textes de test artificiels** pour vérifier le
fonctionnement du code, et ne mesurent pas la performance réelle du détecteur.
Les scores qui peuvent apparaître pendant les tests ne sont pas publiables
comme métriques de détection dans le monde réel.

## ⚠️ Limites et usages déconseillés

Les classificateurs peuvent apprendre le **style du dataset** ou celui d'une
traduction, plutôt que « l'origine IA » proprement dite. Un fort score sur
HC3 ne garantit pas une bonne précision sur des textes d'étudiants, mails,
articles français originaux ou réponses de modèles récents. Les textes
humains très formels peuvent être faussement classés « IA » et inversement.
Les réécritures humaines d'un texte IA rendent toute étiquette binaire
conceptuellement difficile.

**Ne pas utiliser seul pour sanctionner un élève, évaluer un candidat ou
accuser une personne.** Une validation indépendante sur de vraies rédactions
françaises est indispensable avant tout usage sensible.

Études utiles :

- Liang et al. (2023), *GPT detectors are biased against non-native English writers*, Patterns. https://doi.org/10.1016/j.patter.2023.100779
- Weber-Wulff et al. (2023), *Testing of detection tools for AI-generated text*, International Journal for Educational Integrity. https://doi.org/10.1007/s40979-023-00146-z
- Antoun et al. (2023), *Towards a Robust Detection of Language Model-Generated Text: Is ChatGPT that easy to detect?*, https://gitlab.inria.fr/wantoun/robust-chatgpt-detection

## 🗺️ Améliorations possibles

1. Ajouter des textes de **LLM récents** (2025-2026) et des textes humains
   natifs variés (rédactions, mails, articles), d'abord au test natif, puis à
   l'entraînement.
2. Tester la robustesse aux **paraphrases** et aux textes mixtes humain + IA.
3. Comparer avec un **CamemBERT** finement entraîné sur les mêmes splits.
4. Essayer d'autres paires de modèles pour Binoculars : un modèle plus gros
   est plus fiable, mais plus lent sur CPU.

## Licence

Code MIT. Données externes soumises à leurs propres conditions (voir ci-dessus).
