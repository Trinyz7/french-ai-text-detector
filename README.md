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
pip install -r requirements.txt

# Téléchargement gratuit d'un échantillon de HC3 français
python -m detector.download_data

# Entraîner et évaluer le modèle (création de models/detecteur_fr.joblib)
python -m detector.train

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

Pour récupérer moins de données (moins représentatif) :

```powershell
python -m detector.download_data --train-size 600 --validation-size 100 --test-size 100
python -m detector.train
```

Les tailles doivent être **paires et >= 20**. Par défaut, on utilise
**2 000 train + 400 validation + 400 test**, échantillonnés à 50/50 par
classe. La durée dépend de l'accès au serveur public Hugging Face et du PC.

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
Corpus humain + IA (HC3 français)
           |
           v
Train / validation / test (splits du dataset)
           |
           v
Nettoyage, contrôle des labels et suppression des doublons exacts
           |
           v
TF-IDF mots (1-2 grams) + TF-IDF caractères (3-5 grams)
           |
           v
Régression logistique (Scikit-learn)
           |
           v
Seuils de décision ajustés sur la validation
           |
           v
Évaluation F1 / précision / rappel / AUC / faux positifs sur test
           |
           v
Model .joblib + metrics.json -> Streamlit / FastAPI
```

`0` = humain ; `1` = IA. Le score IA est une sortie de modèle, **pas une
probabilité garantie** ni une indication certaine de l'identité de l'auteur.
Les seuils s'appuient sur les distributions des scores de validation :
seuil IA autour du 95e percentile des scores humains (au moins 0,5), seuil
humain autour du 5e percentile des scores IA (au plus 0,5). Les textes entre
ces seuils sont « indéterminés ». Une limitation du taux de faux positifs
**sur cette validation** n'implique pas la même propriété sur du texte réel.

Les métriques calculées sont enregistrées dans `models/metrics.json`.
La matrice de confusion suit l'ordre `[humain, IA]`.

## 📁 Structure

```text
french-ai-text-detector/
├── Dockerfile / compose.yaml  # Déploiement local facultatif avec Docker
├── app.py                     # Interface Streamlit
├── api.py                     # API FastAPI (Swagger /docs)
├── detector/
│   ├── download_data.py       # Données réelles via API publique Hugging Face
│   ├── train.py               # Entraînement et évaluation
│   ├── model.py               # TF-IDF + régression logistique
│   ├── inference.py           # Analyse partagée entre interfaces
│   ├── predict.py             # CLI
│   └── config.py              # Chemins / sources / paramètres
├── data/                      # CSV locaux, ignorés dans Git
├── models/                    # Modèle entraîné localement, ignoré dans Git
├── tests/                     # Tests sans accès Internet
├── .github/workflows/tests.yml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── LICENSE
```

## 📦 Données : source, provenance et licence

**[almanach/hc3_french_ood](https://huggingface.co/datasets/almanach/hc3_french_ood)**
(Inria / ALMAnaCH). Subset `hc3_fr_qa`. Contient des réponses humaines et
générées par ChatGPT avec `answer`, `label`, `source`. Les sources sont
majoritairement **traduites en français depuis l'anglais**, donc leur style
ne représente pas parfaitement des rédactions natives françaises.

Le dépôt du dataset indique **CC BY-SA 4.0**, avec possibilité de conditions
plus restrictives pour les sources initiales. Pour cette raison, ce dépôt
**ne redistribue pas de textes du corpus**. Respecte les attributions,
licences spécifiques et modalités d'utilisation avant de redistribuer un
modèle ou des données. La licence MIT de ce dépôt concerne **uniquement notre
code**, pas les données de tiers.

L'outil de téléchargement passe par l'API publique
[Hugging Face Dataset Viewer](https://huggingface.co/docs/dataset-viewer/en/quick_start)
(`rows`, jusqu'à 100 lignes par requête). En cas d'indisponibilité ou de
limitation du service, réessaie plus tard. Aucune clé payante n'est requise.

### Utiliser tes propres données

Tu peux remplacer les CSV dans `data/` par des fichiers personnels :

- `data/train.csv`, `data/validation.csv`, `data/test.csv`
- Colonnes minimales `text,label` (labels entiers `0` humain, `1` IA)
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

1. Construire un jeu de test externe : **textes natifs français** de différents
   auteurs, modèles IA récents, paraphrases et textes humain+IA.
2. Mesurer la qualité par longueur, sujet, style et source pour identifier les biais.
3. Tester un **CamemBERT** finement entraîné (GPU utile, mais facultatif pour la V1).
4. Comparer un classifieur SVM et un modèle à base de transformers sur les
   **mêmes splits**, sans contamination de données.
5. Mettre en ligne Streamlit pour une démonstration gratuite en respectant
   les ressources et conditions de l'hébergeur (les offres peuvent évoluer).

## Publier sur GitHub

Créer un dépôt vide, puis dans ce dossier :

```powershell
git init
git add .
git commit -m "Initial commit - French AI Text Detector"
git branch -M main
git remote add origin https://github.com/TON-PSEUDO/french-ai-text-detector.git
git push -u origin main
```

Les données et modèles sont ignorés volontairement. Pense à partager une
capture de l'interface **et de vraies métriques obtenues après entraînement**.

## Licence

Code MIT. Données externes soumises à leurs propres conditions (voir ci-dessus).
