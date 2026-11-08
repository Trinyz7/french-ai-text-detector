"""Interface Streamlit : streamlit run app.py"""

import io

import pandas as pd
import streamlit as st

from detector.config import META_PATH, MODEL_PATH
from detector.inference import analyze, load_detector

st.set_page_config(
    page_title="French AI Text Detector",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2.5rem; max-width: 840px;}
h1 {letter-spacing: -0.035em;}
[data-testid="stMetric"] {padding: 0.5rem 0;}
.notice {padding: .9rem 1rem; border: 1px solid rgba(128,128,128,.28);
         border-radius: 10px; font-size: .95rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_detector()


st.title("French AI Text Detector")
st.caption("Analyse NLP / Machine Learning de textes français · Gratuit · Local · Open source")
st.markdown(
    '<div class="notice">Cet outil repère des motifs statistiques appris. '
    '<strong>Il ne peut pas prouver</strong> qu’un texte a été écrit par une IA '
    'ou par une personne. Ne l’utilisez pas pour accuser ou sanctionner quelqu’un.</div>',
    unsafe_allow_html=True,
)
st.write("")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Aucun modèle entraîné n’a encore été trouvé.")
    st.markdown("**Initialisation (depuis le dossier du projet) :**")
    st.code(
        "python -m detector.download_data\n"
        "python -m detector.train\n"
        "streamlit run app.py", language="bash"
    )
    st.info("Le téléchargement nécessite Internet une première fois. Ensuite, l'analyse fonctionne hors ligne.")
    st.stop()

try:
    model, meta = get_model()
except Exception as exc:
    st.error(f"Impossible de charger le modèle local : {exc}")
    st.stop()

with st.sidebar:
    st.header("À propos du modèle")
    st.write("**Langue ciblée :** français")
    st.write("**Méthode :** TF-IDF mots / caractères + régression logistique")
    st.write("**Corpus :** HC3 français (partiellement traduit)")
    test = meta.get("test_seuil_prudent_ia", {})
    if test:
        st.divider()
        st.subheader("Évaluation sur le corpus de test")
        st.metric("Précision IA", f"{test.get('precision_ia', 0):.1%}")
        st.metric("Rappel IA", f"{test.get('rappel_ia', 0):.1%}")
        st.metric("F1 IA", f"{test.get('f1_ia', 0):.1%}")
        st.caption("Scores mesurés sur HC3, pas sur tous les textes du monde réel.")
    st.divider()
    st.markdown("[Documentation du corpus HC3](https://huggingface.co/datasets/almanach/hc3_french_ood)")


def render_result(result):
    st.subheader("Résultat de l'analyse")
    category = result["categorie"]
    if category == "indices_ia":
        st.warning("**Indices plutôt associés à l'IA**")
    elif category == "indices_humains":
        st.success("**Indices plutôt associés à une rédaction humaine**")
    else:
        st.info("**Origine indéterminée**")
    st.write(result["message"])
    st.metric("Score du modèle pour la classe IA", f"{result['score_ia']:.1%}")
    st.progress(result["score_ia"])
    st.caption("Ce score n'est pas la probabilité certaine qu'une IA ait écrit le texte.")
    stats = result["statistiques"]
    a, b, c = st.columns(3)
    a.metric("Mots", stats["mots"])
    b.metric("Caractères", stats["caracteres"])
    c.metric("Phrases (approx.)", stats["phrases_approx"])
    for note in result["avertissements"]:
        st.caption("⚠️ " + note)


tab_text, tab_csv, tab_method = st.tabs(["Analyser un texte", "Analyser un fichier CSV", "Méthode et limites"])
with tab_text:
    entered = st.text_area(
        "Colle ici ton texte en français",
        placeholder="Colle un paragraphe, un article, une réponse rédigée...",
        height=235,
        max_chars=50_000,
    )
    words = len(entered.split())
    st.caption(f"{words} mots — 80 mots ou plus recommandés pour une analyse moins fragile.")
    if st.button("Analyser le texte", type="primary", use_container_width=True):
        if not entered.strip():
            st.warning("Ajoute d'abord un texte à analyser.")
        else:
            render_result(analyze(entered, model, meta))

with tab_csv:
    st.write("Importe un CSV contenant une colonne `text` (jusqu'à 250 lignes par analyse).")
    sample = pd.DataFrame({"text": ["Exemple de texte à analyser."]})
    st.download_button(
        "Télécharger un modèle de CSV",
        sample.to_csv(index=False).encode("utf-8-sig"),
        file_name="modele_textes.csv", mime="text/csv"
    )
    uploaded = st.file_uploader("Choisir un fichier CSV UTF-8", type=["csv"])
    if uploaded is not None and st.button("Analyser le CSV", type="primary"):
        try:
            df = pd.read_csv(io.BytesIO(uploaded.getvalue()), encoding="utf-8-sig")
            if "text" not in df.columns:
                st.error("Il manque la colonne 'text'.")
            elif len(df) > 250:
                st.error("Maximum 250 lignes par analyse.")
            elif df.empty:
                st.warning("Le fichier est vide.")
            else:
                texts = df["text"].fillna("").astype(str).tolist()
                outcomes = []
                for text in texts:
                    if not text.strip():
                        outcomes.append({"categorie": "texte_vide", "score_ia": None})
                    else:
                        r = analyze(text, model, meta)
                        outcomes.append({"categorie": r["categorie"], "score_ia": r["score_ia"]})
                result_df = df.copy()
                result_df["categorie"] = [r["categorie"] for r in outcomes]
                result_df["score_ia"] = [r["score_ia"] for r in outcomes]
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Télécharger les résultats CSV",
                    result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="resultats_detection.csv", mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Impossible de traiter le CSV : {exc}")

with tab_method:
    st.markdown("""
**Comment ça marche ?** Le modèle apprend à différencier des textes étiquetés
humains et des réponses générées par IA. Le TF-IDF encode les mots, les groupes
de mots et les suites de caractères ; une régression logistique classe le texte.

**Seuils :** déterminés sur un jeu de validation indépendant pour limiter les
faux positifs sur ce corpus. Une zone d'incertitude évite de forcer une décision.

**Limites :** le corpus provient majoritairement d'une traduction de textes
anglais vers le français et d'anciens modèles de génération. Les domaines,
styles, longueurs et modèles modernes diffèrent. Un texte retouché, traduit,
technique, créatif ou très court peut être mal classé. Les résultats ne doivent
jamais être utilisés comme seule base d'une décision académique ou professionnelle.

**Confidentialité :** l'analyse s'effectue localement. Aucun texte analysé
n'est envoyé à un fournisseur d'IA payant. L'application ne conserve pas les
textes soumis dans une base de données.
""")
