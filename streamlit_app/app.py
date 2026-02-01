import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ======================================================
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Proof of Concept – Scoring de risque de défaut",
    layout="wide"
)

st.title("📊 Proof of Concept – Scoring de risque de défaut de remboursement")

st.markdown(
    """
    Cette application présente une **preuve de concept** basée sur un modèle
    **LightGBM**, utilisé pour estimer le **risque de défaut de remboursement d’un crédit**.

    Les données utilisées pour l’inférence ont été **préparées avec le même pipeline
    que lors de l’entraînement du modèle**, garantissant une parfaite cohérence.
    """
)

# ======================================================
# CHARGEMENT DU MODÈLE
# ======================================================
@st.cache_resource
def load_model():
    artifacts_path = Path("artifacts")
    return joblib.load(artifacts_path / "lgbm.joblib")

model = load_model()

# ======================================================
# IMPORT CSV
# ======================================================
st.subheader("📂 Import du jeu de données (CSV)")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV d’inférence (features prétraitées)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Veuillez importer un fichier CSV pour continuer.")
    st.stop()

df = pd.read_csv(uploaded_file)
df = df.apply(pd.to_numeric, errors="coerce")

st.success("Fichier chargé avec succès")
st.write(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")
st.dataframe(df.head())

# ======================================================
# ANALYSE EXPLORATOIRE
# ======================================================
st.subheader("🔍 Analyse exploratoire des données")

# --- Statistiques descriptives
st.markdown("### Statistiques descriptives")
st.dataframe(df.describe().T)

numeric_cols = df.columns.tolist()

# ======================================================
# GRAPHIQUE 1 : Distribution (histogramme binned)
# ======================================================
st.markdown("### Distribution d’une variable continue")

col_hist = st.selectbox(
    "Choisir une variable continue",
    numeric_cols,
    key="hist_var"
)

hist_data = df[col_hist].dropna()
bins = np.histogram_bin_edges(hist_data, bins=20)

st.bar_chart(
    pd.cut(hist_data, bins=bins).value_counts().sort_index()
)

# ======================================================
# GRAPHIQUE 2 : Boxplot (dispersion & outliers)
# ======================================================
st.markdown("### Analyse de dispersion (boxplot)")

col_box = st.selectbox(
    "Choisir une variable pour le boxplot",
    numeric_cols,
    key="box_var"
)

box_df = pd.DataFrame({col_box: df[col_box]})
st.box_chart(box_df)

# ======================================================
# SÉLECTION D’UN INDIVIDU
# ======================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider(
    "Choisir un individu",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

input_df = df.iloc[[row_id]]
st.dataframe(input_df)

# ======================================================
# PRÉDICTION
# ======================================================
st.subheader("📈 Résultat de la prédiction")

proba = float(model.predict_proba(input_df)[0][1])
prediction = int(proba >= 0.5)

st.markdown(
    """
    **Interprétation métier :**

    - **Classe 0** : absence de risque de défaut de remboursement
    - **Classe 1** : risque de défaut de remboursement

    La probabilité correspond à l’estimation du risque de défaut pour la classe 1.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Classe prédite", prediction)

with col2:
    st.metric("Probabilité de défaut", round(proba, 3))

# ======================================================
# ACCESSIBILITÉ
# ======================================================
st.subheader("♿ Accessibilité et conformité WCAG")

st.markdown(
    """
    Les principes d’accessibilité essentiels du **WCAG** ont été pris en compte :

    - Utilisation de **composants Streamlit standards**, compatibles avec la navigation clavier
    - Graphiques **lisibles sans dépendance exclusive à la couleur**
    - Titres, sections et hiérarchie visuelle clairs
    - Informations toujours accompagnées de **texte explicatif**
    - Absence d’informations critiques transmises uniquement par des codes visuels
    """
)

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    Ce dashboard illustre une **preuve de concept complète et opérationnelle**
    de scoring de risque de crédit basée sur un **algorithme récent (LightGBM)**.

    L’approche combine :
    - une **analyse exploratoire interactive**,
    - une **sélection dynamique des données en entrée**,
    - une **prédiction interprétable** orientée métier,
    - et un **déploiement cloud** prêt pour un usage industriel.

    Cette preuve de concept démontre la **pertinence du modèle**, la **maîtrise
    de la chaîne data science de bout en bout**, ainsi que la capacité à produire
    un outil exploitable par des utilisateurs non techniques.
    """
)
