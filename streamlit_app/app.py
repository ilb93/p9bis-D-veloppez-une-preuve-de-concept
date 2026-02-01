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
    model = joblib.load(artifacts_path / "lgbm.joblib")
    return model


model = load_model()

# ======================================================
# IMPORT CSV (DÉJÀ PRÉPROCESSÉ)
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

# Sécurisation des types (LightGBM only numeric)
df = df.apply(pd.to_numeric, errors="coerce")

st.success("Fichier chargé avec succès")
st.write(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")
st.dataframe(df.head())

# ======================================================
# ANALYSE EXPLORATOIRE
# ======================================================
st.subheader("🔍 Analyse exploratoire des données")

st.markdown("### Statistiques descriptives")
st.dataframe(df.describe().T)

numeric_cols = df.columns.tolist()

st.markdown("### Distribution d’une variable numérique")
selected_col = st.selectbox(
    "Choisir une variable",
    numeric_cols
)
st.bar_chart(df[selected_col].value_counts().sort_index())

st.markdown("### Variables avec le plus de valeurs manquantes")
missing_ratio = df.isna().mean().sort_values(ascending=False).head(20)
st.bar_chart(missing_ratio)

st.info(
    "Les valeurs manquantes sont courantes dans ce type de données "
    "et sont **nativement prises en charge par LightGBM**."
)

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

    - **Classe 0** : le client ne présente **pas de risque de défaut de remboursement**
    - **Classe 1** : le client présente un **risque de défaut de remboursement**

    La probabilité correspond à **l’estimation du risque de défaut** pour la classe 1.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Risque de défaut estimé", prediction)

with col2:
    st.metric("Probabilité de défaut", round(proba, 3))

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    Ce dashboard illustre l’utilisation d’un **modèle de machine learning récent
    (LightGBM)** pour le **scoring de risque de crédit**.

    Dans cette preuve de concept, les données d’inférence sont volontairement
    fournies **après le même pipeline de préparation que celui utilisé à l’entraînement**,
    ce qui correspond aux **bonnes pratiques industrielles**.
    """
)
