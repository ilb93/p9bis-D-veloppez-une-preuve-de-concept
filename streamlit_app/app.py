import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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

    Le dashboard combine :
    - une **analyse exploratoire des données** ;
    - la **sélection d’un individu** ;
    - l’**estimation du risque de défaut**.
    """
)

# ======================================================
# CHARGEMENT DES ARTEFACTS
# ======================================================
@st.cache_resource
def load_artifacts():
    artifacts_path = Path("artifacts")

    lgbm_model = joblib.load(artifacts_path / "lgbm.joblib")

    with open(artifacts_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return lgbm_model, metadata


lgbm_model, metadata = load_artifacts()

RAW_COLS = metadata["raw_feature_columns"]
COL_MAP = metadata["column_mapping_raw_to_lgbm"]

# ======================================================
# IMPORT DU CSV
# ======================================================
st.subheader("📂 Import du jeu de données (CSV)")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV (données d’inférence)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Veuillez importer un fichier CSV pour continuer.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("Fichier chargé avec succès")
st.write(f"Nombre de lignes : {df.shape[0]} | Nombre de colonnes : {df.shape[1]}")
st.dataframe(df.head())

# ======================================================
# CONTRÔLE DES COLONNES
# ======================================================
missing_cols = set(RAW_COLS) - set(df.columns)
extra_cols = set(df.columns) - set(RAW_COLS)

if missing_cols:
    st.error(f"Colonnes manquantes : {missing_cols}")
    st.stop()

if extra_cols:
    st.warning(f"Colonnes ignorées : {extra_cols}")

df = df[RAW_COLS]

# ======================================================
# ANALYSE EXPLORATOIRE
# ======================================================
st.subheader("🔍 Analyse exploratoire des données")

st.markdown("### Statistiques descriptives")
st.dataframe(df.describe().T)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

st.markdown("### Distribution d’une variable numérique")
selected_col = st.selectbox("Choisir une variable numérique", numeric_cols)

st.bar_chart(df[selected_col].value_counts().sort_index())

st.markdown("### Variables avec le plus de valeurs manquantes")
missing_ratio = df.isna().mean().sort_values(ascending=False).head(20)
st.bar_chart(missing_ratio)

st.info(
    "Les valeurs manquantes sont fréquentes dans ce type de données "
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
# PRÉDICTION LIGHTGBM (SAFE)
# ======================================================
st.subheader("📈 Résultat de la prédiction")

# Renommage des colonnes
X_lgbm = input_df.rename(columns=COL_MAP).copy()

# 🔒 SÉCURISATION ABSOLUE DES DTYPES
for col in X_lgbm.columns:
    X_lgbm[col] = pd.to_numeric(X_lgbm[col], errors="coerce")

# Prédiction
proba = float(lgbm_model.predict_proba(X_lgbm)[0][1])
prediction = int(proba >= 0.5)

# ======================================================
# AFFICHAGE MÉTIER
# ======================================================
st.markdown("### Interprétation de la prédiction")

st.markdown(
    """
    - **Classe 0** : le client ne présente **pas de risque de défaut de remboursement**
    - **Classe 1** : le client présente un **risque de défaut de remboursement**

    La probabilité correspond à **l’estimation du risque de défaut pour la classe 1**.
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
    Ce dashboard illustre l’utilisation d’un **modèle récent (LightGBM)** pour le
    **scoring de risque de crédit**.

    - L’analyse exploratoire permet de comprendre la structure et la qualité des données.
    - La prédiction individuelle illustre concrètement l’apport du modèle.
    - Le seuil de décision présenté ici est arbitraire et pourrait être ajusté selon
      la politique de risque d’un établissement financier.
    """
)
