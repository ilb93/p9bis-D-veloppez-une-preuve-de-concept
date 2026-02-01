import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ======================================================
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Proof of Concept – Amélioration ML",
    layout="wide"
)

st.title("📊 Proof of Concept – Amélioration d’un modèle ML")

st.markdown(
    """
    Cette application présente une **preuve de concept** comparant :

    - 🔹 **Modèle baseline** : RidgeClassifier  
    - 🚀 **Modèle récent** : LightGBM  

    🎯 Objectif : démontrer une **amélioration de performance** via un dashboard simple.
    """
)

# ======================================================
# CHARGEMENT DES ARTEFACTS
# ======================================================
@st.cache_resource
def load_artifacts():
    artifacts_path = Path("artifacts")

    ridge_pipeline = joblib.load(artifacts_path / "ridge_pipeline.joblib")
    lgbm_model = joblib.load(artifacts_path / "lgbm.joblib")

    with open(artifacts_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return ridge_pipeline, lgbm_model, metadata


ridge_pipeline, lgbm_model, metadata = load_artifacts()

RAW_COLS = metadata["raw_feature_columns"]
COL_MAP = metadata["column_mapping_raw_to_lgbm"]

# ======================================================
# IMPORT DU CSV D’INFÉRENCE
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
st.write(f"Shape : {df.shape}")
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
# CAST NUMÉRIQUE GLOBAL (CRITIQUE)
# ======================================================
df = df.apply(pd.to_numeric, errors="coerce")

# ======================================================
# SÉLECTION D’UNE LIGNE
# ======================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider(
    "Choisir une ligne du dataset",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

input_df = df.iloc[[row_id]]

st.write("Données utilisées pour la prédiction")
st.dataframe(input_df)

# ======================================================
# PREPROCESSING
# ======================================================
def preprocess_for_ridge(df_row):
    """
    Pipeline Ridge :
    - cast numérique
    - imputation + scaling internes
    """
    X = df_row.copy()
    X = X[ridge_pipeline.feature_names_in_]
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def preprocess_for_lgbm(df_row):
    """
    LightGBM :
    - accepte les NaN
    - colonnes renommées
    """
    X = df_row.copy()
    X = X.rename(columns=COL_MAP)
    return X

# ======================================================
# CHOIX DU MODÈLE
# ======================================================
st.subheader("⚙️ Choix du modèle")

model_choice = st.radio(
    "Sélectionner le modèle",
    [
        "Baseline – RidgeClassifier",
        "Nouveau modèle – LightGBM"
    ]
)

# ======================================================
# PRÉDICTION
# ======================================================
if st.button("🔮 Lancer la prédiction"):

    if model_choice == "Baseline – RidgeClassifier":
        X_ridge = preprocess_for_ridge(input_df)

        prediction = ridge_pipeline.predict(X_ridge)[0]
        score = ridge_pipeline.decision_function(X_ridge)[0]

    else:
        X_lgbm = preprocess_for_lgbm(input_df)

        prediction = lgbm_model.predict(X_lgbm)[0]
        score = lgbm_model.predict_proba(X_lgbm)[0][1]

    st.success("Prédiction effectuée")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Classe prédite", int(prediction))

    with col2:
        st.metric("Score / Probabilité", round(float(score), 4))

# ======================================================
# COMPARAISON DES MODÈLES
# ======================================================
st.subheader("📊 Comparaison des modèles")

comparison_df = pd.DataFrame({
    "Modèle": ["RidgeClassifier (baseline)", "LightGBM (récent)"],
    "Gestion des NaN": ["❌ Non", "✅ Oui"],
    "Relations non-linéaires": ["❌ Non", "✅ Oui"],
    "Performance": ["Référence", "Supérieure"]
})

st.table(comparison_df)

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    - Le **modèle LightGBM** capture des relations complexes.
    - Le **RidgeClassifier**, encapsulé dans un pipeline, est **stable et robuste en production**.
    - Cette application constitue une **preuve de concept fonctionnelle et déployable**.
    """
)
