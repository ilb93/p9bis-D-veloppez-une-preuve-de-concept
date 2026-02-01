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

    - 🔹 **Modèle baseline** : DummyClassifier  
    - 🚀 **Modèle récent** : LightGBM  

    🎯 Objectif : démontrer l’intérêt d’un modèle plus avancé
    par rapport à une baseline naïve.
    """
)

# ======================================================
# CHARGEMENT DES ARTEFACTS
# ======================================================
@st.cache_resource
def load_artifacts():
    artifacts_path = Path("artifacts")

    dummy_model = joblib.load(artifacts_path / "dummy_classifier.joblib")
    lgbm_model = joblib.load(artifacts_path / "lgbm.joblib")

    with open(artifacts_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return dummy_model, lgbm_model, metadata


dummy_model, lgbm_model, metadata = load_artifacts()

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
# CAST NUMÉRIQUE GLOBAL
# ======================================================
df = df.apply(pd.to_numeric, errors="coerce")

# ======================================================
# 🔴 GUARD CRITIQUE AVANT SLIDER (CORRECTION ICI)
# ======================================================
if df.empty:
    st.error("Le fichier CSV ne contient aucune ligne exploitable.")
    st.stop()

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
# PREPROCESSING LIGHTGBM
# ======================================================
def preprocess_for_lgbm(df_row):
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
        "Baseline – DummyClassifier",
        "Nouveau modèle – LightGBM"
    ]
)

# ======================================================
# PRÉDICTION
# ======================================================
if st.button("🔮 Lancer la prédiction"):

    if model_choice == "Baseline – DummyClassifier":
        prediction = dummy_model.predict(input_df)[0]
        score = dummy_model.predict_proba(input_df)[0][1]

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
    "Modèle": ["DummyClassifier (baseline)", "LightGBM (récent)"],
    "Complexité": ["Très faible", "Élevée"],
    "Relations non-linéaires": ["❌ Non", "✅ Oui"],
    "Performance attendue": ["Faible (référence)", "Supérieure"]
})

st.table(comparison_df)

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    - Le **DummyClassifier** fournit une baseline naïve mais indispensable.
    - Le **modèle LightGBM** apporte une amélioration significative.
    - Cette application constitue une **preuve de concept stable et déployable**.
    """
)
