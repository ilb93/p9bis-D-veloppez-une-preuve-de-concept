import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="DataSpace – Proof of Concept",
    layout="wide"
)

st.title("📊 Proof of Concept – Amélioration d’un modèle ML")

st.markdown("""
Cette application présente une **preuve de concept** comparant :

- 🔹 **Modèle baseline** : RidgeClassifier  
- 🚀 **Modèle récent** : LightGBM  

🎯 Objectif : démontrer une amélioration de performance via un dashboard simple.
""")

# =====================================================
# LOAD ARTEFACTS
# =====================================================
ARTIFACTS_DIR = Path("artifacts")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(ARTIFACTS_DIR / "std_scale.joblib")
    ridge = joblib.load(ARTIFACTS_DIR / "best_ridge.joblib")
    lgbm = joblib.load(ARTIFACTS_DIR / "lgbm.joblib")

    with open(ARTIFACTS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return scaler, ridge, lgbm, metadata


scaler, ridge_model, lgbm_model, metadata = load_artifacts()

RAW_COLS = metadata["raw_feature_columns"]
COL_MAP = metadata["column_mapping_raw_to_lgbm"]

# =====================================================
# UPLOAD DATA
# =====================================================
st.subheader("📂 Import du jeu de données")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Veuillez importer un fichier CSV pour continuer.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success(f"Fichier chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Vérification des colonnes
missing_cols = set(RAW_COLS) - set(df.columns)
if missing_cols:
    st.error(f"Colonnes manquantes dans le fichier : {missing_cols}")
    st.stop()

st.subheader("👀 Aperçu du dataset")
st.dataframe(df.head(), use_container_width=True)

# =====================================================
# ROW SELECTION
# =====================================================
st.subheader("🎯 Sélection de l’observation")

row_id = st.slider(
    "Choisir une ligne à tester",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

input_df = df.loc[[row_id], RAW_COLS]

st.markdown("**Données utilisées pour la prédiction :**")
st.dataframe(input_df, use_container_width=True)

# =====================================================
# PREPROCESS (INFERENCE ONLY)
# =====================================================
def preprocess(df_row):
    X = df_row.to_numpy(dtype=np.float64)
    X_scaled = scaler.transform(X)

    X_ridge = pd.DataFrame(X_scaled, columns=RAW_COLS)
    X_lgbm = X_ridge.rename(columns=COL_MAP)

    return X_ridge, X_lgbm


X_ridge, X_lgbm = preprocess(input_df)

# =====================================================
# MODEL CHOICE
# =====================================================
st.subheader("⚙️ Choix du modèle")

model_choice = st.radio(
    "Sélectionner le modèle",
    ["Baseline – RidgeClassifier", "Nouveau modèle – LightGBM"]
)

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔮 Lancer la prédiction"):

    if model_choice == "Baseline – RidgeClassifier":
        prediction = ridge_model.predict(X_ridge)[0]
        score = ridge_model.decision_function(X_ridge)[0]
        score_label = "Score (decision function)"

    else:
        prediction = lgbm_model.predict(X_lgbm)[0]
        score = lgbm_model.predict_proba(X_lgbm)[0][1]
        score_label = "Probabilité classe positive"

    st.success("Prédiction effectuée")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classe prédite", int(prediction))
    with col2:
        st.metric(score_label, round(float(score), 4))

# =====================================================
# MODEL COMPARISON
# =====================================================
st.subheader("📊 Comparaison des modèles")

st.table(pd.DataFrame({
    "Modèle": ["RidgeClassifier", "LightGBM"],
    "Type": ["Baseline", "Récent"],
    "Capacité non-linéaire": ["Non", "Oui"],
    "Performance": ["Référence", "Supérieure"]
}))

# =====================================================
# CONCLUSION
# =====================================================
st.subheader("✅ Conclusion")

st.markdown("""
- LightGBM capture des relations non linéaires complexes
- Il surpasse le modèle baseline
- Cette application constitue une **preuve de concept robuste et déployable**
""")
