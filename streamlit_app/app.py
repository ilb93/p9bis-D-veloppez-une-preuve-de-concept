import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===============================
# CONFIG STREAMLIT
# ===============================
st.set_page_config(
    page_title="DataSpace – Proof of Concept",
    layout="wide"
)

st.title("📊 Proof of Concept – Amélioration d’un modèle ML")

st.markdown(
    """
    Cette application présente une **preuve de concept** comparant :
    - un **modèle baseline** (RidgeClassifier)
    - un **modèle récent et plus performant** (LightGBM)

    L’objectif est de démontrer **l’amélioration des performances** à travers un dashboard interactif.
    """
)

# ===============================
# CHEMINS
# ===============================
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# ===============================
# CHARGEMENT DES ARTEFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    std_scaler = joblib.load(ARTIFACTS_DIR / "std_scale.joblib")
    imputer = joblib.load(ARTIFACTS_DIR / "imputer_median.joblib")
    ridge_model = joblib.load(ARTIFACTS_DIR / "best_ridge.joblib")
    lgbm_model = joblib.load(ARTIFACTS_DIR / "lgbm.joblib")

    with open(ARTIFACTS_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return std_scaler, imputer, ridge_model, lgbm_model, metadata


std_scaler, imputer, ridge_model, lgbm_model, metadata = load_artifacts()

RAW_COLS = metadata["raw_feature_columns"]
COL_MAP = metadata["column_mapping_raw_to_lgbm"]

# ===============================
# SAISIE UTILISATEUR (POC SANS DATASET)
# ===============================
st.subheader("🧮 Données utilisées pour la prédiction")

st.info(
    "Les valeurs ci-dessous simulent une observation du dataset. "
    "Cela permet de démontrer le fonctionnement des modèles **sans exposer les données réelles**."
)

input_data = {}

cols = st.columns(3)
for idx, col_name in enumerate(RAW_COLS):
    with cols[idx % 3]:
        input_data[col_name] = st.number_input(
            col_name,
            value=0.0,
            format="%.2f"
        )

input_df = pd.DataFrame([input_data])

st.dataframe(input_df, use_container_width=True)

# ===============================
# PREPROCESSING (STRICTEMENT IDENTIQUE AU NOTEBOOK)
# ===============================
def preprocess(df_row: pd.DataFrame):
    """
    Pipeline d'inférence :
    1. Imputation (numpy array obligatoire)
    2. Standardisation
    3. Mapping colonnes LightGBM
    """

    # 1️⃣ Imputation (IMPORTANT : passer un numpy array)
    X_imputed_array = imputer.transform(df_row.values)

    X_imputed = pd.DataFrame(
        X_imputed_array,
        columns=RAW_COLS
    )

    # 2️⃣ Standardisation
    X_scaled_array = std_scaler.transform(X_imputed)

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=RAW_COLS
    )

    # 3️⃣ Mapping pour LightGBM
    X_lgbm = X_scaled.rename(columns=COL_MAP)

    return X_scaled, X_lgbm


X_ridge, X_lgbm = preprocess(input_df)

# ===============================
# CHOIX DU MODÈLE
# ===============================
st.subheader("⚙️ Choix du modèle")

model_choice = st.radio(
    "Sélectionner le modèle à utiliser",
    [
        "Baseline – RidgeClassifier",
        "Nouveau modèle – LightGBM"
    ]
)

# ===============================
# PRÉDICTION
# ===============================
st.subheader("🔮 Prédiction")

if st.button("Lancer la prédiction"):

    if model_choice == "Baseline – RidgeClassifier":
        prediction = ridge_model.predict(X_ridge)[0]
        score = ridge_model.decision_function(X_ridge)[0]

    else:
        prediction = lgbm_model.predict(X_lgbm)[0]
        score = lgbm_model.predict_proba(X_lgbm)[0][1]

    st.success("Prédiction effectuée avec succès")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classe prédite", int(prediction))
    with col2:
        st.metric("Score / Probabilité", round(float(score), 3))

# ===============================
# COMPARAISON DES MODÈLES
# ===============================
st.subheader("📊 Comparaison des modèles")

comparison_df = pd.DataFrame({
    "Modèle": ["RidgeClassifier (baseline)", "LightGBM (récent)"],
    "Type": ["Linéaire", "Gradient Boosting"],
    "Capacité non-linéaire": ["Non", "Oui"],
    "Gestion des interactions": ["Faible", "Élevée"],
    "Performance": ["Référence", "Supérieure"]
})

st.table(comparison_df)

# ===============================
# CONCLUSION
# ===============================
st.subheader("✅ Conclusion")

st.markdown(
    """
    - Le **modèle LightGBM**, issu d’une veille récente, permet de capturer des relations complexes.
    - Il surpasse le modèle baseline en performance globale.
    - Cette application constitue une **preuve de concept complète**, reproductible et prête à être industrialisée.
    """
)
