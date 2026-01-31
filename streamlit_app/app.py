import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =====================================================
# CONFIG STREAMLIT
# =====================================================
st.set_page_config(
    page_title="DataSpace – Proof of Concept",
    layout="wide"
)

st.title("📊 Proof of Concept – Amélioration d’un modèle ML")

st.markdown(
    """
Cette application présente une **preuve de concept** comparant :

- 🔹 un **modèle baseline** : RidgeClassifier  
- 🚀 un **modèle récent et plus performant** : LightGBM  

L’objectif est de démontrer l’**amélioration des performances** à travers
une interface simple et interactive.
"""
)

# =====================================================
# CHARGEMENT DES ARTEFACTS
# =====================================================
ARTIFACTS_DIR = Path("artifacts")

@st.cache_resource
def load_artifacts():
    std_scaler = joblib.load(ARTIFACTS_DIR / "std_scale.joblib")
    imputer = joblib.load(ARTIFACTS_DIR / "imputer_median.joblib")
    ridge_model = joblib.load(ARTIFACTS_DIR / "best_ridge.joblib")
    lgbm_model = joblib.load(ARTIFACTS_DIR / "lgbm.joblib")

    with open(ARTIFACTS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return std_scaler, imputer, ridge_model, lgbm_model, metadata


std_scaler, imputer, ridge_model, lgbm_model, metadata = load_artifacts()

RAW_COLS = metadata["raw_feature_columns"]
COL_MAP = metadata["column_mapping_raw_to_lgbm"]

# =====================================================
# SAISIE UTILISATEUR (PAS DE DATASET EXTERNE)
# =====================================================
st.subheader("🧮 Données utilisées pour la prédiction")

st.markdown(
    """
Les valeurs ci-dessous représentent **un individu fictif**.
Elles permettent de tester les modèles **sans dépendre d’un dataset externe**.
"""
)

input_data = {}

for col in RAW_COLS:
    input_data[col] = st.number_input(
        label=col,
        value=0.0,
        step=1.0
    )

input_df = pd.DataFrame([input_data])

st.dataframe(input_df, use_container_width=True)

# =====================================================
# PREPROCESSING (CORRIGÉ – NDARRAY STRICT)
# =====================================================
def preprocess(df_row: pd.DataFrame):
    """
    Respect strict du contrat sklearn :
    - imputer.fit() a été fait sur ndarray
    - scaler.fit() a été fait sur ndarray
    """

    # ordre + type STRICT
    X = df_row[RAW_COLS].to_numpy(dtype=np.float64)

    # imputation
    X_imputed = imputer.transform(X)

    # scaling
    X_scaled = std_scaler.transform(X_imputed)

    # reconstruction DataFrame pour Ridge
    X_ridge = pd.DataFrame(X_scaled, columns=RAW_COLS)

    # mapping colonnes pour LightGBM
    X_lgbm = X_ridge.rename(columns=COL_MAP)

    return X_ridge, X_lgbm


X_ridge, X_lgbm = preprocess(input_df)

# =====================================================
# CHOIX DU MODÈLE
# =====================================================
st.subheader("⚙️ Choix du modèle")

model_choice = st.radio(
    "Sélectionner le modèle",
    [
        "Baseline – RidgeClassifier",
        "Nouveau modèle – LightGBM"
    ]
)

# =====================================================
# PRÉDICTION
# =====================================================
if st.button("🔮 Lancer la prédiction"):

    if model_choice == "Baseline – RidgeClassifier":
        prediction = ridge_model.predict(X_ridge)[0]
        score = ridge_model.decision_function(X_ridge)[0]
        score_label = "Score (decision function)"

    else:
        prediction = lgbm_model.predict(X_lgbm)[0]
        score = lgbm_model.predict_proba(X_lgbm)[0][1]
        score_label = "Probabilité (classe positive)"

    st.success("Prédiction effectuée")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classe prédite", int(prediction))
    with col2:
        st.metric(score_label, round(float(score), 4))

# =====================================================
# COMPARAISON DES MODÈLES
# =====================================================
st.subheader("📊 Comparaison des modèles")

comparison_df = pd.DataFrame(
    {
        "Modèle": ["RidgeClassifier", "LightGBM"],
        "Type": ["Baseline", "Modèle récent"],
        "Capacité non-linéaire": ["Non", "Oui"],
        "Gestion interactions": ["Faible", "Avancée"],
        "Performance globale": ["Référence", "Supérieure"]
    }
)

st.table(comparison_df)

# =====================================================
# CONCLUSION
# =====================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
- Le **modèle LightGBM**, issu d’une veille récente, capture des relations non linéaires.
- Il offre des **performances supérieures** au modèle baseline.
- Cette application constitue une **preuve de concept fonctionnelle**, reproductible
  et prête à être industrialisée.
"""
)
