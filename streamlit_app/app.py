import streamlit as st
import joblib
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Proof of Concept – Amélioration d’un modèle ML",
    layout="wide"
)

ARTIFACTS_DIR = Path("artifacts")

# ===============================
# LOAD ARTIFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    ridge_model = joblib.load(ARTIFACTS_DIR / "best_ridge.joblib")
    lgbm_model = joblib.load(ARTIFACTS_DIR / "lgbm.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "std_scale.joblib")
    imputer = joblib.load(ARTIFACTS_DIR / "imputer_median.joblib")

    with open(ARTIFACTS_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)

    return ridge_model, lgbm_model, scaler, imputer, metadata


ridge_model, lgbm_model, scaler, imputer, metadata = load_artifacts()

# ===============================
# UI – HEADER
# ===============================
st.title("📊 Proof of Concept – Amélioration d’un modèle ML")

st.markdown(
    """
Cette application présente une **preuve de concept** comparant :

- un **modèle baseline** : RidgeClassifier  
- un **modèle récent et plus performant** : LightGBM  

🎯 Objectif : **démontrer l’amélioration des performances** via un dashboard simple.
"""
)

# ===============================
# MODEL SELECTION
# ===============================
st.sidebar.header("Paramètres")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Baseline – RidgeClassifier", "Modèle avancé – LightGBM"]
)

# ===============================
# FEATURE INPUT (SIMPLIFIÉ)
# ===============================
st.subheader("Entrée utilisateur (exemple)")

raw_features = metadata["raw_feature_columns"]

user_input = {}
for feature in raw_features:
    user_input[feature] = st.number_input(
        feature,
        value=0.0
    )

X_input = pd.DataFrame([user_input])

# Preprocessing
X_imputed = imputer.transform(X_input)
X_scaled = scaler.transform(X_imputed)

# ===============================
# PREDICTION
# ===============================
if st.button("Lancer la prédiction"):

    if model_choice.startswith("Baseline"):
        score = ridge_model.decision_function(X_scaled)[0]
        prediction = int(score > 0)
        st.success("Modèle utilisé : RidgeClassifier (baseline)")
    else:
        score = lgbm_model.predict_proba(X_input)[0, 1]
        prediction = int(score > 0.5)
        st.success("Modèle utilisé : LightGBM (modèle récent)")

    st.metric("Score du modèle", round(float(score), 4))
    st.metric("Classe prédite", prediction)

# ===============================
# METADATA DISPLAY
# ===============================
with st.expander("📂 Informations techniques (features & mapping)"):
    st.write("Features utilisées :")
    st.write(metadata["raw_feature_columns"])

    st.write("Mapping features (LightGBM) :")
    st.write(metadata["column_mapping_raw_to_lgbm"])

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "POC réalisé dans le cadre d’un test technique DataScience – démonstration de veille et d’amélioration de modèle."
)
