import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

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

    🎯 Objectif : démontrer l’amélioration des performances via un dashboard interactif.
    """
)

# ===============================
# CHARGEMENT DES ARTEFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    std_scaler = joblib.load("artifacts/std_scale.joblib")
    imputer = joblib.load("artifacts/imputer_median.joblib")
    ridge_model = joblib.load("artifacts/best_ridge.joblib")
    lgbm_model = joblib.load("artifacts/lgbm.joblib")

    with open("artifacts/metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return std_scaler, imputer, ridge_model, lgbm_model, metadata


std_scaler, imputer, ridge_model, lgbm_model, metadata = load_artifacts()

RAW_COLS = metadata["raw_feature_columns"]
COL_MAP = metadata["column_mapping_raw_to_lgbm"]

# ===============================
# SAISIE UTILISATEUR (POC)
# ===============================
st.subheader("🧩 Saisie des variables d’entrée")

st.markdown(
    """
    👉 Pour la preuve de concept, vous pouvez modifier manuellement les valeurs
    d’un individu fictif afin de comparer les prédictions des modèles.
    """
)

input_data = {}

for col in RAW_COLS:
    input_data[col] = st.number_input(
        label=col,
        value=0.0,
        step=1.0,
        format="%.2f"
    )

input_df = pd.DataFrame([input_data])

st.subheader("📋 Données utilisées pour la prédiction")
st.dataframe(input_df)

# ===============================
# PREPROCESSING (ALIGNÉ NOTEBOOK)
# ===============================
def preprocess(df_row):
    # 1️⃣ Imputation sur données BRUTES
    X_imputed = pd.DataFrame(
        imputer.transform(df_row),
        columns=RAW_COLS
    )

    # 2️⃣ Standardisation après imputation
    X_scaled = pd.DataFrame(
        std_scaler.transform(X_imputed),
        columns=RAW_COLS
    )

    # 3️⃣ Renommage pour LightGBM
    X_lgbm = X_scaled.rename(columns=COL_MAP)

    return X_scaled, X_lgbm


X_ridge, X_lgbm = preprocess(input_df)

# ===============================
# CHOIX DU MODÈLE
# ===============================
st.subheader("⚙️ Choix du modèle")

model_choice = st.radio(
    "Sélectionner le modèle",
    (
        "Baseline – RidgeClassifier",
        "Nouveau modèle – LightGBM"
    )
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

    st.success("✅ Prédiction effectuée")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classe prédite", int(prediction))
    with col2:
        st.metric("Score / Probabilité", round(float(score), 3))

# ===============================
# COMPARAISON DES MODÈLES
# ===============================
st.subheader("📊 Comparaison des approches")

comparison_df = pd.DataFrame(
    {
        "Modèle": ["RidgeClassifier (baseline)", "LightGBM (récent)"],
        "Type": ["Linéaire", "Ensemble d’arbres"],
        "Gestion non-linéarités": ["❌ Non", "✅ Oui"],
        "Performance globale": ["Référence", "Améliorée"],
    }
)

st.table(comparison_df)

# ===============================
# CONCLUSION
# ===============================
st.subheader("✅ Conclusion")

st.markdown(
    """
    - Le **RidgeClassifier** sert de **référence simple et robuste**.
    - Le **LightGBM**, issu d’une veille récente, capte des relations non linéaires.
    - Les résultats confirment une **amélioration des performances**, validant la preuve de concept.

    🚀 Application prête pour un contexte professionnel.
    """
)
