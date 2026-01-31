import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ===============================
# CONFIG
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

    L’objectif est de démontrer **l’amélioration des performances**
    via une interface simple et interactive.
    """
)

# ===============================
# CHARGEMENT DES ARTEFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    std_scaler = joblib.load("artifacts/std_scale.joblib")
    imputer = joblib.load("artifacts/imputer_median.joblib")
    ridge = joblib.load("artifacts/best_ridge.joblib")
    lgbm = joblib.load("artifacts/lgbm.joblib")
    metadata = json.load(open("artifacts/metadata.json", encoding="utf-8"))
    return std_scaler, imputer, ridge, lgbm, metadata


std_scaler, imputer, ridge_model, lgbm_model, metadata = load_artifacts()

RAW_COLS = metadata["raw_feature_columns"]
COL_MAP = metadata["column_mapping_raw_to_lgbm"]

# ===============================
# SAISIE UTILISATEUR
# ===============================
st.subheader("🧾 Données utilisées pour la prédiction")

input_data = {}

cols = st.columns(3)
for i, feature in enumerate(RAW_COLS):
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            step=1.0
        )

input_df = pd.DataFrame([input_data])

st.dataframe(input_df)

# ===============================
# PREPROCESSING (CORRECT, STABLE)
# ===============================
def preprocess(df_row: pd.DataFrame):
    # Sécurité absolue : ordre + type
    X = df_row[RAW_COLS].astype(float)

    # Imputation (DOIT être un DataFrame)
    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=RAW_COLS
    )

    # Scaling
    X_scaled = pd.DataFrame(
        std_scaler.transform(X_imputed),
        columns=RAW_COLS
    )

    # Mapping LightGBM
    X_lgbm = X_scaled.rename(columns=COL_MAP)

    return X_scaled, X_lgbm


# ===============================
# CHOIX DU MODÈLE
# ===============================
st.subheader("⚙️ Choix du modèle")

model_choice = st.radio(
    "Sélectionner le modèle",
    ["Baseline – RidgeClassifier", "Nouveau modèle – LightGBM"]
)

# ===============================
# PRÉDICTION
# ===============================
if st.button("🔮 Lancer la prédiction"):
    X_ridge, X_lgbm = preprocess(input_df)

    if model_choice == "Baseline – RidgeClassifier":
        prediction = ridge_model.predict(X_ridge)[0]
        score = ridge_model.decision_function(X_ridge)[0]

    else:
        prediction = lgbm_model.predict(X_lgbm)[0]
        score = lgbm_model.predict_proba(X_lgbm)[0][1]

    st.success("Prédiction effectuée")

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
    "Complexité": ["Faible", "Élevée"],
    "Non-linéarité": ["Non", "Oui"],
    "Performance": ["Référence", "Supérieure"]
})

st.table(comparison_df)

# ===============================
# CONCLUSION
# ===============================
st.subheader("✅ Conclusion")

st.markdown(
    """
    - Le **modèle LightGBM**, issu d’une veille récente,
      capture des relations non linéaires.
    - Il montre une **amélioration claire des performances**
      par rapport au modèle baseline.
    - Cette application constitue une **preuve de concept complète,
      reproductible et déployable**.
    """
)
