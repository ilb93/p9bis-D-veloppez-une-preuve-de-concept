import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

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

    L’objectif est de démontrer **l’amélioration des performances** à travers un dashboard simple.
    """
)

# ===============================
# CHARGEMENT DES ARTEFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    std_scale = joblib.load("artifacts/std_scale.joblib")
    imputer = joblib.load("artifacts/imputer_median.joblib")
    ridge = joblib.load("artifacts/best_ridge.joblib")
    lgbm = joblib.load("artifacts/lgbm.joblib")
    metadata = json.load(open("artifacts/metadata.json", encoding="utf-8"))
    return std_scale, imputer, ridge, lgbm, metadata

std_scale, imputer, ridge_model, lgbm_model, metadata = load_artifacts()

RAW_COLS = metadata["raw_feature_columns"]
COL_MAP = metadata["column_mapping_raw_to_lgbm"]

# ===============================
# CHARGEMENT DATASET (POC)
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("data/data_clean.csv")  # adapte si besoin

df = load_data()

st.subheader("🔎 Aperçu du jeu de données")
st.write(df.head())

# ===============================
# ANALYSE EXPLORATOIRE (EDA)
# ===============================
st.subheader("📈 Analyse exploratoire rapide")

col1, col2 = st.columns(2)

with col1:
    feature = st.selectbox("Choisir une variable", RAW_COLS)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], bins=30, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution de {feature}")
    st.pyplot(fig)

with col2:
    missing_rate = df.isna().mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    missing_rate.plot(kind="bar", ax=ax, color="#DD8452")
    ax.set_title("Top 10 variables avec valeurs manquantes")
    st.pyplot(fig)

# ===============================
# SÉLECTION D’UNE OBSERVATION
# ===============================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider(
    "Choisir une ligne du dataset",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

input_df = df.loc[[row_id], RAW_COLS]

st.write("Variables sélectionnées :")
st.dataframe(input_df)

# ===============================
# PREPROCESSING
# ===============================
def preprocess(df_row):
    X_scaled = pd.DataFrame(
        std_scale.transform(df_row),
        columns=RAW_COLS
    )
    X_imputed = pd.DataFrame(
        imputer.transform(X_scaled),
        columns=RAW_COLS
    )
    X_lgbm = X_imputed.rename(columns=COL_MAP)
    return X_imputed, X_lgbm

X_ridge, X_lgbm = preprocess(input_df)

# ===============================
# CHOIX DU MODÈLE
# ===============================
st.subheader("⚙️ Choix du modèle")

model_choice = st.radio(
    "Sélectionner le modèle à utiliser",
    ["Baseline – RidgeClassifier", "Nouveau modèle – LightGBM"]
)

# ===============================
# PRÉDICTION
# ===============================
if st.button("🔮 Lancer la prédiction"):

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
    "Capacité non-linéaire": ["Non", "Oui"],
    "Performance globale": ["Référence", "Supérieure"]
})

st.table(comparison_df)

# ===============================
# CONCLUSION
# ===============================
st.subheader("✅ Conclusion")

st.markdown(
    """
    - Le **modèle LightGBM**, issu d’une veille récente, permet de capturer des relations non linéaires.
    - Il offre de **meilleures performances** que le modèle baseline.
    - Cette application constitue une **preuve de concept fonctionnelle** et déployable.
    """
)
