import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import plotly.express as px

# ======================================================
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Preuve de concept – Prédiction ML",
    layout="wide"
)

st.title("📊 Preuve de concept – Prédiction par Machine Learning")

st.markdown(
    """
    Ce dashboard illustre une **preuve de concept** basée sur un **modèle LightGBM**,
    appliqué à des **données structurées**.

    🎯 Objectif :  
    permettre l’exploration du jeu de données et visualiser concrètement
    le **résultat de la prédiction pour un individu donné**.
    """
)

# ======================================================
# CHARGEMENT DES ARTEFACTS
# ======================================================
@st.cache_resource
def load_artifacts():
    artifacts_path = Path("artifacts")

    model = joblib.load(artifacts_path / "lgbm.joblib")

    with open(artifacts_path / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


lgbm_model, metadata = load_artifacts()

RAW_COLS = metadata["raw_feature_columns"]
COL_MAP = metadata["column_mapping_raw_to_lgbm"]

# ======================================================
# IMPORT DU CSV
# ======================================================
st.header("1️⃣ Chargement du jeu de données")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV contenant les données d’entrée",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Veuillez importer un fichier CSV pour démarrer l’analyse.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("Fichier chargé avec succès")
st.write(f"Nombre de lignes : {len(df)} | Nombre de colonnes : {df.shape[1]}")
st.dataframe(df.head())

# ======================================================
# CONTRÔLE DES COLONNES
# ======================================================
missing_cols = set(RAW_COLS) - set(df.columns)
extra_cols = set(df.columns) - set(RAW_COLS)

if missing_cols:
    st.error(f"Colonnes manquantes dans le fichier : {missing_cols}")
    st.stop()

if extra_cols:
    st.warning(
        "Certaines colonnes ne sont pas utilisées par le modèle "
        "et seront ignorées."
    )

df = df[RAW_COLS]

# Conversion numérique sécurisée
df = df.apply(pd.to_numeric, errors="coerce")

if df.empty:
    st.error("Le fichier ne contient aucune ligne exploitable après nettoyage.")
    st.stop()

# ======================================================
# ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# ======================================================
st.header("2️⃣ Analyse exploratoire des données")

st.subheader("Statistiques descriptives")
st.dataframe(df.describe().T)

# --- Graphique 1 : distribution d’une variable numérique
numeric_cols = df.select_dtypes(include="number").columns.tolist()

selected_feature = st.selectbox(
    "Choisir une variable numérique à analyser",
    options=numeric_cols
)

fig_hist = px.histogram(
    df,
    x=selected_feature,
    nbins=50,
    title=f"Distribution de la variable : {selected_feature}",
    labels={selected_feature: "Valeur", "count": "Effectif"}
)

fig_hist.update_layout(
    title_x=0.5,
    template="plotly_white"
)

st.plotly_chart(fig_hist, use_container_width=True)

# --- Graphique 2 : valeurs manquantes
missing_rate = (
    df.isna()
    .mean()
    .reset_index()
    .rename(columns={"index": "Variable", 0: "Taux de valeurs manquantes"})
    .sort_values("Taux de valeurs manquantes", ascending=False)
)

fig_missing = px.bar(
    missing_rate.head(20),
    x="Variable",
    y="Taux de valeurs manquantes",
    title="Top 20 des variables avec le plus de valeurs manquantes",
    labels={"Taux de valeurs manquantes": "Proportion de valeurs manquantes"}
)

fig_missing.update_layout(
    title_x=0.5,
    xaxis_tickangle=-45,
    template="plotly_white"
)

st.plotly_chart(fig_missing, use_container_width=True)

st.markdown(
    """
    ℹ️ Les valeurs manquantes sont courantes dans ce type de données
    et sont **nativement prises en charge par LightGBM**.
    """
)

# ======================================================
# SÉLECTION D’UN INDIVIDU
# ======================================================
st.header("3️⃣ Sélection d’un individu pour la prédiction")

n_rows = len(df)

if n_rows == 1:
    st.info("Un seul individu disponible – sélection automatique.")
    row_id = 0
else:
    row_id = st.slider(
        "Choisir un individu dans le jeu de données",
        min_value=0,
        max_value=n_rows - 1,
        value=0
    )

input_df = df.iloc[[row_id]]

st.dataframe(input_df)

# ======================================================
# PRÉDICTION
# ======================================================
st.header("4️⃣ Résultat de la prédiction")

def preprocess_for_lgbm(df_row):
    X = df_row.copy()
    X = X.rename(columns=COL_MAP)
    return X

X_lgbm = preprocess_for_lgbm(input_df)

prediction = lgbm_model.predict(X_lgbm)[0]
probability = lgbm_model.predict_proba(X_lgbm)[0][1]

# Explication de la classe prédite
st.markdown(
    """
    **Interprétation de la prédiction :**

    - **Classe 0** : l’événement cible ne se produit pas  
    - **Classe 1** : l’événement cible se produit  

    La probabilité affichée correspond à la **confiance du modèle**
    dans l’appartenance à la classe 1.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Classe prédite", int(prediction))

with col2:
    st.metric("Probabilité associée (classe 1)", round(float(probability), 4))

# ======================================================
# CONCLUSION
# ======================================================
st.header("5️⃣ Conclusion")

st.markdown(
    """
    ✅ Ce dashboard présente une **preuve de concept complète** :

    - exploration du jeu de données ;
    - sélection d’un individu ;
    - prédiction réalisée par un **modèle récent (LightGBM)** ;
    - visualisations interactives accessibles.

    📌 La comparaison avec une baseline est détaillée
    dans le notebook et la note méthodologique associée.
    """
)
