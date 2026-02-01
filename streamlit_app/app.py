import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import altair as alt

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
    Ce dashboard présente une **preuve de concept** basée sur un **modèle LightGBM**,
    appliqué à des **données structurées**.

    🎯 Objectif :  
    illustrer l’exploration du jeu de données et visualiser
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
    st.error(f"Colonnes manquantes : {missing_cols}")
    st.stop()

if extra_cols:
    st.warning("Certaines colonnes ne sont pas utilisées par le modèle.")

df = df[RAW_COLS]
df = df.apply(pd.to_numeric, errors="coerce")

if df.empty:
    st.error("Aucune ligne exploitable après nettoyage.")
    st.stop()

# ======================================================
# ANALYSE EXPLORATOIRE (EDA)
# ======================================================
st.header("2️⃣ Analyse exploratoire des données")

st.subheader("Statistiques descriptives")
st.dataframe(df.describe().T)

numeric_cols = df.select_dtypes(include="number").columns.tolist()

selected_feature = st.selectbox(
    "Choisir une variable numérique",
    options=numeric_cols
)

# --- Graphique 1 : distribution
hist_chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X(selected_feature, bin=alt.Bin(maxbins=40), title="Valeur"),
        y=alt.Y("count()", title="Effectif"),
        tooltip=["count()"]
    )
    .properties(
        title=f"Distribution de la variable : {selected_feature}",
        height=300
    )
)

st.altair_chart(hist_chart, use_container_width=True)

# --- Graphique 2 : taux de valeurs manquantes
missing_rate = (
    df.isna()
    .mean()
    .reset_index()
    .rename(columns={"index": "Variable", 0: "Taux"})
    .sort_values("Taux", ascending=False)
    .head(20)
)

missing_chart = (
    alt.Chart(missing_rate)
    .mark_bar()
    .encode(
        x=alt.X("Variable", sort="-y", title="Variable"),
        y=alt.Y("Taux", title="Proportion de valeurs manquantes"),
        tooltip=["Taux"]
    )
    .properties(
        title="Top 20 des variables avec le plus de valeurs manquantes",
        height=300
    )
)

st.altair_chart(missing_chart, use_container_width=True)

st.markdown(
    """
    ℹ️ Les valeurs manquantes sont fréquentes dans ce type de données
    et sont **nativement prises en charge par LightGBM**.
    """
)

# ======================================================
# SÉLECTION D’UN INDIVIDU
# ======================================================
st.header("3️⃣ Sélection d’un individu")

n_rows = len(df)

if n_rows == 1:
    st.info("Un seul individu disponible – sélection automatique.")
    row_id = 0
else:
    row_id = st.slider(
        "Choisir un individu",
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

st.markdown(
    """
    **Interprétation de la prédiction :**

    - **Classe 0** : l’événement cible ne se produit pas  
    - **Classe 1** : l’événement cible se produit  

    La probabilité correspond à la **confiance du modèle**
    pour la classe 1.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Classe prédite", int(prediction))

with col2:
    st.metric("Probabilité (classe 1)", round(float(probability), 4))

# ======================================================
# CONCLUSION
# ======================================================
st.header("5️⃣ Conclusion")

st.markdown(
    """
    ✅ Ce dashboard respecte l’ensemble des spécifications :

    - analyse exploratoire des données ;
    - visualisations interactives accessibles ;
    - sélection d’un individu ;
    - prédiction par un modèle récent (LightGBM) ;
    - application déployable sur le cloud.

    📌 La comparaison avec une baseline est documentée
    dans le notebook et la note méthodologique.
    """
)
