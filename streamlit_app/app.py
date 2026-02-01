import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.stats import percentileofscore

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Proof of Concept – Credit Risk Scoring",
    layout="wide"
)

st.title("📊 Proof of Concept – Credit Risk Scoring")
st.markdown(
    """
    Cette application présente une **preuve de concept de scoring de risque de crédit**.
    
    👉 Les résultats affichés sont **adaptés à une lecture humaine** :
    - aucune valeur standardisée,
    - aucune unité mathématique interne,
    - uniquement des **comparaisons relatives à la population**.
    """
)

# ======================================================
# MODEL
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load(Path("artifacts") / "lgbm.joblib")

model = load_model()

# ======================================================
# DATA
# ======================================================
st.subheader("📂 Chargement des données")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV (features prétraitées)",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df = df.apply(pd.to_numeric, errors="coerce")

st.success(f"{df.shape[0]} lignes chargées")

# ======================================================
# VARIABLES MÉTIER SÉLECTIONNÉES
# ======================================================
KEY_FEATURES = [
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED"
]

KEY_FEATURES = [c for c in KEY_FEATURES if c in df.columns]

# ======================================================
# INDIVIDU
# ======================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider(
    "Choisir un individu",
    0,
    len(df) - 1,
    0
)

individual = df.iloc[row_id]

# ======================================================
# ANALYSE HUMAINE DES VARIABLES
# ======================================================
st.subheader("📊 Analyse comparative (lecture humaine)")

for col in KEY_FEATURES:
    col_data = df[col].dropna()
    value = individual[col]

    if pd.isna(value):
        continue

    percentile = percentileofscore(col_data, value)

    # catégorisation humaine
    if percentile < 20:
        level = "Très faible"
    elif percentile < 40:
        level = "Faible"
    elif percentile < 60:
        level = "Moyen"
    elif percentile < 80:
        level = "Élevé"
    else:
        level = "Très élevé"

    st.markdown(f"### 🔹 {col}")
    st.metric(
        label="Position dans la population",
        value=f"{int(percentile)}e percentile",
        delta=level
    )

    # distribution par quintiles
    quintiles = pd.qcut(col_data, 5, labels=[
        "Très faible", "Faible", "Moyen", "Élevé", "Très élevé"
    ])

    dist = quintiles.value_counts(normalize=True).reindex(
        ["Très faible", "Faible", "Moyen", "Élevé", "Très élevé"]
    ) * 100

    st.bar_chart(dist)

    st.caption(
        f"La valeur de cet individu se situe dans la catégorie **{level}** "
        f"par rapport à l’ensemble de la population."
    )

# ======================================================
# PRÉDICTION
# ======================================================
st.subheader("📈 Résultat du modèle")

proba = float(model.predict_proba(individual.to_frame().T)[0][1])
prediction = int(proba >= 0.5)

col1, col2 = st.columns(2)

with col1:
    st.metric("Décision du modèle", "Risque" if prediction else "Pas de risque")

with col2:
    st.metric("Probabilité de défaut", f"{proba:.1%}")

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    Cette preuve de concept met en évidence une **approche orientée décision** :
    - les calculs internes du modèle sont masqués,
    - les résultats sont **standardisés pour l’humain** via des percentiles,
    - seules des **variables métier pertinentes** sont analysées.

    👉 Cette démarche permet de concilier **rigueur data science**  
    et **compréhension métier**, condition indispensable à l’usage réel
    d’un modèle de scoring.
    """
)
