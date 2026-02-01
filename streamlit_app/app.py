import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="POC – Scoring risque de défaut",
    layout="wide"
)

st.title("📊 Proof of Concept – Scoring de risque de défaut")

# ======================================================
# CHARGEMENT MODÈLE
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load(Path("artifacts") / "lgbm.joblib")

model = load_model()

# ======================================================
# UPLOAD CSV
# ======================================================
uploaded_file = st.file_uploader(
    "Importer le CSV d’inférence",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

# ======================================================
# VARIABLES MÉTIER & RECONSTRUCTION HUMAINE
# ======================================================
FEATURES = {
    "DAYS_BIRTH": {
        "label": "Âge (années)",
        "mean": 43,
        "std": 11
    },
    "DAYS_EMPLOYED": {
        "label": "Ancienneté emploi (années)",
        "mean": 7,
        "std": 8
    },
    "AMT_CREDIT": {
        "label": "Montant du crédit (€)",
        "mean": 600_000,
        "std": 400_000
    },
    "AMT_GOODS_PRICE": {
        "label": "Prix du bien (€)",
        "mean": 540_000,
        "std": 370_000
    },
    "AMT_ANNUITY": {
        "label": "Annuité du crédit (€ / an)",
        "mean": 27_000,
        "std": 14_000
    }
}

# Reconstruction humaine
df_human = pd.DataFrame()
for col, meta in FEATURES.items():
    df_human[col] = df[col] * meta["std"] + meta["mean"]

# ======================================================
# GRAPHIQUE 1 – DISTRIBUTION POPULATION
# ======================================================
st.subheader("📊 Distribution de la population")

var1 = st.selectbox(
    "Choisir une variable",
    list(FEATURES.keys()),
    format_func=lambda x: FEATURES[x]["label"]
)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.hist(df_human[var1], bins=40)
ax1.set_xlabel(FEATURES[var1]["label"])
ax1.set_ylabel("Nombre d’individus")
st.pyplot(fig1)

# ======================================================
# SÉLECTION INDIVIDU
# ======================================================
st.subheader("🎯 Sélection d’un individu")
row_id = st.slider("Index individu", 0, len(df) - 1, 0)

input_df = df.iloc[[row_id]]
input_human = df_human.iloc[row_id]

# ======================================================
# GRAPHIQUE 2 – POSITION INDIVIDU
# ======================================================
st.subheader("📍 Position de l’individu dans la population")

var2 = st.selectbox(
    "Choisir une variable",
    list(FEATURES.keys()),
    key="var2",
    format_func=lambda x: FEATURES[x]["label"]
)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.hist(df_human[var2], bins=40, alpha=0.7)
ax2.axvline(
    input_human[var2],
    linewidth=3
)
ax2.set_xlabel(FEATURES[var2]["label"])
ax2.set_ylabel("Population")
st.pyplot(fig2)

st.metric(
    FEATURES[var2]["label"],
    f"{int(input_human[var2]):,}".replace(",", " ")
)

# ======================================================
# PRÉDICTION
# ======================================================
st.subheader("📈 Prédiction du modèle")

proba = float(model.predict_proba(input_df)[0][1])

st.metric("Probabilité de défaut", f"{proba:.2%}")
st.metric("Décision modèle", "RISQUE" if proba >= 0.5 else "PAS DE RISQUE")

# ======================================================
# NOTE MÉTHODOLOGIQUE
# ======================================================
st.info(
    "⚠️ Les valeurs affichées sont **reconstruites à partir de variables standardisées** "
    "afin d’offrir une lecture métier lisible dans le cadre de cette preuve de concept."
)
