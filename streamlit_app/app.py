import streamlit as st
import pandas as pd
import numpy as np
import torch
import boto3
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ======================================================
# CONFIG STREAMLIT
# ======================================================

st.set_page_config(
    page_title="Proof of Concept – Scoring de risque de crédit",
    layout="wide"
)

st.title("📊 Proof of Concept – Scoring de risque de défaut de remboursement")

st.markdown(
    """
Cette application présente une **preuve de concept** de scoring de risque basée sur un modèle **SAINT (Transformer tabulaire)**.

- Les **graphiques** affichent des **valeurs métier lisibles (années / euros)**
- La **prédiction** repose sur un modèle Deep Learning (SAINT)
- Le modèle est chargé dynamiquement depuis **AWS S3**
"""
)

# ======================================================
# CONFIG S3
# ======================================================

BUCKET_NAME = "projetmodelsaint"
WEIGHTS_KEY = "saint_weights.pth"
LOCAL_WEIGHTS = "saint_weights.pth"

# ⚠️ IMPORTANT :
# Tu dois avoir saint_model.py dans ton projet
# avec la classe SAINT identique à l'entraînement

from saint_model import SAINT


def download_from_s3():
    s3 = boto3.client("s3")
    s3.download_file(BUCKET_NAME, WEIGHTS_KEY, LOCAL_WEIGHTS)


@st.cache_resource
def load_model():

    # Télécharger si absent
    if not os.path.exists(LOCAL_WEIGHTS):
        download_from_s3()

    # ⚠️ Mets EXACTEMENT les mêmes paramètres qu'à l'entraînement
    model = SAINT(
        # Exemple :
        # dim=256,
        # depth=6,
        # heads=8,
        # attn_dropout=0.1,
        # ff_dropout=0.1,
        # categories=...,
        # num_continuous=...,
        # etc.
    )

    model.load_state_dict(
        torch.load(LOCAL_WEIGHTS, map_location="cpu")
    )

    model.eval()
    return model


model = load_model()

# ======================================================
# UPLOAD CSV
# ======================================================

st.subheader("📂 Import du fichier CSV")

uploaded_file = st.file_uploader(
    "Importer le fichier CSV unifié",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = [c.strip() for c in df.columns]

st.success("Fichier chargé avec succès")
st.write(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")

st.markdown("### 📈 Statistiques descriptives")
st.dataframe(df.describe().T, use_container_width=True)

# ======================================================
# OUTILS FORMATAGE
# ======================================================

def euro_fmt(x, pos=None):
    try:
        return f"{x:,.0f} €".replace(",", " ")
    except Exception:
        return ""

def clean_employment_years(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s < 900, np.nan)
    s = s.where(s >= 0, 0)
    return s

def clean_age_years(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s <= 120), np.nan)

def clean_money(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 0, np.nan)

# ======================================================
# VARIABLES LISIBLES
# ======================================================

human_df = pd.DataFrame({
    "Âge (années)": clean_age_years(df["age_years"]),
    "Ancienneté emploi (années)": clean_employment_years(df["employment_years"]),
    "Montant du crédit (€)": clean_money(df["AMT_CREDIT"]),
    "Prix du bien (€)": clean_money(df["AMT_GOODS_PRICE"]),
    "Annuité du crédit (€)": clean_money(df["AMT_ANNUITY"]),
})

# ======================================================
# ANALYSE EXPLORATOIRE
# ======================================================

st.subheader("📊 Analyse exploratoire – population")

var_label = st.selectbox("Choisir une variable", human_df.columns)
series = human_df[var_label].dropna()

col_plot, col_info = st.columns([2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(series, bins=30, edgecolor="black")
    ax.set_title(f"Distribution — {var_label}")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Nombre d'individus")

    if "€" in var_label:
        ax.xaxis.set_major_formatter(FuncFormatter(euro_fmt))

    st.pyplot(fig)

with col_info:
    st.markdown("### 📌 Repères")
    st.metric("Min", euro_fmt(series.min()) if "€" in var_label else f"{series.min():.1f}")
    st.metric("Médiane", euro_fmt(series.median()) if "€" in var_label else f"{series.median():.1f}")
    st.metric("Max", euro_fmt(series.max()) if "€" in var_label else f"{series.max():.1f}")

# ======================================================
# SÉLECTION INDIVIDU
# ======================================================

st.subheader("🎯 Sélection d’un individu")

row_id = st.slider("Choisir un individu", 0, len(df) - 1, 0)

# ======================================================
# POSITION DANS LA POPULATION
# ======================================================

st.markdown("### 📍 Position de l’individu dans la population")

val = human_df.loc[row_id, var_label]

fig2, ax2 = plt.subplots(figsize=(9, 2))
ax2.hist(series, bins=30, edgecolor="black", alpha=0.7)
ax2.axvline(val, color="red", linewidth=2)
ax2.set_xlabel(var_label)
ax2.set_ylabel("Population")

if "€" in var_label:
    ax2.xaxis.set_major_formatter(FuncFormatter(euro_fmt))

st.pyplot(fig2)

# ======================================================
# PRÉPARATION DONNÉES POUR SAINT
# ======================================================

X_row = df.iloc[[row_id]].astype(float)

# ======================================================
# PRÉDICTION SAINT
# ======================================================

with torch.no_grad():
    input_tensor = torch.tensor(
        X_row.values,
        dtype=torch.float32
    )

    outputs = model(input_tensor)
    proba = torch.softmax(outputs, dim=1)[0][1].item()

# ======================================================
# RÉSULTAT
# ======================================================

if proba < 0.3:
    verdict = "Faible risque de crédit"
elif proba < 0.6:
    verdict = "Risque de crédit modéré"
else:
    verdict = "Risque de crédit élevé"

st.subheader("📈 Résultat de la prédiction")

c1, c2 = st.columns(2)
c1.metric("Évaluation du profil", verdict)
c2.metric("Probabilité de défaut", f"{proba:.2%}")

# ======================================================
# CONCLUSION
# ======================================================

st.subheader("✅ Conclusion")

st.markdown(
    """
Cette preuve de concept démontre une approche moderne du scoring de crédit,
reposant sur un **modèle Transformer tabulaire (SAINT)** déployé
dynamiquement via AWS S3.

Elle illustre la capacité à :
- industrialiser un modèle Deep Learning,
- séparer code et artefacts,
- déployer proprement en environnement cloud.
"""
)
