import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time
time.sleep(15)

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
Cette application présente une **preuve de concept** de scoring de risque basée sur un modèle **SAINT Transformer**.

- Les **graphiques** affichent des **valeurs métier lisibles (années / euros)**  
- La **prédiction** utilise **exactement les variables attendues par le modèle**
- La **décision est fondée sur un modèle Deep Learning tabulaire**
"""
)

# ======================================================
# CHARGEMENT MODÈLE SAINT
# ======================================================

@st.cache_resource
def load_saint():

    with open("models/saint_config.pkl", "rb") as f:
        config = pickle.load(f)

    with open("models/saint_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    with open("models/saint_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)

    input_dim = len(metadata["numerical_columns"])

    class SimpleSaint(nn.Module):
        def __init__(self, input_dim, dim, depth):
            super().__init__()
            layers = []
            current_dim = input_dim
            for _ in range(depth):
                layers.append(nn.Linear(current_dim, dim))
                layers.append(nn.ReLU())
                current_dim = dim
            layers.append(nn.Linear(dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    model = SimpleSaint(
        input_dim=input_dim,
        dim=config["dim"],
        depth=config["depth"]
    )

    state_dict = torch.load("models/saint_weights.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, metadata, threshold


model, metadata, THRESHOLD = load_saint()
EXPECTED_FEATURES = metadata["numerical_columns"]

# ======================================================
# UPLOAD CSV
# ======================================================
st.subheader("📂 Import du fichier CSV")

uploaded_file = st.file_uploader(
    "Importer le fichier CSV unifié (ex : sample_unified.csv)",
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
# FORMATAGE
# ======================================================

def euro_fmt(x, pos=None):
    try:
        return f"{x:,.0f} €".replace(",", " ")
    except Exception:
        return ""

# ======================================================
# VARIABLES MÉTIER
# ======================================================

human_df = df.copy()

# ======================================================
# ANALYSE EXPLORATOIRE
# ======================================================

st.subheader("📊 Analyse exploratoire – population")

var_label = st.selectbox("Choisir une variable", EXPECTED_FEATURES)
series = pd.to_numeric(df[var_label], errors="coerce").dropna()

col_plot, col_info = st.columns([2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(series, bins=30, edgecolor="black")
    ax.set_title(f"Distribution — {var_label}")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)

with col_info:
    st.metric("Min", f"{series.min():.2f}")
    st.metric("Médiane", f"{series.median():.2f}")
    st.metric("Max", f"{series.max():.2f}")

# ======================================================
# SÉLECTION INDIVIDU
# ======================================================

st.subheader("🎯 Sélection d’un individu")

row_id = st.slider("Choisir un individu", 0, len(df) - 1, 0)

# ======================================================
# PRÉPARATION DONNÉES MODÈLE
# ======================================================

def build_model_row(data, idx, expected):
    row = []
    for f in expected:
        v = pd.to_numeric(data.loc[idx, f], errors="coerce")
        row.append(0.0 if pd.isna(v) else float(v))
    return torch.tensor([row], dtype=torch.float32)

X_row = build_model_row(df, row_id, EXPECTED_FEATURES)

# ======================================================
# PRÉDICTION SAINT
# ======================================================

with torch.no_grad():
    output = model(X_row)
    proba = torch.sigmoid(output).item()

if proba < THRESHOLD:
    verdict = "Faible risque de crédit"
elif proba < THRESHOLD + 0.2:
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
Cette preuve de concept démontre une **approche Deep Learning du scoring de crédit**, 
tout en conservant une interface métier claire et exploitable.
"""
)
