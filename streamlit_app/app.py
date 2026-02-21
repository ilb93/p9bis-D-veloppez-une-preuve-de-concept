import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import requests
import tempfile
import os

# ==========================================================
# CONFIG STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="Proof of Concept – Scoring SAINT",
    layout="wide"
)

st.title("📊 Proof of Concept – Scoring de risque (SAINT Transformer)")

st.markdown("""
Cette application présente une **preuve de concept** basée sur un modèle **SAINT (Transformer tabulaire)**.

• Modèle chargé dynamiquement depuis AWS S3 (lecture publique)  
• Deep Learning pour données tabulaires  
• Seuil optimisé via F1-score  
""")

# ==========================================================
# URLS S3 PUBLIQUES
# ==========================================================
MODEL_URL = "https://projetmodelsaint.s3.eu-north-1.amazonaws.com/saint_full_model.pth"
THRESHOLD_URL = "https://projetmodelsaint.s3.eu-north-1.amazonaws.com/saint_threshold.pkl"
META_URL = "https://projetmodelsaint.s3.eu-north-1.amazonaws.com/saint_metadata.pkl"

# ==========================================================
# CHARGEMENT DU MODELE
# ==========================================================
@st.cache_resource
def load_model():

    temp_dir = tempfile.mkdtemp()

    model_path = os.path.join(temp_dir, "model.pth")
    threshold_path = os.path.join(temp_dir, "threshold.pkl")
    meta_path = os.path.join(temp_dir, "metadata.pkl")

    # téléchargement modèle
    r = requests.get(MODEL_URL)
    if r.status_code != 200:
        st.error("Impossible de télécharger le modèle depuis S3.")
        st.stop()
    with open(model_path, "wb") as f:
        f.write(r.content)

    # téléchargement threshold
    r = requests.get(THRESHOLD_URL)
    if r.status_code != 200:
        st.error("Impossible de télécharger le threshold depuis S3.")
        st.stop()
    with open(threshold_path, "wb") as f:
        f.write(r.content)

    # téléchargement metadata
    r = requests.get(META_URL)
    if r.status_code != 200:
        st.error("Impossible de télécharger les metadata depuis S3.")
        st.stop()
    with open(meta_path, "wb") as f:
        f.write(r.content)

    model = torch.load(model_path, map_location="cpu")
    model.eval()

    with open(threshold_path, "rb") as f:
        threshold = pickle.load(f)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return model, threshold, metadata


model, threshold, metadata = load_model()

# ==========================================================
# UPLOAD CSV
# ==========================================================
st.subheader("📂 Import du fichier CSV")

uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = [c.strip() for c in df.columns]

st.success("Fichier chargé avec succès")
st.write(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")

# ==========================================================
# SELECTION INDIVIDU
# ==========================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider("Choisir un individu", 0, len(df) - 1, 0)
row = df.iloc[row_id]

# ==========================================================
# PREPARATION INPUT SAINT
# ==========================================================
categorical_dims = metadata["categorical_dims"]
numerical_columns = metadata["numerical_columns"]

x_categ = []
x_cont = []

for col in df.columns:
    if col in categorical_dims:
        value = int(row[col])
        x_categ.append(value)
    elif col in numerical_columns:
        value = float(row[col])
        x_cont.append(value)

if len(x_categ) == 0 or len(x_cont) == 0:
    st.error("Les colonnes du CSV ne correspondent pas au modèle entraîné.")
    st.stop()

x_categ = torch.tensor([x_categ], dtype=torch.long)
x_cont = torch.tensor([x_cont], dtype=torch.float)

# ==========================================================
# PREDICTION
# ==========================================================
with torch.no_grad():
    output = model(x_categ, x_cont)

    if output.shape[1] == 1:
        proba = torch.sigmoid(output).item()
    else:
        proba = torch.softmax(output, dim=1)[0][1].item()

prediction = 1 if proba >= threshold else 0

# ==========================================================
# RESULTATS
# ==========================================================
st.subheader("📈 Résultat de la prédiction")

col1, col2 = st.columns(2)

if prediction == 0:
    verdict = "Faible risque de défaut"
else:
    verdict = "Risque élevé de défaut"

col1.metric("Décision du modèle", verdict)
col2.metric("Probabilité de défaut", f"{proba:.2%}")

st.markdown(f"Seuil appliqué : **{threshold:.4f}**")

st.success("Modèle SAINT chargé dynamiquement depuis AWS S3.")
