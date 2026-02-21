import streamlit as st
import pandas as pd
import numpy as np
import torch
import boto3
import pickle
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

- Modèle chargé dynamiquement depuis AWS S3  
- Deep Learning pour données tabulaires  
- Seuil optimisé via F1-score  
""")

# ==========================================================
# CONFIG S3
# ==========================================================
BUCKET_NAME = "projetmodelsaint"

MODEL_KEY = "saint_full_model.pth"
THRESHOLD_KEY = "saint_threshold.pkl"
META_KEY = "saint_metadata.pkl"

# ==========================================================
# FONCTION DE CHARGEMENT S3
# ==========================================================
@st.cache_resource
def load_from_s3():

    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_REGION"]
    )

    temp_dir = tempfile.mkdtemp()

    model_path = os.path.join(temp_dir, MODEL_KEY)
    threshold_path = os.path.join(temp_dir, THRESHOLD_KEY)
    meta_path = os.path.join(temp_dir, META_KEY)

    s3.download_file(BUCKET_NAME, MODEL_KEY, model_path)
    s3.download_file(BUCKET_NAME, THRESHOLD_KEY, threshold_path)
    s3.download_file(BUCKET_NAME, META_KEY, meta_path)

    model = torch.load(model_path, map_location="cpu")
    model.eval()

    with open(threshold_path, "rb") as f:
        threshold = pickle.load(f)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return model, threshold, metadata


model, threshold, metadata = load_from_s3()

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

# ==========================================================
# SÉLECTION D’UN INDIVIDU
# ==========================================================
row_id = st.slider("Choisir un individu", 0, len(df) - 1, 0)

row = df.iloc[row_id]

# ==========================================================
# PRÉPARATION INPUT SAINT
# ==========================================================
categorical_dims = metadata["categorical_dims"]
numerical_columns = metadata["numerical_columns"]

x_categ = []
x_cont = []

for col in df.columns:
    if col in categorical_dims:
        x_categ.append(int(row[col]))
    elif col in numerical_columns:
        x_cont.append(float(row[col]))

x_categ = torch.tensor([x_categ], dtype=torch.long)
x_cont = torch.tensor([x_cont], dtype=torch.float)

# ==========================================================
# PRÉDICTION
# ==========================================================
with torch.no_grad():
    output = model(x_categ, x_cont)

    if output.shape[1] == 1:
        proba = torch.sigmoid(output).item()
    else:
        proba = torch.softmax(output, dim=1)[0][1].item()

prediction = 1 if proba >= threshold else 0

# ==========================================================
# AFFICHAGE RÉSULTATS
# ==========================================================
st.subheader("📈 Résultat de la prédiction")

c1, c2 = st.columns(2)

if prediction == 0:
    verdict = "Faible risque"
else:
    verdict = "Risque élevé"

c1.metric("Décision du modèle", verdict)
c2.metric("Probabilité de défaut", f"{proba:.2%}")

st.markdown(f"Seuil utilisé : **{threshold:.4f}**")

# ==========================================================
# FIN
# ==========================================================
st.success("Modèle SAINT chargé dynamiquement depuis AWS S3.")
