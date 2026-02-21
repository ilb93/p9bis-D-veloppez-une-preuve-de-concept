import streamlit as st
import torch
import pandas as pd
import numpy as np
import requests
import pickle
import tempfile
import os

st.set_page_config(page_title="SAINT Scoring", layout="wide")

st.title("📊 Proof of Concept – SAINT Transformer")

st.markdown("""
• Modèle chargé depuis AWS S3  
• Deep Learning tabulaire  
• Seuil optimisé F1-score  
""")

# ==========================================================
# URLS S3 (ADAPTE SI BESOIN)
# ==========================================================

BASE_URL = "https://projetmodelsaint.s3.eu-north-1.amazonaws.com"

MODEL_URL = f"{BASE_URL}/saint_full_model.pth"
META_URL = f"{BASE_URL}/saint_metadata.pkl"
THRESHOLD_URL = f"{BASE_URL}/saint_threshold.pkl"

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_everything():

    tmp = tempfile.mkdtemp()

    model_path = os.path.join(tmp, "model.pth")
    meta_path = os.path.join(tmp, "meta.pkl")
    threshold_path = os.path.join(tmp, "threshold.pkl")

    # téléchargement modèle
    with open(model_path, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

    with open(meta_path, "wb") as f:
        f.write(requests.get(META_URL).content)

    with open(threshold_path, "wb") as f:
        f.write(requests.get(THRESHOLD_URL).content)

    # ⚠️ chargement robuste
    model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model.eval()

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    with open(threshold_path, "rb") as f:
        threshold = pickle.load(f)

    return model, metadata, threshold


try:
    model, metadata, threshold = load_everything()
    st.success("✅ Modèle SAINT chargé avec succès")
except Exception as e:
    st.error(f"Erreur chargement modèle : {e}")
    st.stop()

# ==========================================================
# CSV INPUT
# ==========================================================

st.subheader("📂 Charger un CSV")

file = st.file_uploader("Importer votre dataset", type=["csv"])

if file is None:
    st.stop()

df = pd.read_csv(file)

st.write("Aperçu des données :", df.head())

# ==========================================================
# SELECTION INDIVIDU
# ==========================================================

index = st.slider("Choisir une ligne", 0, len(df)-1, 0)
row = df.iloc[index]

# ==========================================================
# PREP INPUT
# ==========================================================

categorical_dims = metadata["categorical_dims"]
numerical_columns = metadata["numerical_columns"]

x_cat = []
x_num = []

for col in df.columns:
    if col in categorical_dims:
        x_cat.append(int(row[col]))
    elif col in numerical_columns:
        x_num.append(float(row[col]))

x_cat = torch.tensor([x_cat], dtype=torch.long)
x_num = torch.tensor([x_num], dtype=torch.float)

# ==========================================================
# PREDICTION
# ==========================================================

with torch.no_grad():
    output = model(x_cat, x_num)

    if output.shape[1] == 1:
        prob = torch.sigmoid(output).item()
    else:
        prob = torch.softmax(output, dim=1)[0][1].item()

prediction = 1 if prob >= threshold else 0

# ==========================================================
# DISPLAY RESULT
# ==========================================================

st.subheader("📈 Résultat")

if prediction == 1:
    st.error(f"⚠️ Risque élevé ({prob:.2%})")
else:
    st.success(f"✅ Faible risque ({prob:.2%})")

st.write(f"Seuil utilisé : {threshold:.4f}")
