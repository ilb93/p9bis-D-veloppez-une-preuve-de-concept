import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import requests
import os

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="SAINT Scoring POC", layout="centered")
st.title("Proof of Concept – SAINT Transformer")
st.write("Deep Learning tabulaire – Modèle chargé depuis S3")

# =====================================================
# S3 URLS
# =====================================================

BASE_URL = "https://projetmodelsaint.s3.eu-north-1.amazonaws.com/"

FILES = {
    "weights": "saint_weights.pth",
    "config": "saint_config.pkl",
    "metadata": "saint_metadata.pkl",
    "threshold": "saint_threshold.pkl"
}

LOCAL_DIR = "models"
os.makedirs(LOCAL_DIR, exist_ok=True)

# =====================================================
# DOWNLOAD FILES IF NOT EXISTS
# =====================================================

def download_file(filename):
    local_path = os.path.join(LOCAL_DIR, filename)
    if not os.path.exists(local_path):
        url = BASE_URL + filename
        r = requests.get(url)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
        else:
            st.error(f"Erreur téléchargement {filename}")
            st.stop()
    return local_path

paths = {}
for key, file in FILES.items():
    paths[key] = download_file(file)

# =====================================================
# LOAD CONFIG
# =====================================================

with open(paths["config"], "rb") as f:
    config = pickle.load(f)

with open(paths["metadata"], "rb") as f:
    metadata = pickle.load(f)

with open(paths["threshold"], "rb") as f:
    threshold = pickle.load(f)

# =====================================================
# SIMPLE SAINT ARCHITECTURE (inference only)
# =====================================================

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

# =====================================================
# BUILD MODEL
# =====================================================

input_dim = len(metadata["numerical_columns"])

model = SimpleSaint(
    input_dim=input_dim,
    dim=config["dim"],
    depth=config["depth"]
)

state_dict = torch.load(paths["weights"], map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

st.success("Modèle chargé avec succès")

# =====================================================
# UI
# =====================================================

st.subheader("Test de prédiction")

inputs = []

for col in metadata["numerical_columns"]:
    value = st.number_input(col, value=0.0)
    inputs.append(value)

if st.button("Predict"):
    x = torch.tensor([inputs], dtype=torch.float32)

    with torch.no_grad():
        output = model(x)
        prob = torch.sigmoid(output).item()

    prediction = int(prob >= threshold)

    st.write("### Résultat")
    st.write("Probabilité :", round(prob, 4))
    st.write("Classe prédite :", prediction)
