import streamlit as st
import torch
import requests
import io
import pickle
import numpy as np
from lit_saint.model import Saint
from lit_saint.config import SaintConfig

# ==========================================================
# CONFIGURATION
# ==========================================================

WEIGHTS_URL = "https://projetmodelsaint.s3.eu-north-1.amazonaws.com/saint_weights.pth"
CONFIG_URL = "https://projetmodelsaint.s3.eu-north-1.amazonaws.com/saint_config.pkl"
META_URL = "https://projetmodelsaint.s3.eu-north-1.amazonaws.com/saint_metadata.pkl"
THRESHOLD_URL = "https://projetmodelsaint.s3.eu-north-1.amazonaws.com/saint_threshold.pkl"

st.set_page_config(page_title="SAINT Risk Scoring", layout="centered")

st.title("📊 Proof of Concept – SAINT Transformer")
st.markdown("""
- Modèle chargé depuis AWS S3  
- Deep Learning tabulaire  
- Seuil optimisé F1-score  
""")

# ==========================================================
# CHARGEMENT UTILITAIRES
# ==========================================================

def load_pickle_from_s3(url):
    response = requests.get(url)
    response.raise_for_status()
    return pickle.load(io.BytesIO(response.content))


def load_weights_from_s3(url):
    response = requests.get(url)
    response.raise_for_status()
    buffer = io.BytesIO(response.content)
    return torch.load(buffer, map_location="cpu")


@st.cache_resource
def load_model():
    try:
        # Charger config
        config_dict = load_pickle_from_s3(CONFIG_URL)

        cfg = SaintConfig()
        cfg.depth = config_dict["depth"]
        cfg.heads = config_dict["heads"]
        cfg.dim = config_dict["dim"]
        cfg.lr = config_dict["lr"]
        cfg.dropout = config_dict["dropout"]
        cfg.class_weights = config_dict["class_weights"]

        # Charger metadata
        metadata = load_pickle_from_s3(META_URL)

        # Créer modèle
        model = Saint(
            categories=metadata["categorical_dims"],
            continuous=metadata["numerical_columns"],
            config=cfg,
            dim_target=metadata["dim_target"]
        )

        # Charger poids
        state_dict = load_weights_from_s3(WEIGHTS_URL)
        model.load_state_dict(state_dict)

        model.eval()

        # Charger threshold
        threshold = load_pickle_from_s3(THRESHOLD_URL)

        return model, threshold, metadata

    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None, None, None


model, threshold, metadata = load_model()

# ==========================================================
# INTERFACE SIMPLE DE TEST
# ==========================================================

if model is not None:

    st.subheader("Test prédiction")

    # Création input dummy compatible
    numerical_input = []
    for col in metadata["numerical_columns"]:
        value = st.number_input(f"{col}", value=0.0)
        numerical_input.append(value)

    if st.button("Prédire"):

        with torch.no_grad():

            x_num = torch.tensor([numerical_input], dtype=torch.float32)

            # Dummy catégories (si pas utilisées)
            if len(metadata["categorical_dims"]) > 0:
                x_cat = torch.zeros((1, len(metadata["categorical_dims"])), dtype=torch.long)
            else:
                x_cat = None

            output = model(x_cat, x_num)
            prob = torch.sigmoid(output).item()

            prediction = int(prob >= threshold)

            st.write("### Résultat")
            st.write(f"Probabilité : {prob:.4f}")
            st.write(f"Classe prédite : {prediction}")
