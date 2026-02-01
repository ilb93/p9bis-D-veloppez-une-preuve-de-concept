import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ======================================================
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Proof of Concept – Scoring de risque de défaut",
    layout="wide"
)

st.title("📊 Proof of Concept – Scoring de risque de défaut de remboursement")

st.markdown(
    """
Cette application présente une **preuve de concept** basée sur un modèle
**LightGBM**, utilisé pour estimer le **risque de défaut de remboursement d’un crédit**.

Les données utilisées pour l’inférence ont été **préparées avec le même pipeline
que lors de l’entraînement du modèle**, garantissant une parfaite cohérence.
"""
)

# ======================================================
# CHARGEMENT DU MODÈLE
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load(Path("artifacts") / "lgbm.joblib")

model = load_model()

# ======================================================
# IMPORT CSV (PRÉTRAITÉ)
# ======================================================
st.subheader("📂 Import du jeu de données (CSV d’inférence)")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV (features numériques)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Veuillez importer un fichier CSV pour continuer.")
    st.stop()

df = pd.read_csv(uploaded_file)
df = df.apply(pd.to_numeric, errors="coerce")

st.success("Fichier chargé avec succès")
st.write(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")
st.dataframe(df.head())

# ======================================================
# SÉLECTION DES 5 VARIABLES MÉTIER
# ======================================================
FEATURES = {
    "Âge (années)": "DAYS_BIRTH",
    "Ancienneté emploi (années)": "DAYS_EMPLOYED",
    "Montant du crédit (€)": "AMT_CREDIT",
    "Prix du bien (€)": "AMT_GOODS_PRICE",
    "Mensualité (€)": "AMT_ANNUITY",
}

missing = [v for v in FEATURES.values() if v not in df.columns]
if missing:
    st.error(f"Variables manquantes dans le CSV : {missing}")
    st.stop()

# ======================================================
# CONSTRUCTION DF LISIBLE HUMAINEMENT
# ======================================================
df_human = df[list(FEATURES.values())].copy()

# Conversion jours → années (Home Credit = valeurs négatives)
df_human["DAYS_BIRTH"] = (-df_human["DAYS_BIRTH"]) / 365.25
df_human["DAYS_EMPLOYED"] = (-df_human["DAYS_EMPLOYED"]) / 365.25

df_human.rename(columns={v: k for k, v in FEATURES.items()}, inplace=True)

# ======================================================
# ANALYSE EXPLORATOIRE – GRAPHIQUE 1
# ======================================================
st.subheader("🔍 Analyse exploratoire – Distribution de la population")

var_label_1 = st.selectbox(
    "Choisir une variable à analyser",
    list(FEATURES.keys()),
    key="dist_var"
)

data_1 = df_human[var_label_1].dropna()

fig1, ax1 = plt.subplots()
ax1.hist(data_1, bins=30)
ax1.set_title(f"Distribution de la population – {var_label_1}")
ax1.set_xlabel(var_label_1)
ax1.set_ylabel("Nombre d'individus")

st.pyplot(fig1, clear_figure=True)

# ======================================================
# SÉLECTION D’UN INDIVIDU
# ======================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider(
    "Choisir un individu",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

st.markdown("**Profil de l’individu sélectionné (valeurs lisibles)**")
st.dataframe(df_human.iloc[[row_id]])

# ======================================================
# ANALYSE EXPLORATOIRE – GRAPHIQUE 2
# ======================================================
st.subheader("📍 Position de l’individu dans la population")

var_label_2 = st.selectbox(
    "Choisir une variable",
    list(FEATURES.keys()),
    key="pos_var"
)

data_2 = df_human[var_label_2].dropna()
value_ind = df_human.loc[row_id, var_label_2]

fig2, ax2 = plt.subplots()
ax2.hist(data_2, bins=30)
ax2.axvline(value_ind, linewidth=2, label="Individu sélectionné")
ax2.set_title(f"Position de l’individu – {var_label_2}")
ax2.set_xlabel(var_label_2)
ax2.set_ylabel("Nombre d'individus")
ax2.legend()

st.pyplot(fig2, clear_figure=True)

# ======================================================
# PRÉDICTION
# ======================================================
st.subheader("📈 Résultat de la prédiction")

input_df = df.iloc[[row_id]]
proba = float(model.predict_proba(input_df)[0][1])
prediction = int(proba >= 0.5)

col1, col2 = st.columns(2)

with col1:
    st.metric("Classe prédite", prediction)

with col2:
    st.metric("Probabilité de défaut", f"{proba:.3f}")

st.markdown(
    """
**Interprétation métier :**

- **Classe 0** : pas de risque de défaut
- **Classe 1** : risque de défaut de remboursement
"""
)

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
Ce dashboard présente une **preuve de concept claire et lisible** :

- 5 **variables métier explicites**
- Données **exprimées en unités compréhensibles** (années, euros)
- 2 **visualisations complémentaires** :
  - distribution globale
  - position de l’individu
- Résultat de prédiction **cohérent avec le pipeline de production**

L’objectif n’est pas l’explicabilité avancée, mais la **démonstration fonctionnelle**
d’un moteur de scoring intégré dans une interface décisionnelle.
"""
)
