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

st.markdown("""
Cette application présente une **preuve de concept** basée sur un modèle **LightGBM**.
L’analyse exploratoire ci-dessous est volontairement réalisée sur des **variables métier
exprimées en unités compréhensibles (années, euros)**, indépendamment des transformations
utilisées par le modèle.
""")

# ======================================================
# CHARGEMENT DU MODÈLE
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load(Path("artifacts") / "lgbm.joblib")

model = load_model()

# ======================================================
# IMPORT CSV
# ======================================================
st.subheader("📂 Import du jeu de données")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

# ======================================================
# VARIABLES MÉTIER (HUMAINES)
# ======================================================
eda_df = pd.DataFrame()

eda_df["Âge (années)"] = (-df["DAYS_BIRTH"] / 365).clip(18, 100)
eda_df["Ancienneté emploi (années)"] = (
    df["DAYS_EMPLOYED"]
    .where(df["DAYS_EMPLOYED"] < 0, np.nan) * -1 / 365
)

eda_df["Montant du crédit (€)"] = df["AMT_CREDIT"]
eda_df["Prix du bien (€)"] = df["AMT_GOODS_PRICE"]
eda_df["Annuité du crédit (€ / an)"] = df["AMT_ANNUITY"]

eda_df = eda_df.dropna()

st.success("Variables métier reconstruites pour l’analyse exploratoire")

# ======================================================
# ANALYSE EXPLORATOIRE – GRAPH 1
# ======================================================
st.subheader("🔍 Analyse exploratoire – Distribution de la population")

var_eda = st.selectbox(
    "Choisir une variable à analyser",
    eda_df.columns
)

fig, ax = plt.subplots()
ax.hist(eda_df[var_eda], bins=30)
ax.set_xlabel(var_eda)
ax.set_ylabel("Nombre d'individus")
ax.set_title(f"Distribution de la population – {var_eda}")

st.pyplot(fig)

# ======================================================
# ANALYSE EXPLORATOIRE – GRAPH 2
# ======================================================
st.subheader("📊 Statistiques de position")

stats = eda_df[var_eda].describe()[["min", "25%", "50%", "75%", "max"]]

fig2, ax2 = plt.subplots()
ax2.bar(stats.index, stats.values)
ax2.set_title(f"Profil statistique – {var_eda}")
ax2.set_ylabel(var_eda)

st.pyplot(fig2)

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

input_df = df.iloc[[row_id]]
st.dataframe(input_df)

# ======================================================
# PRÉDICTION
# ======================================================
st.subheader("📈 Résultat de la prédiction")

proba = float(model.predict_proba(input_df)[0][1])
prediction = int(proba >= 0.5)

col1, col2 = st.columns(2)

with col1:
    st.metric("Risque de défaut estimé", "Oui" if prediction == 1 else "Non")

with col2:
    st.metric("Probabilité de défaut", f"{proba:.2%}")

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown("""
Cette application illustre une **preuve de concept complète** de scoring de risque de crédit.

- Le **modèle** repose sur des variables transformées pour optimiser la performance.
- L’**analyse exploratoire** est réalisée sur des **variables métier exprimées en unités
compréhensibles**, afin de faciliter l’interprétation humaine.
- Cette séparation reflète les **bonnes pratiques industrielles** en data science.
""")
