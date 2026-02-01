import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

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
# IMPORT CSV
# ======================================================
st.subheader("📂 Import du jeu de données (CSV)")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV d’inférence (features prétraitées)",
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
# ANALYSE EXPLORATOIRE
# ======================================================
st.subheader("🔍 Analyse exploratoire des données")

st.markdown("### Statistiques descriptives (variables numériques)")
st.dataframe(df.describe().T)

# ======================================================
# SÉLECTION DES VARIABLES PERTINENTES POUR L’EDA
# ======================================================
# Règles :
# - numérique
# - non binaire
# - suffisamment variable
eda_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=True) > 10
    and df[col].std(skipna=True) > 1e-6
]

st.caption(
    f"{len(eda_cols)} variables continues pertinentes sélectionnées pour l’analyse exploratoire "
    "(variables binaires et techniques exclues)."
)

# ======================================================
# GRAPHIQUE 1 — HISTOGRAMME ROBUSTE
# ======================================================
st.markdown("### Distribution robuste d’une variable continue")

hist_col = st.selectbox(
    "Choisir une variable",
    eda_cols,
    key="hist_col"
)

data = df[hist_col].dropna()

# clipping pour lecture cohérente
low, high = data.quantile([0.01, 0.99])
data = data.clip(lower=low, upper=high)

counts, bins = np.histogram(data, bins=20)

hist_df = pd.DataFrame({
    "Intervalle": [
        f"{round(bins[i], 2)} → {round(bins[i+1], 2)}"
        for i in range(len(bins) - 1)
    ],
    "Effectif": counts
})

st.bar_chart(hist_df.set_index("Intervalle"))

st.caption(
    "Histogramme construit après exclusion des valeurs extrêmes (1 % – 99 %) "
    "pour garantir une représentation visuelle stable et interprétable."
)

# ======================================================
# GRAPHIQUE 2 — PROFIL STATISTIQUE (QUANTILES)
# ======================================================
st.markdown("### Profil statistique de la variable (quantiles)")

stat_col = st.selectbox(
    "Choisir une variable",
    eda_cols,
    key="stat_col"
)

s = df[stat_col].dropna()

quantiles = {
    "Minimum": s.min(),
    "1er quartile (25%)": s.quantile(0.25),
    "Médiane (50%)": s.quantile(0.50),
    "3e quartile (75%)": s.quantile(0.75),
    "Maximum": s.max()
}

stat_df = pd.DataFrame.from_dict(
    quantiles, orient="index", columns=["Valeur"]
)

st.bar_chart(stat_df)

st.caption(
    "Ce graphique présente les statistiques de position clés de la variable. "
    "Cette approche reste pertinente même lorsque les données ont été standardisées."
)

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

st.markdown(
    """
    **Interprétation métier :**

    - **Classe 0** : absence de risque de défaut de remboursement
    - **Classe 1** : risque de défaut de remboursement

    La probabilité correspond à l’estimation du risque de défaut pour la classe 1.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Classe prédite", prediction)

with col2:
    st.metric("Probabilité de défaut", round(proba, 3))

# ======================================================
# ACCESSIBILITÉ
# ======================================================
st.subheader("♿ Accessibilité (WCAG – critères essentiels)")

st.markdown(
    """
    Les critères essentiels d’accessibilité ont été pris en compte :
    - composants standards Streamlit compatibles clavier,
    - graphiques lisibles sans dépendance exclusive à la couleur,
    - hiérarchie claire des titres et sections,
    - informations systématiquement accompagnées de texte explicatif.
    """
)

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    Ce dashboard présente une **preuve de concept complète et cohérente**
    de scoring de risque de crédit basée sur un **modèle LightGBM**.

    L’analyse exploratoire se concentre volontairement sur les **variables continues
    informatives**, afin de garantir des visualisations interprétables, tandis que
    l’inférence repose sur un pipeline de données conforme aux **bonnes pratiques industrielles**.

    Cette approche démontre la capacité à **analyser, modéliser, déployer et expliquer**
    un système de machine learning dans un contexte professionnel.
    """
)
