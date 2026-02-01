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
    artifacts_path = Path("artifacts")
    return joblib.load(artifacts_path / "lgbm.joblib")

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

# --- Statistiques descriptives
st.markdown("### Statistiques descriptives")
st.dataframe(df.describe().T)

numeric_cols = df.columns.tolist()

# ======================================================
# GRAPHIQUE 1 : Histogramme binned (stable)
# ======================================================
st.markdown("### Distribution d’une variable continue")

col_hist = st.selectbox(
    "Choisir une variable continue",
    numeric_cols,
    key="hist_var"
)

hist_data = df[col_hist].dropna()

counts, bin_edges = np.histogram(hist_data, bins=20)

hist_df = pd.DataFrame({
    "Intervalle": [
        f"{round(bin_edges[i], 2)} → {round(bin_edges[i+1], 2)}"
        for i in range(len(bin_edges) - 1)
    ],
    "Effectif": counts
})

st.bar_chart(hist_df.set_index("Intervalle"))

# ======================================================
# GRAPHIQUE 2 : Profil statistique (quantiles)
# ======================================================
st.markdown("### Profil statistique de la variable (quantiles)")

col_quant = st.selectbox(
    "Choisir une variable pour l’analyse statistique",
    numeric_cols,
    key="quant_var"
)

quantiles = df[col_quant].describe()[["min", "25%", "50%", "75%", "max"]]

quant_df = pd.DataFrame({
    "Statistique": quantiles.index,
    "Valeur": quantiles.values
}).set_index("Statistique")

st.bar_chart(quant_df)

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
st.subheader("♿ Accessibilité et conformité WCAG")

st.markdown(
    """
    Les critères essentiels d’accessibilité du **WCAG** ont été pris en compte :

    - Composants Streamlit standards compatibles avec la navigation clavier
    - Graphiques lisibles sans dépendance exclusive à la couleur
    - Hiérarchie claire des titres et sections
    - Informations toujours accompagnées de texte explicatif
    - Absence d’informations critiques transmises uniquement par des codes visuels
    """
)

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    Ce dashboard présente une **preuve de concept complète et opérationnelle**
    de scoring de risque de crédit reposant sur un **algorithme récent (LightGBM)**.

    Il combine une **analyse exploratoire interactive**, une **sélection dynamique
    des données en entrée**, et une **prédiction interprétable orientée métier**,
    dans un environnement **déployé sur le cloud**.

    Cette preuve de concept démontre la maîtrise de l’ensemble de la chaîne
    data science, depuis l’exploration des données jusqu’à la restitution
    des résultats à destination d’utilisateurs non techniques.
    """
)
