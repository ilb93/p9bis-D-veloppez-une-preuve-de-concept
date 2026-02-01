import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ======================================================
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Proof of Concept – Credit Risk Scoring",
    layout="wide"
)

st.title("📊 Proof of Concept – Credit Risk Scoring")

st.markdown(
    """
    Cette application présente une **preuve de concept de scoring de risque de crédit**.
    Les visualisations sont volontairement affichées en **unités compréhensibles**
    (années, euros) afin de garantir une lecture analytique claire.
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
# CHARGEMENT DES DONNÉES
# ======================================================
st.subheader("📂 Chargement des données")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV (features prétraitées)",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df = df.apply(pd.to_numeric, errors="coerce")

# ======================================================
# VARIABLES MÉTIER
# ======================================================
FEATURES = {
    "Montant du crédit (€)": "AMT_CREDIT",
    "Prix du bien (€)": "AMT_GOODS_PRICE",
    "Annuité (€)": "AMT_ANNUITY",
    "Âge du client (années)": "DAYS_BIRTH",
    "Ancienneté emploi (années)": "DAYS_EMPLOYED"
}

# ======================================================
# CONVERSION HUMAINE DES DONNÉES
# ======================================================
df_human = df.copy()

df_human["DAYS_BIRTH"] = -df_human["DAYS_BIRTH"] / 365
df_human["DAYS_EMPLOYED"] = -df_human["DAYS_EMPLOYED"] / 365

# ======================================================
# SÉLECTION D’UN INDIVIDU (ON NE CASSE RIEN)
# ======================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider(
    "Choisir un individu",
    min_value=0,
    max_value=len(df_human) - 1,
    value=0
)

individual = df_human.iloc[row_id]
individual_raw = df.iloc[row_id]  # pour le modèle

# ======================================================
# GRAPHIQUE 1 — DISTRIBUTION POPULATION
# ======================================================
st.subheader("📊 Distribution de la population")

feature_label_1 = st.selectbox(
    "Choisir une variable",
    list(FEATURES.keys()),
    key="dist_pop"
)

col_1 = FEATURES[feature_label_1]
data_1 = df_human[col_1].dropna()

# winsorisation analytique
low, high = data_1.quantile([0.01, 0.99])
data_1 = data_1.clip(low, high)

counts, bins = np.histogram(data_1, bins=25)

hist_df_1 = pd.DataFrame({
    "Intervalle": [
        f"{bins[i]:.0f} – {bins[i+1]:.0f}"
        for i in range(len(bins) - 1)
    ],
    "Effectif": counts
})

st.bar_chart(hist_df_1.set_index("Intervalle"))

# ======================================================
# GRAPHIQUE 2 — POSITION DE L’INDIVIDU
# ======================================================
st.subheader("📈 Position de l’individu dans la distribution")

feature_label_2 = st.selectbox(
    "Choisir une variable",
    list(FEATURES.keys()),
    key="pos_ind"
)

col_2 = FEATURES[feature_label_2]
data_2 = df_human[col_2].dropna()

low, high = data_2.quantile([0.01, 0.99])
data_2 = data_2.clip(low, high)

counts, bins = np.histogram(data_2, bins=25)

hist_df_2 = pd.DataFrame({
    "Centre de classe": [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)],
    "Effectif": counts
})

st.line_chart(hist_df_2.set_index("Centre de classe"))

st.metric(
    label=f"Valeur de l’individu – {feature_label_2}",
    value=f"{individual[col_2]:.0f}"
)

# ======================================================
# PRÉDICTION DU MODÈLE (INCHANGÉE)
# ======================================================
st.subheader("📈 Résultat du modèle")

proba = float(model.predict_proba(individual_raw.to_frame().T)[0][1])
prediction = int(proba >= 0.5)

col1, col2 = st.columns(2)

with col1:
    st.metric("Décision du modèle", "Risque" if prediction else "Pas de risque")

with col2:
    st.metric("Probabilité de défaut", f"{proba:.2%}")

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    Cette preuve de concept combine :
    - une **analyse exploratoire lisible humainement**,
    - une **comparaison individuelle vs population**,
    - et une **prédiction de risque robuste** issue d’un modèle LightGBM.

    Les transformations mathématiques internes au modèle sont volontairement
    **dissociées des visualisations**, afin de garantir une compréhension claire
    pour un public métier ou décisionnel.
    """
)
