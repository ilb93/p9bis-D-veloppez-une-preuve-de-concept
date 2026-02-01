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
    page_title="Proof of Concept – Scoring de risque de crédit",
    layout="wide"
)

st.title("📊 Proof of Concept – Scoring de risque de défaut de remboursement")

st.markdown(
    """
    Cette application présente une **preuve de concept** de scoring de risque
    basée sur un modèle **LightGBM**.

    🔹 Les **graphiques** utilisent des **données métiers réelles**
    🔹 Les **prédictions** utilisent des **features prétraitées**
    
    👉 Cette séparation respecte les **bonnes pratiques industrielles**
    (interprétabilité / MLOps).
    """
)

# ======================================================
# CHARGEMENT MODÈLE
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load(Path("artifacts") / "lgbm.joblib")

model = load_model()

# ======================================================
# CHARGEMENT DATA POPULATION (HUMAINE)
# ======================================================
st.subheader("📂 Données population (visualisation métier)")

population_file = st.file_uploader(
    "Importer le fichier population humaine",
    type=["csv"],
    key="population"
)

if population_file is None:
    st.info("Veuillez importer le fichier `sample_population_human.csv`.")
    st.stop()

pop_df = pd.read_csv(population_file)

st.success("Données population chargées")
st.write(f"{pop_df.shape[0]} individus")
st.dataframe(pop_df.head())

# ======================================================
# VARIABLES MÉTIER
# ======================================================
VARIABLES = {
    "Âge (années)": "age_years",
    "Ancienneté emploi (années)": "employment_years",
    "Montant du crédit (€)": "AMT_CREDIT",
    "Prix du bien (€)": "AMT_GOODS_PRICE",
    "Annuité du crédit (€)": "AMT_ANNUITY",
}

# ======================================================
# ANALYSE EXPLORATOIRE
# ======================================================
st.subheader("📊 Analyse exploratoire – population")

var_label = st.selectbox(
    "Choisir une variable",
    list(VARIABLES.keys())
)

var_col = VARIABLES[var_label]

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pop_df[var_col], bins=30, edgecolor="black")
    ax.set_title(f"Distribution — {var_label}")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)

with col2:
    st.markdown("### ℹ️ Interprétation")

    if "Âge" in var_label:
        st.info(
            "Variable démographique réelle.\n\n"
            "La population est majoritairement composée d'adultes actifs."
        )
    elif "Ancienneté" in var_label:
        st.info(
            "Ancienneté professionnelle en années.\n\n"
            "Présence de carrières longues et de profils récents."
        )
    else:
        st.info(
            "Variable monétaire réelle.\n\n"
            "Distribution asymétrique typique des données financières."
        )

# ======================================================
# CHARGEMENT DATA INFÉRENCE (ML)
# ======================================================
st.subheader("🎯 Prédiction individuelle (modèle ML)")

inference_file = st.file_uploader(
    "Importer le fichier d'inférence ML",
    type=["csv"],
    key="inference"
)

if inference_file is None:
    st.info("Veuillez importer le fichier `sample_inference_clean.csv`.")
    st.stop()

X_inf = pd.read_csv(inference_file)
X_inf = X_inf.apply(pd.to_numeric, errors="coerce").fillna(0)

st.success("Données d'inférence chargées")
st.write(f"{X_inf.shape[0]} individus | {X_inf.shape[1]} features")

# ======================================================
# SÉLECTION INDIVIDU
# ======================================================
row_id = st.slider(
    "Choisir un individu",
    0,
    len(X_inf) - 1,
    0
)

input_row = X_inf.iloc[[row_id]]

# ======================================================
# PRÉDICTION
# ======================================================
proba = float(model.predict_proba(input_row)[0][1])
prediction = int(proba >= 0.5)

st.markdown("### 📈 Résultat de la prédiction")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "Classe prédite",
        "Risque de défaut" if prediction == 1 else "Pas de défaut"
    )

with c2:
    st.metric(
        "Probabilité de défaut",
        f"{proba:.2%}"
    )

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    ✔️ Les **graphiques** reposent sur des données **métier interprétables**  
    ✔️ Les **prédictions** reposent sur des **features standardisées**  
    ✔️ La séparation des usages garantit **robustesse et crédibilité**

    👉 Cette architecture correspond aux **standards professionnels**
    en data science appliquée.
    """
)
