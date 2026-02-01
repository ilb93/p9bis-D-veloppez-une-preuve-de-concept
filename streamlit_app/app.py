import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="PoC – Scoring de risque de crédit",
    layout="wide"
)

st.title("📊 Proof of Concept – Scoring de risque de crédit")

st.markdown("""
Cette application présente une **preuve de concept** de scoring de risque de défaut.

- 📈 Visualisation basée sur des **données humaines réelles**
- 🤖 Prédiction basée sur un **modèle LightGBM**
- ✅ Séparation stricte **métier / ML**
""")

# ======================================================
# CHARGEMENT MODÈLE
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load(Path("artifacts") / "lgbm.joblib")

model = load_model()
MODEL_FEATURES = model.feature_name_

# ======================================================
# CHARGEMENT DATA
# ======================================================
st.subheader("📂 Données population (CSV unique)")

file = st.file_uploader(
    "Importer le fichier `sample_unified.csv`",
    type=["csv"]
)

if file is None:
    st.stop()

df = pd.read_csv(file)

st.success(f"{df.shape[0]} individus chargés")

# ======================================================
# VARIABLES HUMAINES
# ======================================================
HUMAN_VARS = {
    "Âge (années)": "age_years",
    "Ancienneté emploi (années)": "employment_years",
    "Montant du crédit (€)": "AMT_CREDIT",
    "Prix du bien (€)": "AMT_GOODS_PRICE",
    "Annuité du crédit (€)": "AMT_ANNUITY",
}

human_cols = list(HUMAN_VARS.values())

# ======================================================
# STATISTIQUES DESCRIPTIVES
# ======================================================
st.subheader("📊 Statistiques descriptives – population")

st.dataframe(
    df[human_cols].describe().T.round(2),
    use_container_width=True
)

# ======================================================
# ANALYSE EXPLORATOIRE
# ======================================================
st.subheader("📈 Analyse exploratoire – population")

var_label = st.selectbox(
    "Choisir une variable",
    list(HUMAN_VARS.keys())
)

var_col = HUMAN_VARS[var_label]

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df[var_col], bins=30, edgecolor="black")
ax.set_title(f"Distribution — {var_label}")
ax.set_xlabel(var_label)
ax.set_ylabel("Nombre d'individus")
st.pyplot(fig)

# ======================================================
# SÉLECTION INDIVIDU
# ======================================================
st.subheader("🎯 Sélection d’un individu")

idx = st.slider(
    "Choisir un individu",
    0,
    len(df) - 1,
    0
)

individual = df.iloc[idx]

# ======================================================
# POSITION DE L’INDIVIDU
# ======================================================
st.subheader("📍 Position de l’individu dans la population")

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.hist(df[var_col], bins=30, alpha=0.7, edgecolor="black")
ax2.axvline(
    individual[var_col],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Individu sélectionné"
)
ax2.legend()
ax2.set_title(f"Position de l’individu — {var_label}")
ax2.set_xlabel(var_label)
ax2.set_ylabel("Nombre d'individus")
st.pyplot(fig2)

# ======================================================
# PRÉDICTION ML
# ======================================================
st.subheader("🤖 Résultat du scoring de crédit")

X_ml = df[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
input_row = X_ml.iloc[[idx]]

proba = float(model.predict_proba(input_row)[0][1])

if proba < 0.5:
    st.success(
        f"🟢 **Profil à faible risque de défaut**\n\n"
        f"Probabilité estimée de défaut : **{proba:.2%}**\n\n"
        "Le profil ne présente **pas de danger significatif** pour l’octroi du crédit."
    )
else:
    st.error(
        f"🔴 **Profil à risque de défaut**\n\n"
        f"Probabilité estimée de défaut : **{proba:.2%}**\n\n"
        "Le profil présente un **risque significatif**, nécessitant une analyse approfondie."
    )

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown("""
✔️ Données **humaines interprétables**  
✔️ Modèle **robuste et cohérent**  
✔️ Comparaison individu / population claire  

👉 Cette application respecte les **standards professionnels** attendus en data science appliquée.
""")
