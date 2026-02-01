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

    ✅ **Un seul fichier CSV** est importé (valeurs humaines + features modèle).
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
# IMPORT CSV (UN SEUL)
# ======================================================
st.subheader("📂 Import du jeu de données (CSV)")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV (sample_unified.csv)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Veuillez importer un fichier CSV pour continuer.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("Fichier chargé avec succès")
st.write(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")
st.dataframe(df.head())

# ======================================================
# COLONNES HUMAINES (pour graphes)
# ======================================================
HUMAN_VARS = {
    "Âge (années)": ("age_years", "démographique"),
    "Ancienneté emploi (années)": ("employment_years", "carrière"),
    "Montant du crédit (€)": ("AMT_CREDIT", "monétaire"),
    "Prix du bien (€)": ("AMT_GOODS_PRICE", "monétaire"),
    "Annuité du crédit (€)": ("AMT_ANNUITY", "monétaire"),
}

missing_human = [col for (col, _) in HUMAN_VARS.values() if col not in df.columns]
if missing_human:
    st.error(
        "Ton CSV n'a pas les colonnes humaines attendues pour les graphiques : "
        f"{missing_human}\n\n"
        "➡️ Utilise le fichier généré `sample_unified.csv`."
    )
    st.stop()

# ======================================================
# ANALYSE EXPLORATOIRE (GRAPHIQUES PROPREMENT)
# ======================================================
st.subheader("🔍 Analyse exploratoire – données population (valeurs humaines)")

var_label = st.selectbox("Choisir une variable", list(HUMAN_VARS.keys()))
var_col, var_type = HUMAN_VARS[var_label]
data = pd.to_numeric(df[var_col], errors="coerce").dropna()

c1, c2 = st.columns([2, 1])

with c1:
    fig, ax = plt.subplots(figsize=(9, 4))

    # Hist propre (pas value_counts sur des continues)
    ax.hist(data, bins=30, edgecolor="black")

    ax.set_title(f"Distribution — {var_label}")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)

with c2:
    st.markdown("### 📌 Repères")
    # Stats lisibles
    if "€" in var_label:
        st.metric("Min", f"{data.min():,.0f} €".replace(",", " "))
        st.metric("Médiane", f"{data.median():,.0f} €".replace(",", " "))
        st.metric("Max", f"{data.max():,.0f} €".replace(",", " "))
    else:
        st.metric("Min", f"{data.min():.1f}")
        st.metric("Médiane", f"{data.median():.1f}")
        st.metric("Max", f"{data.max():.1f}")

    st.markdown("### ℹ️ Interprétation")
    if var_type == "monétaire":
        st.info("Variable monétaire réelle. Distribution asymétrique typique.")
    elif var_type == "démographique":
        st.info("Âge en années (valeur réelle), directement interprétable.")
    else:
        st.info("Ancienneté en années (valeur réelle), directement interprétable.")

# ======================================================
# SÉLECTION D’UN INDIVIDU (inchangé)
# ======================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider(
    "Choisir un individu",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

# Affichage humain pour l'individu
st.markdown("### Profil client (valeurs humaines)")
human_cols = [col for (col, _) in HUMAN_VARS.values()]
st.dataframe(df.loc[[row_id], human_cols])

# ======================================================
# PRÉDICTION (inchangée, mais on retire colonnes humaines)
# ======================================================
st.subheader("📈 Résultat de la prédiction")

X_model = df.drop(columns=human_cols, errors="ignore")

# sécurisation numeric
X_model = X_model.apply(pd.to_numeric, errors="coerce").fillna(0)

input_df = X_model.iloc[[row_id]]

proba = float(model.predict_proba(input_df)[0][1])
prediction = int(proba >= 0.5)

st.markdown(
    """
    **Interprétation métier :**

    - **Classe 0** : pas de risque de défaut
    - **Classe 1** : risque de défaut
    """
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Risque de défaut estimé", "Oui" if prediction == 1 else "Non")

with col2:
    st.metric("Probabilité de défaut", f"{proba:.2%}")

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    Ce dashboard illustre l’utilisation d’un modèle **LightGBM** pour le scoring.

    ✔️ Les graphiques affichent des valeurs **humaines** (années / euros)  
    ✔️ La prédiction utilise les **features prétraitées** attendues par le modèle  
    ✔️ **Un seul CSV** est utilisé de bout en bout
    """
)
