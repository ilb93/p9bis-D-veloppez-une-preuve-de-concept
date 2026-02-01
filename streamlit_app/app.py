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

    👉 Les visualisations présentées ci-dessous sont **volontairement adaptées à une lecture humaine**
    et **ne montrent pas les valeurs mathématiques internes utilisées par le modèle**.
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

# ======================================================
# SÉLECTION DES VARIABLES PERTINENTES POUR L’EDA
# ======================================================
eda_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=True) > 20
    and df[col].std(skipna=True) > 1e-6
]

# ======================================================
# ANALYSE EXPLORATOIRE — VERSION HUMAINE
# ======================================================
st.subheader("🔍 Analyse exploratoire (lecture humaine)")

st.markdown(
    """
    Les graphiques suivants positionnent les individus **relativement à la population**
    (faible / moyen / élevé), sans afficher de valeurs mathématiques non interprétables.
    """
)

# ======================================================
# GRAPHIQUE 1 — POSITIONNEMENT RELATIF
# ======================================================
st.markdown("### Positionnement relatif d’une variable")

rel_col = st.selectbox(
    "Choisir une variable",
    eda_cols,
    key="rel_col"
)

s = df[rel_col].dropna()

bins = [-np.inf, s.quantile(0.2), s.quantile(0.4),
        s.quantile(0.6), s.quantile(0.8), np.inf]

labels = [
    "Très faible",
    "Faible",
    "Moyen",
    "Élevé",
    "Très élevé"
]

categories = pd.cut(s, bins=bins, labels=labels)
cat_df = categories.value_counts(normalize=True).reindex(labels).fillna(0) * 100

st.bar_chart(cat_df)

st.caption(
    "Répartition de la population par niveau relatif. "
    "Les catégories sont basées sur les quantiles de la population."
)

# ======================================================
# GRAPHIQUE 2 — PROFIL SIMPLIFIÉ
# ======================================================
st.markdown("### Profil global de la variable")

profile = pd.Series({
    "En dessous de la moyenne": (s < s.median()).mean() * 100,
    "Autour de la moyenne": ((s >= s.quantile(0.4)) & (s <= s.quantile(0.6))).mean() * 100,
    "Au-dessus de la moyenne": (s > s.median()).mean() * 100
})

st.bar_chart(profile)

st.caption(
    "Ce graphique synthétise la position de la population par rapport à la moyenne, "
    "sans afficher de valeurs numériques brutes."
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

    - **Classe 0** : pas de risque de défaut
    - **Classe 1** : risque de défaut

    La probabilité indique le **niveau de risque estimé**.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Décision du modèle", prediction)

with col2:
    st.metric("Probabilité de défaut", f"{proba:.1%}")

# ======================================================
# ACCESSIBILITÉ
# ======================================================
st.subheader("♿ Accessibilité (WCAG – critères essentiels)")

st.markdown(
    """
    - Graphiques sans dépendance exclusive à la couleur  
    - Libellés textuels explicites  
    - Hiérarchie claire des sections  
    - Aucune information transmise uniquement par des valeurs numériques abstraites
    """
)

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    Ce dashboard présente une **preuve de concept orientée décision**, conçue pour être
    **compréhensible par des utilisateurs non techniques**.

    Les données utilisées par le modèle sont volontairement **traduites en catégories lisibles**
    pour l’analyse exploratoire, tandis que la prédiction repose sur un pipeline
    **mathématiquement rigoureux et industriel**.

    Cette approche garantit à la fois **performance du modèle** et **clarté métier**.
    """
)
