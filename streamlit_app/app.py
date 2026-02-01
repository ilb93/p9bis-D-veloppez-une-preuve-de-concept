import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
Cette application présente une **preuve de concept** de scoring de risque basée sur un modèle **LightGBM**.

- Les **graphiques** affichent des **valeurs métier lisibles (années / euros)**  
- La **prédiction** utilise **les features exactement attendues par le modèle**
"""
)

# ======================================================
# CHARGEMENT MODÈLE
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load(Path("artifacts") / "lgbm.joblib")

model = load_model()

def get_expected_features(m):
    if hasattr(m, "booster_") and m.booster_ is not None:
        return list(m.booster_.feature_name())
    if hasattr(m, "feature_name_"):
        return list(m.feature_name_)
    raise RuntimeError("Impossible de récupérer les features du modèle.")

EXPECTED_FEATURES = get_expected_features(model)

# ======================================================
# UPLOAD CSV UNIQUE
# ======================================================
st.subheader("📂 Import du fichier CSV")

uploaded_file = st.file_uploader(
    "Importer le fichier CSV unifié (ex: sample_unified.csv)",
    type=["csv"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = [c.strip() for c in df.columns]

st.success("Fichier chargé avec succès")
st.write(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")

st.markdown("### 📈 Statistiques descriptives")
st.dataframe(df.describe().T, use_container_width=True)

# ======================================================
# OUTILS
# ======================================================
def euro_fmt(x, pos=None):
    try:
        return f"{x:,.0f} €".replace(",", " ")
    except Exception:
        return ""

def clean_employment_years(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s < 900, np.nan)
    s = s.where(s >= 0, 0)
    return s

def clean_age_years(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s <= 120), np.nan)

def clean_money(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 0, np.nan)

# ======================================================
# VARIABLES HUMAINES
# ======================================================
human_df = pd.DataFrame({
    "Âge (années)": clean_age_years(df["age_years"]),
    "Ancienneté emploi (années)": clean_employment_years(df["employment_years"]),
    "Montant du crédit (€)": clean_money(df["AMT_CREDIT"]),
    "Prix du bien (€)": clean_money(df["AMT_GOODS_PRICE"]),
    "Annuité du crédit (€)": clean_money(df["AMT_ANNUITY"]),
})

# ======================================================
# ANALYSE EXPLORATOIRE
# ======================================================
st.subheader("📊 Analyse exploratoire – population")

var_label = st.selectbox("Choisir une variable", human_df.columns)
series = human_df[var_label].dropna()

col_plot, col_info = st.columns([2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(series, bins=30, edgecolor="black")
    ax.set_title(f"Distribution — {var_label}")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Nombre d'individus")

    if "€" in var_label:
        ax.xaxis.set_major_formatter(FuncFormatter(euro_fmt))

    st.pyplot(fig)

with col_info:
    st.markdown("### 📌 Repères")
    st.metric("Min", euro_fmt(series.min()) if "€" in var_label else f"{series.min():.1f}")
    st.metric("Médiane", euro_fmt(series.median()) if "€" in var_label else f"{series.median():.1f}")
    st.metric("Max", euro_fmt(series.max()) if "€" in var_label else f"{series.max():.1f}")

# ======================================================
# SÉLECTION INDIVIDU
# ======================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider("Choisir un individu", 0, len(df) - 1, 0)

# ======================================================
# GRAPHIQUE POSITION INDIVIDU (TAILLE RÉDUITE DE MOITIÉ)
# ======================================================
st.markdown("### 📍 Position de l’individu dans la population")

val = human_df.loc[row_id, var_label]

fig2, ax2 = plt.subplots(figsize=(9, 2))  # ⬅️ taille divisée par 2
ax2.hist(series, bins=30, edgecolor="black", alpha=0.7)
ax2.axvline(val, color="red", linewidth=2)
ax2.set_xlabel(var_label)
ax2.set_ylabel("Population")

if "€" in var_label:
    ax2.xaxis.set_major_formatter(FuncFormatter(euro_fmt))

st.pyplot(fig2)

# ======================================================
# PRÉDICTION
# ======================================================
def build_model_row(data, idx, expected):
    row = {}
    for f in expected:
        if f in data.columns:
            v = pd.to_numeric(data.loc[idx, f], errors="coerce")
            row[f] = 0.0 if pd.isna(v) else float(v)
        else:
            row[f] = 0.0
    return pd.DataFrame([row], columns=expected)

X_row = build_model_row(df, row_id, EXPECTED_FEATURES)

proba = float(model.predict_proba(X_row)[0][1])

if proba < 0.3:
    verdict = "Faible risque de crédit"
elif proba < 0.6:
    verdict = "Risque de crédit modéré"
else:
    verdict = "Risque de crédit élevé"

st.subheader("📈 Résultat de la prédiction")

c1, c2 = st.columns(2)
c1.metric("Évaluation du profil", verdict)
c2.metric("Probabilité de défaut", f"{proba:.2%}")

# ======================================================
# CONCLUSION (RÉÉCRITE)
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
Cette preuve de concept démontre une **approche professionnelle du scoring de risque de crédit**, articulée autour de :

- une **analyse exploratoire métier**, fondée sur des variables directement interprétables,
- une **évaluation individuelle**, positionnée par rapport à la population globale,
- une **prédiction algorithmique robuste**, reposant sur l’ensemble des variables du modèle LightGBM.

L’interface a été conçue pour **transformer des résultats statistiques complexes en information décisionnelle compréhensible**, à destination de profils non techniques.

### ♿ Accessibilité & inclusion

La réalisation des graphiques prend en compte les **besoins des personnes en situation de handicap**, en couvrant des critères essentiels des **recommandations WCAG**, notamment :
- lisibilité des axes et contrastes suffisants,
- limitation de la surcharge visuelle,
- hiérarchisation claire de l’information,
- réduction de la charge cognitive.

Ces choix garantissent une **accessibilité équitable à l’information**, indépendamment des capacités visuelles ou cognitives.

### 🎯 Cadre projet

Cette application s’inscrit dans le cadre d’un **projet de preuve de concept en data science**, visant à démontrer la capacité à :
- répondre à un besoin métier réel,
- respecter les contraintes techniques d’un modèle industriel,
- intégrer des considérations d’accessibilité et de responsabilité numérique.
"""
)
