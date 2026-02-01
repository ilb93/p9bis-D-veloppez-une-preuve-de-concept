import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ======================================================
# CONFIG STREAMLIT
# ======================================================
st.set_page_config(page_title="Proof of Concept – Credit Risk Scoring", layout="wide")
st.title("📊 Proof of Concept – Credit Risk Scoring")

st.markdown(
    """
Cette application présente une **preuve de concept** de scoring de risque de défaut.

- ✅ La **prédiction** utilise le CSV **prétraité** (comme en production).
- ✅ L’**analyse exploratoire (EDA)** tente d’afficher des valeurs **humaines** (€, années) :
  - si un **préprocesseur inverse** est disponible (scaler/pipeline sauvegardé), on reconvertit ;
  - sinon, on ne “fabrique” pas des unités : on l’indique clairement.
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
# CHARGEMENT (OPTIONNEL) D'UN PRÉPROCESSEUR POUR INVERSE_TRANSFORM
# ======================================================
@st.cache_resource
def load_optional_preprocessor():
    """
    On tente plusieurs noms usuels.
    L'objectif : disposer d'un objet avec .inverse_transform(X) et idéalement feature_names_in_.
    """
    artifacts = Path("artifacts")
    candidates = [
        artifacts / "preprocessor.joblib",
        artifacts / "pipeline.joblib",
        artifacts / "scaler.joblib",
        artifacts / "transformer.joblib",
    ]
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass
    return None

preprocessor = load_optional_preprocessor()

# ======================================================
# IMPORT CSV (PRÉTRAITÉ)
# ======================================================
st.subheader("📂 Import du CSV d’inférence (prétraité)")

uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
if uploaded_file is None:
    st.info("Veuillez importer un fichier CSV pour continuer.")
    st.stop()

df_model = pd.read_csv(uploaded_file).apply(pd.to_numeric, errors="coerce")

st.success("Fichier chargé")
st.write(f"Lignes : {df_model.shape[0]} | Colonnes : {df_model.shape[1]}")
st.dataframe(df_model.head())

# ======================================================
# 5 VARIABLES MÉTIER (FIXES)
# ======================================================
FEATURES = {
    "AMT_CREDIT (€)": "AMT_CREDIT",
    "AMT_GOODS_PRICE (€)": "AMT_GOODS_PRICE",
    "AMT_ANNUITY (€)": "AMT_ANNUITY",
    "AGE (années)": "DAYS_BIRTH",
    "ANCIENNETÉ EMPLOI (années)": "DAYS_EMPLOYED",
}

# On garde seulement celles présentes
FEATURES = {k: v for k, v in FEATURES.items() if v in df_model.columns}
if len(FEATURES) == 0:
    st.error("Aucune des 5 variables métier attendues n'est présente dans ton CSV.")
    st.stop()

# ======================================================
# CONSTRUCTION D'UN DF "HUMAIN" POUR L'EDA
# ======================================================
def to_human_units(df_preprocessed: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    """
    Retourne (df_eda, ok_human, message)
    - Si inverse_transform possible : on reconvertit approximativement en unités originales.
    - Sinon : on renvoie les valeurs telles quelles + message explicite.
    """
    df_eda = df_preprocessed.copy()

    # 1) Cas : pas de préprocesseur => impossible de revenir en unités €
    if preprocessor is None:
        return df_eda, False, (
            "Préprocesseur inverse non trouvé dans artifacts/. "
            "Impossible de reconvertir les features en unités € / jours. "
            "Pour une EDA en unités humaines, ajoute un fichier artifacts/preprocessor.joblib "
            "(pipeline/scaler utilisé à l’entraînement) ou utilise un dataset brut pour l’EDA."
        )

    # 2) On tente un inverse_transform sur TOUTES les colonnes numériques
    #    (si ça marche, on récupère une matrice reconvertie)
    try:
        X = df_preprocessed.values
        X_inv = preprocessor.inverse_transform(X)

        df_inv = pd.DataFrame(X_inv, columns=df_preprocessed.columns, index=df_preprocessed.index)

        # Conversion jours -> années si les colonnes sont bien en "jours"
        # (Home Credit a DAYS_BIRTH négatif : -age_en_jours ; idem DAYS_EMPLOYED souvent négatif)
        if "DAYS_BIRTH" in df_inv.columns:
            # age en années positif
            df_inv["DAYS_BIRTH"] = (-df_inv["DAYS_BIRTH"]) / 365.25
        if "DAYS_EMPLOYED" in df_inv.columns:
            df_inv["DAYS_EMPLOYED"] = (-df_inv["DAYS_EMPLOYED"]) / 365.25

        return df_inv, True, "Reconvertion en unités humaines effectuée via inverse_transform()."

    except Exception:
        # Si inverse_transform échoue, on n'invente rien
        return df_eda, False, (
            "Préprocesseur trouvé mais inverse_transform() a échoué. "
            "Donc on ne peut pas revenir de façon fiable aux unités € / jours."
        )

df_eda, human_ok, human_msg = to_human_units(df_model)

st.subheader("🔍 Analyse exploratoire (EDA)")
if human_ok:
    st.success(human_msg)
else:
    st.warning(human_msg)

# ======================================================
# SÉLECTION D’UN INDIVIDU (PRÉDICTION INTACTE)
# ======================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider("Choisir un individu", 0, len(df_model) - 1, 0)
x_row_model = df_model.iloc[[row_id]]
x_row_eda = df_eda.iloc[[row_id]]

st.markdown("**Données de l’individu (pour lecture humaine si disponible)**")
st.dataframe(x_row_eda[list(FEATURES.values())], use_container_width=True)

# ======================================================
# 2 GRAPHIQUES DIFFÉRENTS + MENU DÉROULANT SUR LES 5 VARIABLES
# ======================================================

# ---------- Helper: winsorize ----------
def winsorize(s: pd.Series, p_low=0.01, p_high=0.99) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    lo, hi = s.quantile([p_low, p_high])
    return s.clip(lo, hi)

# ---------- Graphique 1: Histogramme population (unité humaine si possible) ----------
st.subheader("📊 Graphique 1 — Distribution de la population (Histogramme)")

var1_label = st.selectbox("Variable (graphique 1)", list(FEATURES.keys()), key="g1")
var1 = FEATURES[var1_label]

s1 = winsorize(df_eda[var1])

fig1, ax1 = plt.subplots()
ax1.hist(s1.values, bins=30)
ax1.set_title(f"Distribution — {var1_label}")
ax1.set_ylabel("Effectif")
ax1.set_xlabel(var1_label)

# repère l'individu
v_ind_1 = x_row_eda[var1].iloc[0]
if pd.notna(v_ind_1):
    ax1.axvline(v_ind_1, linewidth=2, label="Individu sélectionné")
    ax1.legend()

st.pyplot(fig1, clear_figure=True)

# ---------- Graphique 2: Boxplot + médiane/IQR (vraie lecture analyste) ----------
st.subheader("📈 Graphique 2 — Résumé statistique (Boxplot)")

var2_label = st.selectbox("Variable (graphique 2)", list(FEATURES.keys()), key="g2")
var2 = FEATURES[var2_label]

s2 = winsorize(df_eda[var2])

fig2, ax2 = plt.subplots()
ax2.boxplot(s2.values, vert=False, showfliers=False)
ax2.set_title(f"Boxplot (sans outliers extrêmes) — {var2_label}")
ax2.set_xlabel(var2_label)

v_ind_2 = x_row_eda[var2].iloc[0]
if pd.notna(v_ind_2):
    ax2.axvline(v_ind_2, linewidth=2, label="Individu sélectionné")
    ax2.legend()

st.pyplot(fig2, clear_figure=True)

# ======================================================
# PRÉDICTION (INCHANGÉE)
# ======================================================
st.subheader("📌 Prédiction du modèle")

proba = float(model.predict_proba(x_row_model)[0][1])
prediction = int(proba >= 0.5)

c1, c2 = st.columns(2)
with c1:
    st.metric("Classe prédite", prediction)
with c2:
    st.metric("Probabilité de défaut", f"{proba:.3f}")

st.markdown(
    """
**Interprétation :**
- Classe 0 : pas de risque de défaut
- Classe 1 : risque de défaut
"""
)

# ======================================================
# ACCESSIBILITÉ
# ======================================================
st.subheader("♿ Accessibilité (WCAG – essentiels)")

st.markdown(
    """
- Structure en sections claires (titres, sous-titres)
- Composants standards Streamlit (compatibles navigation clavier)
- Graphiques lisibles + légendes textuelles (l’information ne repose pas uniquement sur la couleur)
- Valeur de l’individu matérialisée par un repère + texte
"""
)

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
Ce dashboard combine :
- une **EDA** focalisée sur **5 variables métier**, avec **2 visualisations analytiques distinctes** ;
- une **sélection d’individu** et l’affichage du **résultat de prédiction** ;
- un affichage en **unités humaines** lorsque l’inverse du prétraitement est disponible.

👉 Si tu veux une EDA 100% métier (euros, années, etc.), la bonne pratique industrielle est :
- soit d’avoir un **préprocesseur sauvegardé** (inverse_transform),
- soit d’utiliser un **dataset brut dédié à l’EDA** et garder le prétraité pour l’inférence.
"""
)
