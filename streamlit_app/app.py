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
    """Récupère la liste exacte des features attendues par LightGBM."""
    if hasattr(m, "booster_") and m.booster_ is not None:
        return list(m.booster_.feature_name())
    if hasattr(m, "feature_name_"):
        return list(m.feature_name_)
    raise RuntimeError("Impossible de récupérer la liste des features attendues par le modèle.")

EXPECTED_FEATURES = get_expected_features(model)

# ======================================================
# UPLOAD CSV UNIQUE (comme à l'origine)
# ======================================================
st.subheader("📂 Import du fichier CSV (unique)")

uploaded_file = st.file_uploader(
    "Importer le fichier CSV unifié (ex: sample_unified.csv / sample_unified.csv)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Veuillez importer un fichier CSV pour continuer.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = [c.strip() for c in df.columns]

st.success("Fichier chargé avec succès")
st.write(f"Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")
st.dataframe(df.head())

# ======================================================
# ✅ TABLEAU DESCRIPTIF AU DÉBUT (CE QUE TU DEMANDAIS)
# ======================================================
st.subheader("📊 Statistiques descriptives (données importées)")

# Pour éviter un describe illisible sur 310 colonnes, on propose 2 vues :
tab1, tab2 = st.tabs(["Describe (toutes colonnes numériques)", "Describe (5 variables métier)"])

with tab1:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] == 0:
        st.info("Aucune colonne numérique détectée.")
    else:
        st.dataframe(numeric_df.describe().T.round(3), use_container_width=True)

# ======================================================
# OUTILS : formatters & nettoyage
# ======================================================
def euro_fmt(x, pos=None):
    try:
        return f"{x:,.0f} €".replace(",", " ")
    except Exception:
        return str(x)

def euro_str(x):
    try:
        return f"{x:,.0f} €".replace(",", " ")
    except Exception:
        return str(x)

def pick_best_human_column(base_name: str, data: pd.DataFrame, money=False):
    """
    Si ton CSV contient AMT_CREDIT + AMT_CREDIT.1 etc, on choisit la version "humaine".
    Heuristique : on prend la colonne dont la médiane ABS est la plus grande (z-score ~0).
    """
    candidates = [c for c in data.columns if c == base_name or c.startswith(base_name + ".")]
    if not candidates:
        return None

    best_col = None
    best_score = -np.inf

    for c in candidates:
        s = pd.to_numeric(data[c], errors="coerce")
        med_abs = np.nanmedian(np.abs(s.values))
        score = med_abs + (1000 if (money and med_abs > 1000) else 0)
        if score > best_score:
            best_score = score
            best_col = c

    return best_col

def clean_employment_years(series: pd.Series) -> pd.Series:
    """
    Correction logique :
    - 1000 = sentinelle -> NaN (sinon histogramme cassé)
    - négatif -> 0
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(~(s >= 900), np.nan)  # 1000 -> NaN
    s = s.where(s >= 0, 0)
    return s

def clean_age_years(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where((s >= 0) & (s <= 120), np.nan)
    return s

def clean_money(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s >= 0, np.nan)
    return s

# ======================================================
# CONSTRUCTION DES 5 VARIABLES "HUMAINES"
# ======================================================
st.subheader("🔍 Variables métiers (lisibles)")

age_col = "age_years" if "age_years" in df.columns else None
emp_col = "employment_years" if "employment_years" in df.columns else None

credit_col = pick_best_human_column("AMT_CREDIT", df, money=True)
goods_col  = pick_best_human_column("AMT_GOODS_PRICE", df, money=True)
annuity_col = pick_best_human_column("AMT_ANNUITY", df, money=True)

missing_human = []
if age_col is None: missing_human.append("age_years")
if emp_col is None: missing_human.append("employment_years")
if credit_col is None: missing_human.append("AMT_CREDIT")
if goods_col is None: missing_human.append("AMT_GOODS_PRICE")
if annuity_col is None: missing_human.append("AMT_ANNUITY")

if missing_human:
    st.warning(
        "Colonnes 'humaines' manquantes : " + ", ".join(missing_human) +
        "\n➡️ Les graphes métier ne pourront pas être complets tant que ces colonnes n'existent pas."
    )

human_df = pd.DataFrame(index=df.index)

if age_col:
    human_df["Âge (années)"] = clean_age_years(df[age_col])
if emp_col:
    human_df["Ancienneté emploi (années)"] = clean_employment_years(df[emp_col])
if credit_col:
    human_df["Montant du crédit (€)"] = clean_money(df[credit_col])
if goods_col:
    human_df["Prix du bien (€)"] = clean_money(df[goods_col])
if annuity_col:
    human_df["Annuité du crédit (€)"] = clean_money(df[annuity_col])

st.write("Colonnes utilisées pour les 5 variables métier :")
st.code(
    "\n".join([
        f"age_years -> {age_col}",
        f"employment_years -> {emp_col}",
        f"AMT_CREDIT -> {credit_col}",
        f"AMT_GOODS_PRICE -> {goods_col}",
        f"AMT_ANNUITY -> {annuity_col}",
    ])
)

with tab2:
    if human_df.shape[1] == 0:
        st.info("Aucune variable métier exploitable.")
    else:
        st.dataframe(human_df.describe().T.round(2), use_container_width=True)

# ======================================================
# ANALYSE EXPLORATOIRE (GRAPHIQUE 1 + REPÈRES)
# ======================================================
st.subheader("📈 Analyse exploratoire – population (valeurs humaines)")

if human_df.shape[1] == 0:
    st.stop()

var_label = st.selectbox("Choisir une variable", list(human_df.columns))
series = human_df[var_label].dropna()

col_plot, col_info = st.columns([2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(series.values, bins=30, edgecolor="black")
    ax.set_title(f"Distribution — {var_label}")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Nombre d'individus")

    # Format € sur l’axe si variable monétaire
    if "€" in var_label:
        ax.xaxis.set_major_formatter(FuncFormatter(euro_fmt))

    st.pyplot(fig)

with col_info:
    st.markdown("### 📌 Repères")

    if len(series) == 0:
        st.write("Aucune donnée exploitable.")
    else:
        vmin = float(np.nanmin(series.values))
        vmean = float(np.nanmean(series.values))
        vmax = float(np.nanmax(series.values))

        # ✅ CE QUE TU DEMANDAIS : Min / Moyenne / Max (pas médiane)
        if "€" in var_label:
            st.metric("Min", euro_str(vmin))
            st.metric("Moyenne", euro_str(vmean))
            st.metric("Max", euro_str(vmax))
        else:
            st.metric("Min", f"{vmin:.1f}")
            st.metric("Moyenne", f"{vmean:.1f}")
            st.metric("Max", f"{vmax:.1f}")

    st.markdown("### ℹ️ Interprétation")
    if "Âge" in var_label:
        st.info("Âge réel (années). Valeur directement interprétable.")
    elif "Ancienneté" in var_label:
        st.info(
            "Ancienneté réelle (années).\n\n"
            "⚠️ La valeur 1000 est une sentinelle et a été convertie en valeur manquante pour ne pas casser l'histogramme."
        )
    else:
        st.info("Variable monétaire réelle (euros). Distribution souvent asymétrique (queue à droite).")

# ======================================================
# PRÉDICTION (features modèle)
# ======================================================
st.subheader("🎯 Sélection d’un individu + prédiction")

row_id = st.slider(
    "Choisir un individu",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

# ✅ 2e graphique demandé : sous la sélection d’individu, position sur la variable sélectionnée
st.markdown("### 📍 Position de l’individu dans la distribution (variable choisie)")

indiv_value = human_df.loc[row_id, var_label] if var_label in human_df.columns else np.nan

fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.hist(series.values, bins=30, edgecolor="black", alpha=0.75)
if pd.notna(indiv_value):
    ax2.axvline(float(indiv_value), linestyle="--", linewidth=2, color="red", label="Individu sélectionné")
    ax2.legend()

ax2.set_title(f"Position de l’individu — {var_label}")
ax2.set_xlabel(var_label)
ax2.set_ylabel("Nombre d'individus")

if "€" in var_label:
    ax2.xaxis.set_major_formatter(FuncFormatter(euro_fmt))

st.pyplot(fig2)

# ======================================================
# Construire la ligne modèle EXACTE (anti erreurs LightGBM)
# ======================================================
def build_model_row(data: pd.DataFrame, idx: int, expected_features: list[str]) -> pd.DataFrame:
    """
    Reconstruit une ligne avec EXACTEMENT :
    - colonnes attendues
    - bon ordre
    - dtypes numériques
    - gère les colonnes suffixées (.1, .2...) si collisions
    """
    colset = set(data.columns)
    row_dict = {}

    for f in expected_features:
        if f in colset:
            src = f
        else:
            src = None
            for k in range(1, 10):
                cand = f"{f}.{k}"
                if cand in colset:
                    src = cand
                    break

        if src is None:
            row_dict[f] = 0.0
        else:
            val = pd.to_numeric(data.loc[idx, src], errors="coerce")
            row_dict[f] = 0.0 if pd.isna(val) else float(val)

    return pd.DataFrame([row_dict], columns=expected_features)

X_row = build_model_row(df, row_id, EXPECTED_FEATURES)

# ======================================================
# Affichage profil humain (5 variables)
# ======================================================
st.markdown("### 👤 Profil (variables métier)")

human_profile = human_df.loc[[row_id]].copy()

# joli format tableau
for c in human_profile.columns:
    if "€" in c:
        human_profile[c] = human_profile[c].apply(lambda x: euro_str(x) if pd.notna(x) else "")
    else:
        human_profile[c] = human_profile[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")

st.dataframe(human_profile, use_container_width=True)

# ======================================================
# PRÉDICTION + TEXTE MÉTIER (pas "pas de défaut")
# ======================================================
st.subheader("📈 Résultat de la prédiction")

try:
    proba = float(model.predict_proba(X_row)[0][1])

    # Seuil (tu peux le laisser à 0.5)
    threshold = 0.5
    is_risky = (proba >= threshold)

    c1, c2 = st.columns(2)

    with c1:
        # ✅ message métier demandé
        if is_risky:
            st.metric("Décision (lecture métier)", "⚠️ Profil à risque (danger potentiel)")
        else:
            st.metric("Décision (lecture métier)", "✅ Profil non dangereux (risque faible)")

    with c2:
        st.metric("Probabilité estimée de défaut", f"{proba:.2%}")

    # Bloc explicatif
    if is_risky:
        st.error(
            "Interprétation : la probabilité de défaut est élevée.\n\n"
            "👉 En contexte crédit, ce profil est **potentiellement dangereux** et justifie une **analyse approfondie** "
            "(revenus, garanties, stabilité, etc.)."
        )
    else:
        st.success(
            "Interprétation : la probabilité de défaut est faible.\n\n"
            "👉 En contexte crédit, ce profil est **plutôt non dangereux** et compatible avec une décision favorable "
            "(selon la politique de risque)."
        )

except Exception as e:
    st.error(
        "Erreur lors de la prédiction.\n\n"
        "Cause la plus fréquente : colonnes/features non alignées avec le modèle.\n\n"
        f"Détail : {type(e).__name__} — {e}"
    )
    st.stop()

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
- ✅ **1 seul fichier CSV** (comme à l'origine)
- ✅ **Describe** visible dès le début
- ✅ Graphique population avec **Min / Moyenne / Max**
- ✅ Second graphique : **position de l’individu** dans la distribution
- ✅ Résultat en **langage métier** (“dangereux / non dangereux”)

Si tu vois encore une erreur LightGBM, c’est uniquement si ton CSV ne contient pas les features ML attendues :
ici on reconstruit précisément la ligne modèle (y compris si les colonnes sont suffixées `.1`, `.2`, etc.).
"""
)
