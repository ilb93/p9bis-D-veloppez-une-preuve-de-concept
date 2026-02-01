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
    """
    Récupère la liste exacte des features attendues par LightGBM.
    """
    if hasattr(m, "booster_") and m.booster_ is not None:
        return list(m.booster_.feature_name())
    if hasattr(m, "feature_name_"):
        return list(m.feature_name_)
    # Fallback (rare)
    raise RuntimeError("Impossible de récupérer la liste des features attendues par le modèle.")

EXPECTED_FEATURES = get_expected_features(model)

# ======================================================
# UPLOAD CSV UNIQUE (comme à l'origine)
# ======================================================
st.subheader("📂 Import du fichier CSV (unique)")

uploaded_file = st.file_uploader(
    "Importer le fichier CSV unifié (ex: sample_unified.csv)",
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
# OUTILS : gestion colonnes dupliquées et sélection "humaine"
# ======================================================
def euro_fmt(x, pos=None):
    try:
        return f"{x:,.0f} €".replace(",", " ")
    except Exception:
        return str(x)

def pick_best_human_column(base_name: str, data: pd.DataFrame, money=False):
    """
    Ton CSV peut contenir AMT_CREDIT + AMT_CREDIT.1 etc.
    On choisit la version 'humaine' (= valeurs en euros, pas z-score).
    Heuristique :
      - on prend la colonne dont la médiane absolue est la plus grande
      - (les z-scores ont une médiane proche de 0)
    """
    candidates = [c for c in data.columns if c == base_name or c.startswith(base_name + ".")]
    if not candidates:
        return None

    best_col = None
    best_score = -np.inf

    for c in candidates:
        s = pd.to_numeric(data[c], errors="coerce")
        med = np.nanmedian(np.abs(s.values))
        # petit bonus si money attendu (souvent > 1000)
        score = med + (1000 if (money and med > 1000) else 0)
        if score > best_score:
            best_score = score
            best_col = c

    return best_col

def clean_employment_years(series: pd.Series) -> pd.Series:
    """
    Corrige le problème logique qui te détruit l'histogramme :
    - 1000 = valeur sentinelle (emploi inconnu / anomalie) -> NaN
    - valeurs négatives -> 0 (pas d'ancienneté)
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(~(s >= 900), np.nan)     # 1000 -> NaN
    s = s.where(s >= 0, 0)              # négatif -> 0
    return s

def clean_age_years(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # on protège des âges absurdes
    s = s.where((s >= 0) & (s <= 120), np.nan)
    return s

def clean_money(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s >= 0, np.nan)  # montants négatifs => NaN
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
        "Certaines colonnes 'humaines' ne sont pas trouvées : "
        + ", ".join(missing_human)
        + "\n\n➡️ Les graphes ne pourront pas être complets tant que le CSV ne contient pas ces colonnes."
    )

# dataframe humain pour graphes
human_df = pd.DataFrame()

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

st.write("Colonnes utilisées pour les graphes :")
st.code(
    "\n".join([
        f"age_years -> {age_col}",
        f"employment_years -> {emp_col}",
        f"AMT_CREDIT -> {credit_col}",
        f"AMT_GOODS_PRICE -> {goods_col}",
        f"AMT_ANNUITY -> {annuity_col}",
    ])
)

# ======================================================
# ANALYSE EXPLORATOIRE (GRAPHIQUES HUMAINS)
# ======================================================
st.subheader("📊 Analyse exploratoire – données population (valeurs humaines)")

if human_df.shape[1] == 0:
    st.stop()

var_label = st.selectbox("Choisir une variable", list(human_df.columns))
series = human_df[var_label].dropna()

col_plot, col_info = st.columns([2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(9, 4))

    # Histogramme
    ax.hist(series.values, bins=30, edgecolor="black")
    ax.set_title(f"Distribution — {var_label}")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Nombre d'individus")

    # Format € si besoin
    if "€" in var_label:
        ax.xaxis.set_major_formatter(FuncFormatter(euro_fmt))

    st.pyplot(fig)

with col_info:
    st.markdown("### 📌 Repères")
    if len(series) == 0:
        st.write("Aucune donnée exploitable.")
    else:
        vmin = float(np.nanmin(series.values))
        vmed = float(np.nanmedian(series.values))
        vmax = float(np.nanmax(series.values))

        if "€" in var_label:
            st.metric("Min", euro_fmt(vmin))
            st.metric("Médiane", euro_fmt(vmed))
            st.metric("Max", euro_fmt(vmax))
        else:
            st.metric("Min", f"{vmin:.1f}")
            st.metric("Médiane", f"{vmed:.1f}")
            st.metric("Max", f"{vmax:.1f}")

    st.markdown("### ℹ️ Interprétation")
    if "Âge" in var_label:
        st.info("Âge réel (années). Valeur directement interprétable.")
    elif "Ancienneté" in var_label:
        st.info(
            "Ancienneté réelle (années). "
            "⚠️ La valeur 1000 a été traitée comme 'inconnue' (sinon elle casse l'histogramme)."
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

# On construit X_row STRICTEMENT selon les features attendues par le modèle
def build_model_row(data: pd.DataFrame, idx: int, expected_features: list[str]) -> pd.DataFrame:
    """
    Reconstruit une ligne modèle avec EXACTEMENT :
    - les colonnes attendues
    - le bon ordre
    - les bons dtypes numériques
    - en gérant les colonnes suffixées (.1, .2...) si ton CSV a eu des collisions
    """
    # mapping feature -> colonne réelle existante
    colset = set(data.columns)

    mapping = {}
    for f in expected_features:
        if f in colset:
            mapping[f] = f
        else:
            # tente f.1 / f.2 / f.3 ...
            found = None
            for k in range(1, 6):
                cand = f"{f}.{k}"
                if cand in colset:
                    found = cand
                    break
            mapping[f] = found  # peut être None

    # Construire la ligne
    row_dict = {}
    for f in expected_features:
        src = mapping[f]
        if src is None:
            row_dict[f] = 0.0
        else:
            val = pd.to_numeric(data.loc[idx, src], errors="coerce")
            row_dict[f] = 0.0 if pd.isna(val) else float(val)

    X_row = pd.DataFrame([row_dict], columns=expected_features)
    return X_row

X_row = build_model_row(df, row_id, EXPECTED_FEATURES)

# affichage "humain" de l'individu (les 5 variables)
st.markdown("### 👤 Profil (variables humaines)")
human_profile = {}
for k in human_df.columns:
    v = human_df.loc[row_id, k] if row_id in human_df.index else np.nan
    human_profile[k] = v
human_profile_df = pd.DataFrame([human_profile])

# format plus lisible dans le tableau
for c in human_profile_df.columns:
    if "€" in c:
        human_profile_df[c] = human_profile_df[c].apply(lambda x: euro_fmt(x) if pd.notna(x) else "")
    else:
        human_profile_df[c] = human_profile_df[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")

st.dataframe(human_profile_df, use_container_width=True)

# prédiction
st.subheader("📈 Résultat de la prédiction")

try:
    proba = float(model.predict_proba(X_row)[0][1])
    prediction = int(proba >= 0.5)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Classe prédite", "Risque de défaut" if prediction == 1 else "Pas de défaut")
    with c2:
        st.metric("Probabilité de défaut", f"{proba:.2%}")

except Exception as e:
    st.error(
        "Erreur lors de la prédiction. "
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
- ✅ Graphiques basés sur des **valeurs métiers lisibles**
- ✅ Prédiction basée sur les **features attendues par LightGBM**
- ✅ 1 seul fichier CSV (comme au départ)

Si tu as encore une erreur LightGBM, c’est **forcément** un problème d’alignement des features :  
ce code reconstruit justement **les colonnes attendues** (même si le CSV a `AMT_CREDIT.1`, etc.).
"""
)
