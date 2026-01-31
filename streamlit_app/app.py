import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="DataSpace – Proof of Concept", layout="wide")

st.title("📊 Proof of Concept – Amélioration d’un modèle ML")
st.markdown(
    """
Cette application présente une **preuve de concept** comparant :
- **Modèle baseline : RidgeClassifier**
- **Modèle récent : LightGBM**

👉 Workflow : vous **importez un CSV**, vous choisissez **une ligne**, puis vous lancez une **prédiction** avec le modèle sélectionné.
"""
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


# ======================================================
# CHARGEMENT DES ARTEFACTS
# ======================================================
@st.cache_resource
def load_artifacts():
    # Sécurité : chemins absolus (évite les surprises sur Streamlit Cloud)
    std_scale_path = ARTIFACTS_DIR / "std_scale.joblib"
    imputer_path = ARTIFACTS_DIR / "imputer_median.joblib"
    ridge_path = ARTIFACTS_DIR / "best_ridge.joblib"
    lgbm_path = ARTIFACTS_DIR / "lgbm.joblib"
    metadata_path = ARTIFACTS_DIR / "metadata.json"

    missing = [p.name for p in [std_scale_path, imputer_path, ridge_path, lgbm_path, metadata_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Artefacts manquants dans /artifacts : {missing}. "
            f"Vérifie que ton repo contient bien le dossier artifacts au même niveau que streamlit_app/."
        )

    std_scale = joblib.load(std_scale_path)
    imputer = joblib.load(imputer_path)
    ridge_model = joblib.load(ridge_path)
    lgbm_model = joblib.load(lgbm_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    raw_cols = metadata["raw_feature_columns"]
    col_map = metadata["column_mapping_raw_to_lgbm"]

    return std_scale, imputer, ridge_model, lgbm_model, raw_cols, col_map, metadata


std_scale, imputer, ridge_model, lgbm_model, RAW_COLS, COL_MAP, metadata = load_artifacts()


# ======================================================
# UPLOAD CSV
# ======================================================
st.subheader("📁 Import du jeu de données (CSV)")

uploaded = st.file_uploader(
    "Importer un fichier CSV (doit contenir au minimum les colonnes attendues par le modèle)",
    type=["csv"],
)

if uploaded is None:
    st.info("Veuillez importer un fichier CSV pour continuer.")
    st.stop()

df = pd.read_csv(uploaded)

st.write("Aperçu du fichier importé :")
st.dataframe(df.head(10), use_container_width=True)


# ======================================================
# VALIDATION COLONNES
# ======================================================
missing_cols = [c for c in RAW_COLS if c not in df.columns]
if missing_cols:
    st.error(
        "❌ Colonnes manquantes dans ton CSV.\n\n"
        f"Il manque {len(missing_cols)} colonnes (extrait) : {missing_cols[:20]}\n\n"
        "➡️ Solution : exporte un CSV depuis ton notebook **avec exactement les mêmes features** que le modèle."
    )
    st.stop()


# ======================================================
# SELECTION LIGNE
# ======================================================
st.subheader("🎯 Sélection d’un individu")

row_id = st.slider(
    "Choisir une ligne du dataset",
    min_value=0,
    max_value=max(0, len(df) - 1),
    value=0,
)

input_df = df.loc[[row_id], RAW_COLS].copy()

st.write("Données utilisées pour la prédiction (ligne sélectionnée) :")
st.dataframe(input_df, use_container_width=True)


# ======================================================
# PREPROCESSING (scaler -> imputer) + mapping LightGBM
# ======================================================
def preprocess_for_models(df_row: pd.DataFrame):
    """
    df_row: DataFrame 1 ligne avec colonnes RAW_COLS

    Sorties:
    - X_ridge: np.ndarray shape (1, n_features) prêt pour Ridge
    - X_lgbm:  pd.DataFrame avec colonnes renommées pour LightGBM
    """
    # 1) coercition numérique (évite dtype object)
    X = df_row.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # 2) scaler (comme notebook)
    X_scaled = std_scale.transform(X)

    # 3) imputer (comme notebook) -> IMPORTANT pour Ridge (NaN interdit)
    X_imputed = imputer.transform(X_scaled)

    # Ridge : array
    X_ridge = X_imputed

    # LightGBM : DataFrame + mapping colonnes
    X_lgbm_df = pd.DataFrame(X_imputed, columns=RAW_COLS).rename(columns=COL_MAP)

    return X_ridge, X_lgbm_df


try:
    X_ridge, X_lgbm = preprocess_for_models(input_df)
except Exception as e:
    st.error(
        "Erreur pendant le preprocessing.\n\n"
        "Causes fréquentes : valeurs non numériques, colonnes mal typées, ou incompatibilité sklearn.\n\n"
        f"Détail : {type(e).__name__} — {e}"
    )
    st.stop()


# ======================================================
# CHOIX MODELE + PREDICTION
# ======================================================
st.subheader("⚙️ Choix du modèle")

model_choice = st.radio(
    "Sélectionner le modèle",
    ["Baseline – RidgeClassifier", "Nouveau modèle – LightGBM"],
    index=0,
)

if st.button("🔮 Lancer la prédiction"):
    try:
        if model_choice == "Baseline – RidgeClassifier":
            pred = int(ridge_model.predict(X_ridge)[0])
            score = float(ridge_model.decision_function(X_ridge)[0])
            score_label = "Score (decision_function)"
        else:
            pred = int(lgbm_model.predict(X_lgbm)[0])
            proba = lgbm_model.predict_proba(X_lgbm)[0][1]
            score = float(proba)
            score_label = "Probabilité classe 1"

        st.success("✅ Prédiction effectuée")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Classe prédite", pred)
        with c2:
            st.metric(score_label, round(score, 4))

    except Exception as e:
        st.error(
            "Erreur au moment de la prédiction.\n\n"
            "Si l’erreur mentionne scikit-learn / imputer / attributes : "
            "➡️ c’est quasi sûr que ton environnement Streamlit n’a pas les mêmes versions que ton notebook.\n\n"
            f"Détail : {type(e).__name__} — {e}"
        )


# ======================================================
# COMPARAISON (simple, clair)
# ======================================================
st.subheader("📊 Comparaison des modèles")

comparison_df = pd.DataFrame(
    {
        "Modèle": ["RidgeClassifier (baseline)", "LightGBM (récent)"],
        "Gestion des NaN": ["Non (doit être imputé)", "Oui (mais ici imputé aussi)"],
        "Relations non-linéaires": ["Non", "Oui"],
        "Performance": ["Référence", "Supérieure"],
    }
)

st.dataframe(comparison_df, use_container_width=True)

st.subheader("✅ Conclusion")
st.markdown(
    """
- **RidgeClassifier** sert de **baseline** simple et robuste.
- **LightGBM** (modèle récent) capture des relations **non-linéaires** et améliore les performances.
- Le dashboard démontre la preuve de concept via **import CSV → sélection ligne → choix modèle → prédiction**.
"""
)
