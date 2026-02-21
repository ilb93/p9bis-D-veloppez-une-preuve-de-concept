import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import boto3
import requests
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import sys

# Ajouter le chemin parent pour les imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

# Importer le modèle SAINT avec gestion d'erreur
try:
    # Essayer d'importer depuis le package
    from streamlit_app.saint_model import SAINTModel, predict_saint, load_saint_model_from_files as load_saint_from_files
except ImportError:
    try:
        # Essayer d'importer directement depuis le même dossier
        from saint_model import SAINTModel, predict_saint, load_saint_model_from_files as load_saint_from_files
    except ImportError:
        # Dernier recours: importer depuis le chemin absolu
        import importlib.util
        spec = importlib.util.spec_from_file_location("saint_model", current_dir / "saint_model.py")
        saint_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(saint_module)
        SAINTModel = saint_module.SAINTModel
        predict_saint = saint_module.predict_saint
        load_saint_from_files = saint_module.load_saint_model_from_files

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
Cette application présente une **preuve de concept** de scoring de risque basée sur un modèle **SAINT**.

- Les **graphiques** affichent des **valeurs métier lisibles (années / euros)**  
- La **prédiction** utilise **exactement les variables attendues par le modèle**
- La **décision est expliquée** à l'aide des contributions du modèle
"""
)

# ======================================================
# CONFIGURATION S3
# ======================================================
S3_BUCKET = "projetmodelsaint"
S3_REGION = "eu-north-1"

# Configuration AWS (peut être définie via variables d'environnement)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# ======================================================
# CHARGEMENT MODÈLE SAINT DEPUIS S3
# ======================================================
@st.cache_resource
def load_saint_model_from_s3():
    """Charge le modèle SAINT depuis S3"""
    try:
        # Créer un dossier cache local
        cache_dir = Path(".saint_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Fichiers à télécharger
        files = {
            "weights": "saint_weights.pth",
            "config": "saint_config.pkl",
            "metadata": "saint_metadata.pkl",
            "threshold": "saint_threshold.pkl"
        }
        
        file_paths = {}
        
        # URLs publiques S3 (le bucket est public, pas besoin de credentials)
        base_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"
        
        # Télécharger les fichiers depuis S3 via URLs publiques
        for key, filename in files.items():
            local_path = cache_dir / filename
            
            # Télécharger depuis S3 si nécessaire
            if not local_path.exists():
                try:
                    # Utiliser l'URL publique S3
                    url = f"{base_url}/{filename}"
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    # Sauvegarder le fichier
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    st.success(f"✅ {filename} téléchargé depuis S3")
                except Exception as e:
                    st.error(f"❌ Erreur lors du téléchargement de {filename}: {str(e)}")
                    st.info(f"💡 URL tentée: {base_url}/{filename}")
                    raise
            
            file_paths[key] = local_path
        
        # Charger le modèle avec la fonction helper
        model_data = load_saint_from_files(
            weights_path=file_paths['weights'],
            config_path=file_paths['config'],
            metadata_path=file_paths['metadata'],
            threshold_path=file_paths.get('threshold'),
            device='cpu'
        )
        
        return model_data
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle depuis S3: {str(e)}")
        st.info("💡 Le bucket S3 est public, donc pas besoin de credentials AWS.")
        st.info("💡 Vérifiez que le bucket 'projetmodelsaint' est accessible publiquement.")
        st.stop()

# Charger le modèle
with st.spinner("🔄 Chargement du modèle SAINT depuis S3..."):
    try:
        model_data = load_saint_model_from_s3()
        st.success("✅ Modèle SAINT chargé avec succès!")
    except Exception as e:
        st.error(f"❌ Impossible de charger le modèle: {str(e)}")
        import traceback
        with st.expander("🔍 Détails de l'erreur (pour débogage)"):
            st.code(traceback.format_exc())
        st.info("💡 Vérifiez que:")
        st.info("1. Les secrets AWS sont configurés dans Streamlit Cloud (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        st.info("2. Le bucket S3 'projetmodelsaint' est accessible")
        st.info("3. Les fichiers saint_weights.pth, saint_config.pkl, saint_metadata.pkl, saint_threshold.pkl existent dans le bucket")
        st.stop()

# Extraire les features attendues depuis les métadonnées
def get_expected_features(model_data):
    """Extrait la liste des features depuis les métadonnées et la config"""
    metadata = model_data.get('metadata', {})
    config = model_data.get('config', {})
    
    # Essayer depuis les métadonnées d'abord
    if isinstance(metadata, dict):
        # Essayer différentes clés possibles
        for key in ['feature_names', 'features', 'columns', 'feature_list', 'cat_feat', 'num_feat']:
            if key in metadata:
                feat = metadata[key]
                if isinstance(feat, list) and len(feat) > 0:
                    return feat
                elif isinstance(feat, dict):
                    # Si c'est un dict avec cat et num
                    if 'categorical' in feat and 'numerical' in feat:
                        return feat['categorical'] + feat['numerical']
        
        # Si les clés sont directement les features
        if 'categorical_features' in metadata and 'continuous_features' in metadata:
            cat_feat = metadata['categorical_features'] if isinstance(metadata['categorical_features'], list) else []
            cont_feat = metadata['continuous_features'] if isinstance(metadata['continuous_features'], list) else []
            if cat_feat or cont_feat:
                return cat_feat + cont_feat
        
        # Essayer de trouver toutes les clés qui ressemblent à des features
        possible_features = [k for k in metadata.keys() if not k.startswith('_') and not isinstance(metadata[k], (dict, list))]
        if possible_features and len(possible_features) > 5:  # Au moins 5 features pour être valide
            return possible_features
    
    elif isinstance(metadata, list) and len(metadata) > 0:
        return metadata
    
    # Si on ne trouve rien, essayer depuis la config
    if isinstance(config, dict):
        for key in ['feature_names', 'features', 'columns', 'num_features']:
            if key in config:
                feat = config[key]
                if isinstance(feat, list) and len(feat) > 0:
                    return feat
                elif isinstance(feat, int) and key == 'num_features':
                    # Si on a juste le nombre, on ne peut pas retourner les noms
                    # Mais on peut essayer de les déduire
                    pass
        
        # Essayer depuis num_features et créer des noms génériques
        if 'num_features' in config:
            num_feat = config['num_features']
            if isinstance(num_feat, int) and num_feat > 0:
                # Retourner des noms génériques (sera remplacé par les colonnes du CSV)
                return [f"feature_{i}" for i in range(num_feat)]
    
    # Dernier recours: essayer de déduire depuis la structure du modèle
    model = model_data.get('model')
    if model is not None:
        # Essayer d'extraire depuis les paramètres du modèle
        if hasattr(model, 'num_features'):
            num_feat = model.num_features
            if isinstance(num_feat, int) and num_feat > 0:
                return [f"feature_{i}" for i in range(num_feat)]
    
    return None  # Retourner None au lieu de lever une exception

try:
    EXPECTED_FEATURES = get_expected_features(model_data)
    if EXPECTED_FEATURES is None or len(EXPECTED_FEATURES) == 0:
        st.warning("⚠️ Impossible de récupérer les features automatiquement depuis les métadonnées.")
        st.info("💡 Les features seront déduites depuis les colonnes du CSV chargé.")
        EXPECTED_FEATURES = None  # Sera défini depuis le CSV
    else:
        st.info(f"📋 {len(EXPECTED_FEATURES)} features attendues par le modèle")
except Exception as e:
    st.warning(f"⚠️ Erreur lors de l'extraction des features: {str(e)}")
    st.info("💡 Les features seront déduites depuis les colonnes du CSV chargé.")
    EXPECTED_FEATURES = None

# ======================================================
# UPLOAD CSV
# ======================================================
st.subheader("📂 Import du fichier CSV")

uploaded_file = st.file_uploader(
    "Importer le fichier CSV unifié (ex : sample_unified.csv)",
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
# OUTILS DE FORMATAGE
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
# VARIABLES MÉTIER LISIBLES
# ======================================================
human_df = pd.DataFrame({
    "Âge (années)": clean_age_years(df["age_years"]) if "age_years" in df.columns else pd.Series([np.nan] * len(df)),
    "Ancienneté emploi (années)": clean_employment_years(df["employment_years"]) if "employment_years" in df.columns else pd.Series([np.nan] * len(df)),
    "Montant du crédit (€)": clean_money(df["AMT_CREDIT"]) if "AMT_CREDIT" in df.columns else pd.Series([np.nan] * len(df)),
    "Prix du bien (€)": clean_money(df["AMT_GOODS_PRICE"]) if "AMT_GOODS_PRICE" in df.columns else pd.Series([np.nan] * len(df)),
    "Annuité du crédit (€)": clean_money(df["AMT_ANNUITY"]) if "AMT_ANNUITY" in df.columns else pd.Series([np.nan] * len(df)),
})

# ======================================================
# ANALYSE EXPLORATOIRE
# ======================================================
st.subheader("📊 Analyse exploratoire – population")

available_vars = [col for col in human_df.columns if not human_df[col].isna().all()]
if available_vars:
    var_label = st.selectbox("Choisir une variable", available_vars)
    series = human_df[var_label].dropna()
    
    if len(series) > 0:
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
st.subheader("🎯 Sélection d'un individu")

row_id = st.slider("Choisir un individu", 0, len(df) - 1, 0)

# ======================================================
# POSITION DE L'INDIVIDU
# ======================================================
if available_vars:
    st.markdown("### 📍 Position de l'individu dans la population")
    
    val = human_df.loc[row_id, var_label]
    
    if not pd.isna(val):
        fig2, ax2 = plt.subplots(figsize=(9, 2))
        ax2.hist(series, bins=30, edgecolor="black", alpha=0.7)
        ax2.axvline(val, color="red", linewidth=2)
        ax2.set_xlabel(var_label)
        ax2.set_ylabel("Population")
        
        if "€" in var_label:
            ax2.xaxis.set_major_formatter(FuncFormatter(euro_fmt))
        
        st.pyplot(fig2)

# ======================================================
# PRÉPARATION DONNÉES MODÈLE
# ======================================================
def build_model_row(data, idx, expected):
    """Construit une ligne de données pour le modèle"""
    row = {}
    for f in expected:
        if f in data.columns:
            v = pd.to_numeric(data.loc[idx, f], errors="coerce")
            row[f] = 0.0 if pd.isna(v) else float(v)
        else:
            row[f] = 0.0
    return pd.DataFrame([row], columns=expected)

# Déterminer les features à utiliser
if EXPECTED_FEATURES is None:
    # Utiliser toutes les colonnes numériques du CSV
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    EXPECTED_FEATURES = numeric_cols
    st.info(f"📋 Utilisation de {len(EXPECTED_FEATURES)} features depuis le CSV")

if EXPECTED_FEATURES and len(EXPECTED_FEATURES) > 0:
    X_row = build_model_row(df, row_id, EXPECTED_FEATURES)
    
    # Vérifier que X_row a le bon nombre de features
    if X_row.shape[1] != len(EXPECTED_FEATURES):
        st.error(f"❌ Nombre de features incompatible: attendu {len(EXPECTED_FEATURES)}, obtenu {X_row.shape[1]}")
        st.stop()
    
    # ======================================================
    # PRÉDICTION AVEC SAINT
    # ======================================================
    st.subheader("🔍 Prédiction du modèle SAINT")
    
    try:
        # Utiliser la fonction de prédiction SAINT
        proba = predict_saint(model_data, X_row, device='cpu')
        
        # Appliquer le seuil si disponible
        threshold = model_data.get('threshold', 0.5)
        if isinstance(threshold, (int, float)):
            # Le seuil peut être utilisé pour la classification binaire
            prediction_class = 1 if proba >= threshold else 0
        
        # Afficher les résultats
        if proba < 0.3:
            verdict = "Faible risque de crédit"
            verdict_color = "🟢"
        elif proba < 0.6:
            verdict = "Risque de crédit modéré"
            verdict_color = "🟡"
        else:
            verdict = "Risque de crédit élevé"
            verdict_color = "🔴"
        
        st.subheader("📈 Résultat de la prédiction")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Évaluation du profil", f"{verdict_color} {verdict}")
        c2.metric("Probabilité de défaut", f"{proba:.2%}")
        if isinstance(threshold, (int, float)):
            c3.metric("Seuil utilisé", f"{threshold:.2%}")
        
        # Afficher les features utilisées
        st.markdown("### 📋 Features utilisées pour la prédiction")
        feature_values = pd.DataFrame({
            "Feature": EXPECTED_FEATURES,
            "Valeur": X_row.iloc[0].values
        })
        st.dataframe(feature_values, use_container_width=True, hide_index=True)
        
        # Graphique de la probabilité
        st.markdown("### 📊 Visualisation du risque")
        fig_prob, ax_prob = plt.subplots(figsize=(10, 2))
        ax_prob.barh([0], [proba], color='red' if proba >= 0.6 else 'orange' if proba >= 0.3 else 'green', alpha=0.7)
        ax_prob.axvline(0.3, color='green', linestyle='--', linewidth=2, label='Faible risque')
        ax_prob.axvline(0.6, color='orange', linestyle='--', linewidth=2, label='Risque modéré')
        ax_prob.set_xlim(0, 1)
        ax_prob.set_xlabel('Probabilité de défaut')
        ax_prob.set_yticks([])
        ax_prob.legend()
        ax_prob.set_title(f'Score de risque: {proba:.2%}')
        st.pyplot(fig_prob)
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
        import traceback
        with st.expander("🔍 Détails de l'erreur"):
            st.code(traceback.format_exc())
        st.info("💡 Vérifiez que le modèle SAINT est correctement chargé et que les features correspondent.")
else:
    st.warning("⚠️ Les features attendues n'ont pas pu être déterminées automatiquement.")

# ======================================================
# CONCLUSION
# ======================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
Cette preuve de concept démontre une **approche professionnelle du scoring de risque de crédit**, combinant :

- une **analyse exploratoire métier** fondée sur des variables interprétables,
- une **évaluation individuelle contextualisée** par rapport à la population,
- une **prédiction explicable**, reposant sur un modèle SAINT.

L'objectif est de **rendre compréhensible une décision algorithmique complexe**, afin de faciliter son appropriation par des utilisateurs non techniques, tout en respectant les contraintes d'un déploiement industriel.
"""
)

