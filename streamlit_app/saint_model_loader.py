"""
Module pour charger le modèle SAINT depuis S3
"""
import pickle
import torch
import boto3
import os
from pathlib import Path
import streamlit as st


def download_from_s3(bucket_name, s3_key, local_path):
    """Télécharge un fichier depuis S3 vers le système local"""
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except Exception as e:
        st.error(f"Erreur lors du téléchargement de {s3_key}: {str(e)}")
        return False


@st.cache_resource
def load_saint_model_from_s3(bucket_name="projetmodelsaint", region_name="eu-north-1"):
    """
    Charge le modèle SAINT depuis S3
    
    Args:
        bucket_name: Nom du bucket S3
        region_name: Région AWS
    
    Returns:
        dict contenant le modèle, la config, les métadonnées et le seuil
    """
    # Créer un dossier temporaire pour les fichiers
    cache_dir = Path(".saint_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Fichiers à télécharger
    files = {
        "weights": "saint_weights.pth",
        "config": "saint_config.pkl",
        "metadata": "saint_metadata.pkl",
        "threshold": "saint_threshold.pkl"
    }
    
    model_data = {}
    
    # Télécharger les fichiers depuis S3
    for key, filename in files.items():
        local_path = cache_dir / filename
        
        # Télécharger si le fichier n'existe pas localement
        if not local_path.exists():
            if not download_from_s3(bucket_name, filename, str(local_path)):
                raise FileNotFoundError(f"Impossible de télécharger {filename} depuis S3")
        
        # Charger le fichier
        if key == "weights":
            # Charger les poids PyTorch
            model_data[key] = torch.load(local_path, map_location='cpu')
        else:
            # Charger les fichiers pickle
            with open(local_path, 'rb') as f:
                model_data[key] = pickle.load(f)
    
    return model_data


def get_saint_features(metadata):
    """Extrait la liste des features depuis les métadonnées"""
    if isinstance(metadata, dict):
        # Essayer différentes clés possibles
        for key in ['feature_names', 'features', 'columns', 'feature_list']:
            if key in metadata:
                return metadata[key]
        # Si c'est un dict avec des infos sur les features
        if 'categorical_features' in metadata and 'continuous_features' in metadata:
            return metadata['categorical_features'] + metadata['continuous_features']
    elif isinstance(metadata, list):
        return metadata
    
    # Fallback: essayer d'extraire depuis la config
    raise ValueError("Impossible de déterminer les features depuis les métadonnées")


def predict_saint(model_data, X):
    """
    Effectue une prédiction avec le modèle SAINT
    
    Args:
        model_data: Dict contenant le modèle chargé
        X: DataFrame ou array avec les features
    
    Returns:
        proba: Probabilité de défaut
    """
    weights = model_data['weights']
    config = model_data['config']
    threshold = model_data.get('threshold', 0.5)
    
    # Convertir X en tensor si nécessaire
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    
    # Charger l'architecture du modèle depuis la config
    # Note: Cette partie dépend de la structure exacte de votre modèle SAINT
    # Vous devrez peut-être adapter cette partie selon votre implémentation
    
    # Si les weights contiennent déjà le modèle complet
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        # Le modèle doit être instancié depuis la config
        # Pour l'instant, on suppose que la config contient les infos nécessaires
        pass
    
    # Pour une implémentation basique, on suppose que le modèle est déjà chargé
    # Vous devrez adapter cette partie selon votre architecture SAINT spécifique
    model = weights  # Simplification - à adapter selon votre structure
    
    # Mettre le modèle en mode évaluation
    if hasattr(model, 'eval'):
        model.eval()
    
    # Prédiction
    with torch.no_grad():
        if hasattr(model, 'forward'):
            output = model.forward(X)
        elif hasattr(model, '__call__'):
            output = model(X)
        else:
            raise ValueError("Format de modèle non reconnu")
    
    # Appliquer sigmoid si nécessaire pour obtenir une probabilité
    if output.dim() > 1:
        output = output.squeeze()
    
    # Convertir en probabilité
    if hasattr(torch.nn.functional, 'sigmoid'):
        proba = torch.sigmoid(output).item()
    else:
        proba = torch.special.expit(output).item()
    
    return proba

