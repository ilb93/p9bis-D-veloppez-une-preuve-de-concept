"""
Implémentation du modèle SAINT pour données tabulaires
SAINT: Self-Attention and Intersample Attention Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SAINTModel(nn.Module):
    """
    Modèle SAINT simplifié pour scoring de crédit
    Cette classe peut être adaptée selon votre architecture exacte
    """
    def __init__(self, config=None):
        super(SAINTModel, self).__init__()
        
        # Configuration par défaut
        self.config = config or {}
        
        # Paramètres depuis la config
        self.num_features = self.config.get('num_features', 100)
        self.embedding_dim = self.config.get('embedding_dim', 128)
        self.num_heads = self.config.get('num_heads', 8)
        self.num_layers = self.config.get('num_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.num_categories = self.config.get('num_categories', 0)
        self.num_continuous = self.config.get('num_continuous', self.num_features)
        
        # Embeddings pour features continues
        self.continuous_embedding = nn.Linear(1, self.embedding_dim)
        
        # Embeddings pour features catégorielles (si présentes)
        if self.num_categories > 0:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(cat_size, self.embedding_dim) 
                for cat_size in self.config.get('category_sizes', [])
            ])
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_features, self.embedding_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embedding_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * self.num_features, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, 1)
        )
    
    def forward(self, x):
        """
        Forward pass du modèle
        
        Args:
            x: Tensor de shape (batch_size, num_features)
        
        Returns:
            logits: Tensor de shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Embed les features continues
        # x est de shape (batch_size, num_features)
        x = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        x_emb = self.continuous_embedding(x)  # (batch_size, num_features, embedding_dim)
        
        # Ajouter positional encoding
        x_emb = x_emb + self.pos_encoding
        
        # Passer par le transformer
        x_transformed = self.transformer(x_emb)  # (batch_size, num_features, embedding_dim)
        
        # Flatten pour la classification
        x_flat = x_transformed.reshape(batch_size, -1)  # (batch_size, num_features * embedding_dim)
        
        # Classification
        logits = self.classifier(x_flat)  # (batch_size, 1)
        
        return logits


def load_saint_model_from_files(weights_path, config_path, metadata_path, threshold_path=None, device='cpu'):
    """
    Charge un modèle SAINT depuis les fichiers locaux
    
    Args:
        weights_path: Chemin vers saint_weights.pth
        config_path: Chemin vers saint_config.pkl
        metadata_path: Chemin vers saint_metadata.pkl
        threshold_path: Chemin vers saint_threshold.pkl (optionnel)
        device: Device pour le modèle ('cpu' ou 'cuda')
    
    Returns:
        dict avec 'model', 'config', 'metadata', 'threshold'
    """
    import pickle
    
    # Charger la config
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Charger les métadonnées
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Charger le seuil si disponible
    threshold = 0.5
    if threshold_path:
        try:
            with open(threshold_path, 'rb') as f:
                threshold = pickle.load(f)
        except:
            pass
    
    # Créer le modèle
    model = SAINTModel(config)
    
    # Charger les weights
    # PyTorch 2.6+ a weights_only=True par défaut, mais notre modèle contient des objets LightGBM
    # On doit utiliser weights_only=False pour permettre le chargement
    try:
        # Essayer d'abord avec weights_only=False (nécessaire pour PyTorch 2.6+)
        weights = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        # Pour les versions antérieures de PyTorch qui n'ont pas weights_only
        weights = torch.load(weights_path, map_location=device)
    
    # Vérifier si c'est un modèle complet (nn.Module) ou un state_dict
    if isinstance(weights, torch.nn.Module):
        # Si c'est un modèle complet, l'utiliser directement
        model = weights
    elif isinstance(weights, dict):
        # Si c'est un dictionnaire, extraire le state_dict
        if 'model_state_dict' in weights:
            state_dict = weights['model_state_dict']
        elif 'state_dict' in weights:
            state_dict = weights['state_dict']
        else:
            # Essayer de charger directement comme state_dict
            state_dict = weights
        
        # Charger le state_dict dans le modèle
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            # Si le chargement strict échoue, essayer sans strict
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as e2:
                raise RuntimeError(f"Impossible de charger le state_dict dans le modèle: {str(e2)}")
    else:
        raise ValueError(f"Format de poids non reconnu: {type(weights)}")
    
    # Déplacer le modèle sur le device et le mettre en mode évaluation
    model.to(device)
    model.eval()
    
    return {
        'model': model,
        'config': config,
        'metadata': metadata,
        'threshold': threshold
    }


def predict_saint(model_data, X, device='cpu'):
    """
    Effectue une prédiction avec le modèle SAINT
    
    Args:
        model_data: Dict contenant le modèle chargé
        X: DataFrame ou array avec les features
        device: Device pour le calcul
    
    Returns:
        proba: Probabilité de défaut (float)
    """
    import pandas as pd
    
    model = model_data['model']
    
    # Convertir en tensor si nécessaire
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    
    # S'assurer que X est 2D (batch_size, features)
    if X.dim() == 1:
        X = X.unsqueeze(0)
    
    X = X.to(device)
    
    # Prédiction
    with torch.no_grad():
        try:
            # Vérifier que X a la bonne forme
            if X.shape[1] != model.num_features:
                raise ValueError(f"Nombre de features incompatible: attendu {model.num_features}, obtenu {X.shape[1]}")
            
            # Essayer d'appeler le modèle
            if hasattr(model, 'forward'):
                logits = model.forward(X)
            elif hasattr(model, '__call__'):
                logits = model(X)
            else:
                raise ValueError("Le modèle n'a pas de méthode forward ou __call__")
            
            # Gérer différents formats de sortie
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # Convertir en probabilité avec sigmoid
            if logits.dim() > 1:
                logits = logits.squeeze()
            
            # Si logits est un tensor avec plusieurs valeurs, prendre la première
            if isinstance(logits, torch.Tensor):
                if logits.numel() > 1:
                    logits = logits[0]
                logits_value = logits.item()
            else:
                logits_value = float(logits)
            
            # Appliquer sigmoid
            proba = 1.0 / (1.0 + np.exp(-logits_value))  # sigmoid manuel pour plus de contrôle
            
            # S'assurer que la probabilité est entre 0 et 1
            proba = max(0.0, min(1.0, proba))
            
        except Exception as e:
            import traceback
            raise RuntimeError(f"Erreur lors de la prédiction: {str(e)}\n{traceback.format_exc()}")
    
    return proba

