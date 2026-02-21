# Guide d'adaptation pour le modèle SAINT

Ce document explique comment adapter le code pour votre implémentation spécifique du modèle SAINT.

## Structure des fichiers sur S3

Le modèle SAINT est stocké sur S3 dans le bucket `projetmodelsaint` avec les fichiers suivants:
- `saint_weights.pth` - Poids du modèle PyTorch
- `saint_config.pkl` - Configuration du modèle
- `saint_metadata.pkl` - Métadonnées (features, etc.)
- `saint_threshold.pkl` - Seuil de classification

## Configuration AWS

Pour que le dashboard puisse accéder à S3, vous devez configurer les credentials AWS:

### Sur Streamlit Cloud:
1. Allez dans les paramètres de votre app
2. Ajoutez les secrets suivants:
   - `AWS_ACCESS_KEY_ID`: Votre clé d'accès AWS
   - `AWS_SECRET_ACCESS_KEY`: Votre clé secrète AWS

### Localement:
Créez un fichier `.env` ou configurez les variables d'environnement:
```bash
export AWS_ACCESS_KEY_ID="votre_clé"
export AWS_SECRET_ACCESS_KEY="votre_secret"
```

## Adaptation du code de prédiction

Le fichier `streamlit_app/app.py` contient une section de prédiction qui doit être adaptée selon votre architecture SAINT.

### Si votre modèle SAINT utilise une classe personnalisée:

1. Importez votre classe SAINT dans `app.py`:
```python
from your_saint_module import SAINTModel
```

2. Modifiez la section de chargement du modèle:
```python
# Charger le modèle depuis les weights
weights = model_data['weights']
config = model_data.get('config', {})

# Instancier le modèle avec la config
model = SAINTModel(**config)

# Charger les poids
if isinstance(weights, dict) and 'model_state_dict' in weights:
    model.load_state_dict(weights['model_state_dict'])
else:
    model.load_state_dict(weights)

model.eval()
```

3. Modifiez la section de prédiction:
```python
with torch.no_grad():
    output = model(X_tensor)
    proba = torch.sigmoid(output).item()
```

### Si les métadonnées ont une structure spécifique:

Modifiez la fonction `get_expected_features()` dans `app.py` pour extraire correctement les noms de features depuis vos métadonnées.

## Vérification

Pour vérifier que tout fonctionne:

1. Testez localement:
```bash
streamlit run streamlit_app/app.py
```

2. Vérifiez que les fichiers sont bien téléchargés depuis S3
3. Testez une prédiction avec un fichier CSV de test

## Déploiement sur Streamlit Cloud

1. Poussez votre code sur GitHub
2. Connectez votre repo à Streamlit Cloud
3. Configurez les secrets AWS dans les paramètres
4. Le dashboard devrait se mettre à jour automatiquement

