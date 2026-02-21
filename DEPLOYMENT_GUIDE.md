# Guide de déploiement - Dashboard SAINT sur Streamlit Cloud

## 📋 Résumé des modifications

Votre projet a été adapté pour utiliser le modèle **SAINT** depuis S3 au lieu du modèle LightGBM local.

## 📁 Structure des fichiers créés/modifiés

### Fichiers à la racine:
- ✅ `requirements.txt` - Dépendances Python (PyTorch, boto3, streamlit, etc.)
- ✅ `runtime.txt` - Version Python (3.10.12)
- ✅ `.gitignore` - Fichiers à ignorer dans Git
- ✅ `.streamlit/config.toml` - Configuration Streamlit

### Fichiers dans `streamlit_app/`:
- ✅ `app.py` - Application principale adaptée pour SAINT
- ✅ `saint_model_loader.py` - Module helper pour charger SAINT depuis S3

### Documentation:
- ✅ `SAINT_ADAPTATION.md` - Guide pour adapter le code à votre modèle SAINT spécifique
- ✅ `DEPLOYMENT_GUIDE.md` - Ce fichier

## 🚀 Étapes de déploiement

### 1. Préparer le dépôt GitHub

Assurez-vous que tous les fichiers sont commités:
```bash
git add .
git commit -m "Adaptation pour modèle SAINT depuis S3"
git push origin main
```

### 2. Configurer Streamlit Cloud

1. Allez sur [share.streamlit.io](https://share.streamlit.io)
2. Connectez votre dépôt GitHub
3. Dans les paramètres de l'app, configurez:
   - **Main file path**: `streamlit_app/app.py`
   - **Python version**: `3.10`

### 3. Configurer les secrets AWS

Dans les paramètres de votre app Streamlit Cloud, ajoutez les secrets suivants:

```
AWS_ACCESS_KEY_ID=votre_clé_d_accès
AWS_SECRET_ACCESS_KEY=votre_clé_secrète
```

**Important**: Le bucket S3 `projetmodelsaint` doit être accessible avec ces credentials.

### 4. Vérifier la politique S3

Assurez-vous que votre bucket S3 a la bonne politique (comme montré dans vos screenshots):
- Le bucket doit permettre `s3:GetObject` pour les fichiers du modèle
- Les fichiers doivent être accessibles publiquement OU via les credentials AWS

## 🔧 Configuration S3

D'après vos screenshots, votre bucket `projetmodelsaint` contient:
- `saint_weights.pth` (47.9 Mo)
- `saint_config.pkl` (211.0 o)
- `saint_metadata.pkl` (3.9 Ko)
- `saint_threshold.pkl` (112.0 o)

La politique S3 actuelle permet l'accès public en lecture (`PublicReadModel`), ce qui est parfait pour le déploiement.

## ⚠️ Points d'attention

### Adaptation nécessaire du code de prédiction

Le fichier `streamlit_app/app.py` contient une section de prédiction qui doit être adaptée selon votre architecture SAINT exacte. 

**Ligne ~280-300 dans `app.py`**: Vous devrez adapter cette partie selon:
- La structure de votre classe SAINT
- Comment les weights sont stockés (state_dict ou modèle complet)
- La signature de la méthode forward()

Consultez `SAINT_ADAPTATION.md` pour plus de détails.

### Extraction des features

Le code essaie automatiquement d'extraire les features depuis les métadonnées, mais si la structure de vos fichiers `.pkl` est différente, vous devrez adapter la fonction `get_expected_features()`.

## 🧪 Test local

Avant de déployer, testez localement:

```bash
# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement AWS
export AWS_ACCESS_KEY_ID="votre_clé"
export AWS_SECRET_ACCESS_KEY="votre_secret"

# Lancer l'app
streamlit run streamlit_app/app.py
```

## 📝 Checklist avant déploiement

- [ ] Tous les fichiers sont commités sur GitHub
- [ ] Les secrets AWS sont configurés sur Streamlit Cloud
- [ ] Le code de prédiction SAINT est adapté à votre architecture
- [ ] Les features sont correctement extraites depuis les métadonnées
- [ ] Test local réussi
- [ ] Le bucket S3 est accessible avec les credentials fournis

## 🆘 Dépannage

### Erreur "Impossible de télécharger depuis S3"
- Vérifiez les credentials AWS dans les secrets Streamlit
- Vérifiez que le bucket existe et est dans la région `eu-north-1`
- Vérifiez les permissions IAM

### Erreur "Format de modèle non reconnu"
- Adaptez la section de chargement du modèle dans `app.py`
- Consultez `SAINT_ADAPTATION.md` pour les détails

### Erreur "Impossible de déterminer les features"
- Vérifiez la structure de `saint_metadata.pkl`
- Adaptez la fonction `get_expected_features()` si nécessaire

## 📞 Support

Si vous rencontrez des problèmes, vérifiez:
1. Les logs Streamlit Cloud
2. La structure de vos fichiers sur S3
3. La compatibilité de votre modèle SAINT avec le code

