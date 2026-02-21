# Dashboard SAINT - Guide d'utilisation

## ✅ Ce qui a été fait

Votre dashboard a été **entièrement adapté** pour utiliser le modèle **SAINT** depuis S3 au lieu du modèle LightGBM local.

### Fichiers créés/modifiés

1. **`requirements.txt`** ✅
   - Toutes les dépendances nécessaires (PyTorch, boto3, streamlit, etc.)

2. **`runtime.txt`** ✅
   - Version Python 3.10.12

3. **`streamlit_app/app.py`** ✅
   - Dashboard principal adapté pour SAINT
   - Charge automatiquement le modèle depuis S3
   - Interface utilisateur complète

4. **`streamlit_app/saint_model.py`** ✅
   - Classe `SAINTModel` implémentée
   - Fonctions de chargement et prédiction
   - Gestion automatique des différents formats de modèle

5. **`.streamlit/config.toml`** ✅
   - Configuration Streamlit

6. **`.gitignore`** ✅
   - Exclusion des fichiers de cache

## 🚀 Utilisation

### Test local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement AWS (optionnel si vous avez des credentials par défaut)
export AWS_ACCESS_KEY_ID="votre_clé"
export AWS_SECRET_ACCESS_KEY="votre_secret"

# Lancer l'application
streamlit run streamlit_app/app.py
```

### Déploiement sur Streamlit Cloud

1. **Pousser sur GitHub**:
   ```bash
   git add .
   git commit -m "Adaptation pour modèle SAINT"
   git push origin main
   ```

2. **Configurer Streamlit Cloud**:
   - Allez sur [share.streamlit.io](https://share.streamlit.io)
   - Connectez votre dépôt
   - **Main file path**: `streamlit_app/app.py`
   - **Python version**: `3.10`

3. **Ajouter les secrets AWS**:
   - Dans les paramètres de l'app, ajoutez:
     - `AWS_ACCESS_KEY_ID`
     - `AWS_SECRET_ACCESS_KEY`

## 📋 Structure du modèle SAINT

Le code charge automatiquement depuis S3 (`projetmodelsaint`):
- `saint_weights.pth` - Poids du modèle PyTorch
- `saint_config.pkl` - Configuration du modèle
- `saint_metadata.pkl` - Métadonnées (features, etc.)
- `saint_threshold.pkl` - Seuil de classification

## 🔧 Adaptation du modèle

Si votre modèle SAINT a une architecture différente de celle implémentée dans `saint_model.py`, vous pouvez:

1. **Modifier la classe SAINTModel** dans `streamlit_app/saint_model.py`
2. **Ou créer votre propre classe** et l'importer dans `app.py`

La fonction `load_saint_model_from_files()` gère automatiquement:
- Les state_dict PyTorch
- Les modèles complets
- Différents formats de weights

## ⚠️ Points importants

### Extraction des features

Le code essaie automatiquement d'extraire les features depuis:
1. Les métadonnées (`saint_metadata.pkl`)
2. La configuration (`saint_config.pkl`)
3. La structure du modèle

Si les features ne sont pas trouvées, elles seront déduites depuis les colonnes numériques du CSV chargé.

### Gestion des erreurs

Le code inclut une gestion d'erreurs complète avec:
- Messages d'erreur clairs
- Détails techniques dans un expander
- Suggestions de solutions

## 📊 Fonctionnalités du dashboard

- ✅ Chargement automatique du modèle depuis S3
- ✅ Upload de fichier CSV
- ✅ Analyse exploratoire des données
- ✅ Visualisation de la position d'un individu
- ✅ Prédiction avec le modèle SAINT
- ✅ Affichage des résultats avec seuil
- ✅ Visualisation graphique du risque

## 🐛 Dépannage

### Erreur "Impossible de télécharger depuis S3"
- Vérifiez les credentials AWS dans les secrets Streamlit
- Vérifiez que le bucket `projetmodelsaint` existe et est accessible
- Vérifiez la région (`eu-north-1`)

### Erreur "Format de modèle non reconnu"
- Vérifiez la structure de `saint_weights.pth`
- Adaptez la classe `SAINTModel` si nécessaire
- Consultez `SAINT_ADAPTATION.md` pour plus de détails

### Erreur "Impossible de déterminer les features"
- Vérifiez la structure de `saint_metadata.pkl`
- Le code utilisera automatiquement les colonnes du CSV en fallback

## 📝 Notes

- Le modèle est mis en cache avec `@st.cache_resource` pour éviter les rechargements
- Les fichiers sont téléchargés dans `.saint_cache/` (exclu du git)
- Le code gère automatiquement les différents formats de données

## 🎯 Prochaines étapes

1. ✅ Tester localement avec vos données
2. ✅ Vérifier que les prédictions sont correctes
3. ✅ Adapter la classe SAINTModel si votre architecture est différente
4. ✅ Déployer sur Streamlit Cloud
5. ✅ Configurer les secrets AWS

Tout est prêt ! 🚀

