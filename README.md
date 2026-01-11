# Intelligent Customer Review Analysis

Ce projet permet d'analyser les avis clients de McDonald's à l'aide de modèles de Machine Learning et d'une interface web FastAPI.

## Structure du projet

```
├── app.py                  # Application FastAPI (API + interface web)
├── main.ipynb              # Notebook principal : préparation, entraînement, export des modèles
├── requirements.txt        # Dépendances Python
├── McDonald_s_Reviews.csv  # Jeu de données d'avis clients
├── model_files/            # Fichiers du modèle entraîné (joblib)
│   ├── best_model.joblib
│   ├── label_encoder.joblib
│   └── tfidf_vectorizer.joblib
├── templates/
│   └── index.html          # Template HTML pour l'interface web
```

## Consignes de lancement

1. **Créer et activer l'environnement virtuel**
   ```bash
   python -m venv .venv
   # Sous Windows PowerShell :
   .venv\Scripts\Activate.ps1
   # Sous bash :
   source .venv/bin/activate
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Exécuter le notebook pour entraîner et exporter les modèles**
   - Ouvrir `main.ipynb` dans Jupyter Notebook ou VS Code
   - Exécuter toutes les cellules pour :
     - Charger et prétraiter les données
     - Entraîner les modèles
     - Sauvegarder les fichiers dans `model_files/`

4. **Lancer le serveur FastAPI**
   ```bash
   uvicorn app:app --reload
   ```
   Accéder à l'interface web sur : http://127.0.0.1:8000/

5. **Tester l'API**
   - Documentation interactive : http://127.0.0.1:8000/docs


## Structure détaillée du notebook `main.ipynb`

1. **Import & configuration**
   - Chargement des librairies (pandas, numpy, matplotlib, sklearn, etc.)
   - Réglages d'affichage et configuration de la variable cible

2. **Chargement des données**
   - Lecture du fichier CSV des avis clients
   - Aperçu rapide (shape, head)

3. **Exploration rapide**
   - Analyse des types de colonnes, valeurs manquantes, doublons

4. **Analyse statistique**
   - Statistiques descriptives sur les ratings et les sentiments
   - Analyse par magasin, par longueur de review, et temporelle
   - Corrélations entre variables

5. **Préparation / nettoyage**
   - Nettoyage des noms de colonnes
   - Parsing des champs (rating, rating_count, review_time)
   - Création de variables dérivées (review_length, sentiment)

6. **Visualisation des distributions**
   - Histogrammes et countplots des variables clés
   - Visualisation des valeurs manquantes

7. **Comparaison des méthodes d'imputation**
   - Test de plusieurs stratégies pour gérer les valeurs manquantes (mean, median, KNN, MICE, etc.)
   - Évaluation de l'impact sur la distribution et la performance ML

8. **Prétraitement (imputation + standardisation)**
   - Imputation des variables numériques (médiane)
   - Standardisation (z-score)
   - Comparaison avant/après prétraitement

9. **Split train / validation / test**
   - Séparation stratifiée en 70/15/15
   - Vérification des distributions

10. **Préparation des features pour le ML**
    - Vectorisation du texte (TF-IDF)
    - Encodage des variables catégorielles
    - Gestion du cache et batch processing

11. **Entraînement et évaluation des modèles**
    - Implémentation de plusieurs modèles (Naive Bayes, Logistic Regression, SVM, XGBoost, MLP)
    - Recherche d'hyperparamètres (RandomizedSearchCV)
    - Évaluation sur validation

12. **Comparaison et visualisation des performances**
    - Tableaux et graphiques comparatifs (accuracy, F1, etc.)

13. **Évaluation finale sur le jeu de test**
    - Évaluation des meilleurs modèles sur le jeu de test
    - Matrices de confusion, rapports détaillés

14. **Analyse critique et recommandations**
    - Analyse du surapprentissage, biais/variance, complexité
    - Discussion sur les avantages/limites de chaque approche
    - Sélection multicritères du modèle optimal

15. **Post-traitement et optimisation**
    - Rééquilibrage des classes (SMOTE, class weights)
    - Réentraînement des modèles avec données équilibrées
    - Comparaison avant/après post-traitement
    - Tests statistiques sur les améliorations

16. **Export des objets pour FastAPI**
    - Sauvegarde du meilleur modèle, du vectorizer et du label encoder dans `model_files/`

Chaque étape est commentée et illustrée dans le notebook pour faciliter la compréhension et la reproductibilité.

## Notes
- Les fichiers du modèle doivent être présents dans `model_files/` pour que l'API fonctionne.
- Le template HTML se trouve dans `templates/index.html`.
- Le notebook peut être relancé pour réentraîner et mettre à jour les modèles.

---
