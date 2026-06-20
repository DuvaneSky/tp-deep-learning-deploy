# TP Deep Learning — Déploiement de modèles

Application web de démonstration développée dans le cadre du TP du module **Intelligence Artificielle et Deep Learning** (Informatique Fondamentale, Niveau 4, année académique 2025-2026).

## Description

Cette application permet de tester en ligne les modèles entraînés durant le TP, à travers une interface simple construite avec Streamlit :

- **Bank Marketing** : prédiction de la souscription d'un client à un dépôt à terme, à partir de son profil (modèles Random Forest et ANN).
- **Fashion MNIST** : classification d'une image de vêtement parmi 10 catégories (modèle CNN).

## Lien de l'application

👉 [https://tp-deep-learning.streamlit.app](https://tp-deep-learning.streamlit.app)

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `modelIASky.py` | Code source de l'application Streamlit |
| `random_forest_model.pkl` | Modèle Random Forest entraîné (Bank Marketing) |
| `model_ann.keras` | Modèle ANN entraîné (Bank Marketing) |
| `model_cnn.keras` | Modèle CNN entraîné (Fashion MNIST) |
| `scaler.pkl` | StandardScaler utilisé pour normaliser les données bancaires |
| `label_encoders.pkl` | Encodeurs des variables catégorielles (Bank Marketing) |
| `requirements.txt` | Dépendances Python nécessaires au déploiement |
| `runtime.txt` | Version de Python utilisée (3.11) |

## Jeux de données utilisés

- [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) (UCI Machine Learning Repository)
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research)

## Outils utilisés

Python, scikit-learn, TensorFlow/Keras, Streamlit, Streamlit Community Cloud.

## Auteur

FOZEN POKAM Franck Duvane — Matricule UN21TEL021IY
