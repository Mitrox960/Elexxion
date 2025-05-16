# Projet MSPR TPRE813 – Big Data & Analyse de données

## Objectif du projet

Dans le cadre de la MSPR "Big Data & Analyse de données", nous avons conçu une **preuve de concept (POC)** permettant de prédire les tendances électorales futures en fonction de données socio-économiques.  
Le projet a été réalisé pour la **région Île-de-France**, à l’échelle régionale.

L’idée est d’anticiper, à l’aide d’un modèle d’intelligence artificielle, **le pourcentage de votes** pour un candidat du centre (type Macron), à partir de données sur :

- la **pauvreté**
- la **population**
- l’**emploi**
- etc.

---

## Architecture du projet

Le projet repose sur 3 grands blocs structurés :

 data_lake/ → Données brutes classées par thème
 etl/ → Scripts de traitement pour chaque type de donnée
 output/ → Base de données SQLite et résultats finaux


## Lancement du projet

1. Assurez-vous d’avoir Python 3 installé
2. Installez les dépendances (pandas, openpyxl, etc.) si besoin :
   ```bash
   pip install pandas openpyxl
3. Exécutez tous les scripts de traitement en allant a la racine du projet : python run_all_etl.py

## Lancement du modèle prédictif principal

Pour entraîner et tester le modèle supervisé de régression qui prédit le score électoral du parti du centre (RandomForestRegressor) :
   python prediction_ai.py

Ce script :
   - Charge et fusionne les données des années 2017, 2018 et 2022.
   - Entraîne un modèle supervisé.
   - Affiche le coefficient de détermination R² sur le jeu de test.
   - Affiche l’importance des variables.
   - Réalise des prédictions pour les années 2026, 2027 et 2028.
   - Produit les graphiques de visualisation nécessaires.

## Lancement du modèle alternatif de classification

Pour tester un autre modèle, basé sur une classification des tendances électorales (Gauche, Centre, Droite) avec gestion du déséquilibre via SMOTE, lancez :
   python prediction_ia_modele2.py

Ce script :
   - Charge les données socio-économiques et simulées.
   - Applique le rééquilibrage SMOTE.
   - Entraîne un RandomForestClassifier.
   - Affiche l’accuracy en validation croisée et sur le jeu de test.
   - Affiche la matrice de corrélation et la distribution des classes.
   - Génère des graphiques sur l’importance des variables et les distributions.

## Technologies utilisées

| Couche                  | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| Collecte (Datalake)     | Données publiques téléchargées (INSEE, data.gouv.fr)         |
| Traitement (ETL)        | Scripts Python avec `pandas`, stockés dans `etl/`            |
| Stockage                | Base de données SQLite locale (`output/etude_electorale_idf.db`) |
| Modélisation (IA)       | Entraînement avec un modèle supervisé en Python              |
| Visualisation           | Graphiques `matplotlib` pour présenter les résultats         |

---
