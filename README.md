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

## Technologies utilisées

| Couche                  | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| Collecte (Datalake)     | Données publiques téléchargées (INSEE, data.gouv.fr)         |
| Traitement (ETL)        | Scripts Python avec `pandas`, stockés dans `etl/`            |
| Stockage                | Base de données SQLite locale (`output/etude_electorale_idf.db`) |
| Modélisation (IA)       | Entraînement avec un modèle supervisé en Python              |
| Visualisation           | Graphiques `matplotlib` pour présenter les résultats         |

---
