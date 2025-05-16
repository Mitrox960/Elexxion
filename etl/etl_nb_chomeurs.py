import pandas as pd
import sqlite3
import os

# === Configuration ===
fichier_csv = "data_lake/nb_chomeurs/nb_chomeurs.csv"
nom_table = "chomage_par_commune"
nom_bdd = os.path.join("output", "etude_electorale_idf.db")

# Vérification du fichier
if not os.path.exists(fichier_csv):
    print(f"Le fichier {fichier_csv} est introuvable.")
    exit()

# Lecture du fichier CSV
try:
    df = pd.read_csv(fichier_csv)
except Exception as e:
    print(f"Erreur lors de la lecture du fichier CSV : {e}")
    exit()

# Normalisation des colonnes : nettoyage des noms de colonnes si nécessaire
df.columns = df.columns.str.strip().str.lower()

# Vérifie la présence de la colonne 'insee'
if 'insee' not in df.columns:
    print(f"Colonne 'insee' manquante.")
    print("Colonnes disponibles :", df.columns.tolist())
    exit()

# Enregistrement dans la base SQLite
try:
    conn = sqlite3.connect(nom_bdd)
    df.to_sql(nom_table, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Table insérée : {nom_table} dans {nom_bdd}")
except Exception as e:
    print(f"Erreur lors de l'insertion en base : {e}")
