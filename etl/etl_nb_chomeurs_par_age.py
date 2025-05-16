import pandas as pd
import sqlite3
import os

# === Configuration ===
fichier_csv = "data_lake/nb_chomeurs_par_age/nb_chomeurs_par_age.csv"
nom_table = "chomage_par_age"
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

# Sélection des colonnes pertinentes pour un graphique évolutif
colonnes_utiles = [
    "insee",
    "chom15_19", "chom20_24", "chom25_29", "chom30_49",
    "chom50_54", "chom55_59", "chom60_64", "chom65pl",
    "chom2020"
]

df_selection = df[colonnes_utiles].copy()

# Conversion en numérique (sécurité)
for col in colonnes_utiles[1:]:
    df_selection[col] = pd.to_numeric(df_selection[col], errors="coerce")

# Insertion dans la base SQLite
try:
    conn = sqlite3.connect(nom_bdd)
    df_selection.to_sql(nom_table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Table insérée : {nom_table} dans {nom_bdd}")
except Exception as e:
    print(f"Erreur lors de l'insertion dans la base : {e}")
