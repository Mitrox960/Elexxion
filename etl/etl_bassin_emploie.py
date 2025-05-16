import pandas as pd
import sqlite3
import os

# === Configuration ===
fichier_csv = "data_lake/bassin_emploie/bassin_emploie.csv"
nom_table = "bassins_emploi_idf"
nom_bdd = os.path.join("output", "etude_electorale_idf.db")

# Vérification du fichier
if not os.path.exists(fichier_csv):
    print(f"Le fichier {fichier_csv} est introuvable.")
    exit()

try:
    df = pd.read_csv(fichier_csv)
except Exception as e:
    print(f"Erreur lors de la lecture du fichier CSV : {e}")
    exit()

# Sélection des colonnes pertinentes
colonnes_utiles = [
    "beef_id", "beef_nom", "beef_nbcom", "date", "source",
    "nom_region", "datemaj", "surf", "st_areashape", "st_lengthshape"
]
df = df[colonnes_utiles].copy()

# Renommage pour cohérence
df.rename(columns={
    "beef_id": "ID_Bassin",
    "beef_nom": "Nom_Bassin",
    "beef_nbcom": "Nb_Commune",
    "date": "Date_Creation",
    "source": "Source",
    "nom_region": "Nom_Region",
    "datemaj": "Date_MAJ",
    "surf": "Surface_m2",
    "st_areashape": "Surface_Geom",
    "st_lengthshape": "Longueur_Geom"
}, inplace=True)

# Enregistrement dans la base SQLite
try:
    conn = sqlite3.connect(nom_bdd)
    df.to_sql(nom_table, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Table insérée : {nom_table} dans {nom_bdd}")
except Exception as e:
    print(f"Erreur lors de l'insertion en base : {e}")
