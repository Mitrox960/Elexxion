import pandas as pd
import sqlite3
import os

# Fichiers
fichier_csv = "data_lake/population_active/population_active.csv"
nom_table = "population_active_par_annee"
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

# Garde uniquement les colonnes nécessaires
colonnes_utiles = ['insee'] + [col for col in df.columns if col.startswith("act") and col[3:7].isdigit()]
df_filtré = df[colonnes_utiles]

# Transformation en format long (année, valeur)
df_long = df_filtré.melt(id_vars="insee", var_name="annee", value_name="population_active")

# Nettoyage : extraire l'année depuis le nom de colonne (ex: "act2015" -> 2015)
df_long["annee"] = df_long["annee"].str.extract(r"(\d{4})")

# Conversion des valeurs en numériques
df_long["population_active"] = pd.to_numeric(df_long["population_active"], errors="coerce")

# Retrait des lignes sans année ou valeur
df_long.dropna(subset=["annee", "population_active"], inplace=True)

# Enregistrement dans la base SQLite
try:
    conn = sqlite3.connect(nom_bdd)
    df_long.to_sql(nom_table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Base de données mise à jour : {nom_bdd}")
    print(f"Table insérée : {nom_table}")
except Exception as e:
    print(f"Erreur lors de l'écriture dans la base SQLite : {e}")
