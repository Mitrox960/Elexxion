import pandas as pd
import sqlite3
import os

# Fichier source
fichier_csv = "data_lake/economie/economie.csv"
nom_table = "indicateurs_economiques"
nom_bdd = os.path.join("output", "etude_electorale_idf.db")

# Lecture CSV avec séparateur ; et gestion des guillemets
try:
    df = pd.read_csv(fichier_csv, sep=';', quotechar='"')
except Exception as e:
    print(f"Erreur lors de la lecture du fichier CSV : {e}")
    exit()

# Colonnes importantes
colonnes_utiles = [
    "REF_SECTOR",
    "ACCOUNTING_ENTRY",
    "INSTR_ASSET",
    "TIME_PERIOD",
    "OBS_VALUE"
]

# Filtrage
df_filtré = df[colonnes_utiles]

# Nettoyage : conversion des types
df_filtré["TIME_PERIOD"] = pd.to_numeric(df_filtré["TIME_PERIOD"], errors="coerce")
df_filtré["OBS_VALUE"] = pd.to_numeric(df_filtré["OBS_VALUE"], errors="coerce")

df_filtré.dropna(subset=["TIME_PERIOD", "OBS_VALUE"], inplace=True)

# Connexion et insertion
try:
    conn = sqlite3.connect(nom_bdd)
    df_filtré.to_sql(nom_table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Base de données mise à jour : {nom_bdd}")
    print(f"Table insérée : {nom_table}")
except Exception as e:
    print(f"Erreur lors de l'écriture dans la base SQLite : {e}")
