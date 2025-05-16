import pandas as pd
import sqlite3
import os

# Fichiers
fichier_csv = "data_lake/population_active_socio_professionel/population_active_socio_professionel.csv"
nom_table = "population_active_socio_professionnelle"
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

# Colonnes utiles à conserver
colonnes_utiles = [
    "insee",
    "actagr",       # Agriculteurs exploitants
    "actartcfen",   # Artisans, commerçants et chefs d’entreprise
    "actcadre",     # Cadres et professions intellectuelles supérieures
    "actproint",    # Professions intermédiaires
    "actemplo",     # Employés
    "actouvri",     # Ouvriers
    "actifs"        # Total des actifs
]

# Extraction des colonnes utiles
df_filtré = df[colonnes_utiles]

# Conversion en format long
df_long = df_filtré.melt(id_vars="insee", var_name="categorie_socioprofessionnelle", value_name="population")

# Nettoyage : suppression des lignes sans valeurs numériques
df_long["population"] = pd.to_numeric(df_long["population"], errors="coerce")
df_long.dropna(subset=["population"], inplace=True)

# Enregistrement dans la base SQLite
try:
    conn = sqlite3.connect(nom_bdd)
    df_long.to_sql(nom_table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Base de données mise à jour : {nom_bdd}")
    print(f"Table insérée : {nom_table}")
except Exception as e:
    print(f"Erreur lors de l'écriture dans la base SQLite : {e}")
