import pandas as pd
import sqlite3
import os

fichier_excel = "data_lake/pauvrete/pauvrete_2017.xlsx"
nom_feuille = "DEP"
nom_table = "idf_pauvrete_resume"
nom_bdd = os.path.join("output", "etude_electorale_idf.db")

idf_departements = ['75', '77', '78', '91', '92', '93', '94', '95']

# Vérification fichier
if not os.path.exists(fichier_excel):
    print(f"Le fichier {fichier_excel} est introuvable.")
    exit()

try:
    df_raw = pd.read_excel(fichier_excel, sheet_name=nom_feuille, skiprows=5)
except Exception as e:
    print(f"Erreur lors de la lecture du fichier Excel : {e}")
    exit()

# Filtrer les départements IDF
df_idf = df_raw[df_raw["CODGEO"].astype(str).isin(idf_departements)]

# Vérifier la présence de la colonne de pauvreté uniquement
if "TP6017" not in df_idf.columns:
    print("La colonne TP6017 est introuvable.")
    print("Colonnes disponibles :", df_idf.columns.tolist())
    exit()

# Conversion des données de pauvreté
df_idf["TP6017"] = pd.to_numeric(df_idf["TP6017"], errors="coerce")

# Calcul de la moyenne
pauvrete_moyenne = df_idf["TP6017"].mean()

# Création du résumé
df_resume = pd.DataFrame({
    "Region": ["Île-de-France"],
    "Annee": [2017],
    "Taux_pauvrete_moyen": [pauvrete_moyenne]
})

# Insertion dans la BDD
try:
    conn = sqlite3.connect(nom_bdd)
    df_resume.to_sql(nom_table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Base de données créée : {nom_bdd}")
    print(f"Table insérée : {nom_table}")
except Exception as e:
    print(f"Erreur lors de l'écriture dans la base SQLite : {e}")
