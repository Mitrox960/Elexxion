import pandas as pd
import sqlite3
import os

fichier_csv = "data_lake/emploie_vacant/dares_emploivacants_brut_emploisvacants_trie.csv"
nom_table = "emplois_vacants_resume_par_annee"
nom_bdd = os.path.join("output", "etude_electorale_idf.db")

# Vérification du fichier
if not os.path.exists(fichier_csv):
    print(f"Le fichier {fichier_csv} est introuvable.")
    exit()

try:
    df = pd.read_csv(fichier_csv, sep=';')
except Exception as e:
    print(f"Erreur lors de la lecture du fichier CSV : {e}")
    exit()

# Vérification des colonnes nécessaires
colonnes_requises = ["Date", "Code NAF", "Taux d'emplois vacants (en %)"]
for col in colonnes_requises:
    if col not in df.columns:
        print(f"Colonne manquante : {col}")
        print("Colonnes disponibles :", df.columns.tolist())
        exit()

# Nettoyage des données
df["Taux d'emplois vacants (en %)"] = pd.to_numeric(df["Taux d'emplois vacants (en %)"], errors="coerce")
df["Année"] = df["Date"].str[:4]  # Extraire l'année (ex: "2003" à partir de "2003-T2")

# Moyenne annuelle par Code NAF
df_resume = df.groupby(["Année", "Code NAF"], as_index=False)["Taux d'emplois vacants (en %)"].mean()
df_resume.rename(columns={"Taux d'emplois vacants (en %)": "Taux_vacance_moyen"}, inplace=True)

# Enregistrement dans la base SQLite
try:
    conn = sqlite3.connect(nom_bdd)
    df_resume.to_sql(nom_table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Base de données mise à jour : {nom_bdd}")
    print(f"Table insérée : {nom_table}")
except Exception as e:
    print(f"Erreur lors de l'écriture dans la base SQLite : {e}")
