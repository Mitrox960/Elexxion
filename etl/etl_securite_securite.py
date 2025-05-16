import pandas as pd
import sqlite3
import os

# === Configuration ===
fichier_csv = "data_lake/securite/EvolutionSecuriteParis.csv"
nom_table = "evolution_securite_par_zone"
nom_bdd = os.path.join("output", "etude_electorale_idf.db")

# Vérification du fichier
if not os.path.exists(fichier_csv):
    print(f"Le fichier {fichier_csv} est introuvable.")
    exit()

try:
    df = pd.read_csv(fichier_csv, sep=';', quotechar='"')
except Exception as e:
    print(f"Erreur lors de la lecture du fichier CSV : {e}")
    exit()

# Nettoyage colonnes (notamment les sauts de ligne)
df.columns = [col.strip().replace('\n', ' ') for col in df.columns]

# Conversion des colonnes numériques (virgule -> point)
for col in df.columns[1:]:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Transformation en format long (année / zone / taux)
df_long = df.melt(id_vars=['année'], var_name='Zone', value_name='Taux_securite')
df_long.rename(columns={'année': 'Annee'}, inplace=True)

# Insertion dans la base SQLite
try:
    conn = sqlite3.connect(nom_bdd)
    df_long.to_sql(nom_table, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Table insérée : {nom_table} dans {nom_bdd}")
except Exception as e:
    print(f"Erreur lors de l'insertion en base : {e}")
