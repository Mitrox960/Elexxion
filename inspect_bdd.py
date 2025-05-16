import sqlite3
import os
import pandas as pd

# === Configuration ===
db_path = os.path.join("output", "etude_electorale_idf.db")

# Vérifie que la base existe
if not os.path.exists(db_path):
    print(f"La base de données n'existe pas à ce chemin : {db_path}")
    exit()

# Connexion à la base
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Récupération des noms de tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

if not tables:
    print("Aucune table trouvée dans la base.")
    conn.close()
    exit()

print(f"Tables trouvées dans {db_path} :\n")
for table in tables:
    print(f"Table : {table}")
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        print(df.head(), "\n")  # Affiche les premières lignes
    except Exception as e:
        print(f"Erreur lors de la lecture de {table} : {e}\n")

conn.close()
print("Inspection terminée.")
