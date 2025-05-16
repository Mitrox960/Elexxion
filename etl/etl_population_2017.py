# Réexécution nécessaire après réinitialisation de l'environnement
import pandas as pd
import sqlite3
import os
os.makedirs("output", exist_ok=True)

# Rechargement du fichier depuis le bon chemin
file_path = "data_lake/population/population_2017.csv"
df = pd.read_csv(file_path, sep=';', engine='python')

# === TRAITEMENT ===

# Départements de l'Île-de-France
idf_departements = ['75', '77', '78', '91', '92', '93', '94', '95']

# Filtrage : année 2017, mesure = POP, départements IDF
df = df[
    (df["TIME_PERIOD"] == 2017) &
    (df["EP_MEASURE"] == "POP") &
    (df["GEO"].isin(idf_departements))
]

# Calcul des agrégats
hommes = df[df["SEX"] == "M"]["OBS_VALUE"].sum()
femmes = df[df["SEX"] == "F"]["OBS_VALUE"].sum()
total = df[df["SEX"] == "_T"]["OBS_VALUE"].sum()

# Moyenne d'âge pondérée approximative
df_age = df[(df["SEX"] == "_T") & (df["AGE"].str.startswith("Y"))].copy()
df_age["AGE_NUM"] = df_age["AGE"].str.extract(r'Y(\d+)').astype(float)
df_age["OBS_VALUE"] = pd.to_numeric(df_age["OBS_VALUE"], errors="coerce")
moyenne_age = (df_age["AGE_NUM"] * df_age["OBS_VALUE"]).sum() / df_age["OBS_VALUE"].sum()

# Création du DataFrame résumé
df_resume = pd.DataFrame([{
    "Region": "Île-de-France",
    "Annee": 2017,
    "Population_totale": total,
    "Nombre_hommes": hommes,
    "Nombre_femmes": femmes,
    "Moyenne_age_estimee": round(moyenne_age, 2)
}])

# Insertion dans SQLite
db_path = os.path.join("output", "etude_electorale_idf.db")
conn = sqlite3.connect(db_path)
df_resume.to_sql("idf_population_2017", conn, if_exists="replace", index=False)

# Vérification
df_check = pd.read_sql("SELECT * FROM idf_population_2017", conn)
conn.close()

