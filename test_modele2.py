import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- 1. Connexion et extraction des données ---
conn = sqlite3.connect("output/etude_electorale_idf.db")

# Extraction communes IDF
insee_codes = pd.read_sql_query("SELECT DISTINCT insee FROM chomage_par_commune", conn)
insee_codes['dept'] = insee_codes['insee'].astype(str).str[:2]
idf_departements = ['75', '77', '78', '91', '92', '93', '94', '95']
idf_communes = insee_codes[insee_codes['dept'].isin(idf_departements)]['insee'].tolist()

# Extraction chômage 2017, 2018, 2020
chomage = pd.read_sql_query(
    "SELECT insee, chom2017, chom2018, chom2020 FROM chomage_par_commune",
    conn
)
chomage = chomage[chomage['insee'].isin(idf_communes)]
chomage = chomage.dropna(subset=['chom2017', 'chom2018', 'chom2020'])

# Gestion colonne chom2022 (extrapolation si inexistante)
if 'chom2022' in chomage.columns:
    chomage['chom2022'] = chomage['chom2022'].fillna(
        chomage['chom2020'] + (chomage['chom2020'] - chomage['chom2018']) / 2
    )
else:
    chomage['chom2022'] = chomage['chom2020'] + (chomage['chom2020'] - chomage['chom2018']) / 2

# Population socio-professionnelle
pop_socio = pd.read_sql_query(
    "SELECT insee, SUM(population) AS pop_socio_totale FROM population_active_socio_professionnelle GROUP BY insee",
    conn
)
pop_socio = pop_socio[pop_socio['insee'].isin(idf_communes)].dropna()

# Emploi vacant (2017, 2018, 2022)
emplois_vacants = pd.read_sql_query(
    "SELECT Année, AVG(Taux_vacance_moyen) AS taux_vacance_moyen FROM emplois_vacants_resume_par_annee "
    "WHERE Année IN ('2017','2018','2022') GROUP BY Année", conn
)
emplois_vacants = emplois_vacants.dropna()

# Population totale IDF 2017 (extrapolation possible)
population = pd.read_sql_query("SELECT Population_totale FROM idf_population_2017", conn).dropna()
population_totale = population.iloc[0, 0]

# Pauvreté IDF 2017
pauvrete = pd.read_sql_query("SELECT Taux_pauvrete_moyen FROM idf_pauvrete_resume", conn).dropna()
taux_pauvrete = pauvrete.iloc[0, 0]

# Sécurité France entière (2017, 2018, 2022 si dispo)
securite = pd.read_sql_query(
    "SELECT Annee, Taux_securite FROM evolution_securite_par_zone WHERE Zone='France entière' AND Annee IN (2017,2018,2022)",
    conn
)
conn.close()

# Extraire valeurs sécurité
securite_2017 = securite[securite['Annee'] == 2017]['Taux_securite'].values[0]
securite_2018 = securite[securite['Annee'] == 2018]['Taux_securite'].values[0]
if 2022 in securite['Annee'].values:
    securite_2022 = securite[securite['Annee'] == 2022]['Taux_securite'].values[0]
else:
    securite_2022 = securite_2018 + (securite_2018 - securite_2017) * 4

# Emploi vacant par année
emploiv_2017 = emplois_vacants[emplois_vacants['Année'] == '2017']['taux_vacance_moyen'].values[0]
emploiv_2018 = emplois_vacants[emplois_vacants['Année'] == '2018']['taux_vacance_moyen'].values[0]
if '2022' in emplois_vacants['Année'].values:
    emploiv_2022 = emplois_vacants[emplois_vacants['Année'] == '2022']['taux_vacance_moyen'].values[0]
else:
    emploiv_2022 = (emploiv_2017 + emploiv_2018) / 2

# Fonction de préparation dataset
def prepare_df(year, chom_col, pop_mult, pau_mult, securite_val, emploi_vac_val):
    df_tmp = pd.merge(
        chomage[['insee', chom_col]].rename(columns={chom_col: 'chomage_2017'}),
        pop_socio, on='insee', how='inner'
    ).dropna()
    df_tmp['annee'] = year
    df_tmp['population'] = population_totale * pop_mult
    df_tmp['pauvrete'] = taux_pauvrete * pau_mult
    df_tmp['securite'] = securite_val
    df_tmp['emploi_vacant'] = emploi_vac_val
    return df_tmp

df_2017 = prepare_df(2017, 'chom2017', 1.0, 1.0, securite_2017, emploiv_2017)
df_2018 = prepare_df(2018, 'chom2018', 1.02, 0.97, securite_2018, emploiv_2018)
df_2022 = prepare_df(2022, 'chom2022', 1.05, 0.95, securite_2022, emploiv_2022)


# Concaténation des années
df = pd.concat([df_2017, df_2018, df_2022], ignore_index=True)

# Simulation score électoral (parti centre)
np.random.seed(42)
df['score'] = np.where(
    df['annee'] == 2017,
    35 + np.random.normal(0, 3, len(df)),
    np.where(
        df['annee'] == 2018,
        38 + np.random.normal(0, 3, len(df)),
        41 + np.random.normal(0, 3, len(df))  # hypothèse 2022 légèrement plus haute
    )
)

# Variables d'interaction
df['chom_pauvrete'] = df['chomage_2017'] * df['pauvrete']
df['pop_active_ratio'] = df['pop_socio_totale'] / df['population']
df['securite_emploi'] = df['securite'] * df['emploi_vacant']

features = [
    'chomage_2017', 'securite', 'population', 'pauvrete', 'emploi_vacant', 'pop_socio_totale',
    'chom_pauvrete', 'pop_active_ratio', 'securite_emploi'
]

# Visualisation score moyen
plt.figure(figsize=(10,6))
sns.lineplot(x='annee', y='score', data=df.groupby('annee')['score'].mean().reset_index(), marker='o')
plt.title("Score électoral moyen (simulation)")
plt.xlabel("Année")
plt.ylabel("Score (%)")
plt.grid()
plt.show()

# Matrice corrélation
plt.figure(figsize=(12,10))
sns.heatmap(df[features + ['score']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

# Modèle supervisé
X = df[features]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R² sur test : {r2_score(y_test, y_pred):.3f}")

# Importance variables
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,8))
plt.barh([features[i] for i in indices], importances[indices], color='skyblue')
plt.title("Importance des variables")
plt.xlabel("Importance")
plt.ylabel("Variables")
plt.show()

# Prédictions 2026-2028
scenarios = pd.DataFrame({
    'chomage_2017': [df['chomage_2017'].mean()*0.95, df['chomage_2017'].mean()*0.93, df['chomage_2017'].mean()*0.90],
    'securite': [df['securite'].mean()+0.15, df['securite'].mean()+0.20, df['securite'].mean()+0.25],
    'population': [population_totale*1.05, population_totale*1.06, population_totale*1.07],
    'pauvrete': [taux_pauvrete*0.90, taux_pauvrete*0.88, taux_pauvrete*0.85],
    'emploi_vacant': [df['emploi_vacant'].mean()*0.90, df['emploi_vacant'].mean()*0.88, df['emploi_vacant'].mean()*0.85],
    'pop_socio_totale': [df['pop_socio_totale'].mean()*1.02, df['pop_socio_totale'].mean()*1.03, df['pop_socio_totale'].mean()*1.04],
})
scenarios['chom_pauvrete'] = scenarios['chomage_2017'] * scenarios['pauvrete']
scenarios['pop_active_ratio'] = scenarios['pop_socio_totale'] / scenarios['population']
scenarios['securite_emploi'] = scenarios['securite'] * scenarios['emploi_vacant']

pred_future = model.predict(scenarios[features])

plt.figure(figsize=(10,6))
means = df.groupby('annee')['score'].mean()
plt.plot(means.index, means.values, 'o-', label='Historique moyen', color='orange')
plt.plot([2026, 2027, 2028], pred_future, 'o-', label='Prédiction', color='red')
plt.title("Prédiction du score électoral (parti du centre)")
plt.xlabel("Année")
plt.ylabel("Score (%)")
plt.legend()
plt.grid()
plt.show()
