import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter

# Connexion BDD
conn = sqlite3.connect("output/etude_electorale_idf.db")

# Liste INSEE pour l’Île-de-France (communes des départements 75, 77, 78, 91, 92, 93, 94, 95)
idf_departements = ['75', '77', '78', '91', '92', '93', '94', '95']

# Récupération INSEE communes IDF dans chomage_par_commune
df_insee = pd.read_sql_query("SELECT DISTINCT CAST(insee AS TEXT) as insee FROM chomage_par_commune", conn)
df_insee["dept"] = df_insee["insee"].str[:2]
idf_insee = df_insee[df_insee["dept"].isin(idf_departements)]["insee"].tolist()
insee_str = ",".join(f"'{code}'" for code in idf_insee)

print(f"Nombre de communes IDF récupérées : {len(idf_insee)}")

# Chargement données chomage (2018)
chomage = pd.read_sql_query(
    f"SELECT CAST(insee AS TEXT) as insee, chom2018 FROM chomage_par_commune WHERE CAST(insee AS TEXT) IN ({insee_str})",
    conn,
)

# Vie associative simulée (à remplacer par données réelles si dispo)
np.random.seed(42)
vie_associative = pd.DataFrame({
    'insee': idf_insee,
    'indice_vie_associative': np.random.uniform(0.5, 1.5, len(idf_insee))
})

# Population active 2017
pop = pd.read_sql_query(
    f"SELECT CAST(insee AS TEXT) as insee, population_active FROM population_active_par_annee WHERE CAST(insee AS TEXT) IN ({insee_str}) AND annee=2017",
    conn,
)

conn.close()

# Merge dataframes
df = chomage.rename(columns={"chom2018": "taux_chomage"}).copy()
df = df.merge(vie_associative, on="insee", how="left")
df = df.merge(pop, on="insee", how="left")

df.dropna(inplace=True)
if df.empty:
    raise ValueError("DataFrame vide après nettoyage.")

# Simulation cible vote_tendance - à remplacer par données réelles si disponibles
np.random.seed(42)
df["vote_tendance"] = np.random.choice(["Gauche", "Centre", "Droite"], size=len(df))

features = [
    "taux_chomage",
    "indice_vie_associative",
    "population_active",
]

X = df[features]
y = df["vote_tendance"]

print("\n--- Statistiques descriptives ---")
print(df[features].describe())

print("\n--- Distribution des classes ---")
print(df["vote_tendance"].value_counts())

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE avec k_neighbors dynamique
counts = Counter(y_enc)
print("\nDistribution classes avant SMOTE :", {le.inverse_transform([k])[0]: v for k, v in counts.items()})
min_class_size = min(counts.values())
k_neighbors = min(3, max(1, min_class_size - 1))
print(f"SMOTE k_neighbors = {k_neighbors}")

smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_res, y_res = smote.fit_resample(X_scaled, y_enc)

print("\n--- Après SMOTE ---")
print("Taille dataset rééquilibré:", len(y_res))
unique, counts_res = np.unique(y_res, return_counts=True)
print("Répartition classes:", dict(zip(le.inverse_transform(unique), counts_res)))

# Cross-validation stratifiée
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='accuracy')
print("\nCross-validation accuracy:", scores)
print("Moyenne accuracy:", scores.mean())

# Entraînement final
model.fit(X_res, y_res)

# Test final
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)
y_pred = model.predict(X_test)

print("\n--- Rapport classification (test set) ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Accuracy test set:", accuracy_score(y_test, y_pred))

# Importance variables
importances = model.feature_importances_
print("\n--- Importance des variables ---")
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.4f}")

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Importance des variables dans le modèle")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.show()

# Corrélation
df["vote_tendance_enc"] = le.transform(df["vote_tendance"])
corr = df[features + ["vote_tendance_enc"]].corr()
print("\n--- Matrice de corrélation ---")
print(corr)

plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de corrélation entre variables et cible encodée")
plt.show()

# Distribution des classes
plt.figure(figsize=(6,4))
sns.countplot(x="vote_tendance", data=df)
plt.title("Distribution des classes de tendance électorale")
plt.show()

# Scatter plots variables importantes
top_vars = [features[i] for i in np.argsort(importances)[::-1][:3]]

plt.figure(figsize=(15, 5))
for i, col in enumerate(top_vars, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=col, y="taux_chomage", hue="vote_tendance", data=df, alpha=0.7)
    plt.title(f"{col} vs taux de chômage")
plt.tight_layout()
plt.show()

# Prédictions évolutives
print("\n--- Prédictions évolutives ---")
for annee in [1, 2, 3]:
    scenario = pd.Series(X.mean(axis=0), index=features)
    scenario["taux_chomage"] *= max(0, 1 - 0.02 * annee)
    scenario["indice_vie_associative"] *= 1 + 0.01 * annee
    scenario["population_active"] *= 1 + 0.01 * annee

    scenario_df = pd.DataFrame([scenario])
    scenario_scaled = scaler.transform(scenario_df)
    pred = model.predict(scenario_scaled)
    print(f"Prédiction à {annee} an(s) : {le.inverse_transform(pred)[0]}")
