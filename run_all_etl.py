import os
import subprocess

# Dossier contenant tous les scripts ETL
dossier_etl = "etl"

# Lister tous les fichiers Python dans le dossier
scripts = sorted([
    f for f in os.listdir(dossier_etl)
    if f.endswith(".py") and not f.startswith("__")
])

if not scripts:
    print("Aucun script ETL trouvé dans le dossier 'etl/'.")
    exit()

print("=== Exécution de tous les scripts ETL ===\n")

# Exécution de chaque script
for script in scripts:
    chemin_script = os.path.join(dossier_etl, script)
    print(f"Lancement de {script}...")
    try:
        subprocess.run(["python", chemin_script], check=True)
        print(f"Terminé : {script}\n")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script} : {e}\n")

print("Tous les scripts ont été traités.")
