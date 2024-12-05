import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# Fonction pour charger les données depuis un fichier CSV
def load_precision_recall_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if 'Recall' not in df.columns or 'Precision' not in df.columns:
        raise ValueError(f"Le fichier {csv_file_path} doit contenir les colonnes 'Recall' et 'Precision'.")
    return df['Recall'], df['Precision']

# Dictionnaire des fichiers CSV
csv_files = {
    'KMeans_ART': 'curve/KMeans_ART.csv',
    'KMeans_E34': 'curve/KMeans_E34.csv',
    'KMeans_GFD': 'curve/KMeans_GFD.csv',
    'KMeans_Yang': 'curve/KMeans_Yang.csv',
    'KMeans_Zernike7': 'curve/KMeans_Zernike7.csv',
    'KNN_ART': 'curve/KNN_ART.csv',
    'KNN_E34': 'curve/KNN_E34.csv',
    'KNN_GFD': 'curve/KNN_GFD.csv',
    'KNN_Yang': 'curve/KNN_Yang.csv',
    'KNN_Zernike7': 'curve/KNN_Zernike7.csv',
    'SVM_ART': 'curve/SVM_ART.csv',
    'SVM_E34': 'curve/SVM_E34.csv',
    'SVM_GFD': 'curve/SVM_GFD.csv',
    'SVM_Yang': 'curve/SVM_Yang.csv',
    'SVM_Zernike7': 'curve/SVM_Zernike7.csv'
    }

# Attribuer des couleurs similaires pour chaque modèle principal
colors = {
    'KMeans': cm.Blues,
    'KNN': cm.Greens,
    'SVM': cm.Reds
}

# Initialiser le graphique
plt.figure(figsize=(12, 8))

# Tracer les courbes pour chaque fichier CSV
for label, csv_path in csv_files.items():
    # Extraire le nom du modèle principal (KMeans, KNN, SVM)
    model = label.split('_')[0]  # Exemple : 'KMeans_ART' -> 'KMeans'
    colormap = colors[model]
    
    # Générer une teinte unique pour chaque sous-méthode
    sub_method_index = list(csv_files.keys()).index(label)
    color = colormap(0.2 + 0.15 * (sub_method_index % 5))  # Ajuster la teinte pour chaque méthode

    # Charger les données et tracer la courbe
    recall, precision = load_precision_recall_data(csv_path)
    plt.plot(recall, precision, label=label, color=color, linewidth=2)

# Ajouter les labels, le titre, et la légende
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Comparaison des courbes Precision-Recall pour différents modèles', fontsize=16)
plt.legend(title='Modèles et Méthodes', loc='lower left', bbox_to_anchor=(1, 0.5))
plt.grid(True)

# Enregistrer le graphique dans un fichier
plt.savefig('Figure_1.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

