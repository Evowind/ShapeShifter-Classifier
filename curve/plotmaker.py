import pandas as pd
import matplotlib.pyplot as plt

# Fonction pour charger le CSV et retourner les données Recall et Precision
def load_precision_recall_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if 'Recall' not in df.columns or 'Precision' not in df.columns:
        raise ValueError(f"Le fichier {csv_file_path} doit contenir les colonnes 'Recall' et 'Precision'.")
    return df['Recall'], df['Precision']

# Dictionnaire des fichiers CSV à comparer
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

# Couleurs pour chaque modèle
color_map = {
    'KMeans': 'tab:blue',  # Bleu pour KMeans
    'KNN': 'tab:orange',   # Orange pour KNN
    'SVM': 'tab:green'     # Vert pour SVM
}

# Initialiser un graphique
plt.figure(figsize=(10, 8))

# Charger et tracer chaque fichier avec des couleurs spécifiques pour chaque modèle
for label, csv_path in csv_files.items():
    recall, precision = load_precision_recall_data(csv_path)
    
    # Déterminer le modèle (KMeans, KNN, SVM) et assigner une couleur
    if 'KMeans' in label:
        color = color_map['KMeans']
    elif 'KNN' in label:
        color = color_map['KNN']
    elif 'SVM' in label:
        color = color_map['SVM']
    
    plt.plot(recall, precision, label=label, color=color)

# Personnaliser le graphique
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Comparaison des courbes Precision-Recall pour différents modèles', fontsize=16)
plt.legend(title='Modèles', loc='lower left', bbox_to_anchor=(1, 0.5))
plt.grid(True)

# Afficher le graphique
plt.tight_layout()
plt.show()
