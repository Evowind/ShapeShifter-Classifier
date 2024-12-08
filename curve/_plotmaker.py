import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# Fonction pour charger les données depuis un fichier CSV
def load_precision_recall_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # Vérifier les colonnes présentes
    if set(df.columns) != {'Precision', 'Recall'}:
        raise ValueError(f"Le fichier {csv_file_path} doit contenir exactement les colonnes 'Precision' et 'Recall'.")
    # Inverser l'ordre des colonnes pour correspondre à 'Recall' en premier et 'Precision' en second
    return df['Precision'], df['Recall']  # Inversion des colonnes ici

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
    'SVM_Zernike7': 'curve/SVM_Zernike7.csv',
    'MLP_ART': 'curve/MLP_ART.csv',
    'MLP_E34': 'curve/MLP_E34.csv',
    'MLP_GFD': 'curve/MLP_GFD.csv',
    'MLP_Yang': 'curve/MLP_Yang.csv',
    'MLP_Zernike7': 'curve/MLP_Zernike7.csv'
}

# Attribuer des couleurs similaires pour chaque modèle principal
colors = {
    'KMeans': cm.Blues,
    'KNN': cm.Greens,
    'SVM': cm.Reds,
    'MLP': cm.Purples
}

# Marqueurs uniques pour chaque courbe
markers = ['o', 's', 'D', '^', 'v']

# Graphique combiné pour tous les modèles
plt.figure(figsize=(12, 8))

# Tracer les courbes pour tous les fichiers
for label, csv_path in csv_files.items():
    # Identifier le modèle et choisir la couleur
    model = label.split('_')[0]
    colormap = colors[model]
    sub_method_index = list(csv_files.keys()).index(label)
    color = colormap(0.2 + 0.15 * (sub_method_index % 5))  # Ajuster la teinte pour chaque méthode
    marker = markers[sub_method_index % len(markers)]  # Marqueur unique
    
    # Charger les données et tracer la courbe
    precision, recall = load_precision_recall_data(csv_path)  # Inverser ici aussi
    plt.plot(recall, precision, label=label, color=color, linewidth=2, marker=marker, markersize=6)

# Ajouter les labels, le titre, et la légende
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Courbes Precision-Recall combinées pour tous les modèles', fontsize=16)
plt.legend(title='Modèles et Méthodes', loc='lower left', bbox_to_anchor=(1, 0.5))
plt.grid(True)

# Enregistrer le graphique dans un fichier
plt.savefig('Figure_All_Models.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# Générer un graphique unique pour chaque modèle principal
models = set([label.split('_')[0] for label in csv_files.keys()])
for model in models:
    plt.figure(figsize=(10, 6))
    
    # Tracer les courbes pour les fichiers associés au modèle
    for label, csv_path in csv_files.items():
        if label.startswith(model):
            colormap = colors[model]
            sub_method_index = list(csv_files.keys()).index(label)
            color = colormap(0.2 + 0.15 * (sub_method_index % 5))  # Ajuster la teinte pour chaque méthode
            marker = markers[sub_method_index % len(markers)]  # Marqueur unique
            
            # Charger les données et tracer la courbe
            precision, recall = load_precision_recall_data(csv_path)  # Inverser ici aussi
            plt.plot(recall, precision, label=label, color=color, linewidth=2, marker=marker, markersize=6)

    # Ajouter les labels, le titre, et la légende
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Courbes Precision-Recall pour le modèle {model}', fontsize=16)
    plt.legend(title='Méthodes', loc='lower left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    
    # Enregistrer le graphique dans un fichier
    plt.savefig(f'Figure_{model}.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
