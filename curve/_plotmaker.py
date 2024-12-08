import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
# Load data from a CSV file
def load_precision_recall_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # Check the columns present
    if set(df.columns) != {'Precision', 'Recall'}:
        raise ValueError(f"The file {csv_file_path} must contain exactly the columns 'Precision' and 'Recall'.")
    # Reverse the column order to match 'Recall' first and 'Precision' second
    return df['Precision'], df['Recall']  # Reverse columns here

# Dictionary of CSV files
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

# Assign similar colors for each main model
colors = {
    'KMeans': cm.Blues,
    'KNN': cm.Greens,
    'SVM': cm.Reds,
    'MLP': cm.Purples
}

# Unique markers for each curve
markers = ['o', 's', 'D', '^', 'v']

# Combined plot for all models
plt.figure(figsize=(12, 8))

# Plot the curves for all files
for label, csv_path in csv_files.items():
    # Identify the model and choose the color
    model = label.split('_')[0]
    colormap = colors[model]
    sub_method_index = list(csv_files.keys()).index(label)
    color = colormap(0.2 + 0.15 * (sub_method_index % 5))  # Adjust the shade for each method
    marker = markers[sub_method_index % len(markers)]  # Unique marker
    
    # Load the data and plot the curve
    precision, recall = load_precision_recall_data(csv_path)  # Reverse here too
    plt.plot(recall, precision, label=label, color=color, linewidth=2, marker=marker, markersize=6)

# Add labels, title, and legend
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Combined Precision-Recall Curves for All Models', fontsize=16)
plt.legend(title='Models and Methods', loc='lower left', bbox_to_anchor=(1, 0.5))
plt.grid(True)

# Save the plot to a file
plt.savefig('Figure_All_Models.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# Generate a separate plot for each main model
models = set([label.split('_')[0] for label in csv_files.keys()])
for model in models:
    plt.figure(figsize=(10, 6))
    
    # Plot the curves for the files associated with the model
    for label, csv_path in csv_files.items():
        if label.startswith(model):
            colormap = colors[model]
            sub_method_index = list(csv_files.keys()).index(label)
            color = colormap(0.2 + 0.15 * (sub_method_index % 5))  # Adjust the shade for each method
            marker = markers[sub_method_index % len(markers)]  # Unique marker
            
            # Load the data and plot the curve
            precision, recall = load_precision_recall_data(csv_path)  # Reverse here too
            plt.plot(recall, precision, label=label, color=color, linewidth=2, marker=marker, markersize=6)

    # Add labels, title, and legend
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curves for Model {model}', fontsize=16)
    plt.legend(title='Methods', loc='lower left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(f'Figure_{model}.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

