#include <iostream>
#include <vector>
#include <string>
#include "KMeansClassifier.cpp"

// Fonction utilitaire pour vérifier si un fichier existe
bool fileExists(const std::string& path) {
    std::ifstream file(path.c_str());
    return file.good();
}

// Fonction pour formater le numéro de fichier (s01, s02, etc.)
std::string formatFileNumber(int num) {
    std::string result = "s";
    if (num < 10) {
        result += "0";
    }
    result += std::to_string(num);
    return result;
}

// Fonction pour obtenir l'extension de fichier selon la méthode
std::string getExtension(const std::string& method) {
    if (method == "ART") return ".art";
    if (method == "E34") return ".e34";
    if (method == "GFD") return ".gfd";
    if (method == "Yang") return ".yng";
    if (method == "Zernike7") return ".zrk.txt";
    return "";
}

// Fonction de lecture des données depuis les dossiers SharvitB2
std::vector<DataPoint> loadData(const std::string& basePath) {
    std::vector<DataPoint> data;
    
    // Noms des sous-dossiers contenant les signatures
    std::vector<std::string> methods = {"ART", "E34", "GFD", "Yang", "Zernike7"};
    
    // Vérifier si le chemin de base existe
    if (!fileExists(basePath)) {
        std::cerr << "Erreur: Le chemin de base n'existe pas: " << basePath << std::endl;
        return data;
    }

    for (const std::string& method : methods) {
        // Construire le chemin avec les séparateurs "="
        std::string methodPath = basePath + "/=" + method;
        
        // Pour chaque classe (s01 à s10)
        for (int i = 1; i <= 10; ++i) {
            // Construire le chemin complet du fichier avec la bonne extension
            std::string filename = formatFileNumber(i) + "n001" + getExtension(method);
            std::string fullPath = methodPath + "/" + filename;
            
            std::cout << "Tentative d'ouverture du fichier: " << fullPath << std::endl;
            
            std::ifstream file(fullPath);
            if (!file.is_open()) {
                std::cerr << "Attention: Impossible d'ouvrir le fichier: " << fullPath << std::endl;
                continue;
            }

            // Lecture du fichier
            DataPoint point;
            point.label = i;  // Assigner le label (1-10)
            
            double value;
            while (file >> value) {
                point.features.push_back(value);
            }
            
            // Vérifier que nous avons lu des caractéristiques
            if (!point.features.empty()) {
                data.push_back(point);
                std::cout << "Fichier lu avec succès: " << filename << " (" 
                         << point.features.size() << " caractéristiques)" << std::endl;
            }
            
            file.close();
        }
    }
    
    // Vérifier si nous avons chargé des données
    if (data.empty()) {
        std::cerr << "Attention: Aucune donnée n'a été chargée!" << std::endl;
    } else {
        std::cout << "Nombre total de points de données chargés: " << data.size() << std::endl;
    }
    
    return data;
}

int main() {
    // Définir le chemin correct vers les fichiers de signatures
    std::string basePath = "data/=SharvitB2/=SharvitB2/=Signatures";
    
    std::cout << "Chargement des données depuis: " << basePath << std::endl;
    
    // Charger les données
    std::vector<DataPoint> trainingData = loadData(basePath);
    
    // Vérifier si nous avons des données avant de continuer
    if (trainingData.empty()) {
        std::cerr << "Erreur: Aucune donnée n'a pu être chargée. Arrêt du programme." << std::endl;
        return 1;
    }
    
    // Initialiser le classifieur K-means avec k=10 clusters et 100 itérations max
    KMeansClassifier kmeans(10, 100);
    
    std::cout << "Début de l'entraînement..." << std::endl;
    
    // Entraîner le modèle sur les données
    kmeans.train(trainingData);
    
    std::cout << "Entraînement terminé. Affichage des résultats:" << std::endl;
    
    // Tester le modèle
    kmeans.testAndDisplayResults(trainingData);
    
    return 0;
}