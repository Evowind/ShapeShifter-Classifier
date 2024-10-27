#include <iostream>
#include <vector>
#include <string>
#include "KMeansClassifier.cpp"
#include "KNNClassifier.cpp"
#include "SVMClassifier.cpp"

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
    std::vector<std::string> methods = {"=ART", "=E34", "=GFD", "=Yang", "=Zernike7"};
    
    // Vérifier si le chemin de base existe
    std::cout << "Vérification du chemin: " << basePath << std::endl;
    if (!fileExists(basePath)) {
        std::cerr << "Erreur: Le chemin de base n'existe pas: " << basePath << std::endl;
        return data;
    }

    for (const std::string& method : methods) {
        // Construire le chemin avec les séparateurs "="
        std::string methodPath = basePath + method;
        
        std::cout << "Tentative de chargement à partir de: " << methodPath << std::endl;
        // Vérifiez si le chemin de méthode existe
        if (!fileExists(methodPath)) {
            std::cerr << "Erreur: Le chemin de méthode n'existe pas: " << methodPath << std::endl;
            continue;
        }

        // Pour chaque classe (s01 à s10)
        for (int i = 1; i <= 10; ++i) {
            // Construire le chemin complet du fichier avec la bonne extension
            std::string filename = formatFileNumber(i) + "n001" + getExtension(method.substr(1)); // Retirer '=' pour obtenir le bon type de fichier
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
    std::string basePath = "../data/=SharvitB2/=SharvitB2/=Signatures/";

    std::cout << "Chargement des données depuis: " << basePath << std::endl;

    // Charger les données
    std::vector<DataPoint> trainingData = loadData(basePath);

    // Vérifier si nous avons des données avant de continuer
    if (trainingData.empty()) {
        std::cerr << "Erreur: Aucune donnée n'a pu être chargée. Arrêt du programme." << std::endl;
        return 1;
    }

    // Choisir le modèle
    std::cout << "Choisissez le modèle de classification:" << std::endl;
    std::cout << "1. KMeans" << std::endl;
    std::cout << "2. KNN" << std::endl;
    std::cout << "3. SVM" << std::endl;
    std::cout << "Entrez votre choix (1/2/3): ";
    
    int choice;
    std::cin >> choice;

    if (choice < 1 || choice > 3) {
        std::cerr << "Choix invalide. Arrêt du programme." << std::endl;
        return 1;
    }

    // Initialiser le classifieur choisi
    if (choice == 1) {
        KMeansClassifier kmeans(10, 100);
        std::cout << "Début de l'entraînement avec KMeans..." << std::endl;
        kmeans.train(trainingData);
        kmeans.testAndDisplayResults(trainingData);
    } else if (choice == 2) {
        std::cout << "Attention: L'entraînement avec KNN n'est pas implémenté. Arrête du programme." << std::endl;
        /*
        KNNClassifier knn;  // Assurez-vous de passer les paramètres nécessaires
        std::cout << "Début de l'entraînement avec KNN..." << std::endl;
        knn.train(trainingData);
        knn.testAndDisplayResults(trainingData);
        */
    } else if (choice == 3) {
        std::cout << "Attention: L'entraînement avec SVM n'est pas implémenté. Arrêt du programme." << std::endl;
        /*
        SVMClassifier svm;  // Assurez-vous de passer les paramètres nécessaires
        std::cout << "Début de l'entraînement avec SVM..." << std::endl;
        svm.train(trainingData);
        svm.testAndDisplayResults(trainingData);
        */
    }

    return 0;
}