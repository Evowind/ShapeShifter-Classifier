//Compile: g++ -std=c++17 main.cpp -o shape_recognition
//Execute: ./shape_recognition

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "KMeansClassifier.cpp"
#include "KNNClassifier.h" // Include KNNClassifier
// Note: Include the header files, not the .cpp files
//#include "KNNClassifier.h"  // Uncomment when implemented
//#include "SVMClassifier.h"  // Uncomment when implemented

// Utility function to check if a file exists
bool fileExists(const std::string& path) {
    std::ifstream file(path.c_str());
    return file.good();
}

// Function to format the file number (s01, s02, etc.)
std::string formatFileNumber(int num) {
    std::string result = "s";
    if (num < 10) {
        result += "0";
    }
    result += std::to_string(num);
    return result;
}

// Function to get the file extension based on the method
std::string getExtension(const std::string& method) {
    if (method == "ART") return ".art";
    if (method == "E34") return ".e34";
    if (method == "GFD") return ".gfd";
    if (method == "Yang") return ".yng";
    if (method == "Zernike7") return ".zrk.txt";
    return "";
}

// Function to read data for a specific method and store it in a dedicated vector
std::vector<DataPoint> loadMethodData(const std::string& basePath, const std::string& method) {
    std::vector<DataPoint> methodData;
    std::string methodPath = basePath + method;

    if (!fileExists(methodPath)) {
        std::cerr << "Error: Method path does not exist: " << methodPath << std::endl;
        return methodData;
    }

    for (int i = 1; i <= 10; ++i) {
        std::string filename = formatFileNumber(i) + "n001" + getExtension(method.substr(1)); // Remove '=' for correct file type
        std::string fullPath = methodPath + "/" + filename;

        std::ifstream file(fullPath);
        if (!file.is_open()) {
            std::cerr << "Warning: Unable to open file: " << fullPath << std::endl;
            continue;
        }

        DataPoint point;
        point.label = i;
        double value;
        while (file >> value) {
            point.features.push_back(value);
        }

        if (!point.features.empty()) {
            methodData.push_back(point);
        }

        file.close();
    }
    return methodData;
}

int main() {
    try {
        std::string basePath = "../data/=SharvitB2/=SharvitB2/=Signatures/";

        // Separate vectors for each method
        std::vector<DataPoint> artData = loadMethodData(basePath, "=ART");
        std::vector<DataPoint> e34Data = loadMethodData(basePath, "=E34");
        std::vector<DataPoint> gfdData = loadMethodData(basePath, "=GFD");
        std::vector<DataPoint> yangData = loadMethodData(basePath, "=Yang");
        std::vector<DataPoint> zernike7Data = loadMethodData(basePath, "=Zernike7");

        std::cout << "\nChoose the classification model or comparison mode:" << std::endl;
        std::cout << "1. KMeans" << std::endl;
        std::cout << "2. KNN" << std::endl;
        std::cout << "3. SVM" << std::endl;
        std::cout << "4. Compare all classifiers" << std::endl;
        std::cout << "Enter your choice (1/2/3/4): ";

        int choice;
        std::cin >> choice;

        if (choice < 1 || choice > 4) {
            std::cerr << "Invalid choice. Stopping program." << std::endl;
            return 1;
        }

        // Define a lambda to apply a classifier to all datasets for comparison mode
        auto applyClassifierToAllData = [&](auto& classifier, const std::string& name) {
            std::cout << "Training and testing " << name << " on ART data..." << std::endl;
            classifier.train(artData);
            classifier.testAndDisplayResults(artData);

            std::cout << "Training and testing " << name << " on E34 data..." << std::endl;
            classifier.train(e34Data);
            classifier.testAndDisplayResults(e34Data);

            std::cout << "Training and testing " << name << " on GFD data..." << std::endl;
            classifier.train(gfdData);
            classifier.testAndDisplayResults(gfdData);

            std::cout << "Training and testing " << name << " on Yang data..." << std::endl;
            classifier.train(yangData);
            classifier.testAndDisplayResults(yangData);

            std::cout << "Training and testing " << name << " on Zernike7 data..." << std::endl;
            classifier.train(zernike7Data);
            classifier.testAndDisplayResults(zernike7Data);
        };

        switch (choice) {
            case 1: {
                KMeansClassifier kmeans(10, 100);
                std::cout << "Starting KMeans training on data..." << std::endl;
                applyClassifierToAllData(kmeans, "KMeans");
                break;
            }
            case 2: {
                // std::cout << "Warning: Training with KNN is not yet implemented." << std::endl;
                KNNClassifier knn(3);
                std::cout<<"Starting KNN Classification on data..."<<std :: endl;
                applyClassifierToAllData(knn, "KNN");
                
                break;
            }
            case 3:
                std::cout << "Warning: Training with SVM is not yet implemented." << std::endl;
                break;
            case 4:
                std::cout << "Comparing all classifiers..." << std::endl;
                {
                    KMeansClassifier kmeans(10, 100);
                    applyClassifierToAllData(kmeans, "KMeans");

                    // Uncomment and implement once KNN and SVM are available
                    // KNNClassifier knn;
                    // applyClassifierToAllData(knn, "KNN");

                    // SVMClassifier svm;
                    // applyClassifierToAllData(svm, "SVM");
                }
                // Comparison with KNN
                  {
                      KNNClassifier knn(3); // Initialize KNN with k = 3
                      std::cout << "\n--- KNN Results ---" << std::endl;
                      applyClassifierToAllData(knn, "KNN");
                  }
                break;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    
    
    
    
    
    
    
}
