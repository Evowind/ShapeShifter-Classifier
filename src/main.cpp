#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "KMeansClassifier.cpp"
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

// Function to read data from SharvitB2 folders
std::vector<DataPoint> loadData(const std::string& basePath) {
    std::vector<DataPoint> data;

    // Names of subfolders containing signatures
    std::vector<std::string> methods = {"=ART", "=E34", "=GFD", "=Yang", "=Zernike7"};
    
    // Check if the base path exists
    std::cout << "Checking path: " << basePath << std::endl;
    if (!fileExists(basePath)) {
        std::cerr << "Error: Base path does not exist: " << basePath << std::endl;
        return data;
    }

    for (const std::string& method : methods) {
        // Build path with separators "="
        std::string methodPath = basePath + method;
        
        std::cout << "Attempting to load from: " << methodPath << std::endl;
        // Check if the method path exists
        if (!fileExists(methodPath)) {
            std::cerr << "Error: Method path does not exist: " << methodPath << std::endl;
            continue;
        }

        // For each class (s01 to s10)
        for (int i = 1; i <= 10; ++i) {
            // Build full path with the correct extension
            std::string filename = formatFileNumber(i) + "n001" + getExtension(method.substr(1)); // Remove '=' for correct file type
            std::string fullPath = methodPath + "/" + filename;
            
            std::cout << "Attempting to open file: " << fullPath << std::endl;
            
            std::ifstream file(fullPath);
            if (!file.is_open()) {
                std::cerr << "Warning: Unable to open file: " << fullPath << std::endl;
                continue;
            }

            // Read the file
            DataPoint point;
            point.label = i;  // Assign label (1-10)
            
            double value;
            while (file >> value) {
                point.features.push_back(value);
            }
            
            // Check that we read some features
            if (!point.features.empty()) {
                data.push_back(point);
                std::cout << "File read successfully: " << filename << " (" 
                         << point.features.size() << " features)" << std::endl;
            }
            
            file.close();
        }
    }
    
    // Check if we loaded some data
    if (data.empty()) {
        std::cerr << "Warning: No data loaded!" << std::endl;
    } else {
        std::cout << "Total number of loaded data points: " << data.size() << std::endl;
    }
    
    return data;
}

int main() {
    try {
        // Define the correct path to the signature files
        std::string basePath = "../data/=SharvitB2/=SharvitB2/=Signatures/";

        std::cout << "Loading data from: " << basePath << std::endl;

        // Load the data
        std::vector<DataPoint> trainingData = loadData(basePath);

        // Check if we have some data before continuing
        if (trainingData.empty()) {
            std::cerr << "Error: No data could be loaded. Stopping program." << std::endl;
            return 1;
        }

        // Display information about the loaded data
        std::cout << "Data loaded successfully:" << std::endl;
        std::cout << "Number of points: " << trainingData.size() << std::endl;
        if (!trainingData.empty()) {
            std::cout << "Number of features per point: " << trainingData[0].features.size() << std::endl;
        }

        // Choose the model
        std::cout << "\nChoose the classification model:" << std::endl;
        std::cout << "1. KMeans" << std::endl;
        std::cout << "2. KNN" << std::endl;
        std::cout << "3. SVM" << std::endl;
        std::cout << "Enter your choice (1/2/3): ";
        
        int choice;
        std::cin >> choice;

        if (choice < 1 || choice > 3) {
            std::cerr << "Invalid choice. Stopping program." << std::endl;
            return 1;
        }

        // Initialize and run the chosen classifier
        switch (choice) {
            case 1: {
                KMeansClassifier kmeans(10, 100);
                std::cout << "Starting KMeans training..." << std::endl;
                kmeans.train(trainingData);
                std::cout << "Evaluating results..." << std::endl;
                kmeans.testAndDisplayResults(trainingData);
                break;
            }
            case 2:
                std::cout << "Warning: Training with KNN is not yet implemented." << std::endl;
                break;
            case 3:
                std::cout << "Warning: Training with SVM is not yet implemented." << std::endl;
                break;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
