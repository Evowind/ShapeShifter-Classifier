//Compile: g++ -std=c++17 main.cpp -o shape_recognition
//Execute: ./shape_recognition

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "../include/DataPoint.h"
#include "../include/SVMClassifier.h"
#include "../include/KNNClassifier.h"
#include "../include/KMeansClassifier.h"
#include "../include/PythonHelper.h"
#include "../include/ClassifierEvaluation.h"
/*
 #include "ClassifierEvaluation.cpp"
 #include "KMeansClassifier.cpp"
 #include "KNNClassifier.cpp"  // Uncomment when implemented
 #include "SVMClassifier.cpp"
 */

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

std::string formatSampleNumber(int sampleNum) {
    std::ostringstream oss;
    oss << "n" << std::setw(3) << std::setfill('0') << sampleNum;
    return oss.str();
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

    // Parcours des 10 premières classes
    for (int i = 1; i <= 10; ++i) {
        for (int j = 1; j <= 12; ++j) {  // Parcours des 12 échantillons
            std::string filename = formatFileNumber(i) + formatSampleNumber(j) + getExtension(method.substr(1));
            std::string fullPath = methodPath + "/" + filename;

            std::ifstream file(fullPath);
            if (!file.is_open()) {
                std::cerr << "Warning: Unable to open file: " << fullPath << std::endl;
                continue;
            }

            DataPoint point;
            point.label = i;  // Associer la classe (s01 -> 1, s02 -> 2, ...)
            double value;
            while (file >> value) {
                point.features.push_back(value);
            }

            if (!point.features.empty()) {
                methodData.push_back(point);
            }

            file.close();
        }
    }
    return methodData;
}

int main() {
    runPythonVisualization();
     
    try {
        std::string basePath = "../data/=SharvitB2/=SharvitB2/=Signatures/";

        // Separate vectors for each method
        std::vector<DataPoint> artData = loadMethodData(basePath, "=ART");
        std::vector<DataPoint> e34Data = loadMethodData(basePath, "=E34");
        std::vector<DataPoint> gfdData = loadMethodData(basePath, "=GFD");
        std::vector<DataPoint> yangData = loadMethodData(basePath, "=Yang");
        std::vector<DataPoint> zernike7Data = loadMethodData(basePath, "=Zernike7");


        // Using ClassifierEvaluation to split data
        auto [artTrain, artTest] = ClassifierEvaluation::splitTrainTest(artData);
        auto [e34Train, e34Test] = ClassifierEvaluation::splitTrainTest(e34Data);
        auto [gfdTrain, gfdTest] = ClassifierEvaluation::splitTrainTest(gfdData);
        auto [yangTrain, yangTest] = ClassifierEvaluation::splitTrainTest(yangData);
        auto [zernike7Train, zernike7Test] = ClassifierEvaluation::splitTrainTest(zernike7Data);

        
        
  
        
        /*
        // Unpacking Tuple
        std::tuple<std::vector<DataPoint>, std::vector<DataPoint>> artTrainTest = ClassifierEvaluation::splitTrainTest(artData);
        std::vector<DataPoint> artTrain = std::get<0>(artTrainTest);
        std::vector<DataPoint> artTest = std::get<1>(artTrainTest);

        std::tuple<std::vector<DataPoint>, std::vector<DataPoint>> e34TrainTest = ClassifierEvaluation::splitTrainTest(e34Data);
        std::vector<DataPoint> e34Train = std::get<0>(e34TrainTest);
        std::vector<DataPoint> e34Test = std::get<1>(e34TrainTest);

        std::tuple<std::vector<DataPoint>, std::vector<DataPoint>> gfdTrainTest = ClassifierEvaluation::splitTrainTest(gfdData);
        std::vector<DataPoint> gfdTrain = std::get<0>(gfdTrainTest);
        std::vector<DataPoint> gfdTest = std::get<1>(gfdTrainTest);

        std::tuple<std::vector<DataPoint>, std::vector<DataPoint>> yangTrainTest = ClassifierEvaluation::splitTrainTest(yangData);
        std::vector<DataPoint> yangTrain = std::get<0>(yangTrainTest);
        std::vector<DataPoint> yangTest = std::get<1>(yangTrainTest);

        std::tuple<std::vector<DataPoint>, std::vector<DataPoint>> zernike7TrainTest = ClassifierEvaluation::splitTrainTest(zernike7Data);
        std::vector<DataPoint> zernike7Train = std::get<0>(zernike7TrainTest);
        std::vector<DataPoint> zernike7Test = std::get<1>(zernike7TrainTest);
        */

        
        
        
        
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
            auto applyClassifierToAllData = [artTrain, artTest, e34Train, e34Test, gfdTrain, gfdTest, yangTrain, yangTest, zernike7Train, zernike7Test](auto& classifier, const std::string& name) { 
            std::cout << "Training and testing " << name << " on ART data..." << std::endl;
            classifier.train(artTrain);
            ClassifierEvaluation::testAndDisplayResults(classifier, artTest);

            std::cout << "Training and testing " << name << " on E34 data..." << std::endl;
            classifier.train(e34Train);
            ClassifierEvaluation::testAndDisplayResults(classifier, e34Test);

            std::cout << "Training and testing " << name << " on GFD data..." << std::endl;
            classifier.train(gfdTrain);
            ClassifierEvaluation::testAndDisplayResults(classifier, gfdTest);

            std::cout << "Training and testing " << name << " on Yang data..." << std::endl;
            classifier.train(yangTrain);
            ClassifierEvaluation::testAndDisplayResults(classifier, yangTest);

            std::cout << "Training and testing " << name << " on Zernike7 data..." << std::endl;
            classifier.train(zernike7Train);
            ClassifierEvaluation::testAndDisplayResults(classifier, zernike7Test);
        };

        switch (choice) {
            case 1: {
                KMeansClassifier kmeans(10, 100);
                std::cout << "Starting KMeans training on data..." << std::endl;
                applyClassifierToAllData(kmeans, "KMeans");
                break;
            }
            case 2: {
                int kValue;
                std::cout << "Enter the value of K for KNN: ";
                std::cin >> kValue;

                KNNClassifier knn(kValue);
                std::cout << "Training KNN on data..." << std::endl;
                applyClassifierToAllData(knn, "KNN");
                break;
            }
            case 3: {
                SVMClassifier svm(0.01, 1000);
                std::cout << "Starting SVM training on data..." << std::endl;
                applyClassifierToAllData(svm, "SVM");
                break;
            }
            case 4:
                std::cout << "Comparing all classifiers..." << std::endl;
                {
                    KMeansClassifier kmeans(10, 100);
                    applyClassifierToAllData(kmeans, "KMeans");

                    // Uncomment and implement once KNN and SVM are available
                    // KNNClassifier knn;
                    // applyClassifierToAllData(knn, "KNN");

                    /*
                    SVMClassifier svm;
                    applyClassifierToAllData(svm, "SVM");
                    */
                }
                break;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
