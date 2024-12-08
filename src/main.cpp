// Compile: g++ -std=c++17 -I../include main.cpp -o shape_recognition
// Execute: ./shape_recognition

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "ClassifierEvaluation.cpp"
#include "KMeansClassifier.cpp"
#include "KNNClassifier.cpp"
#include "SVMClassifier.cpp"
#include "MLPClassifier.cpp"
#include "DataPoint.h"

// Utility function to check if a file exists
bool fileExists(const std::string &path)
{
    std::ifstream file(path.c_str());
    return file.good();
}

// Function to format the file number (s01, s02, etc.)
std::string formatFileNumber(int num)
{
    std::string result = "s";
    if (num < 10)
    {
        result += "0";
    }
    result += std::to_string(num);
    return result;
}

std::string formatSampleNumber(int sampleNum)
{
    std::ostringstream oss;
    oss << "n" << std::setw(3) << std::setfill('0') << sampleNum;
    return oss.str();
}

// Function to get the file extension based on the method
std::string getExtension(const std::string &method)
{
    if (method == "ART")
        return ".art";
    if (method == "E34")
        return ".e34";
    if (method == "GFD")
        return ".gfd";
    if (method == "Yang")
        return ".yng";
    if (method == "Zernike7")
        return ".zrk.txt";
    return "";
}

// Function to read data for a specific method and store it in a dedicated vector
std::vector<DataPoint> loadMethodData(const std::string &basePath, const std::string &method)
{
    std::vector<DataPoint> methodData;
    std::string methodPath = basePath + method;

    if (!fileExists(methodPath))
    {
        std::cerr << "Error: Method path does not exist: " << methodPath << std::endl;
        return methodData;
    }

    // Parcours des 10 premières classes
    for (int i = 1; i <= 10; ++i)
    {
        for (int j = 1; j <= 12; ++j)
        { // Parcours des 12 échantillons
            std::string filename = formatFileNumber(i) + formatSampleNumber(j) + getExtension(method.substr(1));
            std::string fullPath = methodPath + "/" + filename;

            std::ifstream file(fullPath);
            if (!file.is_open())
            {
                std::cerr << "Warning: Unable to open file: " << fullPath << std::endl;
                continue;
            }

            DataPoint point;
            point.label = i; // Associer la classe (s01 -> 1, s02 -> 2, ...)
            double value;
            while (file >> value)
            {
                point.features.push_back(value);
            }

            if (!point.features.empty())
            {
                methodData.push_back(point);
            }

            file.close();
        }
    }
    return methodData;
}
int main()
{
    try
    {
        std::string basePath = "../data/=SharvitB2/=SharvitB2/=Signatures/";

        // Charger les données pour chaque méthode
        std::vector<DataPoint> artData = loadMethodData(basePath, "=ART");
        std::vector<DataPoint> e34Data = loadMethodData(basePath, "=E34");
        std::vector<DataPoint> gfdData = loadMethodData(basePath, "=GFD");
        std::vector<DataPoint> yangData = loadMethodData(basePath, "=Yang");
        std::vector<DataPoint> zernike7Data = loadMethodData(basePath, "=Zernike7");

        // Data Preparation Strategy Menu
        std::cout << "\nChoose Data Preparation Strategy:" << std::endl;
        std::cout << "1. Standard Split and Train" << std::endl;
        std::cout << "2. Split and Train with Noise Augmentation" << std::endl;
        std::cout << "3. K-Fold Cross-Validation" << std::endl;
        std::cout << "Enter your choice (1/2/3): ";

        int preparationChoice;
        std::cin >> preparationChoice;

        // Vectors to store prepared data
        std::vector<DataPoint> artTrainData, artTestData;
        std::vector<DataPoint> e34TrainData, e34TestData;
        std::vector<DataPoint> gfdTrainData, gfdTestData;
        std::vector<DataPoint> yangTrainData, yangTestData;
        std::vector<DataPoint> zernike7TrainData, zernike7TestData;

        switch (preparationChoice)
        {
        case 1:
        { // Standard Split and Train
            std::tie(artTrainData, artTestData) = ClassifierEvaluation::splitTrainTest(artData, 0.8, true);
            std::tie(e34TrainData, e34TestData) = ClassifierEvaluation::splitTrainTest(e34Data, 0.8, true);
            std::tie(gfdTrainData, gfdTestData) = ClassifierEvaluation::splitTrainTest(gfdData, 0.8, true);
            std::tie(yangTrainData, yangTestData) = ClassifierEvaluation::splitTrainTest(yangData, 0.8, true);
            std::tie(zernike7TrainData, zernike7TestData) = ClassifierEvaluation::splitTrainTest(zernike7Data, 0.8, true);
            break;
        }
        case 2:
        { // Split and Train with Noise Augmentation
            double noiseLevel;
            std::cout << "Enter noise level (recommended 0.01 - 0.1): ";
            std::cin >> noiseLevel;

            // First split, then augment noise in training data
            std::tie(artTrainData, artTestData) = ClassifierEvaluation::splitTrainTest(artData, 0.5, true);
            std::tie(e34TrainData, e34TestData) = ClassifierEvaluation::splitTrainTest(e34Data, 0.5, true);
            std::tie(gfdTrainData, gfdTestData) = ClassifierEvaluation::splitTrainTest(gfdData, 0.5, true);
            std::tie(yangTrainData, yangTestData) = ClassifierEvaluation::splitTrainTest(yangData, 0.5, true);
            std::tie(zernike7TrainData, zernike7TestData) = ClassifierEvaluation::splitTrainTest(zernike7Data, 0.5, true);

            double augmentationFraction;
            std::cout << "Enter the fraction of data to augment (recommended 0.5): ";
            std::cin >> augmentationFraction;
            // Ajouter du bruit et augmenter la taille des données d'entrainement
            artTrainData = ClassifierEvaluation::augmentNoise(artTrainData, noiseLevel, augmentationFraction);
            e34TrainData = ClassifierEvaluation::augmentNoise(e34TrainData, noiseLevel, augmentationFraction);
            gfdTrainData = ClassifierEvaluation::augmentNoise(gfdTrainData, noiseLevel, augmentationFraction);
            yangTrainData = ClassifierEvaluation::augmentNoise(yangTrainData, noiseLevel, augmentationFraction);
            zernike7TrainData = ClassifierEvaluation::augmentNoise(zernike7TrainData, noiseLevel, augmentationFraction);
            break;
        }
        case 3:
        { // K-Fold Cross-Validation
            int kFolds;
            std::cout << "Enter number of folds (recommended 5 or 10): ";
            std::cin >> kFolds;

            // Note: For K-Fold, we'll use full datasets instead of train/test split
            artTrainData = artData;
            e34TrainData = e34Data;
            gfdTrainData = gfdData;
            yangTrainData = yangData;
            zernike7TrainData = zernike7Data;
            break;
        }
        default:
        {
            std::cerr << "Invalid choice. Exiting." << std::endl;
            return 1;
        }
        }

        std::cout << "\nChoose the classification model:" << std::endl;
        std::cout << "1. KMeans" << std::endl;
        std::cout << "2. KNN" << std::endl;
        std::cout << "3. SVM" << std::endl;
        std::cout << "4. MLP (Multi-Layer Perceptron)" << std::endl;
        std::cout << "Enter your choice (1/2/3/4): ";

        int choice;
        std::cin >> choice;

        if (choice < 1 || choice > 4)
        {
            std::cerr << "Invalid choice. Stopping program." << std::endl;
            return 1;
        }

        // Updated lambda to handle different preparation strategies
        auto applyClassifierToAllData = [&](auto &classifier, const std::string &name)
        {
            ClassifierEvaluation evaluator;

            auto processDataset = [&](const std::vector<DataPoint> &trainData,
                                      const std::vector<DataPoint> &testData,
                                      const std::string &datasetName)
            {
                if (preparationChoice == 3)
                {                    // K-Fold
                    int kFolds = 10; // Try for 10 folds, reduce if overfit
                    evaluator.KFoldCrossValidation(classifier, trainData, kFolds, name, datasetName);
                }
                else
                { // Split and Train
                    classifier.train(trainData);
                    evaluator.testAndDisplayResults(classifier, testData);
                    evaluator.evaluateWithPrecisionRecall(classifier, testData, name + "_" + datasetName + ".csv");
                }
            };

            std::cout << "Processing ART data..." << std::endl;
            processDataset(artTrainData, artTestData, "ART");

            std::cout << "Processing E34 data..." << std::endl;
            processDataset(e34TrainData, e34TestData, "E34");

            std::cout << "Processing GFD data..." << std::endl;
            processDataset(gfdTrainData, gfdTestData, "GFD");

            std::cout << "Processing Yang data..." << std::endl;
            processDataset(yangTrainData, yangTestData, "Yang");

            std::cout << "Processing Zernike7 data..." << std::endl;
            processDataset(zernike7TrainData, zernike7TestData, "Zernike7");
        };

        switch (choice)
        {
        case 1:
        {
            KMeansClassifier kmeans(10, 100);
            std::cout << "Starting KMeans..." << std::endl;
            applyClassifierToAllData(kmeans, "KMeans");
            break;
        }
        case 2:
        {
            int kValue;
            std::cout << "Enter the value of K for KNN: ";
            std::cin >> kValue;
            KNNClassifier knn(kValue);
            std::cout << "Starting KNN..." << std::endl;
            applyClassifierToAllData(knn, "KNN");
            break;
        }
        case 3:
        {
            SVMClassifier svm(0.1, 1000);
            std::cout << "Starting SVM..." << std::endl;
            applyClassifierToAllData(svm, "SVM");
            break;
        }
        case 4:
        {
            int numClasses = 10;
            int inputSize = artTrainData[0].features.size();
            int outputSize = numClasses;
            int hiddenSize = 50;

            MLPClassifier mlp(inputSize, hiddenSize, outputSize);
            std::cout << "Starting MLP..." << std::endl;
            applyClassifierToAllData(mlp, "MLP");
            break;
        }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
