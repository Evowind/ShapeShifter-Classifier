// Compile: g++ -std=c++17 -I../include main.cpp -o shape_recognition
// Execute: ./shape_recognition
// Both: g++ -std=c++17 -I../include main.cpp -o shape_recognition && ./shape_recognition 

#include <iostream>                 // for I/O operations like cout/cin
#include <vector>                   // for dynamic arrays
#include <string>                   // for string manipulation
#include <fstream>                  // for file I/O (ifstream)
#include <sstream>                  // for string streams
#include <filesystem>               // for file/directory operations
#include "ClassifierEvaluation.cpp" // includes evaluation functions
#include "KMeansClassifier.cpp"     // includes KMeans model
#include "KNNClassifier.cpp"        // includes KNN model
#include "SVMClassifier.cpp"        // includes SVM model
#include "MLPClassifier.cpp"        // includes MLP model
#include "../include/DataPoint.h"   // custom class for storing data points

// Utility function to check if a file exists
bool fileExists(const std::string &path)
{
    std::ifstream file(path.c_str()); // Open file
    return file.good();               // Return true if file exists, false if not
}

// Function to format the file number (e.g., s01, s02, etc.)
std::string formatFileNumber(int num)
{
    std::string result = "s"; // Initialize string with 's' prefix
    if (num < 10)             // If the number is less than 10, add leading zero
    {
        result += "0";
    }
    result += std::to_string(num); // Convert the number to a string and append
    return result;                 // Return formatted string (e.g., "s01", "s02")
}

// Function to format the sample number (e.g., n001, n002, etc.)
std::string formatSampleNumber(int sampleNum)
{
    std::ostringstream oss;                                       // Create an output string stream
    oss << "n" << std::setw(3) << std::setfill('0') << sampleNum; // Format number with leading zeros
    return oss.str();                                             // Return formatted sample number as string (e.g., "n001", "n002")
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
    return ""; // Return empty string if no match found
}

// Function to read data for a specific method and store it in a dedicated vector
std::vector<DataPoint> loadMethodData(const std::string &basePath, const std::string &method)
{
    std::vector<DataPoint> methodData;          // Vector to store the data for the method
    std::string methodPath = basePath + method; // Build path to method folder

    if (!fileExists(methodPath)) // Check if the method folder exists
    {
        std::cerr << "Error: Method path does not exist: " << methodPath << std::endl;
        return methodData; // Return empty data if path is not found
    }

    // Loop through 10 classes (s01, s02, ..., s10)
    for (int i = 1; i <= 10; ++i)
    {
        for (int j = 1; j <= 12; ++j) // Loop through 12 samples per class
        {
            // Format the filename (e.g., s01n001.art, s01n002.art, ...)
            std::string filename = formatFileNumber(i) + formatSampleNumber(j) + getExtension(method.substr(1));
            std::string fullPath = methodPath + "/" + filename; // Full file path

            std::ifstream file(fullPath); // Open the file
            if (!file.is_open())          // Check if the file opened successfully
            {
                std::cerr << "Warning: Unable to open file: " << fullPath << std::endl;
                continue; // Skip this file if it cannot be opened
            }

            DataPoint point; // Create a DataPoint object to store features
            point.label = i; // Assign the class label (1, 2, ..., 10)
            double value;
            while (file >> value) // Read each value in the file
            {
                point.features.push_back(value); // Store the value in the features vector
            }

            if (!point.features.empty()) // If the features vector is not empty
            {
                methodData.push_back(point); // Add the DataPoint to the methodData vector
            }

            file.close(); // Close the file after reading
        }
    }
    return methodData; // Return the vector containing all the method data
}
int main()
{
    try
    {
        // Base path to the dataset
        std::string basePath = "../data/=SharvitB2/=SharvitB2/=Signatures/";

        // Load the data for each method
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

        // Declare vectors to store prepared data for each method
        std::vector<DataPoint> artTrainData, artTestData;
        std::vector<DataPoint> e34TrainData, e34TestData;
        std::vector<DataPoint> gfdTrainData, gfdTestData;
        std::vector<DataPoint> yangTrainData, yangTestData;
        std::vector<DataPoint> zernike7TrainData, zernike7TestData;

        // Switch statement to handle different data preparation strategies
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
        { // K-Fold Cross-Validation (e.g., for k = 5, split data into 5 folds)

            int kFolds;
            std::cout << "Enter number of folds (recommended 5 or 10): ";
            std::cin >> kFolds;

            // /!\ Warning: For K-Fold, we'll use full datasets instead of train/test split
            artTrainData = artData;
            e34TrainData = e34Data;
            gfdTrainData = gfdData;
            yangTrainData = yangData;
            zernike7TrainData = zernike7Data;
            break;
        }
        default:
        {
            // Print error message for invalid choice and exit
            std::cerr << "Invalid choice. Exiting." << std::endl;
            return 1;
        }
        }

        // Prompt the user to choose a classification model
        std::cout << "\nChoose the classification model:" << std::endl;
        std::cout << "1. KMeans" << std::endl;
        std::cout << "2. KNN" << std::endl;
        std::cout << "3. SVM" << std::endl;
        std::cout << "4. MLP (Multi-Layer Perceptron)" << std::endl;
        std::cout << "Enter your choice (1/2/3/4): ";

        int choice;
        std::cin >> choice;

        // Check if the choice is valid
        if (choice < 1 || choice > 4)
        {
            std::cerr << "Invalid choice. Stopping program." << std::endl;
            return 1;
        }

        // Lambda function to apply the classifier on all datasets
        auto applyClassifierToAllData = [&](auto &classifier, const std::string &name)
        {
            ClassifierEvaluation evaluator;

            // Lambda function to process a single dataset
            auto processDataset = [&](const std::vector<DataPoint> &trainData,
                                      const std::vector<DataPoint> &testData,
                                      const std::string &datasetName)
            {
                if (preparationChoice == 3)
                {
                    // Perform K-Fold Cross-Validation
                    int kFolds = 10; // Number of folds for cross-validation
                    evaluator.KFoldCrossValidation(classifier, trainData, kFolds, name, datasetName);
                }
                else
                {
                    // Train and test the classifier
                    classifier.train(trainData);
                    evaluator.testAndDisplayResults(classifier, testData);
                    evaluator.evaluateWithPrecisionRecall(classifier, testData, name + "_" + datasetName + ".csv");
                }
            };

            // Process each method's dataset
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

        // Switch statement to handle different classifier choices
        switch (choice)
        {
        case 1:
        {
            // Initialize and apply KMeans classifier
            KMeansClassifier kmeans(10, 100);
            std::cout << "Starting KMeans..." << std::endl;
            applyClassifierToAllData(kmeans, "KMeans");
            break;
        }
        case 2:
        {
            // Initialize and apply KNN classifier
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
            // Initialize and apply SVM classifier
            SVMClassifier svm(0.1, 1000);
            std::cout << "Starting SVM..." << std::endl;
            applyClassifierToAllData(svm, "SVM");
            break;
        }
        case 4:
        {
            // Initialize and apply MLP classifier
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
        // Catch and report any exceptions
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
