#include "../include/ClassifierEvaluation.h"
#include "../include/DataPoint.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <map>
#include "../include/PythonHelper.h"


void ClassifierEvaluation::visualizeResults() {
    //  logic for calling Python visualization
    runPythonVisualization();
}


// changing return type to tuple as code in main.cpp is expecteing tuple in return
std::pair<std::vector<DataPoint>, std::vector<DataPoint>>
ClassifierEvaluation::splitTrainTest(const std::vector<DataPoint>& data, double trainRatio) {
    std::vector<DataPoint> trainData;
    std::vector<DataPoint> testData;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Mélange aléatoire des données
    std::vector<DataPoint> shuffledData = data;
    std::shuffle(shuffledData.begin(), shuffledData.end(), gen);

    // Calculer les tailles en fonction du ratio
    size_t totalSize = shuffledData.size();
    size_t trainSize = static_cast<size_t>(totalSize * trainRatio);
    size_t testSize = totalSize - trainSize;

    // Séparer les données en fonction de la taille calculée
    for (size_t i = 0; i < trainSize; ++i) {
        trainData.push_back(shuffledData[i]);
    }
    for (size_t i = trainSize; i < shuffledData.size(); ++i) {
        testData.push_back(shuffledData[i]);
    }

    // Vérification du ratio
    //TODO: Delete when test is done
    double actualTrainRatio = static_cast<double>(trainData.size()) / totalSize;
    std::cout << "Actual Train/Test Ratio: " << actualTrainRatio << " (Train size: " 
              << trainData.size() << ", Test size: " << testData.size() << ") Delete When Test is Done" << std::endl;

    return {trainData, testData};
}

template <typename Classifier>
void ClassifierEvaluation::testAndDisplayResults(Classifier& classifier, const std::vector<DataPoint>& testData) {
    int numClusters = 10;
    if (testData.empty()) {
        std::cerr << "Test data is empty." << std::endl;
        return;
    }

    std::vector<DataPoint> normalizedTestData = classifier.normalizeData(testData);
    std::vector<std::vector<int>> confusionMatrix(numClusters, std::vector<int>(numClusters, 0));
    int totalPoints = 0;
    int correctAssignments = 0;

    for (const auto& point : normalizedTestData) {
        try {
            int predictedCluster = classifier.predict(point);
            int actualLabel = point.label - 1; // Adjust for 0-based indexing

            if (actualLabel >= 0 && actualLabel < numClusters &&
                predictedCluster >= 0 && predictedCluster < numClusters) {
                confusionMatrix[actualLabel][predictedCluster]++;
                if (predictedCluster == actualLabel) {
                    correctAssignments++;
                }
                totalPoints++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during prediction: " << e.what() << std::endl;
        }
    }

    displayConfusionMatrix(confusionMatrix);

    double accuracy = totalPoints > 0 ? (static_cast<double>(correctAssignments) / totalPoints) * 100 : 0;
    std::cout << "\nAccuracy: " << accuracy << "%\n";

    // Calculate precision, recall, and F1-score for each cluster
    for (int i = 0; i < numClusters; ++i) {
        int truePositive = confusionMatrix[i][i];
        int falsePositive = 0;
        int falseNegative = 0;
        for (int j = 0; j < numClusters; ++j) {
            if (j != i) {
                falsePositive += confusionMatrix[j][i];
                falseNegative += confusionMatrix[i][j];
            }
        }
        int total = truePositive + falsePositive + falseNegative;
        double precision = total > 0 ? static_cast<double>(truePositive) / (truePositive + falsePositive) : 0;
        double recall = total > 0 ? static_cast<double>(truePositive) / (truePositive + falseNegative) : 0;
        double f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;

        std::cout << "Cluster " << i + 1 << ": Precision = " << precision * 100
                  << "%, Recall = " << recall * 100 << "%, F1-score = " << f1 * 100 << "%\n";
    }
}

void ClassifierEvaluation::displayConfusionMatrix(const std::vector<std::vector<int>>& matrix) {
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "Predicted →\nActual ↓\n";
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << std::setw(5) << val << " ";
        }
        std::cout << "\n";
    }
}
