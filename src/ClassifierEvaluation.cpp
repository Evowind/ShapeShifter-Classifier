#include "../include/ClassifierEvaluation.h"
#include "../include/DataPoint.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <map>
#include <fstream>
#include <numeric>

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

void ClassifierEvaluation::computePrecisionRecallCurve(
    const std::vector<DataPoint>& testData,
    const std::vector<double>& scores,
    const std::vector<int>& trueLabels,
    const std::string& outputCsvPath) {
    // Validation des tailles des données
    if (scores.size() != trueLabels.size()) {
        throw std::invalid_argument("Scores and true labels must have the same size.");
    }

    // Création des seuils uniques
    std::vector<double> thresholds = scores;
    std::sort(thresholds.begin(), thresholds.end(), std::greater<double>());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

    // Calcul de la précision et du rappel pour chaque seuil
    std::vector<std::pair<double, double>> precisionRecallCurve;
    size_t positiveCount = std::count(trueLabels.begin(), trueLabels.end(), 1);

    for (const auto& threshold : thresholds) {
        size_t tp = 0, fp = 0, fn = 0;

        for (size_t i = 0; i < scores.size(); ++i) {
            bool predictedPositive = scores[i] >= threshold;
            bool actualPositive = (trueLabels[i] == 1);

            if (predictedPositive && actualPositive) {
                tp++;
            } else if (predictedPositive && !actualPositive) {
                fp++;
            } else if (!predictedPositive && actualPositive) {
                fn++;
            }
        }

        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / positiveCount : 0.0;
        precisionRecallCurve.emplace_back(recall, precision);
    }

    // Calcul de l'AUC (approximation par somme de trapèzes)
    double auc = 0.0;
    for (size_t i = 1; i < precisionRecallCurve.size(); ++i) {
        double xDiff = precisionRecallCurve[i].first - precisionRecallCurve[i - 1].first;
        double yAvg = (precisionRecallCurve[i].second + precisionRecallCurve[i - 1].second) / 2;
        auc += xDiff * yAvg;
    }

    std::cout << "AUC: " << auc << std::endl;

    // Assurez-vous que le dossier ../curve existe
    std::string folderPath = "../curve";
    if (std::filesystem::create_directory(folderPath)) {
        std::cout << "Le dossier ../curve a été créé." << std::endl;
    }

    // Générer le chemin final du fichier CSV
    std::string fullCsvPath = folderPath + "/" + outputCsvPath;

    // Exporter la courbe précision/rappel au format CSV
    std::ofstream csvFile(fullCsvPath);
    if (!csvFile.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + fullCsvPath);
    }

    csvFile << "Recall,Precision\n";
    for (const auto& point : precisionRecallCurve) {
        csvFile << point.first << "," << point.second << "\n";
    }

    csvFile.close();
}

template <typename Classifier>
void ClassifierEvaluation::evaluateWithPrecisionRecall(
    const Classifier& classifier,
    const std::vector<DataPoint>& testData,
    const std::string& outputCsvPath) {
    std::vector<double> scores;
    std::vector<int> trueLabels;

    for (const auto& point : testData) {
        // Ajoute uniquement le score du `predictWithScore`
        scores.push_back(classifier.predictWithScore(point).second);
        trueLabels.push_back(point.label);
    }

    computePrecisionRecallCurve(testData, scores, trueLabels, outputCsvPath);
}
