#include "../include/ClassifierEvaluation.h"
#include "../include/DataPoint.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <map>
#include <fstream>
#include <cmath>
#include <set>
#include <numeric>


std::pair<std::vector<DataPoint>, std::vector<DataPoint>>
ClassifierEvaluation::splitTrainTest(const std::vector<DataPoint>& data, double trainRatio, bool stratified, int minTestSamplesPerClass) {
    std::vector<DataPoint> trainData;
    std::vector<DataPoint> testData;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Stratification
    if (stratified) {
        std::map<int, std::vector<DataPoint>> classMap;

        // Regrouper les données par classe
        for (const auto& point : data) {
            classMap[point.label].push_back(point);
        }

        // Split stratifié
        for (auto& entry : classMap) {
            int label = entry.first;
            std::vector<DataPoint>& classSamples = entry.second;

            // Mélanger les données pour cette classe
            std::shuffle(classSamples.begin(), classSamples.end(), gen);

            // Taille minimale pour le test
            size_t testSize = std::max(static_cast<size_t>(minTestSamplesPerClass), 
                                       static_cast<size_t>(classSamples.size() * (1 - trainRatio)));
            size_t trainSize = classSamples.size() - testSize;

            // Vérifier que les tailles sont valides
            if (testSize > classSamples.size()) {
                std::cerr << "Erreur : pas assez d'échantillons dans la classe " << label << " pour garantir " 
                          << minTestSamplesPerClass << " échantillons dans le jeu de test.\n";
                continue;
            }

            // Ajouter au jeu de données
            trainData.insert(trainData.end(), classSamples.begin(), classSamples.begin() + trainSize);
            testData.insert(testData.end(), classSamples.begin() + trainSize, classSamples.end());
        }
    } else {
        // Standard random split
        std::vector<DataPoint> shuffledData = data;
        std::shuffle(shuffledData.begin(), shuffledData.end(), gen);
        size_t trainSize = static_cast<size_t>(shuffledData.size() * trainRatio);
        trainData.assign(shuffledData.begin(), shuffledData.begin() + trainSize);
        testData.assign(shuffledData.begin() + trainSize, shuffledData.end());
    }

    return {trainData, testData};
}

template <typename Classifier>
void ClassifierEvaluation::testAndDisplayResults(Classifier& classifier, const std::vector<DataPoint>& testData) {
    if (testData.empty()) {
        std::cerr << "Test data is empty." << std::endl;
        return;
    }

    // Déterminer le nombre de classes
    std::set<int> uniqueLabels;
    for (const auto& point : testData) {
        uniqueLabels.insert(point.label);
    }
    int numClasses = uniqueLabels.size();

    // Normalisation des données
    std::vector<DataPoint> normalizedTestData = classifier.normalizeData(testData);

    // Initialiser la matrice de confusion
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));
    int totalPoints = 0;
    int correctAssignments = 0;

    for (const auto& point : normalizedTestData) {
        try {
            int predictedLabel = classifier.predict(point); // Prédiction
            int actualLabel = point.label;                 // Label réel

            if (actualLabel >= 0 && actualLabel < numClasses &&
                predictedLabel >= 0 && predictedLabel < numClasses) {
                confusionMatrix[actualLabel][predictedLabel]++;
                if (predictedLabel == actualLabel) {
                    correctAssignments++;
                }
                totalPoints++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during prediction: " << e.what() << std::endl;
        }
    }

    // Affichage de la matrice de confusion
    displayConfusionMatrix(confusionMatrix);

    // Calcul de la précision globale
    double accuracy = totalPoints > 0 ? (static_cast<double>(correctAssignments) / totalPoints) * 100 : 0;
    std::cout << "\nAccuracy: " << accuracy << "%\n";

    // Calcul des métriques par classe
    double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
    for (int i = 0; i < numClasses; ++i) {
        int truePositive = confusionMatrix[i][i];
        int falsePositive = 0;
        int falseNegative = 0;
        for (int j = 0; j < numClasses; ++j) {
            if (j != i) {
                falsePositive += confusionMatrix[j][i];
                falseNegative += confusionMatrix[i][j];
            }
        }
        int total = truePositive + falsePositive + falseNegative;
        double precision = total > 0 ? static_cast<double>(truePositive) / (truePositive + falsePositive) : 0;
        double recall = total > 0 ? static_cast<double>(truePositive) / (truePositive + falseNegative) : 0;
        double f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;

        totalPrecision += precision;
        totalRecall += recall;
        totalF1 += f1;

        std::cout << "Class " << i + 1 << ": Precision = " << precision * 100
                  << "%, Recall = " << recall * 100 << "%, F1-score = " << f1 * 100 << "%\n";
    }

    // Résultats globaux
    std::cout << "\nMacro Precision: " << (totalPrecision / numClasses) * 100 << "%"
              << ", Macro Recall: " << (totalRecall / numClasses) * 100 << "%"
              << ", Macro F1-score: " << (totalF1 / numClasses) * 100 << "%\n";
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

    // Création des seuils uniques (ajout des limites max et min)
    std::vector<double> thresholds = scores;
    thresholds.push_back(*std::max_element(scores.begin(), scores.end()) + 1); // Au-delà du max
    thresholds.push_back(*std::min_element(scores.begin(), scores.end()) - 1); // En deçà du min
    std::sort(thresholds.begin(), thresholds.end(), std::greater<double>());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

    // Initialisation des variables pour la courbe
    std::vector<std::pair<double, double>> precisionRecallCurve;

    // Nombre total de positifs
    size_t positiveCount = std::count(trueLabels.begin(), trueLabels.end(), 1);

    // Calcul de la précision et du rappel pour chaque seuil
    for (const auto& threshold : thresholds) {
        size_t tp = 0, fp = 0, fn = 0;

        for (size_t i = 1; i < scores.size(); ++i) {
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

        // Calcul de la précision et du rappel
        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        precision = 1.0 - precision; // Inversion de la précision
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / positiveCount : 0.0;

        // Ajout des points valides à la courbe
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
