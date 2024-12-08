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
#include <filesystem>
#include <vector>

// Function to split data into train and test sets
std::pair<std::vector<DataPoint>, std::vector<DataPoint>>
ClassifierEvaluation::splitTrainTest(const std::vector<DataPoint> &data, double trainRatio, bool stratified, int minTestSamplesPerClass)
{
    std::vector<DataPoint> trainData;
    std::vector<DataPoint> testData;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Stratification
    if (stratified)
    {
        std::map<int, std::vector<DataPoint>> classMap;

        // Group data by class
        for (const auto &point : data)
        {
            classMap[point.label].push_back(point);
        }

        for (auto &entry : classMap)
        {
            int label = entry.first;
            std::vector<DataPoint> &classSamples = entry.second;

            // Shuffle the data for this class
            std::shuffle(classSamples.begin(), classSamples.end(), gen);

            // Determine test and train sizes
            size_t testSize = std::max(static_cast<size_t>(minTestSamplesPerClass),
                                       static_cast<size_t>(classSamples.size() * (1 - trainRatio)));
            size_t trainSize = classSamples.size() - testSize;

            if (testSize > classSamples.size())
            {
                std::cerr << "Error: Not enough samples in class " << label
                          << " to guarantee " << minTestSamplesPerClass << " test samples.\n";
                continue;
            }

            trainData.insert(trainData.end(), classSamples.begin(), classSamples.begin() + trainSize);
            testData.insert(testData.end(), classSamples.begin() + trainSize, classSamples.end());

            // Debugging class-wise split
            std::cout << "Class " << label << ": Total = " << classSamples.size()
                      << ", Train = " << trainSize << ", Test = " << testSize << "\n";
        }
    }
    else
    {
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
void ClassifierEvaluation::KFoldCrossValidation(
    Classifier &classifier,
    const std::vector<DataPoint> &data,
    int k,
    const std::string &name,
    const std::string &datasetName)
{
    // Créer une copie non-const du vecteur de données
    std::vector<DataPoint> dataCopy = data;

    // Initialisation des folds
    std::vector<std::vector<DataPoint>> folds(k);

    // Initialisation du générateur aléatoire
    std::random_device rd;
    std::mt19937 gen(rd());

    // Mélanger les données (en utilisant la copie non-const)
    std::shuffle(dataCopy.begin(), dataCopy.end(), gen);

    // Diviser les données en k folds
    for (size_t i = 0; i < dataCopy.size(); ++i)
    {
        folds[i % k].push_back(dataCopy[i]);
    }

    double totalAccuracy = 0;

    // Variables pour stocker les scores et étiquettes
    std::vector<double> allScores;
    std::vector<int> allTrueLabels;

    // Pour chaque fold, entraîner sur k-1 folds et tester sur le fold restant
    for (int i = 0; i < k; ++i)
    {
        std::vector<DataPoint> trainData;
        std::vector<DataPoint> testData = folds[i];

        // Combiner tous les autres folds pour les données d'entraînement
        for (int j = 0; j < k; ++j)
        {
            if (i != j)
            {
                trainData.insert(trainData.end(), folds[j].begin(), folds[j].end());
            }
        }

        // Entraîner le classifieur
        classifier.train(trainData);

        // Tester le classifieur sur les données du fold de test
        for (const auto &point : testData)
        {
            auto [predictedLabel, score] = classifier.predictWithScore(point);
            allScores.push_back(score);
            allTrueLabels.push_back(point.label);
        }

        // Calculer la précision pour ce fold
        double foldAccuracy = computeAccuracy(classifier, testData);
        totalAccuracy += foldAccuracy;
    }

    // Calculer la précision moyenne
    double averageAccuracy = totalAccuracy / k;
    std::cout << "Average Accuracy across " << k << " folds: " << averageAccuracy << "%\n";

    // Générer la courbe précision/rappel et sauvegarder dans un fichier CSV
    computePrecisionRecallCurve(
        allTrueLabels,
        allScores,
        name + "_" + datasetName + ".csv");
}

// Compute accuracy based on correct predictions and total predictions
template <typename Classifier>
double ClassifierEvaluation::computeAccuracy(Classifier &classifier, const std::vector<DataPoint> &testData)
{
    int correctPredictions = 0;
    int totalPredictions = 0;

    for (const auto &point : testData)
    {
        try
        {
            int predictedLabel = classifier.predict(point); // Get the predicted label
            int actualLabel = point.label;                  // Get the actual label

            if (predictedLabel == actualLabel)
            {
                ++correctPredictions; // Increment if prediction is correct
            }
            ++totalPredictions; // Increment total predictions
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during prediction for label " << point.label
                      << ": " << e.what() << "\n";
        }
    }

    if (totalPredictions == 0)
    {
        std::cerr << "Error: No predictions were made.\n";
        return 0.0; // Avoid division by zero
    }

    // Calculate accuracy as the percentage of correct predictions
    double accuracy = static_cast<double>(correctPredictions) / totalPredictions * 100;
    return accuracy;
}

// Add Gaussian noise to the features of each DataPoint
std::vector<DataPoint> ClassifierEvaluation::augmentNoise(
    const std::vector<DataPoint> &data, double noiseLevel, double augmentationFraction)
{

    std::vector<DataPoint> augmentedData = data; // Garder les données originales

    int numAugmented = static_cast<int>(data.size() * augmentationFraction);

    for (int i = 0; i < numAugmented; ++i)
    {
        const auto &point = data[i % data.size()];
        DataPoint noisyPoint = point;

        for (auto &feature : noisyPoint.features)
        {
            feature += (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2 * noiseLevel;
        }

        augmentedData.push_back(noisyPoint);
    }

    std::cout << "Augmented data train size: " << augmentedData.size() << "\n";
    return augmentedData;
}

// Function to test and display results
template <typename Classifier>
void ClassifierEvaluation::testAndDisplayResults(Classifier &classifier, const std::vector<DataPoint> &testData)
{
    if (testData.empty())
    {
        std::cerr << "Test data is empty.\n";
        return;
    }

    // Determine the number of classes (from 1 to 10)
    int numClasses = 10;

    // Normalize data
    std::vector<DataPoint> normalizedTestData = classifier.normalizeData(testData);
    if (normalizedTestData.size() != testData.size())
    {
        std::cerr << "Warning: Normalized test data size (" << normalizedTestData.size()
                  << ") does not match original test data size (" << testData.size() << ").\n";
    }

    // Initialize confusion matrix
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));
    int totalPoints = 0;
    int correctAssignments = 0;

    for (const auto &point : normalizedTestData)
    {
        try
        {
            int predictedLabel = classifier.predict(point);
            int actualLabel = point.label;

            if (actualLabel >= 1 && actualLabel <= numClasses && // Updated to check from 1 to 10
                predictedLabel >= 1 && predictedLabel <= numClasses)
            {                                                           // Updated to check from 1 to 10
                confusionMatrix[actualLabel - 1][predictedLabel - 1]++; // Adjust indexing to start from 0
                if (predictedLabel == actualLabel)
                {
                    correctAssignments++;
                }
                totalPoints++;
            }
            else
            {
                std::cerr << "Skipped sample with actual label " << actualLabel
                          << " or predicted label " << predictedLabel << ".\n";
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during prediction for label " << point.label
                      << ": " << e.what() << "\n";
        }
    }

    if (totalPoints != testData.size())
    {
        std::cerr << "Processed " << totalPoints << " out of " << testData.size() << " samples.\n";
    }

    // Display confusion matrix
    displayConfusionMatrix(confusionMatrix);

    // Calculate overall accuracy
    double accuracy = totalPoints > 0 ? (static_cast<double>(correctAssignments) / totalPoints) * 100 : 0;
    std::cout << "\nAccuracy: " << accuracy << "%\n";

    // Per-class metrics
    double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
    for (int i = 0; i < numClasses; ++i)
    {
        int truePositive = confusionMatrix[i][i];
        int falsePositive = 0;
        int falseNegative = 0;
        for (int j = 0; j < numClasses; ++j)
        {
            if (j != i)
            {
                falsePositive += confusionMatrix[j][i];
                falseNegative += confusionMatrix[i][j];
            }
        }
        double precision = truePositive + falsePositive > 0 ? static_cast<double>(truePositive) / (truePositive + falsePositive) : 0;
        double recall = truePositive + falseNegative > 0 ? static_cast<double>(truePositive) / (truePositive + falseNegative) : 0;
        double f1 = (precision + recall > 0) ? 2 * (precision * recall) / (precision + recall) : 0;

        totalPrecision += precision;
        totalRecall += recall;
        totalF1 += f1;

        std::cout << "Class " << i + 1 << ": Precision = " << precision * 100
                  << "%, Recall = " << recall * 100 << "%, F1-score = " << f1 * 100 << "%\n";
    }

    // Macro metrics
    std::cout << "\nMacro Precision: " << (totalPrecision / numClasses) * 100 << "%"
              << ", Macro Recall: " << (totalRecall / numClasses) * 100 << "%"
              << ", Macro F1-score: " << (totalF1 / numClasses) * 100 << "%\n";
}

// Function to display confusion matrix
void ClassifierEvaluation::displayConfusionMatrix(const std::vector<std::vector<int>> &matrix)
{
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "Predicted →\nActual ↓\n";
    for (const auto &row : matrix)
    {
        for (int val : row)
        {
            std::cout << std::setw(5) << val << " ";
        }
        std::cout << "\n";
    }
}

void ClassifierEvaluation::computePrecisionRecallCurve(
    // const std::vector<DataPoint>& testData,
    const std::vector<int> &trueLabels,
    const std::vector<double> &scores,
    const std::string &outputCsvPath)
{
    // Vérification des tailles
    if (scores.size() != trueLabels.size())
    {
        throw std::invalid_argument("Scores and true labels must have the same size.");
    }

    // Association des scores avec les étiquettes
    std::vector<std::pair<double, int>> scoreLabelPairs;
    for (size_t i = 0; i < scores.size(); ++i)
    {
        scoreLabelPairs.emplace_back(scores[i], trueLabels[i]);
    }

    // Trier les scores par ordre décroissant
    std::sort(scoreLabelPairs.begin(), scoreLabelPairs.end(),
              [](const auto &a, const auto &b)
              { return a.first > b.first; });

    // Variables pour le calcul
    size_t totalPositive = std::count(trueLabels.begin(), trueLabels.end(), 1);
    size_t tp = 0, fp = 0;

    std::vector<std::pair<double, double>> precisionRecallCurve;

    // Itération sur les paires triées pour calculer précision/rappel
    for (const auto &[score, label] : scoreLabelPairs)
    {
        if (label == 1)
        {
            ++tp;
        }
        else
        {
            ++fp;
        }

        double precision = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0;
        precision = 1 - precision;
        double recall = (tp + totalPositive) > 0 ? static_cast<double>(tp) / totalPositive : 0;

        if (precision != 0 && recall != 0)
        {
            precisionRecallCurve.emplace_back(precision, recall);
        }
    }
    // Output to CSV file
    std::filesystem::create_directories("../curve");
    std::ofstream csvFile("../curve/" + outputCsvPath);
    if (!csvFile.is_open())
    {
        throw std::runtime_error("Failed to open CSV file: " + outputCsvPath);
    }

    csvFile << "Precision,Recall\n";
    for (const auto &[precision, recall] : precisionRecallCurve)
    {
        csvFile << precision << "," << recall << "\n";
    }
}

template <typename Classifier>
void ClassifierEvaluation::evaluateWithPrecisionRecall(
    const Classifier &classifier,
    const std::vector<DataPoint> &testData,
    const std::string &outputCsvPath)
{
    std::vector<double> scores;
    std::vector<int> trueLabels;

    for (const auto &point : testData)
    {
        auto [predictedLabel, score] = classifier.predictWithScore(point);
        scores.push_back(score);
        trueLabels.push_back(point.label);
    }

    computePrecisionRecallCurve(trueLabels, scores, outputCsvPath);
}
