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

/**
 * @brief Split the given data into training and test sets based on the given ratio.
 *
 * The split can be either stratified or random. Stratification is done by grouping the data
 * by class and then splitting each group such that the required minimum number of test
 * samples for each class is met. Standard random splitting is done by shuffling the data
 * and then taking the first <code>trainRatio</code> proportion as the training set and
 * the remaining part as the test set.
 *
 * @param data The data to split.
 * @param trainRatio The proportion of the data to use for training.
 * @param stratified Whether to use stratification or standard random splitting.
 * @param minTestSamplesPerClass The minimum number of test samples required for each class
 * when using stratification.
 * @return A pair consisting of the training set and the test set.
 */
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

/**
 * @brief Perform k-fold cross-validation on a classifier.
 *
 * This function will split the provided data into k folds, and for each fold, it will train
 * the classifier on the k-1 remaining folds and test it on the remaining fold. The accuracy
 * of the classifier will be calculated for each fold and the average accuracy across all folds
 * will be printed to the console. The precision-recall curve for the classifier will also be
 * generated and saved to a CSV file.
 *
 * @param classifier The classifier to evaluate.
 * @param data The data to use for the evaluation.
 * @param k The number of folds to use.
 * @param name The name of the classifier to use for the output filename.
 * @param datasetName The name of the dataset to use for the output filename.
 */
template <typename Classifier>
void ClassifierEvaluation::KFoldCrossValidation(
    Classifier &classifier,
    const std::vector<DataPoint> &data,
    int k,
    const std::string &name,
    const std::string &datasetName)
{
    // Create a non-const copy of the data vector
    std::vector<DataPoint> dataCopy = data;

    // Initialize the folds
    std::vector<std::vector<DataPoint>> folds(k);

    // Initialize the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Shuffle the data (using the non-const copy)
    std::shuffle(dataCopy.begin(), dataCopy.end(), gen);

    // Split the data into k folds
    for (size_t i = 0; i < dataCopy.size(); ++i)
    {
        folds[i % k].push_back(dataCopy[i]);
    }

    double totalAccuracy = 0;

    // Variables to store the scores and labels
    std::vector<double> allScores;
    std::vector<int> allTrueLabels;

    // For each fold, train on k-1 folds and test on the remaining fold
    for (int i = 0; i < k; ++i)
    {
        std::vector<DataPoint> trainData;
        std::vector<DataPoint> testData = folds[i];

        // Combine all the other folds for the training data
        for (int j = 0; j < k; ++j)
        {
            if (i != j)
            {
                trainData.insert(trainData.end(), folds[j].begin(), folds[j].end());
            }
        }

        // Train the classifier
        classifier.train(trainData);

        // Test the classifier on the test data for this fold
        for (const auto &point : testData)
        {
            auto [predictedLabel, score] = classifier.predictWithScore(point);
            allScores.push_back(score);
            allTrueLabels.push_back(point.label);
        }

        // Calculate the accuracy for this fold
        double foldAccuracy = computeAccuracy(classifier, testData);
        totalAccuracy += foldAccuracy;
    }

    // Calculate the average accuracy
    double averageAccuracy = totalAccuracy / k;
    std::cout << "Average Accuracy across " << k << " folds: " << averageAccuracy << "%\n";

    // Generate the precision-recall curve and save it to a CSV file
    computePrecisionRecallCurve(
        allTrueLabels,
        allScores,
        name + "_" + datasetName + ".csv");
}

/**
 * @brief Compute the accuracy of a classifier on a given test dataset.
 *
 * This function iterates over the test data and uses the classifier to predict
 * the label for each data point. The accuracy is then calculated as the percentage
 * of correct predictions.
 *
 * @param classifier The classifier to be evaluated.
 * @param testData The test data to evaluate the classifier on.
 * @return The accuracy of the classifier as a percentage.
 */
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

/**
 * @brief Augment the given data with random noise.
 *
 * This function takes in the given data and adds random noise to a specified
 * fraction of the data. The noise is added to each feature of the data points
 * by generating a random number between -1 and 1 and multiplying it by the
 * specified noise level. The augmented data is then returned.
 *
 * @param data The input data to be augmented.
 * @param noiseLevel The maximum amount of noise to be added to each feature.
 * @param augmentationFraction The fraction of the data to be augmented.
 * @return The augmented data.
 */
std::vector<DataPoint> ClassifierEvaluation::augmentNoise(
    const std::vector<DataPoint> &data, double noiseLevel, double augmentationFraction)
{

    std::vector<DataPoint> augmentedData = data; // Copy the original data

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

/**
 * @brief Test a classifier on a given dataset and display the results.
 *
 * This function evaluates the given classifier on the test data by computing
 * the predicted labels and comparing them to the actual labels. The results
 * are displayed in the form of a confusion matrix and overall accuracy.
 * Additionally, per-class precision, recall, and F1-score are computed and
 * displayed. Finally, macro precision, recall, and F1-score are computed and
 * displayed.
 *
 * @param classifier The classifier to be tested.
 * @param testData The test data to evaluate the classifier on.
 */
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

/**
 * @brief Prints the confusion matrix given a 2D vector.
 *
 * The confusion matrix is printed in a readable format with actual class labels
 * on the left and predicted class labels on top. The matrix values are formatted
 * to have 5 spaces each.
 *
 * @param matrix A 2D vector containing the confusion matrix
 */
void ClassifierEvaluation::displayConfusionMatrix(const std::vector<std::vector<int>> &matrix)
{
    int numClasses = matrix.size();
    std::cout << "\nConfusion Matrix (Actual/Predicted): \n";

    // Print header row with class labels
    std::cout << "     "; // Initial padding for the header
    for (int i = 0; i < numClasses; ++i)
    {
        std::cout << std::setw(5) << "P" << std::setfill('0') << std::setw(2) << (i + 1) << std::setfill(' ') << " "; // Print predicted class labels
    }
    std::cout << "\n";

    // Print a horizontal separator line
    std::cout << "     " << std::string(6 * numClasses, '-') << "\n";

    // Print each row with actual class labels
    for (int i = 0; i < numClasses; ++i)
    {
        std::cout << "A" << std::setfill('0') << std::setw(2) << (i + 1) << std::setfill(' ') << " | "; // Print actual class label

        // Print matrix values
        for (int val : matrix[i])
        {
            std::cout << std::setw(5) << val << " "; // Format the matrix values
        }
        std::cout << "\n";
    }

    std::cout << "\n"; // Newline for better readability or I hope so
}

/**
 * @brief Computes the precision-recall curve for a given classifier and test data and writes it to a CSV file.
 *
 * This function takes in the true labels and scores of the test data, and computes the precision-recall curve
 * by iterating over the sorted scores and labels. The precision-recall curve is then written to a CSV file at
 * the specified output path.
 *
 * @param trueLabels The true labels of the test data.
 * @param scores The scores of the test data.
 * @param outputCsvPath The path to the CSV file to write the precision-recall curve to.
 */
void ClassifierEvaluation::computePrecisionRecallCurve(
    // const std::vector<DataPoint>& testData,
    const std::vector<int> &trueLabels,
    const std::vector<double> &scores,
    const std::string &outputCsvPath)
{
    // Check sizes
    if (scores.size() != trueLabels.size())
    {
        throw std::invalid_argument("Scores and true labels must have the same size.");
    }

    // Associate scores with labels
    std::vector<std::pair<double, int>> scoreLabelPairs;
    for (size_t i = 0; i < scores.size(); ++i)
    {
        scoreLabelPairs.emplace_back(scores[i], trueLabels[i]);
    }

    // Sort scores in descending order
    std::sort(scoreLabelPairs.begin(), scoreLabelPairs.end(),
              [](const auto &a, const auto &b)
              { return a.first > b.first; });

    // Variables for calculation
    size_t totalPositive = std::count(trueLabels.begin(), trueLabels.end(), 1);
    size_t tp = 0, fp = 0;

    std::vector<std::pair<double, double>> precisionRecallCurve;

    // Iterate over sorted pairs to calculate precision/recall
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

/**
 * @brief Compute the precision-recall curve for a given classifier and test data and writes it to a CSV file.
 *
 * This function evaluates the given classifier on the test data by computing the score for each data point,
 * and then computes the precision-recall curve using the true labels and the scores. The curve is then written to
 * a CSV file.
 *
 * @param classifier The classifier to be evaluated.
 * @param testData The test data to evaluate the classifier on.
 * @param outputCsvPath The path to the CSV file to write the precision-recall curve to.
 */
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
