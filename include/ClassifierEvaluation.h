#ifndef CLASSIFIER_EVALUATION_H
#define CLASSIFIER_EVALUATION_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <string>
#include <fstream>
#include <filesystem> // For directory management
#include "DataPoint.h"

class ClassifierEvaluation
{
public:
    // Function for k-fold cross-validation
    template <typename Classifier>
    void KFoldCrossValidation(
    Classifier &classifier, 
    const std::vector<DataPoint> &data, 
    int k, 
    const std::string &name, 
    const std::string &datasetName);
    // Function to add noise to the data
    static std::vector<DataPoint> augmentNoise(const std::vector<DataPoint> &data, double noiseLevel, double augmentationFraction);

    // Function to split the data into training and test sets
    static std::pair<std::vector<DataPoint>, std::vector<DataPoint>> splitTrainTest(
        const std::vector<DataPoint> &data, double trainRatio = 0.7, bool stratified = true, int minTestSamplesPerClass = 3);

    // Function to test and display results
    template <typename Classifier>
    static void testAndDisplayResults(Classifier &classifier, const std::vector<DataPoint> &testData);

    // Function to compute the precision-recall curve
    void computePrecisionRecallCurve(
        // const std::vector<DataPoint>& testData,
        const std::vector<int> &trueLabels,
        const std::vector<double> &scores,
        const std::string &outputCsvPath);

    // Function to evaluate the classifier with the precision-recall curve
    template <typename Classifier>
    void evaluateWithPrecisionRecall(
        const Classifier &classifier,
        const std::vector<DataPoint> &testData,
        const std::string &outputCsvPath);

    // Private function to calculate accuracy
    template <typename Classifier>
    static double computeAccuracy(Classifier &classifier, const std::vector<DataPoint> &testData);

private:
    // Private function to display the confusion matrix
    static void displayConfusionMatrix(const std::vector<std::vector<int>> &matrix);
    double computeAUC(const std::vector<int> &trueLabels, const std::vector<double> &scores);
};

#endif

