#include "../include/SVMClassifier.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>

/**
 * @brief Constructs an SVMClassifier with specified learning rate and maximum iterations.
 *
 * Initializes the learning rate, maximum iterations, and bias (set to zero).
 *
 * @param learningRate The learning rate for weight updates during training.
 * @param maxIterations The maximum number of iterations to train the model.
 */
SVMClassifier::SVMClassifier(double learningRate, int maxIterations)
    : learningRate(learningRate), maxIterations(maxIterations), bias(0) {}

/**
 * @brief Trains the SVM classifier using the provided training data.
 *
 * This function uses the Perceptron-like approach to training. In each iteration,
 * it calculates the margin for each training point and updates the weights and
 * bias if the margin is violated (i.e., if the point is on the wrong side of the
 * decision boundary).
 *
 * @param trainingData A vector of DataPoint objects containing features and labels for training.
 */
void SVMClassifier::train(const std::vector<DataPoint> &trainingData)
{
    // Check if training data is empty
    if (trainingData.empty())
        return;

    size_t featureSize = trainingData[0].features.size();
    weights.resize(featureSize, 0.0); // Initialize weights to zero

    // Iterate over the training process for the maximum number of iterations
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        bool updated = false;

        // Go through each data point and update the weights if necessary
        for (const auto &point : trainingData)
        {
            // Calculate the margin for the point
            double dotProduct = std::inner_product(point.features.begin(), point.features.end(), weights.begin(), 0.0);
            double margin = point.label * (dotProduct + bias);

            // Update the weights and bias if the margin condition is violated
            if (margin <= 0)
            {
                for (size_t i = 0; i < featureSize; ++i)
                {
                    weights[i] += learningRate * point.label * point.features[i];
                }
                bias += learningRate * point.label;
                updated = true;
            }
        }

        // Stop if no updates were made in the current iteration (converged)
        if (!updated)
            break;
    }
}

/**
 * @brief Predicts the label for a given data point using the trained SVM model.
 *
 * Computes the decision function (dot product of features and weights + bias)
 * and returns the predicted label based on the sign of the result.
 *
 * @param point The DataPoint for which the label is to be predicted.
 * @return The predicted label (1 or -1).
 */
int SVMClassifier::predict(const DataPoint &point) const
{
    double dotProduct = std::inner_product(point.features.begin(), point.features.end(), weights.begin(), 0.0);
    return (dotProduct + bias >= 0) ? 1 : -1; // Predict 1 if score is >= 0, else -1
}

/**
 * @brief Normalizes the feature values of the given dataset.
 *
 * Each feature in a DataPoint is divided by the Euclidean norm (magnitude) of the feature vector.
 * This ensures that each feature vector has a unit norm, making training more efficient.
 *
 * @param data The dataset to be normalized.
 * @return A new dataset with normalized feature values.
 */
std::vector<DataPoint> SVMClassifier::normalizeData(const std::vector<DataPoint> &data) const
{
    std::vector<DataPoint> normalizedData = data;

    // Normalize each data point's features
    for (auto &point : normalizedData)
    {
        double norm = std::sqrt(std::inner_product(point.features.begin(), point.features.end(), point.features.begin(), 0.0));
        if (norm > 0)
        {
            for (auto &feature : point.features)
            {
                feature /= norm; // Normalize by dividing each feature by the norm
            }
        }
    }

    return normalizedData;
}

/**
 * @brief Predicts the label for a data point and returns the decision score.
 *
 * Similar to the `predict` function, but also returns the score (the result of
 * the decision function), which gives a measure of confidence in the prediction.
 *
 * @param point The DataPoint to be predicted.
 * @return A pair consisting of the predicted label and the decision score.
 */
std::pair<int, double> SVMClassifier::predictWithScore(const DataPoint &point) const
{
    double dotProduct = std::inner_product(point.features.begin(), point.features.end(), weights.begin(), 0.0);
    double score = dotProduct + bias;
    return {(score >= 0) ? 1 : -1, score}; // Return label and score (the score can be used for confidence)
}