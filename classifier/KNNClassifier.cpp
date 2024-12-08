#include "../include/KNNClassifier.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <random>
#include <map>

/**
 * @brief Normalizes the feature values of the given dataset using Z-score normalization.
 *
 * This function computes the mean and standard deviation for each feature across all data points,
 * and then normalizes the feature values for each data point by subtracting the mean and dividing
 * by the standard deviation.
 *
 * @param data The dataset to be normalized.
 * @return A new dataset with normalized feature values.
 */
std::vector<DataPoint> KNNClassifier::normalizeData(const std::vector<DataPoint> &data)
{
    if (data.empty())
        return {};

    size_t featureCount = data[0].features.size();
    std::vector<double> mean(featureCount, 0.0);
    std::vector<double> stdDev(featureCount, 0.0);

    // Calculate the means of each feature
    for (const auto &point : data)
    {
        for (size_t i = 0; i < featureCount; ++i)
        {
            mean[i] += point.features[i];
        }
    }
    for (auto &m : mean)
        m /= data.size(); // Divide by the number of data points to get the mean

    // Calculate the standard deviations of each feature
    for (const auto &point : data)
    {
        for (size_t i = 0; i < featureCount; ++i)
        {
            stdDev[i] += std::pow(point.features[i] - mean[i], 2);
        }
    }
    for (auto &s : stdDev)
        s = std::sqrt(s / data.size()); // Take the square root to get the standard deviation

    // Normalize the data (Z-score normalization)
    std::vector<DataPoint> normalizedData = data;
    for (auto &point : normalizedData)
    {
        for (size_t i = 0; i < featureCount; ++i)
        {
            if (stdDev[i] > 0)
            {
                point.features[i] = (point.features[i] - mean[i]) / stdDev[i]; // Normalize the feature
            }
            else
            {
                point.features[i] = 0.0; // If standard deviation is 0, set the feature to 0
            }
        }
    }

    return normalizedData;
}

/**
 * @brief Trains the KNN classifier by storing the training data.
 *
 * This function simply stores the provided training data for future use when predicting.
 *
 * @param data The training data to be used by the classifier.
 */
void KNNClassifier::train(const std::vector<DataPoint> &data)
{
    trainingData = data; // Save the training data for prediction
}

/**
 * @brief Predicts the label for a given test data point.
 *
 * This function calculates the distance between the test point and each training point,
 * sorts the distances, and returns the label of the majority of the nearest neighbors.
 *
 * @param testPoint The DataPoint for which the label is to be predicted.
 * @return The predicted label for the test point.
 */
int KNNClassifier::predict(const DataPoint &testPoint) const
{
    // Check if the classifier has been trained
    if (trainingData.empty())
    {
        throw std::runtime_error("KNNClassifier is not trained.");
    }

    // Calculate the distance between the test point and each training point
    std::vector<std::pair<double, int>> distances; // (distance, label)
    for (const auto &trainPoint : trainingData)
    {
        double distance = calculateDistance(testPoint, trainPoint);
        distances.emplace_back(distance, trainPoint.label); // Store distance and label
    }

    // Sort the distances in ascending order
    std::sort(distances.begin(), distances.end());

    // Return the label with the highest frequency among the nearest neighbors
    std::map<int, double> labelWeightedCounts;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i)
    {
        double weight = 1.0 / (distances[i].first + 1e-6);  // Avoid division by 0
        labelWeightedCounts[distances[i].second] += weight; // Update weighted count for each label
    }

    // Find the label with the maximum weighted count
    return std::max_element(labelWeightedCounts.begin(), labelWeightedCounts.end(),
                            [](const auto &a, const auto &b)
                            { return a.second < b.second; })
        ->first;
}

/**
 * @brief Calculates the Euclidean distance between two data points.
 *
 * This function calculates the Euclidean distance between the feature vectors of two data points.
 * It assumes that both feature vectors have the same size.
 *
 * @param a The first DataPoint.
 * @param b The second DataPoint.
 * @return The Euclidean distance between the two feature vectors.
 */
double KNNClassifier::calculateDistance(const DataPoint &a, const DataPoint &b) const
{
    // Ensure the feature vectors have the same size
    if (a.features.size() != b.features.size())
    {
        throw std::invalid_argument("Feature vectors must have the same size.");
    }

    // Calculate the Euclidean distance
    double sum = 0.0;
    for (size_t i = 0; i < a.features.size(); ++i)
    {
        double diff = a.features[i] - b.features[i];
        sum += diff * diff; // Sum of squared differences
    }
    return std::sqrt(sum); // Return the square root of the sum (Euclidean distance)
}

/**
 * @brief Predicts the label for a given test data point and returns the decision score.
 *
 * This function predicts the label similar to `predict`, but also calculates a score based on
 * the sum of the distances to the k nearest neighbors. The score is inversely related to the distance.
 *
 * @param testPoint The DataPoint for which the label and score are to be predicted.
 * @return A pair consisting of the predicted label and the decision score.
 */
std::pair<int, double> KNNClassifier::predictWithScore(const DataPoint &testPoint) const
{
    if (trainingData.empty())
    {
        throw std::runtime_error("KNNClassifier is not trained.");
    }

    // Calculate the distance between the test point and each training point
    std::vector<std::pair<double, int>> distances;
    for (const auto &trainPoint : trainingData)
    {
        double distance = calculateDistance(testPoint, trainPoint);
        distances.emplace_back(distance, trainPoint.label); // Store distance and label
    }

    // Sort the distances in ascending order
    std::sort(distances.begin(), distances.end());

    // Store the labels of the k nearest neighbors and calculate the sum of their distances
    std::vector<int> neighborLabels;
    double distanceSum = 0.0;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i)
    {
        neighborLabels.push_back(distances[i].second); // Store label of neighbor
        distanceSum += distances[i].first;             // Sum of distances
    }

    // Count the frequency of each label among the neighbors
    std::map<int, int> labelCounts;
    for (int label : neighborLabels)
    {
        labelCounts[label]++;
    }

    // Find the most frequent label among the neighbors
    int predictedLabel = std::max_element(labelCounts.begin(), labelCounts.end(),
                                          [](const auto &a, const auto &b)
                                          { return a.second < b.second; })
                             ->first;

    // Return the predicted label and the inverse sum of the distances (as score)
    return {predictedLabel, -distanceSum}; // The score is negative to make smaller distances better
}