#include "../include/SVMClassifier.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>

SVMClassifier::SVMClassifier(double learningRate, int maxIterations, double regularization, double gamma)
    : learningRate(learningRate), maxIterations(maxIterations), regularization(regularization), gamma(gamma), bias(0) {}

void SVMClassifier::train(const std::vector<DataPoint>& trainingData) {
    if (trainingData.empty()) {
        std::cerr << "Training data is empty." << std::endl;
        return;
    }

    size_t featureSize = trainingData[0].features.size();
    weights.resize(featureSize, 0.0); // Initialize weights to zero

    for (int iter = 0; iter < maxIterations; ++iter) {
        bool updated = false;

        for (const auto& point : trainingData) {
            // Compute the margin (dot product of weights and features + bias)
            double margin = calculateMargin(point);

            // Update weights and bias if the point is misclassified
            if (point.label * margin < 1) { // Constraint: y * (w·x + b) >= 1
                for (size_t i = 0; i < featureSize; ++i) {
                    weights[i] += learningRate * (point.label * point.features[i] - regularization * weights[i]);
                }
                bias += learningRate * point.label;
                updated = true;
            }
        }

        // Stop early if no updates are made during an iteration
        if (!updated) {
            std::cout << "Converged after " << iter + 1 << " iterations.\n";
            break;
        }
    }
}

int SVMClassifier::predict(const DataPoint& point) const {
    double margin = calculateMargin(point);
    return (margin >= 0) ? 1 : -1;
}

double SVMClassifier::calculateMargin(const DataPoint& point) const {
    // Compute dot product between weights and features, then add the bias
    double sum = std::inner_product(point.features.begin(), point.features.end(), weights.begin(), 0.0);
    return sum + bias;
}

std::vector<DataPoint> SVMClassifier::normalizeData(const std::vector<DataPoint>& data) const {
    std::vector<DataPoint> normalizedData = data;

    for (auto& point : normalizedData) {
        double norm = std::sqrt(std::inner_product(point.features.begin(), point.features.end(), point.features.begin(), 0.0));
        if (norm > 0) {
            for (auto& feature : point.features) {
                feature /= norm;
            }
        }
    }

    return normalizedData;
}

std::pair<int, double> SVMClassifier::predictWithScore(const DataPoint& point) const {
    double score = calculateMargin(point);
    return {(score >= 0) ? 1 : -1, score}; // Return the predicted class and the margin score
}
