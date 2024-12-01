#include "../include/SVMClassifier.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>

SVMClassifier::SVMClassifier(double learningRate, int maxIterations)
    : learningRate(learningRate), maxIterations(maxIterations), bias(0) {}

void SVMClassifier::train(const std::vector<DataPoint>& trainingData) {
    if (trainingData.empty()) return;

    size_t featureSize = trainingData[0].features.size();
    weights.resize(featureSize, 0.0);

    for (int iter = 0; iter < maxIterations; ++iter) {
        bool updated = false;

        for (const auto& point : trainingData) {
            // Calcul de la marge
            double dotProduct = std::inner_product(point.features.begin(), point.features.end(), weights.begin(), 0.0);
            double margin = point.label * (dotProduct + bias);

            // Mettre à jour les poids et le biais si la marge est violée
            if (margin <= 0) {
                for (size_t i = 0; i < featureSize; ++i) {
                    weights[i] += learningRate * point.label * point.features[i];
                }
                bias += learningRate * point.label;
                updated = true;
            }
        }

        // Arrêter si aucune mise à jour n'est effectuée
        if (!updated) break;
    }
}

int SVMClassifier::predict(const DataPoint& point) const {
    double dotProduct = std::inner_product(point.features.begin(), point.features.end(), weights.begin(), 0.0);
    return (dotProduct + bias >= 0) ? 1 : -1;
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
    double dotProduct = std::inner_product(point.features.begin(), point.features.end(), weights.begin(), 0.0);
    double score = dotProduct + bias;
    return {(score >= 0) ? 1 : -1, score}; // Le score est directement utilisable
}
