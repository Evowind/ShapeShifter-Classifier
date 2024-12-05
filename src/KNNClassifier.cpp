#include "../include/KNNClassifier.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <random>
#include <map>

//ZScore variant
std::vector<DataPoint> KNNClassifier::normalizeData(const std::vector<DataPoint>& data) {
    if (data.empty()) return {};

    size_t featureCount = data[0].features.size();
    std::vector<double> mean(featureCount, 0.0);
    std::vector<double> stdDev(featureCount, 0.0);

    // Calcul des moyennes
    for (const auto& point : data) {
        for (size_t i = 0; i < featureCount; ++i) {
            mean[i] += point.features[i];
        }
    }
    for (auto& m : mean) m /= data.size();

    // Calcul des écarts-types
    for (const auto& point : data) {
        for (size_t i = 0; i < featureCount; ++i) {
            stdDev[i] += std::pow(point.features[i] - mean[i], 2);
        }
    }
    for (auto& s : stdDev) s = std::sqrt(s / data.size());

    // Normalisation
    std::vector<DataPoint> normalizedData = data;
    for (auto& point : normalizedData) {
        for (size_t i = 0; i < featureCount; ++i) {
            if (stdDev[i] > 0) {
                point.features[i] = (point.features[i] - mean[i]) / stdDev[i];
            } else {
                point.features[i] = 0.0;
            }
        }
    }

    return normalizedData;
}


void KNNClassifier::train(const std::vector<DataPoint>& data) {
    trainingData = data; // Sauvegarder les données d'entraînement
}

int KNNClassifier::predict(const DataPoint& testPoint) const {
    // Vérification si le modèle est entraîné
    if (trainingData.empty()) {
        throw std::runtime_error("KNNClassifier is not trained.");
    }

    // Calculer la distance entre le point de test et chaque point d'entraînement
    std::vector<std::pair<double, int>> distances; // (distance, label)
    for (const auto& trainPoint : trainingData) {
        double distance = calculateDistance(testPoint, trainPoint);
        distances.emplace_back(distance, trainPoint.label);
    }

    // Trier les distances par ordre croissant
    std::sort(distances.begin(), distances.end());

    // Retourner le label ayant la fréquence maximale
    std::map<int, double> labelWeightedCounts;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
        double weight = 1.0 / (distances[i].first + 1e-6); // Éviter la division par 0
        labelWeightedCounts[distances[i].second] += weight;
    }

    return std::max_element(labelWeightedCounts.begin(), labelWeightedCounts.end(),
                            [](const auto& a, const auto& b) { return a.second < b.second; })
        ->first;

}

double KNNClassifier::calculateDistance(const DataPoint& a, const DataPoint& b) const {
    // Vérifier que les vecteurs de caractéristiques ont la même taille
    if (a.features.size() != b.features.size()) {
        throw std::invalid_argument("Feature vectors must have the same size.");
    }

    // Calculer la distance euclidienne
    double sum = 0.0;
    for (size_t i = 0; i < a.features.size(); ++i) {
        double diff = a.features[i] - b.features[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::pair<int, double> KNNClassifier::predictWithScore(const DataPoint& testPoint) const {
    if (trainingData.empty()) {
        throw std::runtime_error("KNNClassifier is not trained.");
    }

    std::vector<std::pair<double, int>> distances;
    for (const auto& trainPoint : trainingData) {
        double distance = calculateDistance(testPoint, trainPoint);
        distances.emplace_back(distance, trainPoint.label);
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> neighborLabels;
    double distanceSum = 0.0;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
        neighborLabels.push_back(distances[i].second);
        distanceSum += distances[i].first; // Somme des distances des voisins
    }

    std::map<int, int> labelCounts;
    for (int label : neighborLabels) {
        labelCounts[label]++;
    }

    int predictedLabel = std::max_element(labelCounts.begin(), labelCounts.end(),
                                          [](const auto& a, const auto& b) { return a.second < b.second; })
                             ->first;

    // Retourne le label prédit avec la somme des distances comme score (inverse des distances, plus faible est mieux)
    return { predictedLabel, -distanceSum };
}