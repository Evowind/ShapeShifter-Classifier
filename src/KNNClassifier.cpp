#include "../include/KNNClassifier.h"
#include <algorithm>

std::vector<DataPoint> KNNClassifier::normalizeData(const std::vector<DataPoint>& data) {
    if (data.empty()) return {};

    size_t featureCount = data[0].features.size();
    std::vector<double> minValues(featureCount, std::numeric_limits<double>::max());
    std::vector<double> maxValues(featureCount, std::numeric_limits<double>::lowest());

    // Trouver les min et max pour chaque caractéristique
    for (const auto& point : data) {
        for (size_t i = 0; i < featureCount; ++i) {
            minValues[i] = std::min(minValues[i], point.features[i]);
            maxValues[i] = std::max(maxValues[i], point.features[i]);
        }
    }

    // Normaliser les données
    std::vector<DataPoint> normalizedData = data;
    for (auto& point : normalizedData) {
        for (size_t i = 0; i < featureCount; ++i) {
            if (maxValues[i] != minValues[i]) {
                point.features[i] = (point.features[i] - minValues[i]) / (maxValues[i] - minValues[i]);
            } else {
                point.features[i] = 0.0; // Si min == max, la valeur est constante
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

    // Prendre les k voisins les plus proches
    std::vector<int> neighborLabels;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
        neighborLabels.push_back(distances[i].second);
    }

    // Trouver le label majoritaire parmi les k voisins
    std::map<int, int> labelCounts;
    for (int label : neighborLabels) {
        labelCounts[label]++;
    }

    // Retourner le label ayant la fréquence maximale
    return std::max_element(labelCounts.begin(), labelCounts.end(),
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
