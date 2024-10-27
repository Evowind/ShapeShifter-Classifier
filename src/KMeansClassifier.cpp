#include "../include/KMeansClassifier.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>

KMeansClassifier::KMeansClassifier(int k, int maxIterations)
    : k(k), maxIterations(maxIterations) {}

// Fonction pour calculer la distance Euclidienne
double KMeansClassifier::computeDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(sum);
}

// Initialisation et clustering
void KMeansClassifier::train(const std::vector<DataPoint>& data) {
    // Initialisation aléatoire des centroids
    for (int i = 0; i < k; ++i) {
        centroids.push_back(data[i].features);
    }

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        std::vector<std::vector<DataPoint>> clusters(k);

        // Attribution des points au centroid le plus proche
        for (const auto& point : data) {
            int closest = getClosestCentroid(point);
            clusters[closest].push_back(point);
        }

        // Mise à jour des centroids
        updateCentroids(clusters);
    }
}

// Fonction pour trouver le centroid le plus proche d'un point
int KMeansClassifier::getClosestCentroid(const DataPoint& point) {
    int closestIndex = 0;
    double minDistance = std::numeric_limits<double>::max();

    for (int i = 0; i < k; ++i) {
        double distance = computeDistance(point.features, centroids[i]);
        if (distance < minDistance) {
            minDistance = distance;
            closestIndex = i;
        }
    }
    return closestIndex;
}

// Mise à jour des centroids en fonction des clusters
void KMeansClassifier::updateCentroids(const std::vector<std::vector<DataPoint>>& clusters) {
    for (int i = 0; i < k; ++i) {
        if (clusters[i].empty()) continue;

        std::vector<double> newCentroid(clusters[i][0].features.size(), 0.0);
        for (const auto& point : clusters[i]) {
            for (size_t j = 0; j < point.features.size(); ++j) {
                newCentroid[j] += point.features[j];
            }
        }
        for (size_t j = 0; j < newCentroid.size(); ++j) {
            newCentroid[j] /= clusters[i].size();
        }
        centroids[i] = newCentroid;
    }
}

// Prédiction du cluster d'un nouveau point
int KMeansClassifier::predict(const DataPoint& point) {
    return getClosestCentroid(point);
}

// Fonction de test et affichage des résultats
void KMeansClassifier::testAndDisplayResults(const std::vector<DataPoint>& testData) {
    int correct = 0;
    for (const auto& point : testData) {
        int predictedCluster = predict(point);
        std::cout << "Point avec label " << point.label
                  << " assigné au cluster " << predictedCluster << std::endl;
    }
}
