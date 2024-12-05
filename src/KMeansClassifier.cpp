#include "../include/KMeansClassifier.h"
#include <cmath>
#include <limits>
#include <random>
#include <iostream>
#include <algorithm>

KMeansClassifier::KMeansClassifier(int k, int maxIterations, double convergenceThreshold)
    : k(k), maxIterations(maxIterations), convergenceThreshold(convergenceThreshold) {}


std::vector<DataPoint> KMeansClassifier::normalizeData(const std::vector<DataPoint>& rawData) {
    if (rawData.empty()) {
        return rawData;
    }

    // First, find the expected feature dimension
    size_t expectedDim = 0;
    for (const auto& point : rawData) {
        if (!point.features.empty()) {
            expectedDim = point.features.size();
            break;
        }
    }

    if (expectedDim == 0) {
        throw std::runtime_error("Could not determine feature dimension");
    }

    std::cout << "Expected feature dimension: " << expectedDim << std::endl;
        // Copy and validate data
    std::vector<DataPoint> normalizedData;
    for (const auto& point : rawData) {
        if (point.features.size() == expectedDim) {
            normalizedData.push_back(point);
        } else {
            std::cerr << "Skipping point with incorrect dimension: " 
                      << point.features.size() << " (expected " << expectedDim << ")" << std::endl;
        }
    }

    if (normalizedData.empty()) {
        throw std::runtime_error("No valid data points after dimension validation");
    }

    // Calculate mean and standard deviation for each feature
    std::vector<double> means(expectedDim, 0.0);
    std::vector<double> stdDevs(expectedDim, 0.0);

    // Calculate means
    for (const auto& point : normalizedData) {
        for (size_t i = 0; i < expectedDim; ++i) {
            means[i] += point.features[i];
        }
    }
    for (double& mean : means) {
        mean /= normalizedData.size();
    }

    // Calculate standard deviations
    for (const auto& point : normalizedData) {
        for (size_t i = 0; i < expectedDim; ++i) {
            double diff = point.features[i] - means[i];
            stdDevs[i] += diff * diff;
        }
    }
    for (double& stdDev : stdDevs) {
        stdDev = std::sqrt(stdDev / normalizedData.size());
        if (stdDev < 1e-10) stdDev = 1.0; // Prevent division by zero
    }

    // Normalize the data
    for (auto& point : normalizedData) {
        for (size_t i = 0; i < expectedDim; ++i) {
            point.features[i] = (point.features[i] - means[i]) / stdDevs[i];
        }
    }

    std::cout << "Normalized " << normalizedData.size() << " data points" << std::endl;
    return normalizedData;
}


void KMeansClassifier::initializeCentroids(const std::vector<DataPoint>& data) {
    // Initialisation des centroids avec la méthode k-means++
    centroids.clear();
    centroids.push_back(data[0].features); // Choisir un premier centroid aléatoire

    std::random_device rd;
    std::mt19937 gen(rd());

    while (centroids.size() < static_cast<size_t>(k)) {
        std::vector<double> distances(data.size(), std::numeric_limits<double>::max());

        for (size_t i = 0; i < data.size(); ++i) {
            for (const auto& centroid : centroids) {
                distances[i] = std::min(distances[i], computeDistance(data[i].features, centroid));
            }
        }

        std::discrete_distribution<> distribution(distances.begin(), distances.end());
        centroids.push_back(data[distribution(gen)].features);
    }
}

void KMeansClassifier::train(const std::vector<DataPoint>& rawData) {
    if (rawData.empty()) {
        throw std::runtime_error("No training data provided");
    }

    // Normaliser les données
    std::vector<DataPoint> data = normalizeData(rawData);

    // Initialiser les centroids
    initializeCentroids(data);

    bool converged = false;
    int iteration = 0;

    while (!converged && iteration < maxIterations) {
        std::vector<std::vector<const DataPoint*>> clusters(k);

        // Assigner chaque point au centroid le plus proche
        for (const auto& point : data) {
            int closestCluster = getClosestCentroid(point);
            clusters[closestCluster].push_back(&point);
        }

        // Mise à jour des centroids
        converged = true;
        for (int i = 0; i < k; ++i) {
            if (clusters[i].empty()) {
                // Réattribuer un centroid vide avec un point éloigné
                initializeCentroids(data);
                converged = false;
                break;
            }

            std::vector<double> newCentroid(centroids[i].size(), 0.0);
            for (const DataPoint* point : clusters[i]) {
                for (size_t j = 0; j < newCentroid.size(); ++j) {
                    newCentroid[j] += point->features[j];
                }
            }
            for (double& value : newCentroid) {
                value /= clusters[i].size();
            }

            if (computeDistance(newCentroid, centroids[i]) > convergenceThreshold) {
                converged = false;
            }
            centroids[i] = std::move(newCentroid);
        }

        ++iteration;
    }

    std::cout << "Training completed in " << iteration << " iterations." << std::endl;
}

int KMeansClassifier::getClosestCentroid(const DataPoint& point) const {
    int closestIndex = 0;
    double minDistance = std::numeric_limits<double>::max();

    for (size_t i = 0; i < centroids.size(); ++i) {
        double distance = computeDistance(point.features, centroids[i]);
        if (distance < minDistance) {
            minDistance = distance;
            closestIndex = i;
        }
    }

    return closestIndex;
}

int KMeansClassifier::predict(const DataPoint& point) {
    return getClosestCentroid(point);
}

double KMeansClassifier::computeDistance(const std::vector<double>& a, const std::vector<double>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::pair<int, double> KMeansClassifier::predictWithScore(const DataPoint& point) const {
    int closestCentroid = getClosestCentroid(point);
    double distance = computeDistance(point.features, centroids[closestCentroid]);
    return {closestCentroid, -distance};
}