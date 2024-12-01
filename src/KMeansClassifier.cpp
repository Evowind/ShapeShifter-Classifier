#include "../include/KMeansClassifier.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <random>
#include <algorithm>


KMeansClassifier::KMeansClassifier(int k, int maxIterations)
    : k(k), maxIterations(maxIterations) {}

double KMeansClassifier::computeDistance(const std::vector<double>& a, const std::vector<double>& b) const{
    if (a.size() != b.size()) {
        std::cerr << "Vector size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        throw std::runtime_error("Vectors must have the same dimension");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void KMeansClassifier::train(const std::vector<DataPoint>& rawData) {
    if (rawData.empty()) {
        throw std::runtime_error("No training data provided");
    }

    // Normalize and validate data
    std::vector<DataPoint> data = normalizeData(rawData);
    
    if (k > static_cast<int>(data.size())) {
        std::cerr << "Warning: k is larger than the number of data points. Reducing k to match the number of data points." << std::endl;
        k = data.size();  // Adjust k to match the number of available data points
    }

    std::cout << "Starting training with " << data.size() << " points, each with "
              << (data.empty() ? 0 : data[0].features.size()) << " features" << std::endl;

    // Clear previous centroids
    centroids.clear();
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);
    
    // Initialize centroids with random points from the dataset
    std::vector<int> usedIndices;
    while (centroids.size() < static_cast<size_t>(k)) {
        int index = dis(gen);
        if (std::find(usedIndices.begin(), usedIndices.end(), index) == usedIndices.end()) {
            centroids.push_back(data[index].features);
            usedIndices.push_back(index);
        }
    }

    std::cout << "Initialized " << centroids.size() << " centroids" << std::endl;

    bool converged = false;
    int iteration = 0;

    while (!converged && iteration < maxIterations) {
        // Store previous centroids for convergence check
        auto previousCentroids = centroids;
        
        // Reset cluster assignments
        std::vector<std::vector<const DataPoint*>> clusters(k);
        
        // Assign points to nearest centroid
        for (size_t i = 0; i < data.size(); ++i) {
            try {
                int closestCluster = getClosestCentroid(data[i]);
                clusters[closestCluster].push_back(&data[i]);
            } catch (const std::exception& e) {
                std::cerr << "Error assigning point " << i << " to cluster: " << e.what() << std::endl;
                throw;
            }
        }

        // Update centroids
        for (int i = 0; i < k; ++i) {
            if (clusters[i].empty()) {
                // If a cluster is empty, reinitialize it with a random point
                int randomIndex = dis(gen);
                centroids[i] = data[randomIndex].features;
                continue;
            }

            // Calculate new centroid as mean of all points in cluster
            std::vector<double> newCentroid(clusters[i][0]->features.size(), 0.0);
            for (const DataPoint* point : clusters[i]) {
                for (size_t j = 0; j < point->features.size(); ++j) {
                    newCentroid[j] += point->features[j];
                }
            }
            
            for (double& value : newCentroid) {
                value /= clusters[i].size();
            }
            
            centroids[i] = newCentroid;
        }

        // Check for convergence
        converged = true;
        for (size_t i = 0; i < centroids.size(); ++i) {
            if (computeDistance(centroids[i], previousCentroids[i]) > 1e-6) {
                converged = false;
                break;
            }
        }

        if (iteration % 10 == 0) {
            std::cout << "Completed iteration " << iteration << std::endl;
        }

        ++iteration;
    }

    std::cout << "KMeans training completed after " << iteration << " iterations" << std::endl;
}

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

int KMeansClassifier::getClosestCentroid(const DataPoint& point) {
    if (centroids.empty()) {
        throw std::runtime_error("Centroids not initialized");
    }

    int closestIndex = 0;
    double minDistance = std::numeric_limits<double>::max();

    for (size_t i = 0; i < centroids.size(); ++i) {
        try {
            double distance = computeDistance(point.features, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                closestIndex = i;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error computing distance to centroid " << i << ": " << e.what() << std::endl;
            throw;
        }
    }

    return closestIndex;
}
int KMeansClassifier::predict(const DataPoint& point) {
    if (centroids.empty()) {
        throw std::runtime_error("Model not trained yet");
    }
    return getClosestCentroid(point);
}

void KMeansClassifier::test(const std::vector<DataPoint>& testData, std::vector<int>& predictions) {
    if (centroids.empty()) {
        throw std::runtime_error("Model not trained yet!");
    }

    std::vector<DataPoint> normalizedTestData = normalizeData(testData);
    predictions.clear();

    for (const auto& point : normalizedTestData) {
        predictions.push_back(predict(point));
    }
}

std::pair<int, double> KMeansClassifier::predictWithScore(const DataPoint& point) const {
    if (centroids.empty()) {
        throw std::runtime_error("Model not trained yet");
    }

    int closestCentroid = -1;
    double minDistance = std::numeric_limits<double>::max();

    for (size_t i = 0; i < centroids.size(); ++i) {
        double distance = computeDistance(point.features, centroids[i]);
        if (distance < minDistance) {
            minDistance = distance;
            closestCentroid = static_cast<int>(i);
        }
    }

    return {closestCentroid, -minDistance}; // Score est l'opposé de la distance (plus grand = mieux)
}

