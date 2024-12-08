#include "../include/KMeansClassifier.h"
#include <cmath>
#include <limits>
#include <random>
#include <iostream>
#include <algorithm>

/**
 * @brief Construct a new KMeansClassifier object
 *
 * @param k The number of clusters to be formed and the number of centroids to generate.
 * @param maxIterations The maximum number of iterations for the clustering algorithm.
 * @param convergenceThreshold The algorithm will stop iterating when the sum of the squared distances of the samples to their closest centroid is less than or equal to this value.
 */
KMeansClassifier::KMeansClassifier(int k, int maxIterations, double convergenceThreshold)
    : k(k), maxIterations(maxIterations), convergenceThreshold(convergenceThreshold) {}

/**
 * @brief Normalize the input data using Z-score normalization.
 *
 * This function takes the input data and validates each data point to have the
 * expected feature dimension. It then calculates the mean and standard deviation
 * for each feature and normalizes the data by subtracting the mean and dividing
 * by the standard deviation. Data points with invalid dimension are skipped.
 *
 * @param rawData The input data to be normalized.
 * @return The normalized data.
 */
std::vector<DataPoint> KMeansClassifier::normalizeData(const std::vector<DataPoint> &rawData)
{
    if (rawData.empty())
    {
        return rawData;
    }

    // Find the expected feature dimension by inspecting the first valid data point
    size_t expectedDim = 0;
    for (const auto &point : rawData)
    {
        if (!point.features.empty())
        {
            expectedDim = point.features.size();
            break;
        }
    }

    if (expectedDim == 0)
    {
        throw std::runtime_error("Could not determine feature dimension");
    }

    std::cout << "Expected feature dimension: " << expectedDim << std::endl;

    // Validate data points and keep only those with the expected feature dimension
    std::vector<DataPoint> normalizedData;
    for (const auto &point : rawData)
    {
        if (point.features.size() == expectedDim)
        {
            normalizedData.push_back(point);
        }
        else
        {
            std::cerr << "Skipping point with incorrect dimension: "
                      << point.features.size() << " (expected " << expectedDim << ")" << std::endl;
        }
    }

    if (normalizedData.empty())
    {
        throw std::runtime_error("No valid data points after dimension validation");
    }

    // Calculate the mean and standard deviation for each feature
    std::vector<double> means(expectedDim, 0.0);
    std::vector<double> stdDevs(expectedDim, 0.0);

    // Calculate means
    for (const auto &point : normalizedData)
    {
        for (size_t i = 0; i < expectedDim; ++i)
        {
            means[i] += point.features[i];
        }
    }
    for (double &mean : means)
    {
        mean /= normalizedData.size();
    }

    // Calculate standard deviations
    for (const auto &point : normalizedData)
    {
        for (size_t i = 0; i < expectedDim; ++i)
        {
            double diff = point.features[i] - means[i];
            stdDevs[i] += diff * diff;
        }
    }
    for (double &stdDev : stdDevs)
    {
        stdDev = std::sqrt(stdDev / normalizedData.size());
        if (stdDev < 1e-10)
            stdDev = 1.0; // Prevent division by zero
    }

    // Normalize the data using Z-score normalization
    for (auto &point : normalizedData)
    {
        for (size_t i = 0; i < expectedDim; ++i)
        {
            point.features[i] = (point.features[i] - means[i]) / stdDevs[i];
        }
    }

    std::cout << "Normalized " << normalizedData.size() << " data points" << std::endl;
    return normalizedData;
}

/**
 * @brief Initializes centroids using the k-means++ method.
 *
 * This function chooses the first centroid randomly and then iteratively chooses new centroids
 * with probability proportional to the square of the distance from each data point to the closest
 * centroid. The result is a set of centroids that are spread out and cover the input data well.
 *
 * @param data The input data points.
 */
void KMeansClassifier::initializeCentroids(const std::vector<DataPoint> &data)
{
    // Clear any existing centroids and choose the first point randomly
    centroids.clear();
    centroids.push_back(data[0].features);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Choose subsequent centroids based on the k-means++ method
    while (centroids.size() < static_cast<size_t>(k))
    {
        std::vector<double> distances(data.size(), std::numeric_limits<double>::max());

        // Calculate the distance from each point to the closest centroid
        for (size_t i = 0; i < data.size(); ++i)
        {
            for (const auto &centroid : centroids)
            {
                distances[i] = std::min(distances[i], computeDistance(data[i].features, centroid));
            }
        }

        // Choose a new centroid with probability proportional to the square of the distance
        std::discrete_distribution<> distribution(distances.begin(), distances.end());
        centroids.push_back(data[distribution(gen)].features);
    }
}

/**
 * @brief Trains the K-Means classifier using the input data.
 *
 * The training process involves initializing centroids using the k-means++ method,
 * and then iteratively assigning points to clusters and updating centroids until
 * convergence or a maximum number of iterations is reached.
 *
 * @param data The input data points to be used for training.
 */
void KMeansClassifier::train(const std::vector<DataPoint> &data)
{
    if (data.empty())
    {
        throw std::runtime_error("No training data provided");
    }

    // Initialize centroids
    initializeCentroids(data);

    bool converged = false;
    int iteration = 0;

    // Iteratively assign points to clusters and update centroids
    while (!converged && iteration < maxIterations)
    {
        std::vector<std::vector<const DataPoint *>> clusters(k);

        // Assign each point to the closest centroid (cluster)
        for (const auto &point : data)
        {
            int closestCluster = getClosestCentroid(point);
            clusters[closestCluster].push_back(&point);
        }

        // Update centroids based on assigned points
        converged = true;
        for (int i = 0; i < k; ++i)
        {
            if (clusters[i].empty())
            {
                // Reinitialize empty clusters
                initializeCentroids(data);
                converged = false;
                break;
            }

            std::vector<double> newCentroid(centroids[i].size(), 0.0);
            for (const DataPoint *point : clusters[i])
            {
                for (size_t j = 0; j < newCentroid.size(); ++j)
                {
                    newCentroid[j] += point->features[j];
                }
            }
            for (double &value : newCentroid)
            {
                value /= clusters[i].size();
            }

            // Check if centroids have converged
            if (computeDistance(newCentroid, centroids[i]) > convergenceThreshold)
            {
                converged = false;
            }
            centroids[i] = std::move(newCentroid);
        }

        ++iteration;
    }

    std::cout << "Training completed in " << iteration << " iterations." << std::endl;

    // Map clusters to labels
    mapClusterToLabels(data);
}

/**
 * @brief Maps each cluster to the most common label among its points.
 *
 * This function iterates over all clusters and counts the occurrences
 * of each label within the cluster. It assigns the label with the highest
 * frequency to the cluster. The mapping is stored in the clusterToLabel
 * map, where each cluster index is associated with the most common label.
 *
 * @param data The dataset containing the data points with known labels.
 */
void KMeansClassifier::mapClusterToLabels(const std::vector<DataPoint> &data)
{
    clusterToLabel.clear();
    for (int i = 0; i < k; ++i)
    {
        std::map<int, int> labelCount;
        for (const auto &point : data)
        {
            int closestCluster = getClosestCentroid(point);
            if (closestCluster == i)
            {
                labelCount[point.label]++;
            }
        }

        // Assign the most common label to this cluster
        int mostCommonLabel = -1;
        int maxCount = 0;
        for (const auto &labelPair : labelCount)
        {
            if (labelPair.second > maxCount)
            {
                mostCommonLabel = labelPair.first;
                maxCount = labelPair.second;
            }
        }

        clusterToLabel[i] = mostCommonLabel;
    }
}

/**
 * @brief Returns the index of the centroid closest to the given point.
 *
 * This function iterates over all centroids and calculates the Euclidean distance
 * between the point and each centroid. The index of the centroid with the smallest
 * distance is returned.
 *
 * @param point The data point to find the closest centroid for.
 * @return The index of the closest centroid.
 */
int KMeansClassifier::getClosestCentroid(const DataPoint &point) const
{
    int closestIndex = 0;
    double minDistance = std::numeric_limits<double>::max();

    for (size_t i = 0; i < centroids.size(); ++i)
    {
        double distance = computeDistance(point.features, centroids[i]);
        if (distance < minDistance)
        {
            minDistance = distance;
            closestIndex = i;
        }
    }

    return closestIndex;
}

/**
 * @brief Predicts the label of a given data point.
 *
 * This function finds the closest centroid to the given data point and returns the label
 * mapped to that centroid. The mapping of centroids to labels is determined by the
 * `train` function.
 *
 * @param point The data point to predict the label for.
 * @return The predicted label for the given data point.
 */
int KMeansClassifier::predict(const DataPoint &point)
{
    int closestCentroid = getClosestCentroid(point);
    return clusterToLabel[closestCentroid]; // Return the label mapped to the closest centroid
}

/**
 * @brief Computes the Euclidean distance between two vectors.
 *
 * This function takes two vectors `a` and `b` and returns the Euclidean distance
 * between them. The Euclidean distance is the square root of the sum of the squares
 * of the differences between corresponding elements of the two vectors.
 *
 * @param a The first vector.
 * @param b The second vector.
 * @return The Euclidean distance between the two vectors.
 */
double KMeansClassifier::computeDistance(const std::vector<double> &a, const std::vector<double> &b) const
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * @brief Predicts the label and returns the decision score for a given test data point.
 *
 * This function predicts the label similar to `predict`, but also calculates a score based on
 * the Euclidean distance to the closest centroid. The score is the negative distance,
 * so lower scores indicate a better fit.
 *
 * @param point The DataPoint for which the label and score are to be predicted.
 * @return A pair consisting of the predicted label and the decision score.
 */
std::pair<int, double> KMeansClassifier::predictWithScore(const DataPoint &point) const
{
    int closestCentroid = getClosestCentroid(point);
    double distance = computeDistance(point.features, centroids[closestCentroid]);
    return {closestCentroid, -distance}; // Return the centroid index and the negative distance (inverse for better score)
}