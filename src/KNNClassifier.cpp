#include "KNNClassifier.h"

// Constructor
KNNClassifier::KNNClassifier(int kValue) : k(kValue) {}

// Set data for the classifier
void KNNClassifier::setData(const std::vector<DataPoint>& data) {
    trainingData = data;
}

// Normalize the dataset
void KNNClassifier::normalizeData() {
    if (trainingData.empty()) return;

    size_t numFeatures = trainingData[0].features.size();
    std::vector<double> minValues(numFeatures, std::numeric_limits<double>::max());
    std::vector<double> maxValues(numFeatures, std::numeric_limits<double>::lowest());

    // Find min and max for each feature
    for (const auto& point : trainingData) {
        for (size_t i = 0; i < numFeatures; ++i) {
            minValues[i] = std::min(minValues[i], point.features[i]);
            maxValues[i] = std::max(maxValues[i], point.features[i]);
        }
    }

    // Normalize features
    for (auto& point : trainingData) {
        for (size_t i = 0; i < numFeatures; ++i) {
            if (maxValues[i] != minValues[i]) { // Avoid division by zero
                point.features[i] = (point.features[i] - minValues[i]) / (maxValues[i] - minValues[i]);
            }
        }
    }
}
// Train the classifier(store and normalize data)
void KNNClassifier::train(const std::vector<DataPoint>& data) {
    setData(data);
    normalizeData();
}
// Predict the label for a single input point
int KNNClassifier::classify(const std::vector<double>& input) const {
    if (trainingData.empty()) {
        throw std::runtime_error("No training data available");
    }

    // Calculate distances to all training points
    std::vector<std::pair<double, int>> distances; // {distance, label}
    for (const auto& point : trainingData) {
        double distance = 0.0;
        for (size_t i = 0; i < input.size(); ++i) {
            distance += std::pow(input[i] - point.features[i], 2);
        }
        distance = std::sqrt(distance);
        distances.push_back({distance, point.label});
    }

    // Sort distances
    std::sort(distances.begin(), distances.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first < b.first;
              });

    // Find the most common label among the k nearest neighbors
    std::vector<int> labelCounts(100, 0); // Adjust size if know the max label value
    for (int i = 0; i < k && i < distances.size(); ++i) {
        ++labelCounts[distances[i].second];
    }

    // Return the label with the highest count
    return std::distance(labelCounts.begin(), std::max_element(labelCounts.begin(), labelCounts.end()));
}

// Test the classifier on a dataset and display the accuracy
void KNNClassifier::testAndDisplayResults(const std::vector<DataPoint>& testData) {
    if (testData.empty()) {
        std::cerr << "Test dataset is empty." << std::endl;
        return;
    }

    int correct = 0;
    for (const auto& point : testData) {
        int predictedLabel = classify(point.features);
        if (predictedLabel == point.label) {
            ++correct;
        }
    }

    double accuracy = static_cast<double>(correct) / testData.size() * 100.0;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
}

// Getter for Training Data
const std::vector<DataPoint>& KNNClassifier::getData() const {
    return trainingData;
}

// Getter for Labels
std::vector<int> KNNClassifier::getLabels() const {
    std::vector<int> labels;
    for(const auto& point : trainingData) {
        labels.push_back(point.label);
    }
    return labels;
}
