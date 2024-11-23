

#include "KNNClassifier.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <map>

// Constructor
KNNClassifier::KNNClassifier(int k) : k(k) {}

// Function to load data
void KNNClassifier::loadData(const std::string& datasetPath) {
    std::cout << "Loading data from: " << datasetPath << std::endl;

    for (const auto& folder : {"ART", "E34", "GFD", "Yang", "Zernike7"}) {
        std::string folderPath = datasetPath + "/" + folder;
        std::cout << "Processing folder: " << folderPath << std::endl;

        for (int subject = 1; subject <= 18; ++subject) {
            for (int sample = 1; sample <= 12; ++sample) {
                std::string filename = folderPath + "/s" +
                                       (subject < 10 ? "0" : "") + std::to_string(subject) +
                                       "n" + (sample < 10 ? "00" : "0") + std::to_string(sample);

                if (std::string(folder) == "ART")
                {
                    filename += ".art";
                }
                   
                else if (std::string(folder) == "E34")
                {
                    filename += ".e34";
                }
                   
                else if (std::string(folder) == "GFD")
                         {
                    filename += ".gfd";
                }
                    
                else if (std::string(folder) == "Yang")
                         {
                    filename += ".yng";
                }
                    
                else if (std::string(folder) == "Zernike7")
                         {
                    filename += ".zrk.txt";
                }
                    

                std::cout << "Loading file: " << filename << std::endl;

                // Open the file
                std::ifstream file(filename);
                if (!file) {
                    std::cerr << "Failed to open file: " << filename << std::endl;
                    continue;
                }

                std::vector<double> features;
                double value;
                while (file >> value) {
                    features.push_back(value);
                }

                if (!features.empty()) {
                    std::cout << "Loaded " << features.size() << " features from file: " << filename << std::endl;
                }

                data.push_back(features);
                labels.push_back(subject); // Assuming the label corresponds to the subject
            }
        }
    }

    std::cout << "Finished loading data. Total samples: " << data.size() << std::endl;
}

// Function to normalize the data
void KNNClassifier::normalizeData() {
    if (data.empty()) {
        std::cerr << "No data to normalize." << std::endl;
        return;
    }

    size_t featureCount = data[0].size();
    std::vector<double> minValues(featureCount, std::numeric_limits<double>::max());
    std::vector<double> maxValues(featureCount, std::numeric_limits<double>::lowest());

    // Find min and max for each feature
    for (const auto& sample : data) {
        for (size_t i = 0; i < sample.size(); ++i) {
            minValues[i] = std::min(minValues[i], sample[i]);
            maxValues[i] = std::max(maxValues[i], sample[i]);
        }
    }

    // Normalize data
    for (auto& sample : data) {
        for (size_t i = 0; i < sample.size(); ++i) {
            if (maxValues[i] > minValues[i]) {
                sample[i] = (sample[i] - minValues[i]) / (maxValues[i] - minValues[i]);
            }
        }
    }

    std::cout << "Data normalization complete." << std::endl;
}

// Function to calculate Euclidean distance
double KNNClassifier::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Function to classify input
int KNNClassifier::classify(const std::vector<double>& input) {
    if (data.empty() || labels.empty()) {
        std::cerr << "No data available for classification. Load data first." << std::endl;
        return -1;
    }

    if (input.size() != data[0].size()) {
        std::cerr << "Input size does not match feature size. Cannot classify." << std::endl;
        return -1;
    }

    // Debug: print the input vector
    std::cout << "Classifying input: ";
    for (const auto& value : input) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Calculate distances
    std::vector<std::pair<double, int>> distances;
    for (size_t i = 0; i < data.size(); ++i) {
        double dist = euclideanDistance(input, data[i]);
        distances.push_back({dist, labels[i]});
    }

    // Sort distances
    std::sort(distances.begin(), distances.end());

    // Debug: Print top k neighbors
    std::cout << "Top " << k << " nearest neighbors:" << std::endl;
    for (int i = 0; i < k && i < static_cast<int>(distances.size()); ++i) {
        std::cout << "Neighbor " << i + 1 << ": Label = " << distances[i].second
                  << ", Distance = " << distances[i].first << std::endl;
    }

    // Voting mechanism
    std::map<int, int> votes;
    for (int i = 0; i < k && i < static_cast<int>(distances.size()); ++i) {
        votes[distances[i].second]++;
    }

    // Determine the label with the most votes
    int predictedLabel = -1;
    int maxVotes = 0;
    for (const auto& vote : votes) {
        if (vote.second > maxVotes) {
            maxVotes = vote.second;
            predictedLabel = vote.first;
        }
    }

    std::cout << "Predicted Label: " << predictedLabel << " with " << maxVotes << " votes." << std::endl;
    return predictedLabel;
}

