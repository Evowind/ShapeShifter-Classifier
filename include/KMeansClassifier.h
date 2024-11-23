#ifndef KMEANS_CLASSIFIER_H
#define KMEANS_CLASSIFIER_H

#include <vector>
#include <string>

struct DataPoint {
    std::vector<double> features;  // Feature vector for each image
    int label;                     // Actual class of the data, for evaluation
};

class KMeansClassifier {
public:
    KMeansClassifier(int k, int maxIterations = 100);
    void train(const std::vector<DataPoint>& data);
    int predict(const DataPoint& point);
    void testAndDisplayResults(const std::vector<DataPoint>& testData);

private:
    int k; // Number of clusters
    int maxIterations;
    std::vector<std::vector<double>> centroids;

    double computeDistance(const std::vector<double>& a, const std::vector<double>& b);
    int getClosestCentroid(const DataPoint& point);
    std::vector<DataPoint> normalizeData(const std::vector<DataPoint>& rawData);
};




#endif
