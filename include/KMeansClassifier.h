#ifndef KMEANSCLASSIFIER_H
#define KMEANSCLASSIFIER_H

#include <vector>
#include <utility>
#include "DataPoint.h"
#include <map>

class KMeansClassifier
{
public:
    KMeansClassifier(int k, int maxIterations, double convergenceThreshold = 1e-4);

    std::map<int, int> clusterToLabel;

    void train(const std::vector<DataPoint> &rawData);
    int predict(const DataPoint &point);
    void test(const std::vector<DataPoint> &testData, std::vector<int> &predictions);
    std::pair<int, double> predictWithScore(const DataPoint &point) const;
    std::vector<DataPoint> normalizeData(const std::vector<DataPoint> &rawData);
    void mapClusterToLabels(const std::vector<DataPoint> &data);

private:
    int k;
    int maxIterations;
    double convergenceThreshold;
    std::vector<std::vector<double>> centroids;

    double computeDistance(const std::vector<double> &a, const std::vector<double> &b) const;
    int getClosestCentroid(const DataPoint &point) const;
    void initializeCentroids(const std::vector<DataPoint> &data);
};

#endif // KMEANSCLASSIFIER_H
