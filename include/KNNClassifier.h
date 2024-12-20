#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H

#include "DataPoint.h"
#include <vector>
#include <cmath>
#include <algorithm>

class KNNClassifier
{
private:
    std::vector<DataPoint> trainingData;
    int k; // Neighborhood size

public:
    explicit KNNClassifier(int k = 3) : k(k) {}

    void train(const std::vector<DataPoint> &data);
    int predict(const DataPoint &testPoint) const;
    static std::vector<DataPoint> normalizeData(const std::vector<DataPoint> &data);
    std::pair<int, double> predictWithScore(const DataPoint &testPoint) const;

private:
    double calculateDistance(const DataPoint &a, const DataPoint &b) const;
};

#endif // KNNCLASSIFIER_H
