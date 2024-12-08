#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include <vector>
#include "DataPoint.h"

class SVMClassifier
{
private:
    std::vector<double> weights; // SVM weights
    double bias;                 // Bias
    double learningRate;         // Learning rate
    int maxIterations;           // Maximum number of iterations

public:
    SVMClassifier(double learningRate = 0.01, int maxIterations = 1000);

    void train(const std::vector<DataPoint> &trainingData);
    int predict(const DataPoint &point) const;

    std::vector<DataPoint> normalizeData(const std::vector<DataPoint> &data) const;
    std::pair<int, double> predictWithScore(const DataPoint &point) const;
};

#endif // SVMCLASSIFIER_H