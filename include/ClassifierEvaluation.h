#ifndef CLASSIFIER_EVALUATION_H
#define CLASSIFIER_EVALUATION_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include "DataPoint.h"

class ClassifierEvaluation {
public:
    
    // changing return type to tuple
    static std::tuple<std::vector<DataPoint>, std::vector<DataPoint>> splitTrainTest(
        const std::vector<DataPoint>& data, double trainRatio = 0.7);

    template <typename Classifier>
    static void testAndDisplayResults(Classifier& classifier, const std::vector<DataPoint>& testData, int numClusters);

private:
    static void displayConfusionMatrix(const std::vector<std::vector<int>>& matrix);
};

#endif
