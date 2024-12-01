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
    static std::pair<std::vector<DataPoint>, std::vector<DataPoint>> splitTrainTest(
        const std::vector<DataPoint>& data, double trainRatio = 0.7);

    template <typename Classifier>
    static void testAndDisplayResults(Classifier& classifier, const std::vector<DataPoint>& testData);

    void computePrecisionRecallCurve(
                                    const std::vector<DataPoint>& testData,
                                    const std::vector<double>& scores,
                                    const std::vector<int>& trueLabels,
                                    const std::string& outputCsvPath);

    template <typename Classifier>
    void evaluateWithPrecisionRecall(
                                    const Classifier& classifier,
                                    const std::vector<DataPoint>& testData,
                                    const std::string& outputCsvPath);

private:
    static void displayConfusionMatrix(const std::vector<std::vector<int>>& matrix);
};



#endif
