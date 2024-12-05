#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <random>
#include "DataPoint.h"

class MLPClassifier {
public:
    MLPClassifier(int inputSize, int hiddenSize, int outputSize);
    void train(const std::vector<DataPoint>& trainingData, int epochs = 1000, double learningRate = 0.01);
    std::pair<int, double> predictWithScore(const DataPoint& point) const;
    std::vector<DataPoint> normalizeData(const std::vector<DataPoint>& data) const;
    int predict(const DataPoint& point) const;
    std::vector<double> softmax(const std::vector<double>& logits) const;

private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<double> biasHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> biasOutput;

    // Fonction d'activation sigmoïde
    double sigmoid(double x) const;
    double sigmoidDerivative(double x) const;

    // Propagation avant
    std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double>& input) const;
    void backpropagate(const std::vector<DataPoint>& trainingData, double learningRate);
};
