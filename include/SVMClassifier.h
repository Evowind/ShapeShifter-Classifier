#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include <vector>
#include "DataPoint.h"

class SVMClassifier {
private:
    std::vector<double> weights; // Poids du SVM
    double bias;                 // Biais
    double learningRate;         // Taux d'apprentissage
    int maxIterations;           // Nombre maximal d'itérations

public:
    SVMClassifier(double learningRate = 0.01, int maxIterations = 1000);

    void train(const std::vector<DataPoint>& trainingData);
    int predict(const DataPoint& point) const;

    std::vector<DataPoint> normalizeData(const std::vector<DataPoint>& data) const;

    // Getter pour vérifier les poids (optionnel)
    const std::vector<double>& getWeights() const { return weights; }
    double getBias() const { return bias; }
};

#endif // SVMCLASSIFIER_H
