#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include <vector>
#include <cmath>
#include <numeric>
#include <DataPoint.h>

class SVMClassifier {
public:
    // Constructeur avec ajout des paramètres regularization et gamma
    SVMClassifier(double learningRate, int maxIterations, double regularization = 0.01, double gamma = 0.1);

    // Méthodes publiques
    void train(const std::vector<DataPoint>& trainingData);
    int predict(const DataPoint& point) const;
    std::vector<DataPoint> normalizeData(const std::vector<DataPoint>& data) const;
    std::pair<int, double> predictWithScore(const DataPoint& point) const;

private:
    // Variables internes
    std::vector<double> weights;
    double bias;
    double learningRate;
    int maxIterations;
    double regularization; // Paramètre de régularisation
    double gamma;          // Paramètre pour le noyau RBF

    // Méthodes privées
    double calculateMargin(const DataPoint& point) const;
};

#endif // SVMCLASSIFIER_H
