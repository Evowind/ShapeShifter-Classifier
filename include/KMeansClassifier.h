#ifndef KMEANS_CLASSIFIER_H
#define KMEANS_CLASSIFIER_H

#include <vector>
#include <string>

struct DataPoint {
    std::vector<double> features;  // Vecteur de caractéristiques pour chaque image
    int label;                     // Classe réelle de la donnée, pour évaluation
};

class KMeansClassifier {
public:
    KMeansClassifier(int k, int maxIterations = 100);
    void train(const std::vector<DataPoint>& data);
    int predict(const DataPoint& point);
    void testAndDisplayResults(const std::vector<DataPoint>& testData);

private:
    int k; // Nombre de clusters
    int maxIterations;
    std::vector<std::vector<double>> centroids;

    double computeDistance(const std::vector<double>& a, const std::vector<double>& b);
    int getClosestCentroid(const DataPoint& point);
    void updateCentroids(const std::vector<std::vector<DataPoint>>& clusters);
};

#endif
