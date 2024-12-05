#ifndef CLASSIFIER_EVALUATION_H
#define CLASSIFIER_EVALUATION_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <string>
#include <fstream>
#include <filesystem>  // Pour la gestion des répertoires
#include "DataPoint.h"

class ClassifierEvaluation {
public:
    
    // Fonction de séparation des données en ensemble d'entraînement et de test
    static std::pair<std::vector<DataPoint>, std::vector<DataPoint>> splitTrainTest(
        const std::vector<DataPoint>& data, double trainRatio = 0.7, bool stratified = true, int minTestSamplesPerClass = 3);

    // Fonction de test et d'affichage des résultats
    template <typename Classifier>
    static void testAndDisplayResults(Classifier& classifier, const std::vector<DataPoint>& testData);

    // Fonction pour calculer la courbe précision/rappel
    void computePrecisionRecallCurve(
        const std::vector<DataPoint>& testData,
        const std::vector<double>& scores,
        const std::vector<int>& trueLabels,
        const std::string& outputCsvPath);

    // Fonction pour évaluer le classifieur avec la courbe précision/rappel
    template <typename Classifier>
    void evaluateWithPrecisionRecall(
        const Classifier& classifier,
        const std::vector<DataPoint>& testData,
        const std::string& outputCsvPath);
    
private:
    // Fonction privée pour afficher la matrice de confusion
    static void displayConfusionMatrix(const std::vector<std::vector<int>>& matrix);
};

#endif
