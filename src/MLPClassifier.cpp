#include "MLPClassifier.h"

#include <cmath>
#include <algorithm>

MLPClassifier::MLPClassifier(int inputSize, int hiddenSize, int outputSize)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize)
{
    // Initialiser les poids et biais avec des petites valeurs aléatoires
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
    biasHidden.resize(hiddenSize);
    weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
    biasOutput.resize(outputSize);

    // Initialisation aléatoire des poids et biais
    for (int i = 0; i < inputSize; ++i)
    {
        for (int j = 0; j < hiddenSize; ++j)
        {
            weightsInputHidden[i][j] = dis(gen);
        }
    }
    for (int j = 0; j < hiddenSize; ++j)
    {
        biasHidden[j] = dis(gen);
    }
    for (int j = 0; j < hiddenSize; ++j)
    {
        for (int k = 0; k < outputSize; ++k)
        {
            weightsHiddenOutput[j][k] = dis(gen);
        }
    }
    for (int k = 0; k < outputSize; ++k)
    {
        biasOutput[k] = dis(gen);
    }
}

std::vector<DataPoint> MLPClassifier::normalizeData(const std::vector<DataPoint> &data) const
{
    std::vector<DataPoint> normalizedData = data;

    for (auto &point : normalizedData)
    {
        double mean = 0.0;
        double stddev = 0.0;

        // Calcul de la moyenne
        for (double value : point.features)
        {
            mean += value;
        }
        mean /= point.features.size();

        // Calcul de l'écart-type
        for (double value : point.features)
        {
            stddev += (value - mean) * (value - mean);
        }
        stddev = std::sqrt(stddev / point.features.size());

        // Normalisation
        for (double &value : point.features)
        {
            value = (value - mean) / stddev; // Normalisation Z-score
        }
    }

    return normalizedData;
}

double MLPClassifier::sigmoid(double x) const
{
    return 1.0 / (1.0 + std::exp(-x));
}

double MLPClassifier::sigmoidDerivative(double x) const
{
    return x * (1.0 - x);
}

std::pair<std::vector<double>, std::vector<double>> MLPClassifier::forward(const std::vector<double> &input) const
{
    std::vector<double> hidden(hiddenSize);
    for (int j = 0; j < hiddenSize; ++j)
    {
        hidden[j] = 0;
        for (int i = 0; i < inputSize; ++i)
        {
            hidden[j] += input[i] * weightsInputHidden[i][j];
        }
        hidden[j] += biasHidden[j];
        hidden[j] = sigmoid(hidden[j]);
    }

    std::vector<double> logits(outputSize);
    for (int k = 0; k < outputSize; ++k)
    {
        logits[k] = 0.0;
        for (int j = 0; j < hiddenSize; ++j)
        {
            logits[k] += hidden[j] * weightsHiddenOutput[j][k];
        }
        logits[k] += biasOutput[k];
    }

    std::vector<double> output = softmax(logits);
    return {hidden, output};
}

std::vector<double> MLPClassifier::softmax(const std::vector<double> &logits) const
{
    std::vector<double> probabilities(logits.size());
    double maxLogit = *std::max_element(logits.begin(), logits.end());
    double sumExp = 0.0;

    for (double logit : logits)
    {
        sumExp += std::exp(logit - maxLogit);
    }

    for (size_t i = 0; i < logits.size(); ++i)
    {
        probabilities[i] = std::exp(logits[i] - maxLogit) / sumExp;
    }

    return probabilities;
}

void MLPClassifier::train(const std::vector<DataPoint> &trainingData, int epochs, double learningRate)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (const auto &data : trainingData)
        {
            int inputSize = data.features.size(); // Redéfinir la taille de l'entrée pour chaque échantillon

            // Redimensionner les poids et biais en fonction de la nouvelle taille d'entrée
            weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
            biasHidden.resize(hiddenSize);

            // Propagation avant
            auto [hidden, output] = forward(data.features);

            // Calcul du gradient de l'erreur pour chaque classe
            std::vector<double> outputDeltas(outputSize);
            for (int k = 0; k < outputSize; ++k)
            {
                outputDeltas[k] = (data.label == k ? 1.0 : 0.0) - output[k]; // One-hot encoded target
                outputDeltas[k] *= output[k] * (1.0 - output[k]);            // Application de la dérivée de la sigmoid
            }

            std::vector<double> hiddenDeltas(hiddenSize, 0.0);
            for (int k = 0; k < outputSize; ++k)
            {
                for (int j = 0; j < hiddenSize; ++j)
                {
                    hiddenDeltas[j] += outputDeltas[k] * weightsHiddenOutput[j][k];
                }
            }

            // Mise à jour des poids et biais pour la couche de sortie
            for (int k = 0; k < outputSize; ++k)
            {
                for (int j = 0; j < hiddenSize; ++j)
                {
                    weightsHiddenOutput[j][k] += learningRate * outputDeltas[k] * hidden[j];
                }
                biasOutput[k] += learningRate * outputDeltas[k];
            }

            // Mise à jour des poids et biais pour la couche cachée
            for (int j = 0; j < hiddenSize; ++j)
            {
                for (int i = 0; i < inputSize; ++i)
                {
                    weightsInputHidden[i][j] += learningRate * hiddenDeltas[j] * data.features[i];
                }
                biasHidden[j] += learningRate * hiddenDeltas[j];
            }
        }
    }
}

int MLPClassifier::predict(const DataPoint &point) const
{
    auto [hidden, output] = forward(point.features);
    // Trouver l'index de la classe avec la probabilité la plus élevée
    int predictedClass = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    return predictedClass;
}

std::pair<int, double> MLPClassifier::predictWithScore(const DataPoint &point) const
{
    auto [hidden, output] = forward(point.features);

    // Trouver l'indice de la classe ayant la probabilité la plus élevée
    int predictedClass = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    double maxScore = output[predictedClass];

    // Retourner la classe prédite et la probabilité de cette classe
    return {predictedClass, maxScore};
}
