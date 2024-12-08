#include "../include/MLPClassifier.h"

#include <cmath>
#include <algorithm>

/**
 * @brief Constructs an MLPClassifier with specified input, hidden, and output layer sizes.
 *
 * Initializes the weights and biases for the input-to-hidden and hidden-to-output layers
 * with small random values uniformly distributed between -0.5 and 0.5.
 *
 * @param inputSize The number of neurons in the input layer.
 * @param hiddenSize The number of neurons in the hidden layer.
 * @param outputSize The number of neurons in the output layer.
 */
MLPClassifier::MLPClassifier(int inputSize, int hiddenSize, int outputSize)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize)
{
    // Initialize weights and biases with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
    biasHidden.resize(hiddenSize);
    weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
    biasOutput.resize(outputSize);

    // Random initialization of weights and biases
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

/**
 * @brief Normalize the features of a given dataset of DataPoints.
 *
 * For each DataPoint, computes the mean and standard deviation of its features,
 * and then normalizes each feature by subtracting the mean and dividing by the
 * standard deviation, effectively mapping the features to a Z-score scale.
 *
 * @param data The input dataset of DataPoints.
 * @return A new dataset of normalized DataPoints.
 */
std::vector<DataPoint> MLPClassifier::normalizeData(const std::vector<DataPoint> &data) const
{
    std::vector<DataPoint> normalizedData = data;

    for (auto &point : normalizedData)
    {
        double mean = 0.0;
        double stddev = 0.0;

        // Calculate mean
        for (double value : point.features)
        {
            mean += value;
        }
        mean /= point.features.size();

        // Calculate standard deviation
        for (double value : point.features)
        {
            stddev += (value - mean) * (value - mean);
        }
        stddev = std::sqrt(stddev / point.features.size());

        // Normalization
        for (double &value : point.features)
        {
            value = (value - mean) / stddev; // Z-score normalization
        }
    }

    return normalizedData;
}

/**
 * @brief The sigmoid function maps a real-valued number to a value between 0 and 1.
 *
 * @param x The input value.
 * @return The sigmoid of x, i.e. 1 / (1 + exp(-x)).
 */
double MLPClassifier::sigmoid(double x) const
{
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * @brief Forward pass through the neural network.
 *
 * Given a set of input features, computes the output of the network by
 * propagating the input through the hidden layer and the output layer.
 *
 * @param input The input features.
 * @return A std::pair containing the hidden layer output and the output layer
 *         output.
 */
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

/**
 * @brief Computes the softmax of a vector of logits.
 *
 * The softmax function transforms a vector of logits into a probability
 * distribution, ensuring that the output values are non-negative and sum to 1.
 * The transformation also applies a numerical stabilization technique by
 * subtracting the maximum logit value from each logit before exponentiation.
 *
 * @param logits The input vector of logits.
 * @return A vector of probabilities corresponding to the softmax of the logits.
 */
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

/**
 * @brief Trains the MLPClassifier using the provided training data.
 *
 * This function performs training over a specified number of epochs,
 * adjusting the model's weights and biases based on the input features
 * and labels of the training data. The learning rate determines the step
 * size for weight updates during backpropagation.
 *
 * The training process involves forward propagation to compute the
 * outputs, followed by backpropagation to calculate the gradients and
 * update the parameters. The model learns by minimizing the error
 * between the predicted and actual labels using gradient descent.
 *
 * @param trainingData A vector of DataPoints containing input features
 * and corresponding labels for training.
 * @param epochs The number of complete passes through the training dataset.
 * @param learningRate The step size for updating weights during training.
 */
void MLPClassifier::train(const std::vector<DataPoint> &trainingData, int epochs, double learningRate)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (const auto &data : trainingData)
        {
            int inputSize = data.features.size(); // Redefine the input size for each sample

            // Resize weights and biases according to the new input size
            weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
            biasHidden.resize(hiddenSize);

            // Forward propagation
            auto [hidden, output] = forward(data.features);

            // Calculate the error gradient for each class
            std::vector<double> outputDeltas(outputSize);
            for (int k = 0; k < outputSize; ++k)
            {
                outputDeltas[k] = (data.label == k ? 1.0 : 0.0) - output[k]; // One-hot encoded target
                outputDeltas[k] *= output[k] * (1.0 - output[k]);            // Apply sigmoid derivative
            }

            std::vector<double> hiddenDeltas(hiddenSize, 0.0);
            for (int k = 0; k < outputSize; ++k)
            {
                for (int j = 0; j < hiddenSize; ++j)
                {
                    hiddenDeltas[j] += outputDeltas[k] * weightsHiddenOutput[j][k];
                }
            }

            // Update weights and biases for the output layer
            for (int k = 0; k < outputSize; ++k)
            {
                for (int j = 0; j < hiddenSize; ++j)
                {
                    weightsHiddenOutput[j][k] += learningRate * outputDeltas[k] * hidden[j];
                }
                biasOutput[k] += learningRate * outputDeltas[k];
            }

            // Update weights and biases for the hidden layer
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

/**
 * @brief Predicts the class label for a given data point.
 *
 * This function performs a forward pass through the neural network using the
 * provided features of the data point, and returns the class label with the
 * highest predicted probability.
 *
 * @param point The DataPoint containing the input features.
 * @return The predicted class label as an integer.
 */
int MLPClassifier::predict(const DataPoint &point) const
{
    auto [hidden, output] = forward(point.features);
    // Find the index of the class with the highest probability
    int predictedClass = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    return predictedClass;
}

/**
 * @brief Predicts the class label for a given data point and returns the score for the most likely class.
 *
 * This function performs a forward pass through the neural network using the
 * provided features of the data point, and returns the class label with the
 * highest predicted probability, along with the score for the most likely class.
 *
 * @param point The DataPoint containing the input features.
 * @return A std::pair containing the predicted class label as an integer, and the score of the most likely class as a double.
 */
std::pair<int, double> MLPClassifier::predictWithScore(const DataPoint &point) const
{
    auto [hidden, output] = forward(point.features);

    // Find the index of the class with the highest probability
    int predictedClass = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    double maxScore = output[predictedClass];

    // Return the predicted class and the probability of that class
    return {predictedClass, maxScore};
}
