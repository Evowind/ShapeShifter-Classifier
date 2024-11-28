#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H
//#include "KNNClassifier.cpp"  // Include KMeansClassifier to access DataPoint (??)
#include "DataPoint.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>



// KNNClassifier class definition
class KNNClassifier {
    int k; // Number of neighbors
    std::vector<DataPoint> trainingData;

public:
    // Constructor
    KNNClassifier(int kValue);

    // Set data for the classifier
    void setData(const std::vector<DataPoint>& data);

    // Normalize the dataset
    void normalizeData();
    
    //Train the Classifier(Store and Normalize data)
    void train(const std::vector<DataPoint>& data);

    // Predict the label for a single input point
    int classify(const std::vector<double>& input) const;

    // Test the classifier on a dataset and display the accuracy
    void testAndDisplayResults(const std::vector<DataPoint>& testData);
    
    //Getter for training data
    const std::vector<DataPoint>& getData() const ;
    
    //Getter for labels
    std::vector<int> getLabels() const;
};

#endif // KNNCLASSIFIER_H