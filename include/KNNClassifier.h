
// comment correction

#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H
#include <vector>
#include <string>

// KNN Classifier Declaration
class KNNClassifier {
private:
    int k; // Number of neighbors
    std::vector<std::vector<double>> data; // Data points
    std::vector<int> labels; // Corresponding labels

    // Helper function to calculate Euclidean distance
    //double calculateDistance(const std::vector<double>& point1, const std::vector<double>& point2);
    
    //Correction
    //Euclidean Distance
    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) const;


public:
    // Constructor
    KNNClassifier(int neighbors);

    // Function to load data from files
    void loadData(const std::string& folderPath);

    // KNN prediction function
    int classify(const std::vector<double>& input);

    // Utility: Normalize the dataset (optional)
    void normalizeData();
    
    //Correction
    //Getter For Data
    const std::vector<std::vector<double>>& getData() const {
        return data;
    }
    
    // Getter for labels
    const std::vector<int>& getLabels() const{
        return labels;
    }
};



#endif // KNNCLASSIFIER_H
