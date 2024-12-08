#ifndef DATAPOINT_H
#define DATAPOINT_H

#include <vector>

struct DataPoint
{
    int label;                    // The label or class of the data point
    std::vector<double> features; // The feature vector of the data point
};

#endif // DATAPOINT_H
