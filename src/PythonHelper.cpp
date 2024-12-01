
/*#include <Python.h>
#include "../include/PythonHelper.h"



void runPythonVisualization() {
    // Initialize the Python interpreter
    Py_Initialize();

    if (Py_IsInitialized()) {
        // Example Python code to plot using Matplotlib
        PyRun_SimpleString("import matplotlib.pyplot as plt");
        PyRun_SimpleString("plt.plot([1, 2, 3, 4], [10, 20, 25, 30])");
        PyRun_SimpleString("plt.title('Example Plot from C++')");
        PyRun_SimpleString("plt.xlabel('X-axis')");
        PyRun_SimpleString("plt.ylabel('Y-axis')");
        PyRun_SimpleString("plt.show()");
    }

    // Finalize the Python interpreter
    Py_Finalize();
}*/

#include <Python.h>
#include <iostream>
#include <cstdio>  // For fopen and FILE
#include "../include/PythonHelper.h"

void runPythonVisualization() {
    // Initialize the Python interpreter
    Py_Initialize();

    // Set the current working directory to the folder where Python scripts are located
    // This step ensures the script can be found
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./python')");

    // Open the Python script from the python directory
    FILE* file = fopen("python/TestScript.py", "r");
    if (file) {
        std::cout << "Running Python script..." << std::endl;
        // Execute the Python script
        PyRun_SimpleFile(file, "python/TestScript.py");
        fclose(file);
    } else {
        std::cerr << "Failed to open Python script!" << std::endl;
    }

    // Finalize the Python interpreter
    Py_Finalize();
}

