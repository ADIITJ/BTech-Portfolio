QUESTION 1- PERCEPTRON IMPLEMENTATION 

This project implements the perceptron learning algorithm for binary classification tasks. The perceptron model is trained using the training data provided, and the learned weights are saved. Subsequently, the trained model is tested on new data to predict the labels of each sample.

Files

- train.py: This script trains the perceptron model using the provided training data and saves the learned weights.
- test.py: This script tests the trained perceptron model on new data to predict the labels of each sample.

Usage

Training the Perceptron Model

To train the perceptron model, execute the following command in the terminal:

python train.py train.txt

- train.txt: The training data file containing the feature vectors and labels of samples.

Upon execution, the script will train the perceptron model using the specified training data and save the learned weights. Once training is complete, the message "Training Over and Weights are saved" will be displayed.

Testing the Perceptron Model

To test the trained perceptron model on new data, execute the following command in the terminal:

python test.py test.txt

- test.txt: The test data file containing the feature vectors of samples.

Upon execution, the script will load the trained weights from the file "perceptron_weights.npy" and use them to predict the labels of samples in the test data. The predicted labels will be output in a comma-separated form.

Code Overview

- train.py:
  - f(X, w): Computes the dot product of feature vectors and weights.
  - z_norm(X): Performs Z-score normalization on the feature vectors.
  - perceptron_train(X_train, y_train): Trains the perceptron model using the provided training data.
  - read_input_file(filename): Reads the input data from the specified file.

- test.py:
  - f(X, w): Computes the dot product of feature vectors and weights.
  - z_norm(X): Performs Z-score normalization on the feature vectors.
  - perceptron_test(X_test, weights): Tests the trained perceptron model on new data to predict the labels.
  - read_input_file(filename): Reads the input data from the specified file.

Example

# Train the perceptron model
python train.py train.txt

# Test the trained model
python test.py test.txt

Author
Atharva Date
ROLL_NO: B22AI045
