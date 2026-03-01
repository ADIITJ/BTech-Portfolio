import numpy as np

def f(X, w):
    return np.dot(X, w[1:]) + w[0]

def z_norm(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

def perceptron_test(X_test, weights):
    X_test = z_norm(X_test)
    y_pred = f(X_test, weights)
    labels = np.where(y_pred >= 0, 1, 0)
    return labels

def read_input_file(filename):
    with open(filename, 'r') as file:
        num_vectors = int(file.readline().strip())
        vectors = []
        for _ in range(num_vectors):
            vector = list(map(float, file.readline().strip().split()))
            vectors.append(vector)
        return np.array(vectors)
        
if __name__ == "__main__":
    import sys

    # Load test data
    test_file = sys.argv[1]
    X_test = read_input_file(test_file)

    # Load weights
    weights = np.load('perceptron_weights.npy')

    # Perform testing
    y_pred = perceptron_test(X_test, weights)

    # Output predicted labels
    print(",".join(map(str, y_pred)))
