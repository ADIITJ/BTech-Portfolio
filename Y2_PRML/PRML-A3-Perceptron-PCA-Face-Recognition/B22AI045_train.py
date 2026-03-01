import numpy as np

def f(X, w):
    return np.dot(X, w[1:]) + w[0]

def z_norm(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

def perceptron_train(X_train, y_train):
    w = np.random.randn(X_train.shape[1] + 1)

    X_train = z_norm(X_train)

    while True:
        misclassified = False

        for x, y in zip(X_train, y_train):
            y_pred = f(x, w)

            if y_pred >= 0:
                y_pred = 1
            else:
                y_pred = 0

            if y != y_pred:
                misclassified = True
                w[1:] += (y - y_pred)*x
                w[0] += (y - y_pred)

        if not misclassified:
            break

    return w


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

    # Load training data
    train_file = sys.argv[1]
    data = read_input_file(train_file)
    X_train = data[:, :-1]
    y_train = data[:, -1]

    # Train the perceptron
    weights = perceptron_train(X_train, y_train)
    print('Training Over and Weights are saved')
    # Save the weights
    np.save('perceptron_weights.npy', weights)
