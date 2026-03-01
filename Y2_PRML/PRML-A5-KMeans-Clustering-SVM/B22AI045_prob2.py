## Q2
### Task 1
#### a)
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris(as_frame=True)
X = iris.data[['petal length (cm)', 'petal width (cm)']]
y = iris.target

X_binary = X[y != 2]
y_binary = y[y != 2]

scaler = StandardScaler()
X_binary_scaled = scaler.fit_transform(X_binary)

X_train, X_test, y_train, y_test = train_test_split(X_binary_scaled, y_binary, test_size=0.3, random_state=42)
#### b)
# Plot scatter plot
from sklearn import svm
clf = svm.SVC(kernel="linear", gamma=0).fit(X_train, y_train)
x_min, x_max = -3,3
y_min, y_max = -3,3
    
plt.figure(figsize=(10, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Scatter plot of train data \nwith decision boundary')

w = clf.coef_[0]
b = clf.intercept_[0]
x_plot = np.linspace(x_min, x_max, 100)
y_plot = -(w[0] * x_plot + b) / w[1]
plt.plot(x_plot, y_plot, '-r', label='Decision Boundary')

plt.legend()
plt.show()

x_min, x_max = -3,3
y_min, y_max = -3,3
    
plt.figure(figsize=(10, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Scatter plot of test data \nwith decision boundary')

w = clf.coef_[0]
b = clf.intercept_[0]
x_plot = np.linspace(x_min, x_max, 100)
y_plot = -(w[0] * x_plot + b) / w[1]
plt.plot(x_plot, y_plot, '-r', label='Decision Boundary')

plt.legend()
plt.show()
### Task 2
#### a)
from sklearn.datasets import make_moons
X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=42)

#### b)
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay


def plot_training_data_with_decision_boundary(kernel, X, y):
    # Train the SVC
    clf = svm.SVC(kernel=kernel, gamma=1).fit(X, y)

    # Settings for plotting
    _, ax = plt.subplots(figsize=(10, 7))
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=250,
        facecolors="none",
        edgecolors="k",
    )
    # Plot samples by color and add legend
    ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors="k")
    ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")

    _ = plt.show()
plot_training_data_with_decision_boundary(kernel='linear', X=X_moons, y=y_moons)
plot_training_data_with_decision_boundary(kernel='poly', X=X_moons, y=y_moons)
plot_training_data_with_decision_boundary(kernel='rbf', X=X_moons, y=y_moons)
#### c)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

# Define parameter distributions for randomized search
param_dist = {'C': reciprocal(0.1, 100), 'gamma': reciprocal(0.001, 1)}

# Perform randomized search
random_search = RandomizedSearchCV(svm.SVC(kernel='rbf'), param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
random_search.fit(X_moons, y_moons)

# Get best hyperparameters
best_params_random = random_search.best_params_
print("Best hyperparameters:", best_params_random)

#### d)
def plot_best_boundary(kernel, X, y):
    # Train the SVC
    clf = svm.SVC(kernel=kernel,C = best_params_random['C'] , gamma=best_params_random['gamma']).fit(X, y)

    # Settings for plotting
    _, ax = plt.subplots(figsize=(10, 7))
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=250,
        facecolors="none",
        edgecolors="k",
    )
    # Plot samples by color and add legend
    ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors="k")
    ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")

    _ = plt.show()
plot_best_boundary(kernel='rbf', X=X_moons, y=y_moons)
from sklearn.metrics import accuracy_score
clf1 = svm.SVC(kernel='rbf',gamma=5)
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred1)
print("Accuracy score of gamma = 1:", accuracy)

clf2 = svm.SVC(kernel='rbf',gamma=best_params_random['gamma'], C=best_params_random['C'])
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

accuracy = accuracy_score(y_test, y_pred2)
print("Accuracy score of best params:", accuracy)
