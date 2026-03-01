#This is a helper code for problem-1 (Task-1) of PA-4
#Complete this code by writing the function definations
#Compute following terms and print them:\\
#1. Difference of class wise means = ${m_1-m_2}$\\
#2. Total Within-class Scatter Matrix $S_W$\\
#3. Between-class Scatter Matrix $S_B$\\
#4. The EigenVectors of matrix $S_W^{-1}S_B$ corresponding to highest EigenValue\\
#5. For any input 2-D point, print its projection according to LDA.
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def ComputeMeanDiff(X):
    class1_mean = np.mean(X[X[:, -1] == '0.0'][:, :-1].astype(float))
    class2_mean = np.mean(X[X[:, -1] == '1.0'][:, :-1].astype(float))
    return class1_mean - class2_mean

def ComputeSW(X):
    class1_samples = X[X[:, 2] == '0.0'][:, :-1].astype(float)
    class2_samples = X[X[:, 2] == '1.0'][:, :-1].astype(float)
    class1_mean = np.mean(class1_samples, axis=0)
    class2_mean = np.mean(class2_samples, axis=0)
    class1_centered = class1_samples - class1_mean
    class2_centered = class2_samples - class2_mean
    sw_class1 = np.dot(class1_centered.T, class1_centered)
    sw_class2 = np.dot(class2_centered.T, class2_centered)
    return sw_class1 + sw_class2


def ComputeSB(X):
    class1_samples = X[X[:, -1] == '0.0'][:, :-1].astype(float)
    class2_samples = X[X[:, -1] == '1.0'][:, :-1].astype(float)
    class1_mean = np.mean(class1_samples, axis=0)
    class2_mean = np.mean(class2_samples, axis=0)

    overall_mean = np.mean(X[:, :-1].astype(float),axis=0)
    sb_class1 = len(class1_samples) * np.outer((class1_mean - overall_mean), (class1_mean - overall_mean))
    sb_class2 = len(class2_samples) * np.outer((class2_mean - overall_mean), (class2_mean - overall_mean))

    return sb_class1 + sb_class2

def GetLDAProjectionVector(X):
    sw = ComputeSW(X)
    sb = ComputeSB(X)
    eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.inv(sw), sb))
    max_idx = np.argmax(eigvals)
    return eigvecs[:, max_idx]

def project(x, y, w):
    point = np.array([x, y])
    return np.dot(point, w)

#########################################################
###################Helper Code###########################
#########################################################

X = np.empty((0, 3))
with open('data.csv', mode ='r') as file:
    csvFile = csv.reader(file)
    for sample in csvFile:
        X = np.vstack((X, sample))

print(X)
print(X.shape)
# X Contains m samples each of format (x,y) and class label 0.0 or 1.0

opt = int(input("Input your option (1-5): "))

if opt == 1:
    meanDiff = ComputeMeanDiff(X)
    print(meanDiff)
elif opt == 2:
    SW = ComputeSW(X)
    print(SW)
elif opt == 3:
    SB = ComputeSB(X)
    print(SB)
elif opt == 4:
    w = GetLDAProjectionVector(X)
    print(w)
elif opt == 5:
    x = int(input("Input x dimension of a 2-dimensional point: "))
    y = int(input("Input y dimension of a 2-dimensional point: "))
    w = GetLDAProjectionVector(X)
    print(project(x, y, w))

