# ECE475
# Assignment 2
# Jonathan Lam, Tiffany Yu, Harris Paspuleti

# Setting up
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb
#import the South African heart disease dataset

#!wget https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data
dataset = pd.read_csv('SAheart.data', index_col=0)
#take out the column of adiposity and typea
dataset.pop('adiposity')
dataset.pop('typea')
#add a column of ones for the intercept
# dataset.insert(0, 'intercept', 1)
#We can't have the words present/absent so we replace them with 1 if present and 0 if absent
dataset['famhist'] = (dataset['famhist'] == 'Present')*1

#shuffle and split training into 80% train, 10% validation, 10% test.
training, validation, test = np.split(dataset.sample(frac=1), [int(.8*len(dataset)), int(.9*len(dataset))])
#prints out the length of each section of the dataset and the dataset itself
print("Length of dataset:", len(dataset))
print( "Length of training:", len(training))
print("Length of validation:", len(validation))
print("Length of test:", len(test))

#creates a graph like figure 4.12
#sb.pairplot(dataset, hue = 'chd',palette="hls", height = 3)
# plt.show()

#seperates the training, testing, and validation into x and y
#famhist is the y, which is the last column
#all but the last columns are the x data
#iloc is Purely integer-location based indexing for selection by position.
x_train, y_train = training.iloc[:, :-1], training.iloc[:, -1]
x_test, y_test = test.iloc[:, :-1], test.iloc[:,-1]
x_val, y_val = validation.iloc[:, :-1], validation.iloc[:,-1]

#normalize all the data
# x_train = preprocessing.normalize(x_train, axis=0)
# x_train = preprocessing.StandardScaler().fit(x_train).transform(x_train)
# x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)
# x_val = preprocessing.StandardScaler().fit(x_val).transform(x_val)
#add a column of ones for the intercept
x_train = np.concatenate((np.ones((x_train.shape[0], 1)),x_train), axis = 1)
x_test = np.concatenate((np.ones((x_test.shape[0], 1)),x_test), axis = 1)
x_val = np.concatenate((np.ones((x_val.shape[0], 1)),x_val), axis = 1)

# generate initial weight vector
N, P = x_train.shape
P -= 1
theta = np.zeros((P+1, 1))

# theta = np.array([[-4.130, 0.006, 0.080, 0.185, 0.939, -0.035, 0.001, 0.043]]).T

# reshape y_train
y_train = y_train.to_numpy().reshape((N, 1))

# FOR TESTING
# x_train = np.array([[1, 1, 1, 1, 1, 1],
#                     [0, 2, 4, 6, 8, 10],
#                     [0, 2, 0, 2, 0, 2]]).T
# y_train = np.array([[0, 1, 0, 1, 0, 1]]).T
# N, P = x_train.shape
# P -= 1
# theta = np.zeros((P+1, 1))
# END TESTING

# X: N x (P+1)
# y: N x 1
# theta: (P+1) x 1

# hypothesis function
#y_hat
# returns N x 1
def h(theta, X):
    return 1 / (1 + np.exp(-X @ theta))

# update function
# theta_j := theta_j + alpha(y_i -h_theta(x_i)) * x_i_j
# returns (P+1)
#SGD =  j+α(y(i)−hθ(x(i)))x(i)j
def grad(theta, X, y):
    # return gradient of l(theta, X, y) w.r.t. theta
    return X.T @ (y - h(theta, X)) - 2 * lam * np.linalg.norm(theta)

# log likelihood
def l(theta, X, y, lam):
    return y.T @ np.log(h(theta, X)) + (1 - y).T @ np.log(1 - h(theta, X)) - (lam * np.linalg.norm(theta ** 2))

# percent classified wrong
def pctWrong(theta, X, y):
    return np.sum(abs(np.round(h(theta, X)) - y)) / N
   # thetaJ = SGD(thetaJ, alpha, train_Y, hypothesis, train_X)  - 2* lam * thetaJ
   
# alpha = 0.001
# lam = .0000
# for i in range(10000):
#     #weights are the theta's
#     # print(grad(theta, x_train, y_train))
#     theta += alpha * grad(theta, x_train, y_train)
#     if i % 1000 == 0:
#         # alpha *= 0.95
#         print(f'iteration {i}\tclassified wrong: {np.around(pctWrong(theta, x_train, y_train),2)}\tlog likelihood: {np.around(l(theta, x_train, y_train, 0),2)}')

alpha = 0.001
lam = .0000

# adam optimizer
beta1 = 0.9
beta2 = 0.999
ztheta = np.zeros_like(theta)
zthetaSquared = np.zeros_like(theta)
ep = 0.0001

for i in range(10000):
    thetaGrad = grad(theta, x_train, y_train)
    ztheta = beta1 * ztheta + (1 - beta1) * thetaGrad
    zthetaSquared = beta2 * zthetaSquared + (1 - beta2) * thetaGrad ** 2

    theta += alpha * ztheta / (np.sqrt(zthetaSquared) + ep)

    # periodic logging info
    if i % 1000 == 0:
        print(f'iteration {i}\tclassified wrong: {np.around(pctWrong(theta, x_train, y_train),2)}\tlog likelihood: {np.around(l(theta, x_train, y_train, 0),2)}')



print('theta', np.around(theta, 3))

# print(np.hstack((h(theta, x_train), y_train)))
# print(x_train, y_train, h(theta, x_train))